"""
Plagiarism detection tools: internal library deduplication and external web similarity check.
"""
import math
import os
import re
from typing import List, Optional

import structlog

from src.tools.collections import get_collection_names

logger = structlog.get_logger(__name__)

# Imported here so tests can patch src.tools.plagiarism.*
try:
    from kbase.vector.sync_search import semantic_search
    from kbase.vector.sync_embeddings import generate_embedding
except ImportError:
    semantic_search = None  # type: ignore
    generate_embedding = None  # type: ignore

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore

# Environment-configurable thresholds
_DEFAULT_INTERNAL_THRESHOLD = float(os.getenv("PLAGIARISM_INTERNAL_THRESHOLD", "0.85"))
_DEFAULT_EXTERNAL_THRESHOLD = float(os.getenv("PLAGIARISM_EXTERNAL_THRESHOLD", "0.75"))
_DEFAULT_VERDICT_PCT = float(os.getenv("PLAGIARISM_VERDICT_THRESHOLD_PCT", "30.0"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, filter fragments, cap at 30."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if len(s.strip()) >= 20]
    return sentences[:30]


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity using stdlib only."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _jaccard(a: str, b: str) -> float:
    """Character-level Jaccard similarity between two strings."""
    set_a = set(a.lower())
    set_b = set(b.lower())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _select_key_sentences(sentences: List[str], n: int = 3) -> List[str]:
    """
    Select n representative sentences: longest first, then most dissimilar
    to already-selected ones (character-level Jaccard). Minimises Tavily API calls.
    """
    if len(sentences) <= n:
        return sentences

    # Start with the longest sentence
    sorted_by_len = sorted(sentences, key=len, reverse=True)
    selected = [sorted_by_len[0]]

    for candidate in sorted_by_len[1:]:
        if len(selected) >= n:
            break
        # Keep candidate if it's sufficiently different from all already-selected
        max_sim = max(_jaccard(candidate, s) for s in selected)
        if max_sim < 0.7:
            selected.append(candidate)

    # If we still don't have enough (all too similar), just take the longest ones
    if len(selected) < n:
        for s in sorted_by_len:
            if s not in selected:
                selected.append(s)
            if len(selected) >= n:
                break

    return selected


# ---------------------------------------------------------------------------
# Internal similarity check
# ---------------------------------------------------------------------------

def check_internal_similarity(
    text: str,
    threshold: float = _DEFAULT_INTERNAL_THRESHOLD,
    top_k_per_sentence: int = 3,
    verdict_threshold_pct: float = _DEFAULT_VERDICT_PCT,
    user_id: str = "default",
) -> dict:
    """
    Check if a passage is too similar to content already in the writing library.

    Splits the input into sentences and searches each against the user's
    writing_passages Qdrant collection. Returns flagged sentences and an overall similarity score.
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    collection = get_collection_names(user_id)["passages"]
    sentences = _split_sentences(text)

    if not sentences:
        return {"success": False, "error": "No valid sentences found in text (all too short)"}

    flagged_sentences = []
    seen_doc_ids: set = set()

    try:
        for sentence in sentences:
            results = semantic_search(
                collection_name=collection,
                query=sentence,
                limit=top_k_per_sentence,
                score_threshold=threshold,
            )
            if not results:
                continue

            matches = []
            for r in results:
                doc_id = r.get("document_id", "")
                seen_doc_ids.add(doc_id)
                matches.append({
                    "document_id": doc_id,
                    "score": round(r.get("score", 0), 4),
                    "excerpt": r.get("text", "")[:120],
                    "source": r.get("metadata", {}).get("source", ""),
                })

            if matches:
                flagged_sentences.append({
                    "sentence": sentence,
                    "max_score": max(m["score"] for m in matches),
                    "matches": matches,
                })

        overall_pct = round(len(flagged_sentences) / len(sentences) * 100, 1)
        verdict = "flagged" if overall_pct >= verdict_threshold_pct else "clean"

        return {
            "success": True,
            "overall_similarity_pct": overall_pct,
            "verdict": verdict,
            "verdict_threshold_pct": verdict_threshold_pct,
            "sentences_checked": len(sentences),
            "flagged_sentences": flagged_sentences,
        }

    except Exception as e:
        logger.error("Internal similarity check failed", error=str(e))
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# External similarity check
# ---------------------------------------------------------------------------

def check_external_similarity(
    text: str,
    threshold: float = _DEFAULT_EXTERNAL_THRESHOLD,
    max_sentences: int = 3,
    verdict_threshold_pct: float = _DEFAULT_VERDICT_PCT,
) -> dict:
    """
    Search the web for content similar to the given text using the Tavily API.

    Requires TAVILY_API_KEY in the environment. If not set, returns the key
    sentences and fallback instructions for manual search via mcp__MCP_DOCKER__tavily_search
    followed by a call to score_external_similarity().
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    sentences = _split_sentences(text)

    if not sentences:
        return {"success": False, "error": "No valid sentences found in text (all too short)"}

    key_sentences = _select_key_sentences(sentences, n=max_sentences)

    if not api_key:
        return {
            "success": False,
            "reason": "no_api_key",
            "message": (
                "TAVILY_API_KEY not configured. Add it to .env to enable direct external search."
            ),
            "fallback_instructions": (
                "To run the external check manually: "
                "(1) Use mcp__MCP_DOCKER__tavily_search to search each key sentence below. "
                "(2) Collect results as a list of {url, content, title} dicts. "
                "(3) Call score_external_similarity(text=<original text>, search_results=<results>)."
            ),
            "key_sentences": key_sentences,
        }

    flagged_sentences = []

    try:
        for sentence in key_sentences:
            # Query Tavily REST API
            response = _requests.post(
                "https://api.tavily.com/search",
                json={
                    "query": sentence,
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_raw_content": False,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            response.raise_for_status()
            search_data = response.json()

            matches = _score_snippets(
                sentence=sentence,
                results=search_data.get("results", []),
                threshold=threshold,
            )
            if matches:
                flagged_sentences.append({
                    "sentence": sentence,
                    "max_score": max(m["score"] for m in matches),
                    "matches": matches,
                })

        overall_pct = round(len(flagged_sentences) / len(key_sentences) * 100, 1)
        verdict = "flagged" if overall_pct >= verdict_threshold_pct else "clean"

        return {
            "success": True,
            "overall_similarity_pct": overall_pct,
            "verdict": verdict,
            "verdict_threshold_pct": verdict_threshold_pct,
            "sentences_checked": len(key_sentences),
            "flagged_sentences": flagged_sentences,
        }

    except _requests.RequestException as e:
        logger.error("Tavily API request failed", error=str(e))
        return {"success": False, "error": f"Tavily API request failed: {e}"}
    except Exception as e:
        logger.error("External similarity check failed", error=str(e))
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Score external (Option 3 companion — pre-fetched results)
# ---------------------------------------------------------------------------

def score_external_similarity(
    text: str,
    search_results: List[dict],
    threshold: float = _DEFAULT_EXTERNAL_THRESHOLD,
    verdict_threshold_pct: float = _DEFAULT_VERDICT_PCT,
) -> dict:
    """
    Score similarity between text and pre-fetched web search results.

    Use this after manually searching with mcp__MCP_DOCKER__tavily_search
    when TAVILY_API_KEY is not configured. Pass the search results directly.

    Args:
        text: The text to check
        search_results: List of {url, content, title} dicts from a Tavily search
        threshold: Cosine similarity threshold for flagging (default 0.75)
        verdict_threshold_pct: Overall % of sentences flagged to trigger "flagged" verdict

    Returns:
        Same structure as check_external_similarity
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    if not search_results:
        return {
            "success": True,
            "overall_similarity_pct": 0.0,
            "verdict": "clean",
            "verdict_threshold_pct": verdict_threshold_pct,
            "sentences_checked": 0,
            "flagged_sentences": [],
        }

    sentences = _split_sentences(text)
    if not sentences:
        return {"success": False, "error": "No valid sentences found in text (all too short)"}

    flagged_sentences = []

    try:
        for sentence in sentences:
            matches = _score_snippets(
                sentence=sentence,
                results=search_results,
                threshold=threshold,
            )
            if matches:
                flagged_sentences.append({
                    "sentence": sentence,
                    "max_score": max(m["score"] for m in matches),
                    "matches": matches,
                })

        overall_pct = round(len(flagged_sentences) / len(sentences) * 100, 1)
        verdict = "flagged" if overall_pct >= verdict_threshold_pct else "clean"

        return {
            "success": True,
            "overall_similarity_pct": overall_pct,
            "verdict": verdict,
            "verdict_threshold_pct": verdict_threshold_pct,
            "sentences_checked": len(sentences),
            "flagged_sentences": flagged_sentences,
        }

    except Exception as e:
        logger.error("score_external_similarity failed", error=str(e))
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Internal helper: embed + score snippets against a sentence
# ---------------------------------------------------------------------------

def _score_snippets(sentence: str, results: List[dict], threshold: float) -> List[dict]:
    """
    Embed a sentence and each result snippet, compute cosine similarity,
    and return matches above threshold.
    """
    try:
        sentence_vec = generate_embedding(sentence)
    except Exception as e:
        logger.warning("Could not embed sentence for scoring", error=str(e))
        return []

    matches = []
    for result in results:
        snippet = result.get("content", "") or result.get("snippet", "")
        if not snippet:
            continue
        try:
            snippet_vec = generate_embedding(snippet[:500])  # cap snippet length
        except Exception:
            continue

        score = _cosine(sentence_vec, snippet_vec)
        if score >= threshold:
            matches.append({
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "score": round(score, 4),
                "snippet": snippet[:200],
            })

    # Sort by score descending
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches
