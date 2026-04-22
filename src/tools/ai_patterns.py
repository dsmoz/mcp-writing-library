"""
AI writing pattern scorer.

Scores text against known AI-generated prose patterns to identify content that may
be flagged by AI detectors or simply read as robotic. All detection is rule-based —
no external services or Qdrant needed.

Categories detected:
    connector_repetition  — Overused connectors: Furthermore, Additionally, Moreover, etc.
    hollow_intensifiers   — "It is important to note that", "It is crucial that", etc.
    grandiose_openers     — Dramatic paragraph openings typical of AI prose (EN)
    grandiose_openers_pt  — Same pattern in Portuguese
    em_dash_intercalation — Paired em-dashes as parenthetical inserts (AI pattern in PT)
    sentence_monotony     — 3+ consecutive sentences of similar length (±3 words)
    passive_voice         — High density of passive constructions (>25% of sentences)
    paragraph_length      — Paragraphs exceeding configurable per doc_type (default: 5) sentences
    discursive_deficit    — Fewer than configurable per doc_type (default: 1.0/page) discursive expressions
    mechanical_listing    — Paragraph openers: Firstly, Secondly, Thirdly, Finally
    generic_closings      — "In conclusion, this report has shown...", "To summarise..."
"""

import re
from typing import List, Optional, Tuple

import structlog

from src.sentry import capture_tool_error
from src.tools.pattern_store import load_items, load_values
from src.tools.qdrant_errors import handle_qdrant_error

logger = structlog.get_logger(__name__)

try:
    from kbase.vector.sync_search import semantic_search
except ImportError:
    semantic_search = None  # type: ignore


# ---------------------------------------------------------------------------
# Pattern definitions — loaded from data/patterns/*.json via pattern_store
# (two-layer: core defaults + per-user overrides). Seeded by
# scripts/seed_patterns.py; mutated at runtime via manage_patterns.
# ---------------------------------------------------------------------------

def _connectors_en(client_id: str = "default") -> List[str]:
    return load_items("connectors_en", client_id)

def _connectors_pt(client_id: str = "default") -> List[str]:
    return load_items("connectors_pt", client_id)

def _hollow_intensifiers(client_id: str = "default") -> List[str]:
    return load_items("hollow_intensifiers", client_id)

def _grandiose_openers_en(client_id: str = "default") -> List[str]:
    return load_items("grandiose_openers_en", client_id)

def _grandiose_openers_pt(client_id: str = "default") -> List[str]:
    return load_items("grandiose_openers_pt", client_id)

def _generic_closings(client_id: str = "default") -> List[str]:
    return load_items("generic_closings", client_id)

def _discursive_expressions(client_id: str = "default") -> List[str]:
    return load_items("discursive_expressions", client_id)

def _passive_patterns(client_id: str = "default") -> List[str]:
    return load_items("passive_patterns", client_id)

def _pt_function_words(client_id: str = "default") -> set:
    return set(load_items("pt_function_words", client_id))

def _para_limits(client_id: str = "default") -> dict:
    return load_values("para_limits", client_id)

def _discursive_targets(client_id: str = "default") -> dict:
    return load_values("discursive_targets", client_id)

def _hedging_words(language: str, client_id: str = "default") -> List[str]:
    words = load_items("hedging_words_en", client_id)
    if language == "pt":
        words = words + load_items("hedging_words_pt", client_id)
    return words

def _hedging_targets(client_id: str = "default") -> dict:
    return load_values("hedging_targets", client_id)

def _config(client_id: str = "default") -> dict:
    return load_values("config", client_id)

# Social doc_types where discursive_deficit is structurally inapplicable
_SOCIAL_DOC_TYPES = frozenset({"facebook-post", "linkedin-post", "instagram-caption"})

# Creative doc_types — prose pattern detectors are inapplicable or misleading
_CREATIVE_DOC_TYPES = frozenset({
    "haiku", "sonnet", "free-verse", "villanelle", "spoken-word",
    "pop-song", "ballad", "rap-verse", "hymn", "jingle",
    "novel-chapter", "short-story", "flash-fiction", "screenplay",
    "creative-nonfiction",
})


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _detect_language(text: str, client_id: str = "default") -> str:
    """Simple heuristic: count PT function words in lowercased text."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return "en"
    pt_words = _pt_function_words(client_id)
    pt_count = sum(1 for w in words if w in pt_words)
    ratio = pt_count / len(words)
    return "pt" if ratio >= 0.05 else "en"


# ---------------------------------------------------------------------------
# Text segmentation helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split into sentences, filter fragments under 10 chars."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= 10]


def _split_paragraphs(text: str, stanza_mode: bool = False) -> List[str]:
    """Split on blank lines. stanza_mode uses single blank line (poetry/songs)."""
    sep = r"\n[ \t]*\n" if stanza_mode else r"\n\s*\n"
    paras = re.split(sep, text.strip())
    return [p.strip() for p in paras if p.strip()]


def _sentence_word_count(sentence: str) -> int:
    return len(sentence.split())


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _detect_connector_repetition(text: str, language: str, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Flag connectors that appear more than once."""
    en = _connectors_en(client_id)
    connectors = en if language == "en" else en + _connectors_pt(client_id)
    lower = text.lower()
    findings = []
    total_excess_hits = 0

    for connector in connectors:
        pattern = r"\b" + re.escape(connector) + r"\b"
        matches = list(re.finditer(pattern, lower))
        if len(matches) > 1:
            total_excess_hits += len(matches) - 1  # hits above the first are the excess
            for m in matches[:3]:  # show first 3 occurrences max
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 60)
                findings.append({
                    "excerpt": "..." + text[start:end].strip() + "...",
                    "pattern": connector,
                    "count": len(matches),
                })

    sentences = _split_sentences(text)
    sentence_count = max(len(sentences), 1)
    score = min(1.0, total_excess_hits / sentence_count)
    return round(score, 3), findings


def _detect_hollow_intensifiers(text: str, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Detect hollow intensifier phrases."""
    lower = text.lower()
    findings = []
    hit_count = 0

    for pattern in _hollow_intensifiers(client_id):
        for m in re.finditer(pattern, lower):
            hit_count += 1
            start = max(0, m.start())
            end = min(len(text), m.end() + 60)
            findings.append({"excerpt": text[start:end].strip() + "...", "pattern": pattern})

    sentences = _split_sentences(text)
    sentence_count = max(len(sentences), 1)
    score = min(1.0, hit_count / (sentence_count * 0.15))
    return round(score, 3), findings


def _detect_grandiose_openers(text: str, language: str, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Detect grandiose paragraph openings."""
    patterns = _grandiose_openers_pt(client_id) if language == "pt" else _grandiose_openers_en(client_id)
    paragraphs = _split_paragraphs(text)
    findings = []
    hit_count = 0

    for para in paragraphs:
        first_sentence = re.split(r"(?<=[.!?])\s+", para.strip())[0]
        lower = first_sentence.lower()
        for pat in patterns:
            if re.search(pat, lower):
                hit_count += 1
                findings.append({
                    "excerpt": first_sentence[:150],
                    "pattern": pat,
                })
                break  # one match per paragraph

    para_count = max(len(paragraphs), 1)
    score = min(1.0, hit_count / para_count)
    return round(score, 3), findings


def _detect_em_dash_intercalation(text: str) -> Tuple[float, List[dict]]:
    """Detect paired em-dashes used as parenthetical inserts."""
    # Match — ... — or --- ... ---
    patterns = [
        r"—[^—\n]{3,60}—",
        r"---[^-\n]{3,60}---",
        r"--[^-\n]{3,60}--",
    ]
    findings = []
    hit_count = 0

    for pat in patterns:
        for m in re.finditer(pat, text):
            hit_count += 1
            findings.append({"excerpt": m.group(0)[:120]})

    sentences = _split_sentences(text)
    sentence_count = max(len(sentences), 1)
    score = min(1.0, hit_count / (sentence_count * 0.1))
    return round(score, 3), findings


def _detect_sentence_monotony(text: str) -> Tuple[float, List[dict]]:
    """Detect 3+ consecutive sentences of similar length (±3 words)."""
    sentences = _split_sentences(text)
    findings = []
    monotone_runs = 0

    i = 0
    while i < len(sentences):
        run = [sentences[i]]
        base_len = _sentence_word_count(sentences[i])
        j = i + 1
        while j < len(sentences):
            candidate_len = _sentence_word_count(sentences[j])
            if abs(candidate_len - base_len) <= 3:
                run.append(sentences[j])
                j += 1
            else:
                break
        if len(run) >= 3:
            monotone_runs += 1
            findings.append({
                "excerpt": " ".join(run[:3])[:200],
                "run_length": len(run),
                "approx_length": f"~{base_len} words each",
            })
        i = j if j > i else i + 1

    sentences_count = max(len(sentences), 1)
    score = min(1.0, monotone_runs / max(sentences_count / 5, 1))
    return round(score, 3), findings


def _detect_passive_voice(text: str, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Detect high passive voice density (>25% of sentences)."""
    sentences = _split_sentences(text)
    findings = []
    passive_count = 0
    passive_pats = _passive_patterns(client_id)

    for sentence in sentences:
        lower = sentence.lower()
        is_passive = any(re.search(pat, lower) for pat in passive_pats)
        if is_passive:
            passive_count += 1
            findings.append({"excerpt": sentence[:150]})

    sentence_count = max(len(sentences), 1)
    passive_ratio = passive_count / sentence_count
    score = max(0.0, (passive_ratio - 0.25) / 0.75)  # 0 at 25%, 1.0 at 100%
    return round(score, 3), findings if passive_ratio > 0.25 else []


def _detect_paragraph_length(text: str, max_sentences: int = 5, stanza_mode: bool = False) -> Tuple[float, List[dict]]:
    """Detect paragraphs exceeding max_sentences sentences."""
    paragraphs = _split_paragraphs(text, stanza_mode=stanza_mode)
    findings = []
    violation_count = 0

    for para in paragraphs:
        sentences = _split_sentences(para)
        if len(sentences) > max_sentences:
            violation_count += 1
            findings.append({
                "excerpt": para[:200] + "...",
                "sentence_count": len(sentences),
            })

    para_count = max(len(paragraphs), 1)
    score = min(1.0, violation_count / para_count)
    return round(score, 3), findings


def _detect_discursive_deficit(text: str, target: float = 1.0, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Detect fewer than target discursive expressions per ~300-word page."""
    if target == 0.0:
        return 0.0, []

    lower = text.lower()
    word_count = len(text.split())
    pages = max(word_count / 300, 1)

    hit_count = 0
    for pattern in _discursive_expressions(client_id):
        if re.search(pattern, lower):
            hit_count += 1

    density = hit_count / pages

    if density >= target:
        return 0.0, []

    score = min(1.0, (target - density) / target)
    findings = [{
        "detail": f"{round(density, 1)} discursive expressions per page (target: ≥{target})",
        "count_found": hit_count,
        "page_equivalent": round(pages, 1),
    }]
    return round(score, 3), findings


def _detect_mechanical_listing(text: str) -> Tuple[float, List[dict]]:
    """Detect mechanical paragraph openers: Firstly, Secondly, Thirdly, Finally."""
    openers = [
        r"^firstly[,:]", r"^secondly[,:]", r"^thirdly[,:]", r"^fourthly[,:]",
        r"^finally[,:]", r"^lastly[,:]",
        r"^em primeiro lugar", r"^em segundo lugar", r"^em terceiro lugar",
        r"^em (?:último|ultimo) lugar", r"^por último[,:]", r"^por ultimo[,:]",
    ]
    paragraphs = _split_paragraphs(text)
    findings = []
    hit_count = 0

    for para in paragraphs:
        first_line = para.strip().split("\n")[0].strip().lower()
        for pat in openers:
            if re.match(pat, first_line):
                hit_count += 1
                findings.append({"excerpt": para[:100]})
                break

    para_count = max(len(paragraphs), 1)
    score = min(1.0, hit_count / para_count)
    return round(score, 3), findings


def _detect_generic_closings(text: str, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Detect generic AI closing phrases."""
    lower = text.lower()
    findings = []
    hit_count = 0

    for pattern in _generic_closings(client_id):
        for m in re.finditer(pattern, lower):
            hit_count += 1
            start = m.start()
            end = min(len(text), m.end() + 80)
            findings.append({"excerpt": text[start:end].strip()})

    score = min(1.0, hit_count * 0.5)  # 2 hits = score 1.0
    return round(score, 3), findings


def _detect_sentence_burstiness(text: str, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Flag low burstiness: sentence lengths too uniform (CoV below threshold).

    AI-generated prose tends to produce sentences of similar length; human prose
    varies more. Coefficient of variation (stdev/mean) below `burstiness_cov_threshold`
    is scored proportionally. Fewer than 3 sentences → 0.0 (insufficient signal).
    """
    from statistics import mean, pstdev

    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return 0.0, []
    lengths = [_sentence_word_count(s) for s in sentences]
    m = mean(lengths)
    if m == 0:
        return 0.0, []
    cov = pstdev(lengths) / m
    threshold = _config(client_id).get("burstiness_cov_threshold", 0.45)
    if cov >= threshold:
        return 0.0, []
    score = max(0.0, 1.0 - (cov / threshold))
    findings = [{
        "cov": round(cov, 3),
        "mean_words_per_sentence": round(m, 1),
        "threshold": threshold,
        "sentence_count": len(sentences),
    }]
    return round(score, 3), findings


def _detect_hedging_removal(text: str, language: str, doc_type: str, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Flag documents where epistemic hedges are absent.

    AI prose strips hedges ("arguably", "perhaps", "tends to") in favour of
    declarative certainty. Density per 300-word page is compared against a
    per-doc-type target; density below target is scored linearly.
    """
    target = _hedging_targets(client_id).get(doc_type, 1.0)
    if target == 0.0:
        return 0.0, []

    words = _hedging_words(language, client_id)
    lower = text.lower()
    pages = max(len(text.split()) / 300, 1)
    count = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", lower)) for w in words)
    density = count / pages
    if density >= target:
        return 0.0, []
    score = min(1.0, (target - density) / target)
    findings = [{
        "density": round(density, 2),
        "target": target,
        "count_found": count,
        "page_equivalent": round(pages, 1),
    }]
    return round(score, 3), findings


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_ai_patterns(
    text: str,
    language: str = "auto",
    threshold: float = 0.25,
    doc_type: str = "general",
    client_id: str = "default",
) -> dict:
    """
    Score text against known AI writing patterns.

    Args:
        text: The text to score (full document or section)
        language: "en", "pt", or "auto" (auto-detects; default: "auto")
        threshold: Per-category score above which a category is flagged (default: 0.25)
        doc_type: Document type for threshold calibration. One of: concept-note,
                  full-proposal, eoi, executive-summary, general, annual-report,
                  monitoring-report, financial-report, assessment, tor,
                  governance-review, facebook-post, linkedin-post,
                  instagram-caption. Default: "general".

    Returns:
        dict with overall_score, verdict, per-category scores and findings,
        summary, word_count, page_equivalent, doc_type.
        Verdict: "clean" (<0.25), "review" (0.25–0.55), "ai-sounding" (≥0.55)
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    if language not in ("en", "pt", "auto"):
        return {"success": False, "error": f"Invalid language '{language}'. Must be 'en', 'pt', or 'auto'."}

    # Coerce threshold to float and validate range
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return {"success": False, "error": f"Invalid threshold '{threshold}'. Must be a number between 0.0 and 1.0."}
    if not (0.0 <= threshold <= 1.0):
        return {"success": False, "error": f"Invalid threshold {threshold}. Must be between 0.0 and 1.0."}

    para_limits_map = _para_limits(client_id)
    if doc_type not in para_limits_map:
        valid = ", ".join(sorted(para_limits_map.keys()))
        return {"success": False, "error": f"Invalid doc_type '{doc_type}'. Must be one of: {valid}."}

    detected_language = _detect_language(text, client_id) if language == "auto" else language

    word_count = len(text.split())
    page_equivalent = round(word_count / 300, 1)

    para_limit = para_limits_map.get(doc_type, 5)
    discursive_target = _discursive_targets(client_id).get(doc_type, 1.0)

    is_creative = doc_type in _CREATIVE_DOC_TYPES
    stanza_mode = doc_type in {
        "haiku", "sonnet", "free-verse", "villanelle", "spoken-word",
        "pop-song", "ballad", "rap-verse", "hymn", "jingle",
    }

    try:
        # Run all detectors
        connector_score, connector_findings = _detect_connector_repetition(text, detected_language, client_id)
        intensifier_score, intensifier_findings = _detect_hollow_intensifiers(text, client_id)
        if is_creative:
            grandiose_score, grandiose_findings = 0.0, []
        else:
            grandiose_score, grandiose_findings = _detect_grandiose_openers(text, detected_language, client_id)
        em_dash_score, em_dash_findings = _detect_em_dash_intercalation(text)
        monotony_score, monotony_findings = _detect_sentence_monotony(text)
        if is_creative:
            passive_score, passive_findings = 0.0, []
        else:
            passive_score, passive_findings = _detect_passive_voice(text, client_id)
        para_len_score, para_len_findings = _detect_paragraph_length(text, max_sentences=para_limit, stanza_mode=stanza_mode)
        if doc_type in _SOCIAL_DOC_TYPES or is_creative:
            discursive_score, discursive_findings = 0.0, []
        else:
            discursive_score, discursive_findings = _detect_discursive_deficit(text, target=discursive_target, client_id=client_id)
        if is_creative:
            listing_score, listing_findings = 0.0, []
            closing_score, closing_findings = 0.0, []
        else:
            listing_score, listing_findings = _detect_mechanical_listing(text)
            closing_score, closing_findings = _detect_generic_closings(text, client_id)

        # New detectors (Phase 2): burstiness is a universal signal; hedging is
        # skipped for social/creative doc_types where epistemic hedges are not
        # stylistically expected.
        burstiness_score, burstiness_findings = _detect_sentence_burstiness(text, client_id)
        if doc_type in _SOCIAL_DOC_TYPES or is_creative:
            hedging_score, hedging_findings = 0.0, []
        else:
            hedging_score, hedging_findings = _detect_hedging_removal(text, detected_language, doc_type, client_id)

        categories = {
            "connector_repetition": {"score": connector_score, "findings": connector_findings},
            "hollow_intensifiers": {"score": intensifier_score, "findings": intensifier_findings},
            "grandiose_openers": {"score": grandiose_score, "findings": grandiose_findings},
            "em_dash_intercalation": {"score": em_dash_score, "findings": em_dash_findings},
            "sentence_monotony": {"score": monotony_score, "findings": monotony_findings},
            "passive_voice": {"score": passive_score, "findings": passive_findings},
            "paragraph_length": {"score": para_len_score, "findings": para_len_findings},
            "discursive_deficit": {"score": discursive_score, "findings": discursive_findings},
            "mechanical_listing": {"score": listing_score, "findings": listing_findings},
            "generic_closings": {"score": closing_score, "findings": closing_findings},
            "sentence_burstiness": {"score": burstiness_score, "findings": burstiness_findings},
            "hedging_removal": {"score": hedging_score, "findings": hedging_findings},
        }

        scores = [v["score"] for v in categories.values()]
        overall_score = round(sum(scores) / len(scores), 3)

        if overall_score < 0.25:
            verdict = "clean"
        elif overall_score < 0.55:
            verdict = "review"
        else:
            verdict = "ai-sounding"

        flagged = [cat for cat, data in categories.items() if data["score"] >= threshold]
        if flagged:
            summary = f"{len(flagged)} categor{'y' if len(flagged) == 1 else 'ies'} flagged: {', '.join(flagged)}."
        else:
            summary = "No categories flagged above threshold."

        return {
            "success": True,
            "language": detected_language,
            "overall_score": overall_score,
            "verdict": verdict,
            "threshold": threshold,
            "doc_type": doc_type,
            "categories": categories,
            "summary": summary,
            "word_count": word_count,
            "page_equivalent": page_equivalent,
        }

    except Exception as e:
        logger.error("score_ai_patterns failed", error=str(e))
        capture_tool_error(e, tool_name="score_ai_patterns", doc_type=doc_type)
        return {"success": False, "error": str(e)}


def score_semantic_ai_likelihood(
    text: str,
    top_k: int = 10,
    client_id: str = "default",
) -> dict:
    """
    Score how semantically similar text is to known AI-corrected passages vs.
    human-corrected passages stored in the library via record_correction().

    Embeds the input and retrieves the top_k nearest neighbours from each
    labelled sub-corpus (style='ai-corrected' and style='human-corrected').
    Returns the mean similarity to each sub-corpus and a likelihood score
    (0.0 = human-like, 1.0 = AI-like).

    Requires at least 1 stored passage in each sub-corpus; degrades gracefully
    when the library is empty or unreachable.

    Args:
        text: The passage to score
        top_k: Number of neighbours to retrieve per sub-corpus (default 10)

    Returns:
        {
            success, likelihood (0–1), verdict (human-like|ambiguous|ai-like),
            ai_mean_similarity, human_mean_similarity,
            ai_sample_count, human_sample_count,
            method ("semantic" | "insufficient_data"),
            note (present when method=="insufficient_data")
        }
    """
    if semantic_search is None:
        return {
            "success": False,
            "error": "kbase not available — semantic scoring requires the kbase library",
        }

    try:
        from src.tools.collections import get_collection_names
    except ImportError:
        return {"success": False, "error": "collections module not available"}

    collection = get_collection_names(client_id)["passages"]

    def _mean_similarity(results: list) -> Optional[float]:
        scores = [r["score"] for r in results if "score" in r]
        return sum(scores) / len(scores) if scores else None

    try:
        # Search within each labelled sub-corpus using style as a post-filter
        # (kbase-core filter_conditions does not support list-field matching,
        # so we over-fetch and post-filter, same pattern as search_passages)
        fetch_k = top_k * 4  # over-fetch to survive post-filter attrition

        raw_ai = semantic_search(
            collection_name=collection,
            query=text,
            limit=fetch_k,
            filter_conditions={"entry_type": "correction"},
        )
        raw_human = semantic_search(
            collection_name=collection,
            query=text,
            limit=fetch_k,
            filter_conditions={"entry_type": "correction"},
        )

        # Post-filter by style tag
        ai_results = [
            r for r in raw_ai
            if "ai-corrected" in r.get("metadata", {}).get("style", [])
        ][:top_k]
        human_results = [
            r for r in raw_human
            if "human-corrected" in r.get("metadata", {}).get("style", [])
        ][:top_k]

        ai_mean = _mean_similarity(ai_results)
        human_mean = _mean_similarity(human_results)

        if ai_mean is None or human_mean is None:
            return {
                "success": True,
                "likelihood": None,
                "verdict": None,
                "ai_mean_similarity": ai_mean,
                "human_mean_similarity": human_mean,
                "ai_sample_count": len(ai_results),
                "human_sample_count": len(human_results),
                "method": "insufficient_data",
                "note": (
                    "Not enough labelled samples to score. "
                    "Call record_correction() to build the correction corpus."
                ),
            }

        # Likelihood: proportion of similarity explained by the AI sub-corpus
        total = ai_mean + human_mean
        likelihood = round(ai_mean / total, 4) if total > 0 else 0.5

        if likelihood >= 0.55:
            verdict = "ai-like"
        elif likelihood <= 0.45:
            verdict = "human-like"
        else:
            verdict = "ambiguous"

        return {
            "success": True,
            "likelihood": likelihood,
            "verdict": verdict,
            "ai_mean_similarity": round(ai_mean, 4),
            "human_mean_similarity": round(human_mean, 4),
            "ai_sample_count": len(ai_results),
            "human_sample_count": len(human_results),
            "method": "semantic",
        }

    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="score_semantic_ai_likelihood", collection=collection)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("score_semantic_ai_likelihood failed", error=str(e))
        capture_tool_error(e, tool_name="score_semantic_ai_likelihood")
        return {"success": False, "error": str(e)}
