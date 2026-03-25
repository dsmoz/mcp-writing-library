"""
Evidence hallucination detection tools.

Provides claim extraction, corroboration search against Zotero and Cerebellum
knowledge bases, and evidence density scoring without external search.
"""
import re
import sys
from typing import List

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Claim-detection regex patterns
# ---------------------------------------------------------------------------

_PATTERN_NUMBERS = re.compile(
    r'\b\d[\d,\.]*\s*(%|percent|per cent|pct)(?!\w)',
    re.IGNORECASE,
)

_PATTERN_EPISTEMIC = re.compile(
    r'\b(shows? that|indicates? that|demonstrates? that|evidence suggests?'
    r'|data (reveals?|shows?)|according to'
    r'|research (shows?|finds?|suggests?))\b',
    re.IGNORECASE,
)

_PATTERN_CITATION_APA = re.compile(
    r'\([A-Z][\w\s,\.]+\d{4}\)',
)

_PATTERN_CITATION_NUMERIC = re.compile(
    r'\[\d+\]',
)

_PATTERN_PREVALENCE = re.compile(
    r'\b(in Mozambique|in Angola|in SADC|among key populations?'
    r'|HIV prevalence|maternal mortality|under-five mortality'
    r'|adolescent|PLHIV)\b',
    re.IGNORECASE,
)

_ALL_CLAIM_PATTERNS = [
    _PATTERN_NUMBERS,
    _PATTERN_EPISTEMIC,
    _PATTERN_CITATION_APA,
    _PATTERN_CITATION_NUMERIC,
    _PATTERN_PREVALENCE,
]

_CITATION_PATTERNS = [_PATTERN_CITATION_APA, _PATTERN_CITATION_NUMERIC]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering out very short fragments."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= 20]


def _is_claim_sentence(sentence: str) -> bool:
    """Return True if the sentence matches any claim-bearing pattern."""
    return any(p.search(sentence) for p in _ALL_CLAIM_PATTERNS)


def _has_number(sentence: str) -> bool:
    """Return True if the sentence contains a number or percentage claim."""
    return bool(_PATTERN_NUMBERS.search(sentence))


def _has_citation(sentence: str) -> bool:
    """Return True if the sentence contains an explicit citation marker."""
    return any(p.search(sentence) for p in _CITATION_PATTERNS)


# ---------------------------------------------------------------------------
# Lazy cross-server imports with graceful degradation
# ---------------------------------------------------------------------------

def _search_zotero(query: str, top_k: int) -> List[dict]:
    """Search the Zotero knowledge base. Returns empty list on any failure."""
    try:
        sys.path.insert(0, "/Users/danilodasilva/Documents/Programming/mcp-servers/mcp-zotero-qdrant")
        from src.qdrant.search import SemanticSearch  # type: ignore
    except Exception as exc:
        logger.warning("Zotero import failed — skipping", error=str(exc))
        return []

    try:
        results = SemanticSearch(query=query, top_k=top_k)
        normalised = []
        for r in results:
            normalised.append({
                "title": r.get("title", ""),
                "citekey": r.get("citekey"),
                "score": float(r.get("score", 0)),
                "source_type": "zotero",
                "excerpt": (r.get("text", "") or "")[:200],
            })
        return normalised
    except Exception as exc:
        logger.warning("Zotero search failed — returning empty", error=str(exc))
        return []


def _search_cerebellum(query: str, top_k: int) -> List[dict]:
    """Search the Cerebellum knowledge base. Returns empty list on any failure."""
    try:
        sys.path.insert(0, "/Users/danilodasilva/Documents/Programming/mcp-servers/mcp-cerebellum")
        from tools.search import global_search  # type: ignore
    except Exception as exc:
        logger.warning("Cerebellum import failed — skipping", error=str(exc))
        return []

    try:
        results = global_search(query=query, limit=top_k)
        normalised = []
        for r in results:
            normalised.append({
                "title": r.get("title", ""),
                "citekey": None,
                "score": float(r.get("score", 0)),
                "source_type": "cerebellum",
                "excerpt": (r.get("text", "") or "")[:200],
            })
        return normalised
    except Exception as exc:
        logger.warning("Cerebellum search failed — returning empty", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_claims(
    text: str,
    domain: str = "general",
    top_k_per_claim: int = 3,
    corroboration_threshold: float = 0.65,
) -> dict:
    """
    Extract claim-bearing sentences from text and verify each against Zotero
    and Cerebellum knowledge bases.

    Args:
        text: The text to analyse for unsupported claims.
        domain: Thematic domain for context (not used for filtering; reserved
                for future per-domain thresholds).
        top_k_per_claim: Number of sources to retrieve per claim sentence.
        corroboration_threshold: Minimum score from any source to mark a
                                 claim as "verified" (default 0.65).

    Returns:
        dict with overall_evidence_score, verdict, per-claim results, and
        ghost_stat flags for unverified numeric claims.
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    sentences = _split_sentences(text)
    claim_sentences = [s for s in sentences if _is_claim_sentence(s)]

    if not claim_sentences:
        return {
            "success": True,
            "overall_evidence_score": 1.0,
            "verdict": "evidenced",
            "total_claims": 0,
            "verified_count": 0,
            "claims": [],
        }

    claims_output = []
    verified_count = 0

    for sentence in claim_sentences:
        zotero_sources = _search_zotero(sentence, top_k=top_k_per_claim)
        cerebellum_sources = _search_cerebellum(sentence, top_k=top_k_per_claim)

        all_sources = zotero_sources + cerebellum_sources
        best_score = max((s["score"] for s in all_sources), default=0.0)

        if best_score >= corroboration_threshold:
            verdict = "verified"
            ghost_stat = False
            verified_count += 1
        else:
            verdict = "unverified"
            ghost_stat = _has_number(sentence)

        claims_output.append({
            "sentence": sentence,
            "verdict": verdict,
            "ghost_stat": ghost_stat,
            "sources": all_sources,
        })

    total_claims = len(claim_sentences)
    overall_score = round(verified_count / total_claims, 4) if total_claims else 1.0

    if overall_score >= 0.8:
        overall_verdict = "evidenced"
    elif overall_score >= 0.4:
        overall_verdict = "mixed"
    else:
        overall_verdict = "unverified"

    return {
        "success": True,
        "overall_evidence_score": overall_score,
        "verdict": overall_verdict,
        "total_claims": total_claims,
        "verified_count": verified_count,
        "claims": claims_output,
    }


def score_evidence_density(text: str) -> dict:
    """
    Analyse the ratio of evidenced sentences to total sentences without
    calling any external search APIs.

    Counts claim-bearing sentences (using the same regex patterns as
    verify_claims) and sentences with explicit citation markers.

    Args:
        text: The text to score.

    Returns:
        dict with total_sentences, claim_sentences, cited_sentences,
        evidence_density, verdict, and a recommendation string.
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    sentences = _split_sentences(text)
    total = len(sentences)

    if total == 0:
        return {"success": False, "error": "No valid sentences found in text"}

    claim_count = sum(1 for s in sentences if _is_claim_sentence(s))
    cited_count = sum(1 for s in sentences if _has_citation(s))

    density = round(cited_count / claim_count, 4) if claim_count > 0 else 0.0

    if density >= 0.6:
        verdict = "well-evidenced"
        recommendation = (
            "The text has a strong citation ratio. Review remaining uncited claims "
            "for accuracy but the overall evidential foundation is solid."
        )
    elif density >= 0.3:
        verdict = "partially-evidenced"
        recommendation = (
            "Some claims are cited but others lack source attribution. "
            "Identify the uncited claim sentences and add references or qualify the language."
        )
    else:
        verdict = "under-evidenced"
        if claim_count == 0:
            recommendation = (
                "No claim-bearing sentences detected. If the text makes empirical assertions, "
                "revisit the language to ensure claims are explicit and supported."
            )
        else:
            recommendation = (
                "Most claim-bearing sentences lack explicit citations. "
                "Add references for all statistics, prevalence figures, "
                "and causal statements before submission."
            )

    return {
        "success": True,
        "total_sentences": total,
        "claim_sentences": claim_count,
        "cited_sentences": cited_count,
        "evidence_density": density,
        "verdict": verdict,
        "recommendation": recommendation,
    }
