"""
Multi-author voice consistency scorer.

Measures how consistently text sections match a target style profile
and detects authorship shifts within a single document.
"""
import statistics
from typing import List, Optional
import structlog

logger = structlog.get_logger(__name__)

try:
    from kbase.vector.sync_embeddings import generate_embedding
except ImportError:
    generate_embedding = None  # type: ignore

try:
    from kbase.vector.sync_client import get_qdrant_client
except ImportError:
    get_qdrant_client = None  # type: ignore


def _cosine(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x ** 2 for x in a) ** 0.5
    nb = sum(x ** 2 for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _jaccard(a: str, b: str) -> float:
    """Word overlap similarity as fallback when embeddings are unavailable."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def _centroid(embeddings: List[list]) -> list:
    """Compute element-wise mean of a list of vectors."""
    dim = len(embeddings[0])
    result = [0.0] * dim
    n = len(embeddings)
    for emb in embeddings:
        for i, v in enumerate(emb):
            result[i] += v / n
    return result


def score_voice_consistency(
    sections: List[str],
    profile_name: Optional[str] = None,
    top_k_profile: int = 1,
) -> dict:
    """
    Measure how consistently a list of text sections share a voice/style.

    Args:
        sections: List of text sections to compare (2–20 sections).
        profile_name: If provided, compare each section against this saved style profile.
                      If None, sections are compared against each other only.
        top_k_profile: How many top profile matches to return (default 1).
                       Only relevant when profile_name is None.

    Returns:
        {
            success, section_count, inter_section_consistency, consistency_verdict,
            profile_name, profile_consistency, profile_verdict,
            sections: [{index, preview, drift_score, profile_score}],
            highest_drift_section, scoring_method
        }
    """
    # Validate section count
    if not sections or len(sections) < 2:
        return {"success": False, "error": "sections must contain at least 2 items"}
    if len(sections) > 20:
        return {"success": False, "error": "sections must contain at most 20 items"}

    scoring_method = "embedding"

    # Step 1: Embed each section
    if generate_embedding is not None:
        try:
            embeddings = [generate_embedding(s) for s in sections]
        except Exception as e:
            logger.warning("Embedding failed, falling back to Jaccard", error=str(e))
            embeddings = None
            scoring_method = "fallback"
    else:
        embeddings = None
        scoring_method = "fallback"

    # Step 2: Compute pairwise similarities
    n = len(sections)
    profile_scores = None
    profile_consistency = None
    profile_verdict = None

    if scoring_method == "embedding" and embeddings is not None:
        # Pairwise cosine similarity matrix (upper triangle)
        pair_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_sims.append(_cosine(embeddings[i], embeddings[j]))

        inter_section_consistency = sum(pair_sims) / len(pair_sims) if pair_sims else 1.0

        # Step 3: Centroid and drift scores
        cent = _centroid(embeddings)
        drift_scores = [1.0 - _cosine(emb, cent) for emb in embeddings]

        # Step 4: Profile comparison
        if profile_name is not None:
            from src.tools.style_profiles import load_style_profile
            profile_result = load_style_profile(name=profile_name)
            if not profile_result.get("success"):
                return {
                    "success": False,
                    "error": f"Style profile '{profile_name}' not found",
                }
            profile_data = profile_result["profile"]
            excerpts = profile_data.get("sample_excerpts", [])
            excerpt_text = "\n".join(excerpts) if excerpts else profile_data.get("description", "")
            try:
                profile_embedding = generate_embedding(excerpt_text)
                profile_scores = [_cosine(emb, profile_embedding) for emb in embeddings]
                profile_consistency = sum(profile_scores) / len(profile_scores)
            except Exception as e:
                logger.warning("Profile embedding failed", error=str(e))
                profile_scores = None
                profile_consistency = None

    else:
        # Fallback: Jaccard similarity
        pair_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_sims.append(_jaccard(sections[i], sections[j]))
        inter_section_consistency = sum(pair_sims) / len(pair_sims) if pair_sims else 1.0

        # Drift via Jaccard: compare each section to concatenation of all others
        drift_scores = []
        for i in range(n):
            others = " ".join(s for k, s in enumerate(sections) if k != i)
            drift_scores.append(1.0 - _jaccard(sections[i], others))

        # Profile comparison skipped in fallback mode
        if profile_name is not None:
            from src.tools.style_profiles import load_style_profile
            profile_result = load_style_profile(name=profile_name)
            if not profile_result.get("success"):
                return {
                    "success": False,
                    "error": f"Style profile '{profile_name}' not found",
                }
            # Profile comparison unavailable in fallback mode
            profile_scores = None
            profile_consistency = None

    # Consistency verdict
    if inter_section_consistency >= 0.7:
        consistency_verdict = "consistent"
    elif inter_section_consistency >= 0.5:
        consistency_verdict = "moderate"
    else:
        consistency_verdict = "inconsistent"

    # Profile verdict
    if profile_consistency is not None:
        if profile_consistency >= 0.65:
            profile_verdict = "on-voice"
        elif profile_consistency >= 0.45:
            profile_verdict = "near-voice"
        else:
            profile_verdict = "off-voice"

    # Highest drift section
    highest_drift_section = drift_scores.index(max(drift_scores))

    # Build per-section output
    sections_out = []
    for i, section in enumerate(sections):
        sections_out.append({
            "index": i,
            "preview": section[:80],
            "drift_score": round(drift_scores[i], 4),
            "profile_score": round(profile_scores[i], 4) if profile_scores is not None else None,
        })

    return {
        "success": True,
        "section_count": n,
        "inter_section_consistency": round(inter_section_consistency, 4),
        "consistency_verdict": consistency_verdict,
        "profile_name": profile_name,
        "profile_consistency": round(profile_consistency, 4) if profile_consistency is not None else None,
        "profile_verdict": profile_verdict,
        "sections": sections_out,
        "highest_drift_section": highest_drift_section,
        "scoring_method": scoring_method,
    }


def detect_authorship_shift(
    text: str,
    min_segment_length: int = 100,
) -> dict:
    """
    Split a document into segments and flag those that deviate stylistically
    from the majority — suggesting a possible authorship change.

    Args:
        text: Full document text.
        min_segment_length: Minimum character length per segment (default 100).

    Returns:
        {
            success, total_segments, mean_deviation, std_deviation,
            shifted_segments: [{index, preview, deviation, z_score}],
            shift_detected, scoring_method
        }

    Note: Shift detection becomes statistically unreliable with fewer than 5 segments.
    With exactly 3 segments, the 1.5×std threshold may exceed the maximum possible
    deviation, causing real shifts to go undetected. Prefer 5+ segments for reliable
    results.
    """
    # Segment the text
    raw_segments = text.split("\n\n")
    segments = [s.strip() for s in raw_segments if len(s.strip()) >= min_segment_length]

    if len(segments) < 3:
        return {
            "success": False,
            "error": "Not enough segments for authorship analysis (need at least 3)",
        }

    scoring_method = "embedding"

    # Embed segments
    if generate_embedding is not None:
        try:
            embeddings = [generate_embedding(seg) for seg in segments]
        except Exception as e:
            logger.warning("Embedding failed, falling back to Jaccard", error=str(e))
            embeddings = None
            scoring_method = "fallback"
    else:
        embeddings = None
        scoring_method = "fallback"

    # Compute deviations
    if scoring_method == "embedding" and embeddings is not None:
        cent = _centroid(embeddings)
        deviations = [1.0 - _cosine(emb, cent) for emb in embeddings]
    else:
        # Jaccard fallback: deviation = 1 - similarity to all-others concatenation
        scoring_method = "fallback"
        deviations = []
        for i, seg in enumerate(segments):
            others = " ".join(s for k, s in enumerate(segments) if k != i)
            deviations.append(1.0 - _jaccard(seg, others))

    mean_dev = statistics.mean(deviations)
    std_dev = statistics.stdev(deviations) if len(deviations) > 1 else 0.0

    threshold = mean_dev + 1.5 * std_dev

    shifted = []
    for i, dev in enumerate(deviations):
        if dev > threshold:
            z = (dev - mean_dev) / std_dev if std_dev > 0 else 0.0
            shifted.append({
                "index": i,
                "preview": segments[i][:80],
                "deviation": round(dev, 4),
                "z_score": round(z, 4),
            })

    return {
        "success": True,
        "total_segments": len(segments),
        "mean_deviation": round(mean_dev, 4),
        "std_deviation": round(std_dev, 4),
        "shifted_segments": shifted,
        "shift_detected": len(shifted) > 0,
        "scoring_method": scoring_method,
    }
