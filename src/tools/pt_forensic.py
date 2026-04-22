"""Portuguese-specific AI-writing forensic scorer.

Complements ``score_ai_patterns`` with three PT-specific detectors:

- ``juridiques_density``  — bureaucratic/legal jargon over-used by AI in PT
- ``synthetic_passive``   — ``foi realizado``/``são implementadas`` constructions
- ``nominalisation_density`` — suffix-heavy, verb-poor prose (-ção/-mento/-dade)

All patterns load from ``data/patterns/*.json`` via ``pattern_store`` so they
can be tuned per-user at runtime via ``manage_patterns``.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import structlog

from src.sentry import capture_tool_error
from src.tools.ai_patterns import _split_sentences
from src.tools.pattern_store import load_items, load_values

logger = structlog.get_logger(__name__)


def _juridiques_terms(client_id: str = "default") -> List[str]:
    return load_items("juridiques_terms", client_id)


def _pt_passive_patterns(client_id: str = "default") -> List[str]:
    return load_items("pt_passive_patterns", client_id)


def _nominalisation_suffixes(client_id: str = "default") -> List[str]:
    return load_items("nominalisation_suffixes", client_id)


def _config(client_id: str = "default") -> dict:
    return load_values("config", client_id)


def _detect_juridiques(text: str, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Flag bureaucratic / legalistic PT jargon.

    Score: ``min(1.0, hits_per_page / max_hits_per_page)`` using the
    ``juridiques_hits_per_page_max`` config value as the saturation point.
    """
    terms = _juridiques_terms(client_id)
    lower = text.lower()
    findings: List[dict] = []
    hit_count = 0

    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        matches = list(re.finditer(pattern, lower))
        if matches:
            hit_count += len(matches)
            for m in matches[:2]:
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 60)
                findings.append({
                    "term": term,
                    "count": len(matches),
                    "excerpt": "..." + text[start:end].strip() + "...",
                })

    word_count = len(text.split())
    pages = max(word_count / 300, 1)
    max_per_page = _config(client_id).get("juridiques_hits_per_page_max", 2.0)
    density = hit_count / pages
    score = min(1.0, density / max_per_page) if max_per_page > 0 else 0.0
    if score == 0.0:
        findings = []
    return round(score, 3), findings


def _detect_synthetic_passive_pt(text: str, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Flag synthetic-passive density (PT): ``foi realizado``, ``são implementadas``, etc.

    Mirrors the EN passive detector: 0.0 below 25% of sentences, linear to 1.0
    at 100%.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0, []
    patterns = _pt_passive_patterns(client_id)
    findings: List[dict] = []
    passive_count = 0
    for s in sentences:
        lower = s.lower()
        if any(re.search(p, lower) for p in patterns):
            passive_count += 1
            findings.append({"excerpt": s[:150]})

    ratio = passive_count / len(sentences)
    score = max(0.0, (ratio - 0.25) / 0.75)
    return round(score, 3), findings if ratio > 0.25 else []


def _detect_nominalisation_density(text: str, client_id: str = "default") -> Tuple[float, List[dict]]:
    """Flag abstract / verb-poor prose by suffix ratio.

    Computes ratio of tokens whose endings match common PT nominalisation
    suffixes (-ção, -mento, -dade, ...). Below ``nominalisation_low`` → 0.0;
    linear to 1.0 at ``nominalisation_high``.
    """
    suffix_patterns = _nominalisation_suffixes(client_id)
    if not suffix_patterns:
        return 0.0, []
    # Compose one alternation regex for performance
    suffix_alt = "|".join(suffix_patterns)
    suffix_re = re.compile(rf"\b\w*?({suffix_alt})\b", re.IGNORECASE)

    tokens = re.findall(r"\b[\wÀ-ÿ]+\b", text.lower())
    if len(tokens) < 20:
        return 0.0, []
    matches = suffix_re.findall(text.lower())
    ratio = len(matches) / len(tokens)

    cfg = _config(client_id)
    low = cfg.get("nominalisation_low", 0.15)
    high = cfg.get("nominalisation_high", 0.40)
    if ratio <= low or high <= low:
        return 0.0, []
    score = min(1.0, (ratio - low) / (high - low))
    findings = [{
        "ratio": round(ratio, 3),
        "low_threshold": low,
        "high_threshold": high,
        "nominalised_tokens": len(matches),
        "total_tokens": len(tokens),
    }]
    return round(score, 3), findings


def score_pt_forensic(
    text: str,
    language: str = "pt",
    threshold: float = 0.25,
    doc_type: str = "general",
    client_id: str = "default",
) -> dict:
    """Score PT text against forensic AI tells (juridiquês, synthetic passive,
    nominalisation density).

    Returns the same schema as ``score_ai_patterns`` so callers can switch
    transparently via ``score_writing_patterns(mode="pt")``.
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return {"success": False, "error": f"Invalid threshold '{threshold}'. Must be a number between 0.0 and 1.0."}
    if not (0.0 <= threshold <= 1.0):
        return {"success": False, "error": f"Invalid threshold {threshold}. Must be between 0.0 and 1.0."}

    word_count = len(text.split())
    page_equivalent = round(word_count / 300, 1)

    try:
        jur_score, jur_findings = _detect_juridiques(text, client_id)
        passive_score, passive_findings = _detect_synthetic_passive_pt(text, client_id)
        nom_score, nom_findings = _detect_nominalisation_density(text, client_id)

        categories = {
            "juridiques_density": {"score": jur_score, "findings": jur_findings},
            "synthetic_passive": {"score": passive_score, "findings": passive_findings},
            "nominalisation_density": {"score": nom_score, "findings": nom_findings},
        }
        scores = [v["score"] for v in categories.values()]
        overall_score = round(sum(scores) / len(scores), 3)

        if overall_score < 0.25:
            verdict = "clean"
        elif overall_score < 0.55:
            verdict = "review"
        else:
            verdict = "ai-sounding"

        flagged = [c for c, d in categories.items() if d["score"] >= threshold]
        summary = (
            f"{len(flagged)} categor{'y' if len(flagged) == 1 else 'ies'} flagged: {', '.join(flagged)}."
            if flagged else "No categories flagged above threshold."
        )

        return {
            "success": True,
            "language": language,
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
        logger.error("score_pt_forensic failed", error=str(e))
        capture_tool_error(e, tool_name="score_pt_forensic", doc_type=doc_type)
        return {"success": False, "error": str(e)}
