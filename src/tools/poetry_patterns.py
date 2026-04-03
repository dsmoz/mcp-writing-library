"""
Poetry-specific craft pattern detector.

Detectors:
    rhyme_scheme_regularity   — Sonnet ABAB/couplet and villanelle ABA refrains (N/A for other forms)
    meter_regularity          — Syllable-count variance; strict for haiku (5-7-5) and sonnet (~10±2)
    stanza_length_consistency — Flags stanzas with dramatically different line counts (stddev > 2.5)
    line_ending_cliche        — Overused end-words: heart, soul, moon, tears, fire, etc.
    prose_intrusion           — Lines >12 words with subordinating conjunction + finite verb
    forced_rhyme_flag         — Inverted syntax at line end suggesting rhyme-forced phrasing

Use score_poetry_patterns instead of score_ai_patterns for poetry — the ai_patterns
tool checks prose document patterns that are inapplicable or misleading for verse.
"""

import re
from typing import List, Tuple

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Valid doc_types for this scorer
# ---------------------------------------------------------------------------

_VALID_DOC_TYPES = {"haiku", "sonnet", "free-verse", "villanelle", "spoken-word"}

# ---------------------------------------------------------------------------
# Clichéd end-words
# ---------------------------------------------------------------------------

_CLICHE_END_WORDS = {
    "heart", "soul", "moon", "night", "light", "sky", "dream", "tears",
    "fire", "time", "pain", "love", "rain", "true", "free", "sea",
    "star", "stars", "eyes", "mind", "face", "grace", "place", "space",
    "deep", "sleep", "keep", "weep", "breathe", "believe", "leave",
}

# ---------------------------------------------------------------------------
# Forced rhyme inversion markers
# ---------------------------------------------------------------------------

_INVERSION_PATTERNS = [
    r"\bthus (?:did|does|shall|would|could) \w+\b",
    r"\bso (?:did|does|shall|would|could) \w+\b",
    r"\bsuch (?:was|is|were) the \w+\b",
    r"\byet (?:did|does|shall|would) \w+\b",
    r"\bnor (?:did|does|would|shall|could) \w+\b",
    r"\bhence (?:did|does|shall|would) \w+\b",
]

# Subordinating conjunctions signalling prose-like construction
_SUBORDINATORS = r"\b(because|although|though|unless|since|whereas|whenever|wherever|whoever|which|whose|that)\b"

# Finite verb pattern (simple heuristic)
_FINITE_VERB = r"\b(is|are|was|were|has|have|had|does|do|did|will|would|can|could|shall|should|may|might|must)\b"

# ---------------------------------------------------------------------------
# Syllable counting
# ---------------------------------------------------------------------------

# Common exceptions for syllable counting
_SYLLABLE_EXCEPTIONS: dict = {
    "fire": 1, "tired": 2, "every": 3, "even": 2, "seven": 2,
    "heaven": 2, "given": 2, "driven": 2, "river": 2, "never": 2,
    "over": 2, "under": 2, "other": 2, "whether": 2, "together": 3,
    "forever": 3, "whatever": 3, "however": 3, "wherever": 3,
    "fiery": 3, "prayer": 2, "world": 1, "through": 1, "though": 1,
    "thought": 1, "bought": 1, "brought": 1, "caught": 1,
}


def _count_syllables(word: str) -> int:
    """Heuristic syllable counter."""
    word = word.lower().strip(".,!?;:\"'()-")
    if not word:
        return 0
    if word in _SYLLABLE_EXCEPTIONS:
        return _SYLLABLE_EXCEPTIONS[word]
    # Remove silent e at end (but not single-letter words)
    if len(word) > 2 and word.endswith("e") and word[-2] not in "aeiou":
        word = word[:-1]
    # Count vowel clusters
    vowels = re.findall(r"[aeiouy]+", word)
    count = len(vowels)
    return max(1, count)


def _line_syllables(line: str) -> int:
    """Count syllables across all words in a line."""
    words = re.findall(r"\b\w+\b", line)
    return sum(_count_syllables(w) for w in words)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _split_stanzas(text: str) -> List[str]:
    """Split on single blank lines."""
    stanzas = re.split(r"\n[ \t]*\n", text.strip())
    return [s.strip() for s in stanzas if s.strip()]


def _get_lines(text: str) -> List[str]:
    """Return non-empty lines."""
    return [l.strip() for l in text.splitlines() if l.strip()]


def _end_word(line: str) -> str:
    """Extract the last meaningful word from a line."""
    words = re.findall(r"[a-záéíóúàãõâêôçüñ]+", line.lower())
    return words[-1] if words else ""


def _rhyme_token(word: str) -> str:
    """2–3 character suffix as phonetic rhyme proxy."""
    if len(word) >= 3:
        return word[-3:]
    return word


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def _detect_rhyme_scheme_regularity(text: str, doc_type: str) -> Tuple[float, List[dict], bool]:
    """
    Check rhyme scheme for sonnets (ABAB CDCD EFEF GG) and villanelles (ABA × 5, ABAA).
    Returns (score, findings, applicable).
    """
    if doc_type not in ("sonnet", "villanelle"):
        return 0.0, [], False

    stanzas = _split_stanzas(text)
    if not stanzas:
        return 0.0, [], True

    failures = []

    if doc_type == "sonnet":
        # Need at least 14 lines across stanzas
        all_lines = _get_lines(text)
        if len(all_lines) < 14:
            return 0.0, [{"note": f"Only {len(all_lines)} lines found; expected 14"}], True

        # Check ABAB for quatrains (stanzas 1–3)
        for stanza_idx in range(min(3, len(stanzas))):
            lines = _get_lines(stanzas[stanza_idx])
            if len(lines) < 4:
                continue
            tokens = [_rhyme_token(_end_word(l)) for l in lines[:4]]
            # ABAB: tokens[0]==tokens[2], tokens[1]==tokens[3]
            if tokens[0] and tokens[2] and tokens[0] != tokens[2]:
                failures.append({"stanza": stanza_idx + 1, "expected": "ABAB (lines 1,3 rhyme)", "end_words": [_end_word(lines[0]), _end_word(lines[2])]})
            if tokens[1] and tokens[3] and tokens[1] != tokens[3]:
                failures.append({"stanza": stanza_idx + 1, "expected": "ABAB (lines 2,4 rhyme)", "end_words": [_end_word(lines[1]), _end_word(lines[3])]})

        # Check couplet (last stanza, 2 lines rhyme)
        if len(stanzas) >= 4:
            couplet_lines = _get_lines(stanzas[-1])
            if len(couplet_lines) >= 2:
                t0 = _rhyme_token(_end_word(couplet_lines[0]))
                t1 = _rhyme_token(_end_word(couplet_lines[1]))
                if t0 and t1 and t0 != t1:
                    failures.append({"stanza": len(stanzas), "expected": "Couplet (AA rhyme)", "end_words": [_end_word(couplet_lines[0]), _end_word(couplet_lines[1])]})

    elif doc_type == "villanelle":
        # ABA refrains in each tercet
        for i, stanza in enumerate(stanzas[:-1]):
            lines = _get_lines(stanza)
            if len(lines) < 3:
                continue
            t0 = _rhyme_token(_end_word(lines[0]))
            t2 = _rhyme_token(_end_word(lines[2]))
            if t0 and t2 and t0 != t2:
                failures.append({"stanza": i + 1, "expected": "ABA (lines 1,3 rhyme)", "end_words": [_end_word(lines[0]), _end_word(lines[2])]})

    # Score = proportion of expected checks that failed (max 1.0)
    total_checks = 3 * 2 + 1 if doc_type == "sonnet" else max(1, len(stanzas) - 1)
    score = min(1.0, len(failures) / max(total_checks, 1))
    return round(score, 3), failures, True


def _detect_meter_regularity(text: str, doc_type: str) -> Tuple[float, List[dict]]:
    """
    Check syllable counts per line.
    - haiku: strict 5-7-5
    - sonnet: ~10 syllables ±2
    - free-verse / spoken-word: check for near-zero variance (monotony is the anti-pattern)
    - others: normalized variance
    """
    lines = _get_lines(text)
    if not lines:
        return 0.0, []

    syllable_counts = [_line_syllables(l) for l in lines]
    findings = []

    if doc_type == "haiku":
        expected = [5, 7, 5]
        if len(lines) < 3:
            return 0.5, [{"note": "Fewer than 3 lines; haiku expects exactly 3"}]
        violations = 0
        for i, (line, exp) in enumerate(zip(lines[:3], expected)):
            actual = syllable_counts[i]
            if abs(actual - exp) > 1:
                violations += 1
                findings.append({"line": i + 1, "expected_syllables": exp, "counted": actual, "text": line[:60]})
        score = violations / 3
        return round(score, 3), findings

    if doc_type == "sonnet":
        # Expect 10 ± 2 syllables per line
        violations = sum(1 for c in syllable_counts if abs(c - 10) > 2)
        score = violations / len(syllable_counts)
        for i, (count, line) in enumerate(zip(syllable_counts, lines)):
            if abs(count - 10) > 2:
                findings.append({"line": i + 1, "counted": count, "expected_range": "8–12", "text": line[:60]})
        return round(score, 3), findings

    if doc_type in ("free-verse", "spoken-word"):
        # Penalise monotony (uniform line lengths) not variation
        if len(syllable_counts) < 3:
            return 0.0, []
        mean = sum(syllable_counts) / len(syllable_counts)
        variance = sum((c - mean) ** 2 for c in syllable_counts) / len(syllable_counts)
        std = variance ** 0.5
        # Low stddev in free verse = monotone rhythm = mild flag
        if mean > 0 and std / mean < 0.1:
            return 0.3, [{"note": f"Very uniform line lengths (stddev/mean={std/mean:.2f}); free verse benefits from rhythm variation"}]
        return 0.0, []

    # Default: normalized variance
    if len(syllable_counts) < 2:
        return 0.0, []
    mean = sum(syllable_counts) / len(syllable_counts)
    if mean == 0:
        return 0.0, []
    variance = sum((c - mean) ** 2 for c in syllable_counts) / len(syllable_counts)
    std = variance ** 0.5
    score = min(1.0, std / mean)
    return round(score, 3), findings


def _detect_stanza_length_consistency(text: str) -> Tuple[float, List[dict]]:
    """Flag stanzas with dramatically different line counts (stddev > 2.5)."""
    stanzas = _split_stanzas(text)
    if len(stanzas) < 2:
        return 0.0, []

    counts = [len(_get_lines(s)) for s in stanzas]
    mean = sum(counts) / len(counts)
    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
    std = variance ** 0.5

    if std <= 2.5:
        return 0.0, []

    score = min(1.0, std / 4.0)
    findings = [
        {"stanza": i + 1, "lines": c}
        for i, c in enumerate(counts)
        if abs(c - mean) > std
    ]
    return round(score, 3), findings


def _detect_line_ending_cliche(text: str) -> Tuple[float, List[dict]]:
    """Detect overused end-words (heart, soul, moon, tears, etc.)."""
    lines = _get_lines(text)
    if not lines:
        return 0.0, []

    findings = []
    flagged = 0
    for i, line in enumerate(lines):
        word = _end_word(line)
        if word in _CLICHE_END_WORDS:
            flagged += 1
            findings.append({"line": i + 1, "word": word, "text": line[:60]})

    score = min(1.0, flagged / len(lines))
    return round(score, 3), findings


def _detect_prose_intrusion(text: str) -> Tuple[float, List[dict]]:
    """Flag lines >12 words that contain subordinating conjunctions + finite verb."""
    lines = _get_lines(text)
    if not lines:
        return 0.0, []

    findings = []
    flagged = 0
    for i, line in enumerate(lines):
        words = line.split()
        if len(words) > 12:
            lower = line.lower()
            has_subordinator = bool(re.search(_SUBORDINATORS, lower))
            has_finite = bool(re.search(_FINITE_VERB, lower))
            if has_subordinator and has_finite:
                flagged += 1
                findings.append({"line": i + 1, "word_count": len(words), "text": line[:80]})

    score = min(1.0, flagged / len(lines))
    return round(score, 3), findings


def _detect_forced_rhyme(text: str) -> Tuple[float, List[dict]]:
    """Detect inverted syntax at line end suggesting rhyme-forced phrasing."""
    lines = _get_lines(text)
    if not lines:
        return 0.0, []

    findings = []
    flagged = 0
    for i, line in enumerate(lines):
        lower = line.lower()
        for pat in _INVERSION_PATTERNS:
            if re.search(pat, lower):
                flagged += 1
                findings.append({"line": i + 1, "text": line[:80]})
                break

    score = min(1.0, flagged / len(lines))
    return round(score, 3), findings


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_poetry_patterns(
    text: str,
    doc_type: str = "free-verse",
    language: str = "auto",
    threshold: float = 0.25,
) -> dict:
    """
    Score a poem against poetry-specific craft patterns.

    Use this instead of score_ai_patterns for poetry — it checks form-appropriate
    patterns rather than document prose patterns.

    Args:
        text: The poem text
        doc_type: haiku|sonnet|free-verse|villanelle|spoken-word (default: free-verse)
        language: en|pt|auto (default: auto)
        threshold: Per-category score above which a category is flagged (default: 0.25)

    Returns:
        overall_score, verdict (clean|review|craft-issue), per-category scores and findings,
        line_count, stanza_count, word_count, doc_type
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    if doc_type not in _VALID_DOC_TYPES:
        valid = ", ".join(sorted(_VALID_DOC_TYPES))
        return {"success": False, "error": f"Invalid doc_type '{doc_type}'. Must be one of: {valid}."}

    if language not in ("en", "pt", "auto"):
        return {"success": False, "error": f"Invalid language '{language}'. Must be 'en', 'pt', or 'auto'."}

    lines = _get_lines(text)
    stanzas = _split_stanzas(text)
    word_count = len(text.split())

    try:
        rhyme_score, rhyme_findings, rhyme_applicable = _detect_rhyme_scheme_regularity(text, doc_type)
        meter_score, meter_findings = _detect_meter_regularity(text, doc_type)
        stanza_score, stanza_findings = _detect_stanza_length_consistency(text)
        cliche_score, cliche_findings = _detect_line_ending_cliche(text)
        prose_score, prose_findings = _detect_prose_intrusion(text)
        forced_score, forced_findings = _detect_forced_rhyme(text)

        categories = {
            "rhyme_scheme_regularity": {
                "score": rhyme_score,
                "findings": rhyme_findings,
                "applicable": rhyme_applicable,
            },
            "meter_regularity": {"score": meter_score, "findings": meter_findings},
            "stanza_length_consistency": {"score": stanza_score, "findings": stanza_findings},
            "line_ending_cliche": {"score": cliche_score, "findings": cliche_findings},
            "prose_intrusion": {"score": prose_score, "findings": prose_findings},
            "forced_rhyme_flag": {"score": forced_score, "findings": forced_findings},
        }

        # Only include applicable categories in overall score
        applicable_scores = [
            v["score"] for k, v in categories.items()
            if k != "rhyme_scheme_regularity" or v.get("applicable", True)
        ]
        overall_score = round(sum(applicable_scores) / max(len(applicable_scores), 1), 3)

        if overall_score < 0.25:
            verdict = "clean"
        elif overall_score < 0.55:
            verdict = "review"
        else:
            verdict = "craft-issue"

        flagged = [cat for cat, data in categories.items() if data["score"] >= threshold]
        if flagged:
            summary = f"{len(flagged)} categor{'y' if len(flagged) == 1 else 'ies'} flagged: {', '.join(flagged)}."
        else:
            summary = "No categories flagged above threshold."

        return {
            "success": True,
            "doc_type": doc_type,
            "language": language,
            "overall_score": overall_score,
            "verdict": verdict,
            "threshold": threshold,
            "categories": categories,
            "summary": summary,
            "line_count": len(lines),
            "stanza_count": len(stanzas),
            "word_count": word_count,
        }

    except Exception as exc:
        logger.error("poetry_patterns_error", error=str(exc))
        return {"success": False, "error": f"Scoring failed: {exc}"}
