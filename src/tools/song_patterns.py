"""
Songwriting-specific craft pattern detector.

Detectors:
    verse_chorus_structure    — Checks for at least one repeated stanza (chorus/refrain)
    hook_repetition           — Measures whether the hook phrase recurs at expected intervals
    syllable_singability      — Lines >12 syllables flagged as hard to sing in one breath
    abstract_lyric_density    — Over-abstract lyrics with no concrete nouns
    filler_word_density       — "Oh yeah", "na na", "la la" used as whole-line padding
    rhyme_scheme_consistency  — Stanza-to-stanza rhyme scheme (AABB/ABAB) consistency

Use score_song_patterns instead of score_ai_patterns for song lyrics.
"""

import re
from typing import List, Tuple

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Valid doc_types
# ---------------------------------------------------------------------------

_VALID_DOC_TYPES = {"pop-song", "ballad", "rap-verse", "hymn", "jingle"}

# Doc_types where verse/chorus structure and hook_repetition are N/A
_NO_CHORUS_TYPES = {"rap-verse", "jingle"}

# ---------------------------------------------------------------------------
# Concrete noun list (proxy for grounded imagery)
# ---------------------------------------------------------------------------

_CONCRETE_NOUNS = {
    # Body
    "hand", "hands", "eye", "eyes", "heart", "face", "mouth", "voice",
    "skin", "hair", "blood", "bone", "breath", "tears", "lip", "lips",
    # Nature
    "rain", "sun", "moon", "star", "stars", "sky", "sea", "river", "road",
    "tree", "trees", "stone", "fire", "water", "wind", "earth", "snow",
    "cloud", "clouds", "light", "dark", "dark", "night", "day", "dawn",
    # Objects
    "door", "window", "house", "home", "room", "bed", "floor", "wall",
    "car", "train", "boat", "bridge", "phone", "ring", "dress", "coat",
    "letter", "book", "glass", "cup", "table", "chair", "key", "keys",
    # Time/place
    "street", "city", "town", "field", "hill", "lake", "shore", "bridge",
    "summer", "winter", "spring", "autumn", "morning", "evening", "midnight",
}

# ---------------------------------------------------------------------------
# Filler patterns
# ---------------------------------------------------------------------------

_FILLER_PATTERNS = [
    r"^oh+\b",
    r"^yeah+\b",
    r"^na[\s\-]na",
    r"^la[\s\-]la",
    r"^hey[\s,!]+hey",
    r"^mm+\b",
    r"^ba+by\b",
    r"^uh+\b",
    r"^whoa+\b",
    r"^hey+\b",
    r"^ooh+\b",
    r"^ah+\b",
]

# ---------------------------------------------------------------------------
# Syllable counting (shared with poetry_patterns)
# ---------------------------------------------------------------------------

_SYLLABLE_EXCEPTIONS: dict = {
    "fire": 1, "tired": 2, "every": 3, "even": 2, "seven": 2,
    "heaven": 2, "given": 2, "driven": 2, "river": 2, "never": 2,
    "over": 2, "under": 2, "other": 2, "whether": 2, "together": 3,
    "forever": 3, "whatever": 3, "however": 3, "wherever": 3,
    "fiery": 3, "prayer": 2, "world": 1, "through": 1, "though": 1,
    "thought": 1, "bought": 1, "brought": 1, "caught": 1,
}


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:\"'()-")
    if not word:
        return 0
    if word in _SYLLABLE_EXCEPTIONS:
        return _SYLLABLE_EXCEPTIONS[word]
    if len(word) > 2 and word.endswith("e") and word[-2] not in "aeiou":
        word = word[:-1]
    vowels = re.findall(r"[aeiouy]+", word)
    return max(1, len(vowels))


def _line_syllables(line: str) -> int:
    return sum(_count_syllables(w) for w in re.findall(r"\b\w+\b", line))


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _split_stanzas(text: str) -> List[str]:
    stanzas = re.split(r"\n[ \t]*\n", text.strip())
    return [s.strip() for s in stanzas if s.strip()]


def _get_lines(text: str) -> List[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


def _end_word(line: str) -> str:
    words = re.findall(r"[a-záéíóúàãõâêôçüñ]+", line.lower())
    return words[-1] if words else ""


def _rhyme_token(word: str) -> str:
    if len(word) >= 3:
        return word[-3:]
    return word


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def _stanza_similarity(s1: str, s2: str) -> float:
    """Proportion of lines in s1 that appear in s2 (case-insensitive)."""
    lines1 = set(l.lower().strip() for l in _get_lines(s1))
    lines2 = set(l.lower().strip() for l in _get_lines(s2))
    if not lines1:
        return 0.0
    matches = lines1 & lines2
    return len(matches) / len(lines1)


def _detect_verse_chorus_structure(text: str) -> Tuple[float, List[dict]]:
    """Check for at least one repeated stanza (≥60% identical lines)."""
    stanzas = _split_stanzas(text)
    if len(stanzas) < 2:
        return 0.8, [{"note": "Only one stanza found; no chorus/refrain detectable"}]

    # Find any pair with ≥60% similarity
    for i in range(len(stanzas)):
        for j in range(i + 1, len(stanzas)):
            if _stanza_similarity(stanzas[i], stanzas[j]) >= 0.6:
                return 0.0, []  # Chorus found

    return 0.8, [{"note": "No repeated stanza found; chorus or refrain expected in this doc_type"}]


def _detect_hook_repetition(text: str, doc_type: str) -> Tuple[float, List[dict]]:
    """Check that the most-repeated stanza recurs at expected frequency."""
    stanzas = _split_stanzas(text)
    if len(stanzas) < 2:
        return 0.0, []

    # Find the stanza with the most repetitions
    max_reps = 0
    for i, stanza in enumerate(stanzas):
        reps = sum(
            1 for j, other in enumerate(stanzas)
            if i != j and _stanza_similarity(stanza, other) >= 0.6
        )
        max_reps = max(max_reps, reps)

    expected_reps = 1 if doc_type == "hymn" else 2

    if max_reps >= expected_reps:
        return 0.0, []

    return 0.6, [{"note": f"Hook/chorus repeats {max_reps} time(s); expected at least {expected_reps}"}]


def _detect_syllable_singability(text: str) -> Tuple[float, List[dict]]:
    """Flag lines with >12 syllables as hard to sing in one breath."""
    lines = _get_lines(text)
    if not lines:
        return 0.0, []

    findings = []
    flagged = 0
    for i, line in enumerate(lines):
        count = _line_syllables(line)
        if count > 12:
            flagged += 1
            findings.append({"line": i + 1, "syllables": count, "text": line[:80]})

    score = min(1.0, flagged / len(lines))
    return round(score, 3), findings


def _detect_abstract_lyric_density(text: str) -> Tuple[float, List[dict]]:
    """Proportion of lines with zero concrete nouns."""
    lines = _get_lines(text)
    if not lines:
        return 0.0, []

    abstract_lines = []
    for i, line in enumerate(lines):
        words = set(re.findall(r"\b\w+\b", line.lower()))
        if not words & _CONCRETE_NOUNS:
            abstract_lines.append({"line": i + 1, "text": line[:80]})

    score = min(1.0, len(abstract_lines) / len(lines))
    return round(score, 3), abstract_lines


def _detect_filler_word_density(text: str) -> Tuple[float, List[dict]]:
    """Detect lines that are purely filler (oh yeah, na na, la la, etc.)."""
    lines = _get_lines(text)
    if not lines:
        return 0.0, []

    findings = []
    flagged = 0
    for i, line in enumerate(lines):
        lower = line.lower().strip()
        for pat in _FILLER_PATTERNS:
            if re.match(pat, lower):
                # Only flag if the whole line is essentially filler (< 4 real words)
                real_words = [w for w in re.findall(r"\b\w+\b", lower) if len(w) > 2 and not re.match(r"^(oh|yeah|na|la|hey|mm|uh|ooh|ah|whoa)+$", w)]
                if len(real_words) < 2:
                    flagged += 1
                    findings.append({"line": i + 1, "text": line[:80]})
                    break

    # Threshold: > 10% filler lines is excessive
    score = 0.0 if len(lines) == 0 else min(1.0, max(0.0, (flagged / len(lines) - 0.1) / 0.4))
    return round(score, 3), findings


def _classify_rhyme_scheme(stanza: str) -> str:
    """Classify a stanza's rhyme scheme: AABB, ABAB, AAAA, or mixed."""
    lines = _get_lines(stanza)
    if len(lines) < 4:
        return "short"
    tokens = [_rhyme_token(_end_word(l)) for l in lines[:4]]
    if not all(tokens):
        return "unknown"

    # AABB
    if tokens[0] == tokens[1] and tokens[2] == tokens[3] and tokens[0] != tokens[2]:
        return "AABB"
    # ABAB
    if tokens[0] == tokens[2] and tokens[1] == tokens[3] and tokens[0] != tokens[1]:
        return "ABAB"
    # AAAA
    if len(set(tokens)) == 1:
        return "AAAA"
    return "mixed"


def _detect_rhyme_scheme_consistency(text: str) -> Tuple[float, List[dict]]:
    """Check that stanzas use a consistent rhyme scheme."""
    stanzas = _split_stanzas(text)
    if len(stanzas) < 2:
        return 0.0, []

    schemes = [_classify_rhyme_scheme(s) for s in stanzas]
    # Ignore unknown/short stanzas
    valid_schemes = [s for s in schemes if s not in ("unknown", "short")]
    if not valid_schemes:
        return 0.0, []

    # Majority scheme
    from collections import Counter
    majority = Counter(valid_schemes).most_common(1)[0][0]
    inconsistent = [i + 1 for i, s in enumerate(schemes) if s not in ("unknown", "short", majority)]

    if not inconsistent:
        return 0.0, []

    score = min(1.0, len(inconsistent) / len(valid_schemes))
    findings = [{"stanza": i, "scheme": schemes[i - 1]} for i in inconsistent]
    return round(score, 3), findings


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_song_patterns(
    text: str,
    doc_type: str = "pop-song",
    language: str = "auto",
    threshold: float = 0.25,
) -> dict:
    """
    Score song lyrics against songwriting-specific craft patterns.

    Use this instead of score_ai_patterns for song lyrics.

    Args:
        text: The lyrics text
        doc_type: pop-song|ballad|rap-verse|hymn|jingle (default: pop-song)
        language: en|pt|auto (default: auto)
        threshold: Per-category score above which a category is flagged (default: 0.25)

    Returns:
        overall_score, verdict (clean|review|craft-issue), per-category scores,
        stanza_count, line_count, word_count, doc_type
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
        if doc_type in _NO_CHORUS_TYPES:
            chorus_score, chorus_findings = 0.0, []
            hook_score, hook_findings = 0.0, []
        else:
            chorus_score, chorus_findings = _detect_verse_chorus_structure(text)
            hook_score, hook_findings = _detect_hook_repetition(text, doc_type)

        singability_score, singability_findings = _detect_syllable_singability(text)
        abstract_score, abstract_findings = _detect_abstract_lyric_density(text)
        filler_score, filler_findings = _detect_filler_word_density(text)
        rhyme_score, rhyme_findings = _detect_rhyme_scheme_consistency(text)

        categories = {
            "verse_chorus_structure": {
                "score": chorus_score,
                "findings": chorus_findings,
                "applicable": doc_type not in _NO_CHORUS_TYPES,
            },
            "hook_repetition": {
                "score": hook_score,
                "findings": hook_findings,
                "applicable": doc_type not in _NO_CHORUS_TYPES,
            },
            "syllable_singability": {"score": singability_score, "findings": singability_findings},
            "abstract_lyric_density": {"score": abstract_score, "findings": abstract_findings},
            "filler_word_density": {"score": filler_score, "findings": filler_findings},
            "rhyme_scheme_consistency": {"score": rhyme_score, "findings": rhyme_findings},
        }

        applicable_scores = [
            v["score"] for v in categories.values()
            if v.get("applicable", True)
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
            "stanza_count": len(stanzas),
            "line_count": len(lines),
            "word_count": word_count,
        }

    except Exception as exc:
        logger.error("song_patterns_error", error=str(exc))
        return {"success": False, "error": f"Scoring failed: {exc}"}
