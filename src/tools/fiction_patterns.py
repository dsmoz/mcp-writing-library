"""
Prose fiction-specific craft pattern detector.

Detectors:
    show_vs_tell_ratio        — Proportion of "telling" sentences vs "showing" sentences
    dialogue_tag_variety      — Overuse of non-"said" attribution tags (exclaimed, hissed, snapped…)
    adverb_overload           — Adverbs modifying dialogue tags (said quietly, replied angrily)
    filter_word_density       — "She felt / He saw / She noticed" filtering reader from direct experience
    purple_prose_density      — Dense stacking of adjectives + abstract nouns in non-dialogue paragraphs
    narrative_distance        — Distant narration proxy (informational only, not penalised)

Use score_fiction_patterns instead of score_ai_patterns for prose fiction.
creative-nonfiction uses a higher threshold (0.4) for show_vs_tell_ratio to
avoid penalising legitimate introspective passages.
"""

import re
from typing import List, Tuple

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Valid doc_types
# ---------------------------------------------------------------------------

_VALID_DOC_TYPES = {
    "novel-chapter", "short-story", "flash-fiction",
    "screenplay", "creative-nonfiction",
}

# Higher show/tell threshold for creative-nonfiction
_CREATIVE_NONFICTION_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# Emotion adjectives (telling words)
# ---------------------------------------------------------------------------

_EMOTION_ADJECTIVES = {
    "sad", "happy", "angry", "afraid", "nervous", "excited", "upset",
    "furious", "terrified", "thrilled", "devastated", "elated", "anxious",
    "worried", "jealous", "proud", "ashamed", "guilty", "confused",
    "disappointed", "frustrated", "hopeful", "lonely", "miserable",
    "overjoyed", "panicked", "relieved", "surprised", "exhausted",
}

# Telling verbs (filter + state verbs)
_TELLING_VERBS = r"\b(felt|feel|feels|seemed|seems|appear(?:ed|s)?|was|were|is|are)\b"
_FILTER_VERBS = r"\b(felt|feel|saw|see|noticed|notice|watched|watch|heard|hear|thought|think|realized|realise|knew|know|decided|decide|wondered|wonder)\b"

# POV pronouns
_POV_PRONOUNS = r"\b(he|she|they|i)\b"

# Showing patterns (action + sensory verbs)
_SHOWING_PATTERNS = [
    r"\b\w+ed\s+(?:her|his|their|my)\s+\w+",  # "pressed her hands", "tightened his jaw"
    r"\b(?:the\s+room|the\s+air|the\s+space)\s+\w+ed",  # "the room smelled", "the air felt"
    r"\b(?:her|his|their|my)\s+\w+\s+\w+ed",  # "his hands shook", "her voice cracked"
]

# ---------------------------------------------------------------------------
# Non-said dialogue tags
# ---------------------------------------------------------------------------

_SAID_VARIANTS = {
    "exclaimed", "whispered", "hissed", "snapped", "retorted", "murmured",
    "declared", "announced", "breathed", "gasped", "growled", "shrieked",
    "bellowed", "cried", "shouted", "yelled", "snarled", "pleaded",
    "stammered", "stuttered", "muttered", "mused", "pondered", "intoned",
    "purred", "drawled", "barked", "chirped", "cooed", "sighed",
    "grunted", "sniffed", "quipped", "teased", "chided", "admonished",
    "protested", "insisted", "conceded", "admitted", "confessed",
    "elaborated", "clarified", "explained", "continued", "added",
}

# All attribution verbs (said + variants)
_ALL_ATTRIBUTION = _SAID_VARIANTS | {"said", "asked", "replied", "answered", "told", "called"}

# ---------------------------------------------------------------------------
# Abstract nouns (for purple prose detection)
# ---------------------------------------------------------------------------

_ABSTRACT_NOUNS = {
    "darkness", "silence", "emptiness", "eternity", "beauty", "sorrow",
    "longing", "destiny", "fate", "truth", "love", "pain", "grief",
    "joy", "hope", "despair", "loss", "memory", "dream", "fear",
    "regret", "loneliness", "passion", "chaos", "peace", "freedom",
    "innocence", "guilt", "shame", "pride", "wonder", "awe",
}

# ---------------------------------------------------------------------------
# Distant narration markers
# ---------------------------------------------------------------------------

_DISTANT_NARRATION = [
    r"\b(?:one|the narrator|the reader)\s+(?:could|might|would)\s+(?:see|observe|notice|feel)\b",
    r"\bit\s+(?:was|is)\s+(?:clear|evident|obvious)\s+that\b",
    r"\b(?:as|though)\s+(?:if\s+)?(?:seen|viewed|observed)\s+from\b",
    r"\bthe\s+(?:reader|audience|viewer)\s+(?:is|was)\b",
    r"\bone\s+(?:cannot|could not)\s+help\s+but\b",
]

# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paras if p.strip()]


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= 8]


def _is_dialogue(paragraph: str) -> bool:
    """Heuristic: paragraph contains a dialogue quote."""
    return bool(re.search(r'[""«»\'"]{1}[^""«»\'"]{3,}[""«»\'"]{1}', paragraph))


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def _detect_show_vs_tell(text: str, doc_type: str) -> Tuple[float, List[dict]]:
    """
    Proportion of telling sentences (felt/seemed/was + emotion adj) vs total.
    creative-nonfiction uses a relaxed threshold.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0, []

    telling = []
    showing_count = 0

    for i, sent in enumerate(sentences):
        lower = sent.lower()
        # Check for telling pattern: POV pronoun + telling/state verb + emotion adj
        has_pov = bool(re.search(_POV_PRONOUNS, lower))
        has_telling_verb = bool(re.search(_TELLING_VERBS, lower))

        emotion_adj_found = any(f" {adj}" in lower or lower.startswith(adj) for adj in _EMOTION_ADJECTIVES)

        # Showing check
        is_showing = any(re.search(pat, lower) for pat in _SHOWING_PATTERNS)
        if is_showing:
            showing_count += 1

        if has_pov and has_telling_verb and emotion_adj_found:
            telling.append({"sentence": i + 1, "text": sent[:100]})

    total = max(len(sentences), 1)
    raw_score = len(telling) / total
    return round(raw_score, 3), telling


def _detect_dialogue_tag_variety(text: str) -> Tuple[float, List[dict]]:
    """
    Proportion of non-'said' attribution tags out of all attribution verbs.
    Threshold: > 30% non-said is flagged.
    """
    # Find all attribution verb occurrences after a closing quote
    all_tags_found = re.findall(
        r'[""«»\'"][\s,]*\b(\w+)\b(?:\s+\w+)?\s+(?:he|she|they|i|the\s+\w+)',
        text, re.IGNORECASE
    )
    if not all_tags_found:
        return 0.0, []

    non_said = []
    total = 0
    for tag in all_tags_found:
        lower = tag.lower()
        if lower in _ALL_ATTRIBUTION:
            total += 1
            if lower in _SAID_VARIANTS:
                non_said.append(lower)

    if total == 0:
        return 0.0, []

    ratio = len(non_said) / total
    score = max(0.0, (ratio - 0.3) / 0.7)  # 0 at 30%, 1.0 at 100%
    score = min(1.0, round(score, 3))

    findings = []
    if score > 0:
        from collections import Counter
        counts = Counter(non_said)
        findings = [{"tag": tag, "count": count} for tag, count in counts.most_common(10)]
    return score, findings


def _detect_adverb_overload(text: str) -> Tuple[float, List[dict]]:
    """
    Adverbs modifying dialogue tags: 'said quietly', 'replied angrily'.
    Threshold: > 20% of dialogue lines.
    """
    # Patterns: attribution verb + adverb, or adverb + attribution verb
    patterns = [
        r'[""«»\'"]\s*,?\s*\b(\w+)\s+(\w+ly)\b',  # "said quietly"
        r'\b(\w+ly)\s+(\w+)\b(?=\s+[""«»\'"])',  # quietly said "
    ]

    dialogue_lines = [l for l in text.splitlines() if re.search(r'[""«»\'"]', l)]
    if not dialogue_lines:
        return 0.0, []

    findings = []
    flagged = 0
    for i, line in enumerate(dialogue_lines):
        for pat in patterns:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                groups = m.groups()
                # Check that one of the groups is an attribution verb
                verb = groups[0].lower() if groups[0].lower() in _ALL_ATTRIBUTION else (groups[1].lower() if len(groups) > 1 else "")
                if verb:
                    flagged += 1
                    findings.append({"line": line[:100]})
                    break

    score = max(0.0, (flagged / max(len(dialogue_lines), 1) - 0.2) / 0.8)
    return round(min(1.0, score), 3), findings


def _detect_filter_words(text: str) -> Tuple[float, List[dict]]:
    """
    Detect "she felt", "he saw", "she noticed" constructions that filter the reader.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0, []

    pattern = re.compile(
        r"\b(he|she|they|i)\s+" + _FILTER_VERBS,
        re.IGNORECASE
    )
    findings = []
    for i, sent in enumerate(sentences):
        if pattern.search(sent):
            findings.append({"sentence": i + 1, "text": sent[:100]})

    score = min(1.0, len(findings) / max(len(sentences), 1))
    return round(score, 3), findings


def _count_adjectives(text: str) -> int:
    """Heuristic: words ending in common adjective suffixes."""
    adj_pattern = r"\b\w+(?:ful|less|ous|ive|al|ic|ish|ed|ing|ary|ory|ent|ant)\b"
    return len(re.findall(adj_pattern, text.lower()))


def _detect_purple_prose(text: str) -> Tuple[float, List[dict]]:
    """
    Detect dense adjective + abstract noun stacking in non-dialogue paragraphs.
    Flag paragraphs with ≥3 adjectives AND ≥2 abstract nouns.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return 0.0, []

    non_dialogue_paras = [p for p in paragraphs if not _is_dialogue(p)]
    if not non_dialogue_paras:
        return 0.0, []

    findings = []
    flagged = 0
    for para in non_dialogue_paras:
        adj_count = _count_adjectives(para)
        words = set(re.findall(r"\b\w+\b", para.lower()))
        abstract_count = len(words & _ABSTRACT_NOUNS)
        if adj_count >= 3 and abstract_count >= 2:
            flagged += 1
            findings.append({
                "adjectives_found": adj_count,
                "abstract_nouns_found": abstract_count,
                "excerpt": para[:120],
            })

    score = min(1.0, flagged / len(non_dialogue_paras))
    return round(score, 3), findings


def _detect_narrative_distance(text: str, doc_type: str) -> Tuple[float, List[dict]]:
    """
    Informational proxy for distant narration. Not penalised in scoring.
    N/A for screenplay.
    """
    if doc_type == "screenplay":
        return 0.0, []

    sentences = _split_sentences(text)
    if not sentences:
        return 0.0, []

    findings = []
    for i, sent in enumerate(sentences):
        lower = sent.lower()
        for pat in _DISTANT_NARRATION:
            if re.search(pat, lower):
                findings.append({"sentence": i + 1, "text": sent[:100]})
                break

    score = min(1.0, len(findings) / max(len(sentences), 1))
    return round(score, 3), findings


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_fiction_patterns(
    text: str,
    doc_type: str = "short-story",
    language: str = "auto",
    threshold: float = 0.25,
) -> dict:
    """
    Score prose fiction against fiction-specific craft patterns.

    Use this instead of score_ai_patterns for prose fiction.

    Args:
        text: The fiction text (chapter, story, or excerpt)
        doc_type: novel-chapter|short-story|flash-fiction|screenplay|creative-nonfiction
        language: en|pt|auto (default: auto)
        threshold: Per-category score above which a category is flagged (default: 0.25)

    Returns:
        overall_score, verdict (clean|review|craft-issue), per-category scores,
        paragraph_count, sentence_count, word_count, dialogue_line_count, doc_type
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    if doc_type not in _VALID_DOC_TYPES:
        valid = ", ".join(sorted(_VALID_DOC_TYPES))
        return {"success": False, "error": f"Invalid doc_type '{doc_type}'. Must be one of: {valid}."}

    if language not in ("en", "pt", "auto"):
        return {"success": False, "error": f"Invalid language '{language}'. Must be 'en', 'pt', or 'auto'."}

    paragraphs = _split_paragraphs(text)
    sentences = _split_sentences(text)
    dialogue_lines = [l for l in text.splitlines() if re.search(r'[""«»\'"]', l)]
    word_count = len(text.split())

    # creative-nonfiction: apply relaxed show/tell threshold
    show_tell_threshold = _CREATIVE_NONFICTION_THRESHOLD if doc_type == "creative-nonfiction" else threshold

    try:
        show_tell_score, show_tell_findings = _detect_show_vs_tell(text, doc_type)
        tag_score, tag_findings = _detect_dialogue_tag_variety(text)
        adverb_score, adverb_findings = _detect_adverb_overload(text)
        filter_score, filter_findings = _detect_filter_words(text)
        purple_score, purple_findings = _detect_purple_prose(text)
        distance_score, distance_findings = _detect_narrative_distance(text, doc_type)

        categories = {
            "show_vs_tell_ratio": {
                "score": show_tell_score,
                "findings": show_tell_findings,
                "note": f"Threshold for this doc_type: {show_tell_threshold}",
            },
            "dialogue_tag_variety": {"score": tag_score, "findings": tag_findings},
            "adverb_overload": {"score": adverb_score, "findings": adverb_findings},
            "filter_word_density": {"score": filter_score, "findings": filter_findings},
            "purple_prose_density": {"score": purple_score, "findings": purple_findings},
            "narrative_distance": {
                "score": 0.0,  # Informational only — not included in overall score
                "informational_score": distance_score,
                "findings": distance_findings,
                "note": "Informational only — not penalised. High score = distant narration style.",
            },
        }

        # Exclude narrative_distance from overall score (informational only)
        scored_categories = {k: v for k, v in categories.items() if k != "narrative_distance"}
        overall_score = round(
            sum(v["score"] for v in scored_categories.values()) / max(len(scored_categories), 1),
            3,
        )

        if overall_score < 0.25:
            verdict = "clean"
        elif overall_score < 0.55:
            verdict = "review"
        else:
            verdict = "craft-issue"

        # Use doc_type-aware threshold for show_vs_tell flagging
        flagged = []
        for cat, data in categories.items():
            if cat == "narrative_distance":
                continue
            cat_threshold = show_tell_threshold if cat == "show_vs_tell_ratio" else threshold
            if data["score"] >= cat_threshold:
                flagged.append(cat)

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
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "dialogue_line_count": len(dialogue_lines),
            "word_count": word_count,
        }

    except Exception as exc:
        logger.error("fiction_patterns_error", error=str(exc))
        return {"success": False, "error": f"Scoring failed: {exc}"}
