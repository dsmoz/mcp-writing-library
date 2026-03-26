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

logger = structlog.get_logger(__name__)

try:
    from kbase.vector.sync_search import semantic_search
except ImportError:
    semantic_search = None  # type: ignore


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Connectors that become AI markers when repeated
_CONNECTORS_EN = [
    "furthermore", "additionally", "moreover", "in conclusion", "in summary",
    "to summarise", "to summarize", "firstly", "secondly", "thirdly",
    "lastly", "in addition", "it is worth noting",
]

_CONNECTORS_PT = [
    "além disso", "adicionalmente", "ademais", "em conclusão", "em resumo",
    "em primeiro lugar", "em segundo lugar", "em terceiro lugar",
    "por último", "igualmente", "do mesmo modo",
]

# Hollow intensifiers (EN only — PT has different equivalents)
_HOLLOW_INTENSIFIERS = [
    r"it is important to note that",
    r"it is crucial that",
    r"it is essential to recognise",
    r"it is essential to recognize",
    r"it should be noted that",
    r"it is worth noting that",
    r"it is important to highlight",
    r"it bears emphasising",
    r"it bears emphasizing",
    r"it must be acknowledged",
    r"needless to say",
]

# Grandiose paragraph openers (EN)
_GRANDIOSE_OPENERS_EN = [
    r"against this backdrop",
    r"the fundamental insight here is that",
    r"what emerges from this analysis",
    r"the picture that emerges",
    r"these .{0,30} are not mere",
    r"\w+ deserve(?:s)? special attention",
    r"in this context of",
    r"at the heart of this",
    r"this is a pivotal moment",
    r"this represents a watershed",
    r"the evidence is unequivocal",
    r"the data paints? a (?:clear|stark|compelling)",
]

# Grandiose paragraph openers (PT)
_GRANDIOSE_OPENERS_PT = [
    r"contra este pano de fundo",
    r"a percep[çc][ãa]o fundamental aqui [eé] que",
    r"o quadro que emerge",
    r"estas? .{0,30} n[ãa]o s[ãa]o meros?",
    r"\w+ merece(?:m)? destaque",
    r"neste contexto de",
    r"no cerne desta",
    r"este [eé] um momento fulcral",
    r"os dados revelam",
    r"a evid[eê]ncia [eé] inequ[íi]voca",
]

# Generic closings (EN)
_GENERIC_CLOSINGS = [
    r"in conclusion,? this (?:report|document|paper|analysis) has shown",
    r"to summaris[e] the above",
    r"to summarize the above",
    r"as (?:has been )?demonstrated above",
    r"as (?:has been )?shown above",
    r"in summary,? this (?:report|document|analysis)",
    r"the foregoing analysis (?:has shown|demonstrates)",
    r"as outlined (?:above|in this report)",
]

# Discursive expressions (positive — their absence is the signal)
_DISCURSIVE_EXPRESSIONS = [
    # EN stance markers
    r"what (?:this|the analysis|the evidence) (?:reveals?|shows?|suggests?|indicates?)",
    r"what emerges from",
    r"the key (?:insight|finding|implication) (?:here )?is",
    r"what makes this (?:particularly )?significant",
    r"this raises (?:a )?(?:crucial|important|key) question",
    r"the implications extend beyond",
    r"what this means in practice",
    r"to understand why",
    r"consider what this suggests",
    r"the challenge,? then,? is",
    r"building on this",
    r"this is evident in",
    # PT stance markers
    r"o que (?:esta|a análise|os dados) (?:revela?|mostra?|sugere?)",
    r"o que emerge de",
    r"a (?:conclusão|lição|implicação) central (?:aqui )?é",
    r"o que isto significa na prática",
    r"para compreender (?:por que|porquê)",
    r"esta (?:questão|realidade) é evidente em",
    r"construindo sobre isto",
    r"o desafio,? portanto,? é",
]

# Passive voice heuristic patterns (EN)
_PASSIVE_PATTERNS = [
    r"\b(?:was|were|is|are|has been|have been|had been|being)\s+\w+ed\b",
    r"\b(?:was|were|is|are|has been|have been|had been|being)\s+\w+en\b",
]

# Portuguese function words for language detection
_PT_FUNCTION_WORDS = {
    "que", "uma", "para", "com", "por", "são", "também", "mais", "sobre",
    "como", "mas", "dos", "das", "nos", "nas", "quando", "porque", "entre",
    "seus", "suas", "este", "esta", "estes", "estas", "isso", "essa",
}

# Max sentences per paragraph before flagging, keyed by doc_type
_PARA_LIMITS = {
    "concept-note": 4,
    "full-proposal": 4,
    "eoi": 4,
    "executive-summary": 3,
    "general": 5,
    "annual-report": 6,
    "monitoring-report": 7,
    "financial-report": 8,
    "assessment": 7,
    "tor": 6,
    "governance-review": 6,
    # Social media — short-form, single visual blocks
    "facebook-post": 2,
    "linkedin-post": 3,
    "instagram-caption": 1,
}

# Required discursive expressions per 300-word page, keyed by doc_type
_DISCURSIVE_TARGETS = {
    "concept-note": 2.0,
    "full-proposal": 2.0,
    "eoi": 1.5,
    "executive-summary": 2.0,
    "general": 1.0,
    "annual-report": 1.0,
    "monitoring-report": 0.5,
    "financial-report": 0.0,
    "assessment": 1.0,
    "tor": 0.5,
    "governance-review": 1.0,
    # Social media — discursive connectors are not expected in short-form posts
    "facebook-post": 0.0,
    "linkedin-post": 0.5,
    "instagram-caption": 0.0,
}

# Social doc_types where discursive_deficit is structurally inapplicable
_SOCIAL_DOC_TYPES = frozenset({"facebook-post", "linkedin-post", "instagram-caption"})


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    """Simple heuristic: count PT function words in lowercased text."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return "en"
    pt_count = sum(1 for w in words if w in _PT_FUNCTION_WORDS)
    ratio = pt_count / len(words)
    return "pt" if ratio >= 0.05 else "en"


# ---------------------------------------------------------------------------
# Text segmentation helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split into sentences, filter fragments under 10 chars."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= 10]


def _split_paragraphs(text: str) -> List[str]:
    """Split on blank lines."""
    paras = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paras if p.strip()]


def _sentence_word_count(sentence: str) -> int:
    return len(sentence.split())


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _detect_connector_repetition(text: str, language: str) -> Tuple[float, List[dict]]:
    """Flag connectors that appear more than once."""
    connectors = _CONNECTORS_EN if language == "en" else _CONNECTORS_EN + _CONNECTORS_PT
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


def _detect_hollow_intensifiers(text: str) -> Tuple[float, List[dict]]:
    """Detect hollow intensifier phrases."""
    lower = text.lower()
    findings = []
    hit_count = 0

    for pattern in _HOLLOW_INTENSIFIERS:
        for m in re.finditer(pattern, lower):
            hit_count += 1
            start = max(0, m.start())
            end = min(len(text), m.end() + 60)
            findings.append({"excerpt": text[start:end].strip() + "...", "pattern": pattern})

    sentences = _split_sentences(text)
    sentence_count = max(len(sentences), 1)
    score = min(1.0, hit_count / (sentence_count * 0.15))
    return round(score, 3), findings


def _detect_grandiose_openers(text: str, language: str) -> Tuple[float, List[dict]]:
    """Detect grandiose paragraph openings."""
    patterns = _GRANDIOSE_OPENERS_PT if language == "pt" else _GRANDIOSE_OPENERS_EN
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


def _detect_passive_voice(text: str) -> Tuple[float, List[dict]]:
    """Detect high passive voice density (>25% of sentences)."""
    sentences = _split_sentences(text)
    findings = []
    passive_count = 0

    for sentence in sentences:
        lower = sentence.lower()
        is_passive = any(re.search(pat, lower) for pat in _PASSIVE_PATTERNS)
        if is_passive:
            passive_count += 1
            findings.append({"excerpt": sentence[:150]})

    sentence_count = max(len(sentences), 1)
    passive_ratio = passive_count / sentence_count
    score = max(0.0, (passive_ratio - 0.25) / 0.75)  # 0 at 25%, 1.0 at 100%
    return round(score, 3), findings if passive_ratio > 0.25 else []


def _detect_paragraph_length(text: str, max_sentences: int = _PARA_LIMITS["general"]) -> Tuple[float, List[dict]]:
    """Detect paragraphs exceeding max_sentences sentences."""
    paragraphs = _split_paragraphs(text)
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


def _detect_discursive_deficit(text: str, target: float = _DISCURSIVE_TARGETS["general"]) -> Tuple[float, List[dict]]:
    """Detect fewer than target discursive expressions per ~300-word page."""
    if target == 0.0:
        return 0.0, []

    lower = text.lower()
    word_count = len(text.split())
    pages = max(word_count / 300, 1)

    hit_count = 0
    for pattern in _DISCURSIVE_EXPRESSIONS:
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


def _detect_generic_closings(text: str) -> Tuple[float, List[dict]]:
    """Detect generic AI closing phrases."""
    lower = text.lower()
    findings = []
    hit_count = 0

    for pattern in _GENERIC_CLOSINGS:
        for m in re.finditer(pattern, lower):
            hit_count += 1
            start = m.start()
            end = min(len(text), m.end() + 80)
            findings.append({"excerpt": text[start:end].strip()})

    score = min(1.0, hit_count * 0.5)  # 2 hits = score 1.0
    return round(score, 3), findings


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_ai_patterns(
    text: str,
    language: str = "auto",
    threshold: float = 0.25,
    doc_type: str = "general",
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

    if doc_type not in _PARA_LIMITS:
        valid = ", ".join(sorted(_PARA_LIMITS.keys()))
        return {"success": False, "error": f"Invalid doc_type '{doc_type}'. Must be one of: {valid}."}

    detected_language = _detect_language(text) if language == "auto" else language

    word_count = len(text.split())
    page_equivalent = round(word_count / 300, 1)

    para_limit = _PARA_LIMITS.get(doc_type, 5)
    discursive_target = _DISCURSIVE_TARGETS.get(doc_type, 1.0)

    try:
        # Run all detectors
        connector_score, connector_findings = _detect_connector_repetition(text, detected_language)
        intensifier_score, intensifier_findings = _detect_hollow_intensifiers(text)
        grandiose_score, grandiose_findings = _detect_grandiose_openers(text, detected_language)
        em_dash_score, em_dash_findings = _detect_em_dash_intercalation(text)
        monotony_score, monotony_findings = _detect_sentence_monotony(text)
        passive_score, passive_findings = _detect_passive_voice(text)
        para_len_score, para_len_findings = _detect_paragraph_length(text, max_sentences=para_limit)
        if doc_type in _SOCIAL_DOC_TYPES:
            discursive_score, discursive_findings = 0.0, []
        else:
            discursive_score, discursive_findings = _detect_discursive_deficit(text, target=discursive_target)
        listing_score, listing_findings = _detect_mechanical_listing(text)
        closing_score, closing_findings = _detect_generic_closings(text)

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
        return {"success": False, "error": str(e)}


def score_semantic_ai_likelihood(
    text: str,
    top_k: int = 10,
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

    collection = get_collection_names()["passages"]

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
        logger.error("score_semantic_ai_likelihood failed", error=str(e))
        return {"success": False, "error": str(e)}
