"""
Vocabulary intelligence layer: thesaurus-backed detection and suggestion of
AI-pattern words with rich semantic context for both English and Portuguese.
"""
import json
import re
from typing import Optional
from uuid import uuid4

import structlog

from src.sentry import capture_tool_error
from src.tools.collections import get_collection_names, ensure_user_collections_once
from src.tools.qdrant_errors import handle_qdrant_error
from src.tools.registry import VALID_DOMAINS, VALID_LANGUAGES

logger = structlog.get_logger(__name__)

# Module-level imports so tests can patch src.tools.thesaurus.*
try:
    from kbase.vector.sync_indexing import index_document, delete_document_vectors, check_document_indexed
    from kbase.vector.sync_search import semantic_search
    from kbase.vector.sync_client import get_qdrant_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue
except ImportError:
    index_document = None  # type: ignore
    delete_document_vectors = None  # type: ignore
    check_document_indexed = None  # type: ignore
    semantic_search = None  # type: ignore
    get_qdrant_client = None  # type: ignore
    Filter = None  # type: ignore
    FieldCondition = None  # type: ignore
    MatchValue = None  # type: ignore

VALID_PARTS_OF_SPEECH = {"verb", "noun", "adjective", "adverb", "phrase", "literary-device"}
VALID_REGISTERS = {"formal", "neutral", "informal", "institutional", "academic", "lyrical", "poetic", "colloquial"}


def _build_content(entry: dict) -> str:
    """Build the text content indexed for semantic search."""
    alternatives_text = "; ".join(
        f"{a['word']} ({a.get('meaning_nuance', '')})"
        for a in entry.get("alternatives", [])
    )
    parts = [
        f"Headword: {entry['headword']}",
        f"Definition: {entry.get('definition', '')}",
        f"Alternatives: {alternatives_text}" if alternatives_text else "",
        f"Why avoid: {entry.get('why_avoid', '')}",
        f"Collocations: {', '.join(entry.get('collocations', []))}",
        f"Example bad: {entry.get('example_bad', '')}",
        f"Example good: {entry.get('example_good', '')}",
        f"Domain: {entry.get('domain', 'general')}",
        f"Language: {entry.get('language', 'en')}",
    ]
    return "\n".join(p for p in parts if p)


def add_thesaurus_entry(
    headword: str,
    language: str = "en",
    domain: str = "general",
    definition: str = "",
    part_of_speech: str = "verb",
    register: str = "neutral",
    alternatives: Optional[list] = None,
    collocations: Optional[list] = None,
    why_avoid: str = "",
    example_bad: str = "",
    example_good: str = "",
    source: str = "manual",
    client_id: str = "default",
) -> dict:
    """
    Add a new vocabulary entry to the user's per-user writing_thesaurus collection.

    Args:
        headword: The word or phrase to flag/avoid.
        language: Language code (en, pt); must be in VALID_LANGUAGES.
        domain: Domain slug; must be in VALID_DOMAINS.
        definition: Short definition of what the word is.
        part_of_speech: Category (verb, noun, adjective, etc.).
        register: Formality/tone level (formal, neutral, informal, etc.).
        alternatives: List of dicts with word, meaning_nuance, register, when_to_use.
        collocations: List of common word phrases.
        why_avoid: Reason to avoid this word.
        example_bad: Example of poor usage.
        example_good: Example of correct alternative usage.
        source: Where this entry came from (manual, scrape, ai-detection, etc.).
        client_id: User identifier; defaults to "default" in stdio mode.

    Returns:
        {success, document_id, chunks_created, collection} on success.
        {success: False, error, existing_document_id (if duplicate)} on failure.

    Raises:
        Captures exceptions to Sentry via capture_tool_error.

    Example:
        result = add_thesaurus_entry(
            headword="leverage",
            language="en",
            domain="general",
            why_avoid="Overused in donor-speak",
            alternatives=[{"word": "use", "meaning_nuance": "simple, clear"}],
            client_id="user_456"
        )
        assert result["success"]
    """
    if not headword or not headword.strip():
        return {"success": False, "error": "headword cannot be empty"}
    if language not in VALID_LANGUAGES:
        return {"success": False, "error": f"Invalid language '{language}'. Must be one of: {sorted(VALID_LANGUAGES)}"}
    if domain not in VALID_DOMAINS:
        return {"success": False, "error": f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}"}
    if part_of_speech not in VALID_PARTS_OF_SPEECH:
        return {"success": False, "error": f"Invalid part_of_speech '{part_of_speech}'. Must be one of: {sorted(VALID_PARTS_OF_SPEECH)}"}
    if register not in VALID_REGISTERS:
        return {"success": False, "error": f"Invalid register '{register}'. Must be one of: {sorted(VALID_REGISTERS)}"}

    alternatives = alternatives or []
    collocations = collocations or []
    ensure_user_collections_once(client_id)
    collection = get_collection_names(client_id)["thesaurus"]

    # Duplicate check: same headword + language
    try:
        existing = semantic_search(
            collection_name=collection,
            query=headword,
            limit=5,
            filter_conditions={"language": language},
        )
        for hit in existing:
            if hit.get("metadata", {}).get("headword", "").lower() == headword.lower():
                return {
                    "success": False,
                    "error": f"Entry for '{headword}' ({language}) already exists. Use search_thesaurus to find its document_id and delete before re-adding.",
                    "existing_document_id": hit.get("document_id"),
                }
    except Exception:
        pass  # If search fails, proceed with insert

    document_id = str(uuid4())
    entry = {
        "headword": headword.strip(),
        "language": language,
        "domain": domain,
        "definition": definition,
        "part_of_speech": part_of_speech,
        "register": register,
        "alternatives": alternatives,
        "collocations": collocations,
        "why_avoid": why_avoid,
        "example_bad": example_bad,
        "example_good": example_good,
        "source": source,
        "entry_type": "thesaurus",
    }

    content = _build_content(entry)
    metadata = {**entry, "alternatives": json.dumps(alternatives), "collocations": json.dumps(collocations)}

    try:
        point_ids = index_document(
            collection_name=collection,
            document_id=document_id,
            title=headword,
            content=content,
            metadata=metadata,
            context_mode="metadata",
        )
        return {"success": True, "document_id": document_id, "chunks_created": len(point_ids), "collection": collection}
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="add_thesaurus_entry", collection=collection, headword=headword)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("Failed to add thesaurus entry", error=str(e), client_id=client_id)
        capture_tool_error(e, tool_name="add_thesaurus_entry", headword=headword, client_id=client_id)
        return {"success": False, "error": str(e)}


def search_thesaurus(
    query: str,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 8,
    client_id: str = "default",
) -> dict:
    """
    Semantic search across the user's thesaurus entries.

    Args:
        query: Search term (will be embedded and matched).
        language: Optional language filter (en, pt, etc.).
        domain: Optional domain filter.
        top_k: Maximum results to return (default 8).
        client_id: User identifier; defaults to "default" in stdio mode.

    Returns:
        {success, results: [{score, document_id, headword, language, domain, definition,
         part_of_speech, register, alternatives, collocations, why_avoid, example_bad,
         example_good, source}], total} on success.
        {success: False, error, results: []} on failure.

    Raises:
        Captures exceptions to Sentry via capture_tool_error.

    Example:
        result = search_thesaurus(query="leverage", language="en", client_id="user_456")
        assert result["success"]
        for hit in result["results"]:
            print(f"{hit['headword']}: {hit['score']}")
    """
    if not query or not query.strip():
        return {"success": False, "error": "query cannot be empty"}

    ensure_user_collections_once(client_id)
    collection = get_collection_names(client_id)["thesaurus"]
    filter_conditions = {}
    if language:
        filter_conditions["language"] = language
    if domain:
        filter_conditions["domain"] = domain

    try:
        raw = semantic_search(
            collection_name=collection,
            query=query,
            limit=top_k,
            filter_conditions=filter_conditions if filter_conditions else None,
        )
        results = []
        for r in raw:
            meta = r.get("metadata", {})
            results.append({
                "score": round(r["score"], 4),
                "document_id": r.get("document_id"),
                "headword": meta.get("headword", r.get("title", "")),
                "language": meta.get("language"),
                "domain": meta.get("domain"),
                "definition": meta.get("definition", ""),
                "part_of_speech": meta.get("part_of_speech", ""),
                "register": meta.get("register", ""),
                "alternatives": json.loads(meta.get("alternatives", "[]")),
                "collocations": json.loads(meta.get("collocations", "[]")),
                "why_avoid": meta.get("why_avoid", ""),
                "example_bad": meta.get("example_bad", ""),
                "example_good": meta.get("example_good", ""),
                "source": meta.get("source", ""),
            })
        return {"success": True, "results": results, "total": len(results)}
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="search_thesaurus", collection=collection)
        if qdrant_result is not None:
            qdrant_result["results"] = []
            return qdrant_result
        logger.error("Thesaurus search failed", error=str(e), client_id=client_id)
        capture_tool_error(e, tool_name="search_thesaurus", client_id=client_id)
        return {"success": False, "error": str(e), "results": []}


def _search_terms_fallback(word: str, language: str) -> list:
    """Search writing_terms collection for a word; returns simplified alternative list."""
    try:
        from src.tools.terms import search_terms
        result = search_terms(query=word, language=language, top_k=5)
        if not result.get("success"):
            return []
        return [
            {"preferred": r["preferred"], "avoid": r["avoid"], "why": r["why"]}
            for r in result.get("results", [])
            if r.get("preferred")
        ]
    except Exception:
        return []


def suggest_alternatives(
    word: Optional[str] = None,
    language: str = "en",
    domain: str = "general",
    context_sentence: Optional[str] = None,
    top_k: int = 5,
    phrase: Optional[str] = None,
    client_id: str = "default",
) -> dict:
    """
    Look up a word in the user's thesaurus and return rich alternatives with semantic context.

    Falls back to search_terms if the word is not in the thesaurus.

    Args:
        word: The word to look up.
        language: Language code (en, pt, etc.).
        domain: Domain slug.
        context_sentence: Reserved for future semantic re-ranking; currently unused.
        top_k: Maximum alternatives to return (default 5).
        phrase: Alias for word — accepted for backward compatibility.
        client_id: User identifier; defaults to "default" in stdio mode.

    Returns:
        {success, found_in_thesaurus, headword, language, domain, definition, part_of_speech,
         register, why_avoid, alternatives, collocations, example_bad, example_good, source,
         document_id} on success if found in thesaurus.
        {success, found_in_thesaurus: False, headword, language, note, alternatives} if fallback
        to terminology dictionary.
        {success: False, error} on failure.

    Raises:
        Captures exceptions to Sentry via capture_tool_error.

    Example:
        result = suggest_alternatives(word="leverage", language="en", client_id="user_456")
        assert result["success"]
        print(f"Found: {result['found_in_thesaurus']}")
        for alt in result["alternatives"]:
            print(f"{alt['word']}: {alt.get('meaning_nuance', '')}")
    """
    word = word or phrase
    if not word or not word.strip():
        return {"success": False, "error": "word cannot be empty"}
    if language not in VALID_LANGUAGES:
        return {"success": False, "error": f"Invalid language '{language}'. Must be one of: {sorted(VALID_LANGUAGES)}"}
    if domain not in VALID_DOMAINS:
        return {"success": False, "error": f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}"}

    ensure_user_collections_once(client_id)
    collection = get_collection_names(client_id)["thesaurus"]

    try:
        raw = semantic_search(
            collection_name=collection,
            query=word.strip(),
            limit=10,
            filter_conditions={"language": language},
        )
    except Exception as e:
        raw = []
        logger.warning("Thesaurus search failed in suggest_alternatives", error=str(e), client_id=client_id)

    # Find an exact headword match
    match = None
    for r in raw:
        if r.get("metadata", {}).get("headword", "").lower() == word.strip().lower():
            match = r
            break

    if match:
        meta = match.get("metadata", {})
        alternatives = json.loads(meta.get("alternatives", "[]"))[:top_k]
        return {
            "success": True,
            "found_in_thesaurus": True,
            "headword": meta.get("headword"),
            "language": meta.get("language"),
            "domain": meta.get("domain"),
            "definition": meta.get("definition", ""),
            "part_of_speech": meta.get("part_of_speech", ""),
            "register": meta.get("register", ""),
            "why_avoid": meta.get("why_avoid", ""),
            "alternatives": alternatives,
            "collocations": json.loads(meta.get("collocations", "[]")),
            "example_bad": meta.get("example_bad", ""),
            "example_good": meta.get("example_good", ""),
            "source": meta.get("source", ""),
            "document_id": match.get("document_id"),
        }

    # Fallback to terms collection
    fallback = _search_terms_fallback(word, language)
    return {
        "success": True,
        "found_in_thesaurus": False,
        "headword": word,
        "language": language,
        "note": "Word not found in thesaurus. Showing results from terminology dictionary.",
        "alternatives": fallback,
    }


def flag_vocabulary(
    text: str,
    language: str = "en",
    domain: str = "general",
    client_id: str = "default",
) -> dict:
    """
    Scan text for words present in the user's thesaurus as flagged headwords.

    Returns flagged words with occurrence counts and alternative previews.
    Complements score_writing_patterns (structural) with lexical detection.

    Args:
        text: Document text to scan.
        language: Language code (en, pt, etc.).
        domain: Domain slug.
        client_id: User identifier; defaults to "default" in stdio mode.

    Returns:
        {success, flagged_count, verdict, flagged: [{headword, occurrences, why_avoid,
         alternatives_preview, document_id}], language, domain, word_count} on success.
        {success: False, error} on failure.
        Verdict: "clean" (0 flagged), "review" (1–3 flagged), "ai-sounding" (4+ flagged).

    Raises:
        Captures exceptions to Sentry via capture_tool_error.

    Example:
        result = flag_vocabulary(
            text="We will leverage our robust stakeholder network...",
            language="en",
            client_id="user_456"
        )
        assert result["success"]
        print(result["verdict"])  # "clean", "review", or "ai-sounding"
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    ensure_user_collections_once(client_id)
    collection = get_collection_names(client_id)["thesaurus"]

    # --- Step 1: fetch all headwords for this language in one query ---
    try:
        all_entries = semantic_search(
            collection_name=collection,
            query="word vocabulary avoid alternative language",
            limit=500,
            filter_conditions={"language": language},
        )
    except Exception:
        all_entries = []

    headword_map: dict[str, dict] = {}  # headword (lower) -> metadata
    for entry in all_entries:
        meta = entry.get("metadata", {})
        hw = meta.get("headword", "").lower()
        if hw:
            headword_map[hw] = {"meta": meta, "document_id": entry.get("document_id")}

    if not headword_map:
        return {
            "success": True,
            "flagged_count": 0,
            "verdict": "clean",
            "flagged": [],
            "language": language,
            "domain": domain,
            "word_count": 0,
        }

    # --- Step 2: build candidate set from text ---
    tokens = [re.sub(r"[^\w\-]", "", w).lower() for w in text.split()]
    tokens = [t for t in tokens if len(t) > 2]

    candidates: set[str] = set(tokens)
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            candidates.add(" ".join(tokens[i: i + n]))

    # --- Step 3: intersect candidates against known headwords ---
    flagged = []
    for hw in candidates & headword_map.keys():
        entry_data = headword_map[hw]
        meta = entry_data["meta"]
        alternatives_preview = json.loads(meta.get("alternatives", "[]"))[:3]
        if " " in hw or "-" in hw:
            occurrences = text.lower().count(hw)
        else:
            occurrences = tokens.count(hw)
        flagged.append({
            "headword": meta.get("headword"),
            "occurrences": occurrences,
            "why_avoid": meta.get("why_avoid", ""),
            "alternatives_preview": alternatives_preview,
            "document_id": entry_data["document_id"],
        })

    verdict = "clean" if not flagged else ("review" if len(flagged) <= 3 else "ai-sounding")
    return {
        "success": True,
        "flagged_count": len(flagged),
        "verdict": verdict,
        "flagged": flagged,
        "language": language,
        "domain": domain,
        "word_count": len(tokens),
    }
