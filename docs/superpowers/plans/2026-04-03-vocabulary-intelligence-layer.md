# Vocabulary Intelligence Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thesaurus-backed vocabulary intelligence layer to the mcp-writing-library that detects AI-pattern vocabulary and suggests rich, context-aware alternatives in English and Portuguese.

**Architecture:** A new `writing_thesaurus` Qdrant collection stores richly annotated headword entries (definition, register, alternatives with nuance, collocations, why_avoid). Four new MCP tools expose it: `suggest_alternatives`, `add_thesaurus_entry`, `search_thesaurus`, and `flag_vocabulary`. A seed script populates the collection from a curated word list enriched with Wordnik (EN) and Dicionário Aberto XML (PT).

**Tech Stack:** Python 3.11, kbase-core (Qdrant hybrid search), requests (Wordnik API), xml.etree.ElementTree (Dicionário Aberto TEI XML, stdlib — no new dep needed), FastMCP, structlog.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/tools/thesaurus.py` | **Create** | All thesaurus logic: add, search, suggest, flag |
| `src/tools/collections.py` | **Modify** | Register `writing_thesaurus` collection |
| `src/server.py` | **Modify** | Register 4 new MCP tool endpoints |
| `scripts/seed_thesaurus.py` | **Create** | Seeding pipeline: curated list → Wordnik EN → Dicionário Aberto PT → batch import |
| `scripts/data/thesaurus_wordlist.py` | **Create** | The ~80-word AI-pattern seed list with manually curated PT equivalents |
| `tests/test_thesaurus.py` | **Create** | Unit tests for all 4 tool functions |
| `pyproject.toml` | **No change** | `requests` already present; stdlib XML is sufficient for Dicionário Aberto |
| `CLAUDE.md` | **Modify** | Document new thesaurus collection and tools |

---

## Task 1: Register `writing_thesaurus` collection

**Files:**
- Modify: `src/tools/collections.py`
- Test: `tests/test_thesaurus.py` (bootstrap)

- [ ] **Step 1.1: Write a failing test for collection registration**

```python
# tests/test_thesaurus.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock


def test_get_collection_names_includes_thesaurus():
    from src.tools.collections import get_collection_names
    names = get_collection_names()
    assert "thesaurus" in names
    assert names["thesaurus"] == "writing_thesaurus"
```

- [ ] **Step 1.2: Run test to confirm it fails**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
source .venv/bin/activate
python -m pytest tests/test_thesaurus.py::test_get_collection_names_includes_thesaurus -v
```

Expected: `FAILED — KeyError: 'thesaurus'` or `AssertionError`

- [ ] **Step 1.3: Add `thesaurus` to `get_collection_names()` and `setup_collections()`**

In `src/tools/collections.py`, update `get_collection_names()`:

```python
def get_collection_names() -> dict:
    """Return configured collection names from environment."""
    return {
        "passages": os.getenv("COLLECTION_PASSAGES", "writing_passages"),
        "terms": os.getenv("COLLECTION_TERMS", "writing_terms"),
        "style_profiles": os.getenv("COLLECTION_STYLE_PROFILES", "writing_style_profiles"),
        "rubrics": os.getenv("COLLECTION_RUBRICS", "writing_rubrics"),
        "templates": os.getenv("COLLECTION_TEMPLATES", "writing_templates"),
        "thesaurus": os.getenv("COLLECTION_THESAURUS", "writing_thesaurus"),
    }
```

No change needed to `setup_collections()` or `get_stats()` — both already iterate `get_collection_names()` dynamically.

- [ ] **Step 1.4: Run test to confirm it passes**

```bash
python -m pytest tests/test_thesaurus.py::test_get_collection_names_includes_thesaurus -v
```

Expected: `PASSED`

- [ ] **Step 1.5: Commit**

```bash
git add src/tools/collections.py tests/test_thesaurus.py
git commit -m "feat: register writing_thesaurus Qdrant collection"
```

---

## Task 2: Create `src/tools/thesaurus.py` — core functions

**Files:**
- Create: `src/tools/thesaurus.py`
- Modify: `tests/test_thesaurus.py`

This task implements all four tool functions. Tests use mocks for Qdrant — no live connection needed.

- [ ] **Step 2.1: Write failing tests for `add_thesaurus_entry`**

Append to `tests/test_thesaurus.py`:

```python
def test_add_thesaurus_entry_returns_document_id():
    with patch("src.tools.thesaurus.index_document") as mock_index:
        mock_index.return_value = ["point-1"]
        from src.tools.thesaurus import add_thesaurus_entry
        result = add_thesaurus_entry(
            headword="leverage",
            language="en",
            domain="general",
            definition="To use something to maximum advantage.",
            part_of_speech="verb",
            register="institutional",
            alternatives=[
                {"word": "use", "meaning_nuance": "Direct and clear", "register": "neutral", "when_to_use": "Default replacement in most contexts"},
            ],
            collocations=["leverage resources", "leverage partnerships"],
            why_avoid="Overused in AI-generated donor proposals; sounds mechanical.",
            example_bad="We will leverage our networks to ensure impact.",
            example_good="We will draw on our networks to extend reach.",
            source="manual",
        )
        assert result["success"] is True
        assert "document_id" in result


def test_add_thesaurus_entry_rejects_invalid_domain():
    from src.tools.thesaurus import add_thesaurus_entry
    result = add_thesaurus_entry(
        headword="leverage",
        language="en",
        domain="invalid-domain",
        definition="To use something to maximum advantage.",
        part_of_speech="verb",
        register="institutional",
        alternatives=[],
    )
    assert result["success"] is False
    assert "domain" in result["error"]


def test_add_thesaurus_entry_rejects_invalid_language():
    from src.tools.thesaurus import add_thesaurus_entry
    result = add_thesaurus_entry(
        headword="leverage",
        language="fr",
        domain="general",
        definition="...",
        part_of_speech="verb",
        register="neutral",
        alternatives=[],
    )
    assert result["success"] is False
    assert "language" in result["error"]


def test_add_thesaurus_entry_rejects_duplicate(monkeypatch):
    """If headword+language already exists, return error."""
    import src.tools.thesaurus as mod
    monkeypatch.setattr(mod, "semantic_search", lambda **kw: [
        {"metadata": {"headword": "leverage", "language": "en"}, "score": 0.99, "document_id": "abc", "title": "leverage"}
    ])
    result = mod.add_thesaurus_entry(
        headword="leverage",
        language="en",
        domain="general",
        definition="...",
        part_of_speech="verb",
        register="neutral",
        alternatives=[],
    )
    assert result["success"] is False
    assert "already exists" in result["error"]
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_thesaurus.py -k "add_thesaurus" -v
```

Expected: `ModuleNotFoundError: No module named 'src.tools.thesaurus'`

- [ ] **Step 2.3: Create `src/tools/thesaurus.py` with `add_thesaurus_entry`**

```python
"""
Vocabulary intelligence layer: thesaurus-backed detection and suggestion of
AI-pattern words with rich semantic context for both English and Portuguese.
"""
import json
from typing import Optional
from uuid import uuid4

import structlog

from src.tools.collections import get_collection_names
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

VALID_PARTS_OF_SPEECH = {"verb", "noun", "adjective", "adverb", "phrase"}
VALID_REGISTERS = {"formal", "neutral", "informal", "institutional", "academic"}


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
) -> dict:
    """Add a new entry to the writing_thesaurus collection."""
    if not headword or not headword.strip():
        return {"success": False, "error": "headword cannot be empty"}
    if language not in VALID_LANGUAGES:
        return {"success": False, "error": f"Invalid language '{language}'. Must be one of: {sorted(VALID_LANGUAGES)}"}
    if domain not in VALID_DOMAINS:
        return {"success": False, "error": f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}"}

    alternatives = alternatives or []
    collocations = collocations or []
    collection = get_collection_names()["thesaurus"]

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
        logger.error("Failed to add thesaurus entry", error=str(e))
        return {"success": False, "error": str(e)}
```

- [ ] **Step 2.4: Run add tests to confirm they pass**

```bash
python -m pytest tests/test_thesaurus.py -k "add_thesaurus" -v
```

Expected: All 4 tests `PASSED`

- [ ] **Step 2.5: Write failing tests for `search_thesaurus`**

Append to `tests/test_thesaurus.py`:

```python
def test_search_thesaurus_returns_results(monkeypatch):
    import src.tools.thesaurus as mod
    import json
    monkeypatch.setattr(mod, "semantic_search", lambda **kw: [
        {
            "score": 0.92,
            "document_id": "abc-123",
            "title": "leverage",
            "metadata": {
                "headword": "leverage",
                "language": "en",
                "domain": "general",
                "definition": "To use something to maximum advantage.",
                "part_of_speech": "verb",
                "register": "institutional",
                "alternatives": json.dumps([{"word": "use", "meaning_nuance": "Direct", "register": "neutral", "when_to_use": "Default"}]),
                "collocations": json.dumps(["leverage resources"]),
                "why_avoid": "Overused in AI text.",
                "example_bad": "We will leverage our networks.",
                "example_good": "We will draw on our networks.",
                "source": "manual",
                "entry_type": "thesaurus",
            },
        }
    ])
    result = mod.search_thesaurus(query="leverage")
    assert result["success"] is True
    assert len(result["results"]) == 1
    entry = result["results"][0]
    assert entry["headword"] == "leverage"
    assert isinstance(entry["alternatives"], list)
    assert entry["alternatives"][0]["word"] == "use"


def test_search_thesaurus_empty_query():
    from src.tools.thesaurus import search_thesaurus
    result = search_thesaurus(query="")
    assert result["success"] is False
    assert "query" in result["error"]
```

- [ ] **Step 2.6: Run search tests to confirm they fail**

```bash
python -m pytest tests/test_thesaurus.py -k "search_thesaurus" -v
```

Expected: `AttributeError: module 'src.tools.thesaurus' has no attribute 'search_thesaurus'`

- [ ] **Step 2.7: Add `search_thesaurus` to `src/tools/thesaurus.py`**

Append to `src/tools/thesaurus.py`:

```python
def search_thesaurus(
    query: str,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 8,
) -> dict:
    """Semantic search across thesaurus entries."""
    if not query or not query.strip():
        return {"success": False, "error": "query cannot be empty"}

    collection = get_collection_names()["thesaurus"]
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
        logger.error("Thesaurus search failed", error=str(e))
        return {"success": False, "error": str(e), "results": []}
```

- [ ] **Step 2.8: Run search tests to confirm they pass**

```bash
python -m pytest tests/test_thesaurus.py -k "search_thesaurus" -v
```

Expected: Both tests `PASSED`

- [ ] **Step 2.9: Write failing tests for `suggest_alternatives`**

Append to `tests/test_thesaurus.py`:

```python
def test_suggest_alternatives_found(monkeypatch):
    import src.tools.thesaurus as mod
    import json
    monkeypatch.setattr(mod, "semantic_search", lambda **kw: [
        {
            "score": 0.97,
            "document_id": "abc-123",
            "title": "leverage",
            "metadata": {
                "headword": "leverage",
                "language": "en",
                "domain": "general",
                "definition": "To use something to maximum advantage.",
                "part_of_speech": "verb",
                "register": "institutional",
                "alternatives": json.dumps([
                    {"word": "use", "meaning_nuance": "Direct and clear", "register": "neutral", "when_to_use": "Default"},
                    {"word": "draw on", "meaning_nuance": "Implies existing resource", "register": "neutral", "when_to_use": "When referring to existing assets"},
                    {"word": "apply", "meaning_nuance": "More technical", "register": "formal", "when_to_use": "When precision matters"},
                ]),
                "collocations": json.dumps(["leverage resources", "leverage partnerships"]),
                "why_avoid": "Overused in AI text.",
                "example_bad": "We will leverage our networks.",
                "example_good": "We will draw on our networks.",
                "source": "manual",
                "entry_type": "thesaurus",
            },
        }
    ])
    # Also mock search_terms fallback as returning nothing
    monkeypatch.setattr(mod, "_search_terms_fallback", lambda word, language: [])
    result = mod.suggest_alternatives(word="leverage", language="en", domain="general")
    assert result["success"] is True
    assert result["found_in_thesaurus"] is True
    assert len(result["alternatives"]) == 3
    assert result["alternatives"][0]["word"] == "use"
    assert "why_avoid" in result
    assert "definition" in result


def test_suggest_alternatives_not_found_falls_back(monkeypatch):
    """When word not in thesaurus, fall back to search_terms."""
    import src.tools.thesaurus as mod
    monkeypatch.setattr(mod, "semantic_search", lambda **kw: [])
    monkeypatch.setattr(mod, "_search_terms_fallback", lambda word, language: [
        {"preferred": "use", "avoid": "leverage", "why": "More direct"}
    ])
    result = mod.suggest_alternatives(word="leverage", language="en", domain="general")
    assert result["success"] is True
    assert result["found_in_thesaurus"] is False
    assert len(result["alternatives"]) >= 1


def test_suggest_alternatives_empty_word():
    from src.tools.thesaurus import suggest_alternatives
    result = suggest_alternatives(word="", language="en", domain="general")
    assert result["success"] is False
    assert "word" in result["error"]
```

- [ ] **Step 2.10: Run suggest tests to confirm they fail**

```bash
python -m pytest tests/test_thesaurus.py -k "suggest_alternatives" -v
```

Expected: `AttributeError: module 'src.tools.thesaurus' has no attribute 'suggest_alternatives'`

- [ ] **Step 2.11: Add `suggest_alternatives` and `_search_terms_fallback` to `src/tools/thesaurus.py`**

Append to `src/tools/thesaurus.py`:

```python
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
    word: str,
    language: str = "en",
    domain: str = "general",
    context_sentence: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Look up a word in the thesaurus and return rich alternatives with semantic context.

    Falls back to search_terms if the word is not in the thesaurus.
    """
    if not word or not word.strip():
        return {"success": False, "error": "word cannot be empty"}

    collection = get_collection_names()["thesaurus"]

    try:
        raw = semantic_search(
            collection_name=collection,
            query=word.strip(),
            limit=10,
            filter_conditions={"language": language},
        )
    except Exception as e:
        raw = []
        logger.warning("Thesaurus search failed in suggest_alternatives", error=str(e))

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
```

- [ ] **Step 2.12: Run suggest tests to confirm they pass**

```bash
python -m pytest tests/test_thesaurus.py -k "suggest_alternatives" -v
```

Expected: All 3 tests `PASSED`

- [ ] **Step 2.13: Write failing tests for `flag_vocabulary`**

Append to `tests/test_thesaurus.py`:

```python
def test_flag_vocabulary_detects_headwords(monkeypatch):
    import src.tools.thesaurus as mod
    import json

    def mock_search(collection_name, query, limit, filter_conditions=None):
        entries = {
            "leverage": {
                "headword": "leverage",
                "language": "en",
                "domain": "general",
                "why_avoid": "AI overuse.",
                "alternatives": json.dumps([{"word": "use", "meaning_nuance": "Direct", "register": "neutral", "when_to_use": "Default"}]),
                "collocations": "[]",
                "definition": "...", "part_of_speech": "verb", "register": "institutional",
                "example_bad": "", "example_good": "", "source": "manual", "entry_type": "thesaurus",
            },
            "ensure": {
                "headword": "ensure",
                "language": "en",
                "domain": "general",
                "why_avoid": "Weak AI filler.",
                "alternatives": json.dumps([{"word": "guarantee", "meaning_nuance": "Stronger commitment", "register": "formal", "when_to_use": "High-stakes commitments"}]),
                "collocations": "[]",
                "definition": "...", "part_of_speech": "verb", "register": "institutional",
                "example_bad": "", "example_good": "", "source": "manual", "entry_type": "thesaurus",
            },
        }
        hits = []
        for headword, meta in entries.items():
            if headword in query.lower():
                hits.append({"score": 0.95, "document_id": f"id-{headword}", "title": headword, "metadata": meta})
        return hits

    monkeypatch.setattr(mod, "semantic_search", mock_search)
    text = "We will leverage our networks to ensure maximum impact."
    result = mod.flag_vocabulary(text=text, language="en", domain="general")
    assert result["success"] is True
    flagged = [f["headword"] for f in result["flagged"]]
    assert "leverage" in flagged
    assert "ensure" in flagged
    assert result["flagged_count"] == 2


def test_flag_vocabulary_empty_text():
    from src.tools.thesaurus import flag_vocabulary
    result = flag_vocabulary(text="", language="en", domain="general")
    assert result["success"] is False
    assert "text" in result["error"]


def test_flag_vocabulary_clean_text(monkeypatch):
    import src.tools.thesaurus as mod
    monkeypatch.setattr(mod, "semantic_search", lambda **kw: [])
    result = mod.flag_vocabulary(text="The team met on Tuesday to discuss the findings.", language="en", domain="general")
    assert result["success"] is True
    assert result["flagged_count"] == 0
    assert result["verdict"] == "clean"
```

- [ ] **Step 2.14: Run flag tests to confirm they fail**

```bash
python -m pytest tests/test_thesaurus.py -k "flag_vocabulary" -v
```

Expected: `AttributeError: module 'src.tools.thesaurus' has no attribute 'flag_vocabulary'`

- [ ] **Step 2.15: Add `flag_vocabulary` to `src/tools/thesaurus.py`**

Append to `src/tools/thesaurus.py`:

```python
def flag_vocabulary(
    text: str,
    language: str = "en",
    domain: str = "general",
) -> dict:
    """
    Scan text for words present in the thesaurus as AI-pattern headwords.

    Returns flagged words with positions and alternative previews.
    Complements score_ai_patterns (structural) with lexical detection.
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    collection = get_collection_names()["thesaurus"]
    words = text.lower().split()
    seen_headwords: set = set()
    flagged = []

    for word in set(words):  # deduplicate before querying
        clean_word = word.strip(".,;:!?\"'()")
        if not clean_word or len(clean_word) < 3:
            continue
        try:
            hits = semantic_search(
                collection_name=collection,
                query=clean_word,
                limit=5,
                filter_conditions={"language": language},
            )
        except Exception:
            continue

        for hit in hits:
            headword = hit.get("metadata", {}).get("headword", "").lower()
            if headword == clean_word and headword not in seen_headwords:
                seen_headwords.add(headword)
                meta = hit.get("metadata", {})
                alternatives_preview = json.loads(meta.get("alternatives", "[]"))[:3]
                flagged.append({
                    "headword": meta.get("headword"),
                    "occurrences": words.count(clean_word),
                    "why_avoid": meta.get("why_avoid", ""),
                    "alternatives_preview": alternatives_preview,
                    "document_id": hit.get("document_id"),
                })
                break

    verdict = "clean" if not flagged else ("review" if len(flagged) <= 3 else "ai-sounding")
    return {
        "success": True,
        "flagged_count": len(flagged),
        "verdict": verdict,
        "flagged": flagged,
        "language": language,
        "domain": domain,
        "word_count": len(words),
    }
```

- [ ] **Step 2.16: Run all thesaurus tests**

```bash
python -m pytest tests/test_thesaurus.py -v
```

Expected: All tests `PASSED`

- [ ] **Step 2.17: Commit**

```bash
git add src/tools/thesaurus.py tests/test_thesaurus.py
git commit -m "feat: add thesaurus tool module (add, search, suggest, flag)"
```

---

## Task 3: Register thesaurus tools in MCP server

**Files:**
- Modify: `src/server.py`

- [ ] **Step 3.1: Add four tool endpoints to `src/server.py`**

At the top of `src/server.py`, add to the docstring tools list (after `detect_authorship_shift`):

```
    suggest_alternatives     — rich alternatives for a word with meaning, register, and usage context
    add_thesaurus_entry      — add a new AI-pattern word to the thesaurus
    search_thesaurus         — semantic search across thesaurus entries
    flag_vocabulary          — scan text for AI-pattern vocabulary headwords
```

At the end of `src/server.py`, append:

```python
@mcp.tool()
def suggest_alternatives(
    word: str,
    language: str = "en",
    domain: str = "general",
    context_sentence: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Look up a word in the vocabulary thesaurus and return rich alternatives.

    Returns definition, register, meaning nuances, collocations, and why the word sounds AI-generated.
    Falls back to search_terms if the word is not in the thesaurus.
    Use this when drafting or reviewing text and you want to replace an overused or AI-sounding word.

    Args:
        word: The word to look up (e.g. "leverage", "robust", "ensure")
        language: Language of the word: en|pt
        domain: Thematic domain: srhr|governance|climate|general|m-and-e|health|finance|org
        context_sentence: Optional sentence where the word appears (used for ranking)
        top_k: Maximum alternatives to return (default 5)

    Returns:
        definition, why_avoid, alternatives (with word, meaning_nuance, register, when_to_use),
        collocations, example_bad, example_good. found_in_thesaurus flag indicates source.
    """
    from src.tools.thesaurus import suggest_alternatives as _suggest
    return _suggest(word=word, language=language, domain=domain,
                    context_sentence=context_sentence, top_k=top_k)


@mcp.tool()
def add_thesaurus_entry(
    headword: str,
    language: str = "en",
    domain: str = "general",
    definition: str = "",
    part_of_speech: str = "verb",
    register: str = "neutral",
    alternatives: Optional[List[dict]] = None,
    collocations: Optional[List[str]] = None,
    why_avoid: str = "",
    example_bad: str = "",
    example_good: str = "",
    source: str = "manual",
) -> dict:
    """
    Add a new AI-pattern word to the vocabulary thesaurus.

    Use this when you encounter a word that is overused or sounds AI-generated
    and you want to document it with its alternatives for future reference.

    Args:
        headword: The word to flag (e.g. "leverage")
        language: Language: en|pt
        domain: Thematic domain: srhr|governance|climate|general|m-and-e|health|finance|org
        definition: Concise definition of the headword
        part_of_speech: verb|noun|adjective|adverb|phrase
        register: formal|neutral|informal|institutional|academic
        alternatives: List of dicts: [{word, meaning_nuance, register, when_to_use}]
        collocations: Common collocations to flag (e.g. ["robust framework"])
        why_avoid: Why this word sounds AI-generated or overused
        example_bad: Sentence using the headword poorly
        example_good: Sentence using a preferred alternative
        source: Origin: manual|dicionario-aberto|wordnik|harvested

    Returns:
        document_id on success; error if duplicate or invalid input
    """
    from src.tools.thesaurus import add_thesaurus_entry as _add
    return _add(headword=headword, language=language, domain=domain,
                definition=definition, part_of_speech=part_of_speech,
                register=register, alternatives=alternatives or [],
                collocations=collocations or [], why_avoid=why_avoid,
                example_bad=example_bad, example_good=example_good, source=source)


@mcp.tool()
def search_thesaurus(
    query: str,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 8,
) -> dict:
    """
    Semantic search across thesaurus entries.

    Use to explore what AI-pattern words are stored, or to find entries
    related to a concept (e.g. "governance vocabulary", "verbs for action").

    Args:
        query: What you are searching for
        language: Filter by language: en|pt
        domain: Filter by domain: srhr|governance|climate|general|m-and-e|health|finance|org
        top_k: Number of results (default 8)

    Returns:
        List of matching entries with full metadata including alternatives
    """
    from src.tools.thesaurus import search_thesaurus as _search
    return _search(query=query, language=language, domain=domain, top_k=top_k)


@mcp.tool()
def flag_vocabulary(
    text: str,
    language: str = "en",
    domain: str = "general",
) -> dict:
    """
    Scan text for AI-pattern vocabulary headwords present in the thesaurus.

    Use alongside score_ai_patterns (which catches structural patterns) to
    get lexical-level flagging. Returns flagged words with occurrence counts
    and a preview of alternatives.

    Args:
        text: The text to scan
        language: Language of the text: en|pt
        domain: Thematic domain: srhr|governance|climate|general|m-and-e|health|finance|org

    Returns:
        flagged_count, verdict (clean|review|ai-sounding), list of flagged entries
        with headword, occurrences, why_avoid, and alternatives_preview
    """
    from src.tools.thesaurus import flag_vocabulary as _flag
    return _flag(text=text, language=language, domain=domain)
```

- [ ] **Step 3.2: Verify server imports cleanly**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
source .venv/bin/activate
python -c "from src.server import mcp; print('OK')"
```

Expected: `OK` with no errors

- [ ] **Step 3.3: Commit**

```bash
git add src/server.py
git commit -m "feat: register thesaurus MCP endpoints in server"
```

---

## Task 4: Create the seed word list

**Files:**
- Create: `scripts/data/thesaurus_wordlist.py`

This file is the curated seed list — the human-reviewed foundation before API enrichment.

- [ ] **Step 4.1: Create `scripts/data/__init__.py`**

```bash
touch /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library/scripts/data/__init__.py
```

- [ ] **Step 4.2: Create `scripts/data/thesaurus_wordlist.py`**

```python
"""
Curated seed list of AI-pattern words for the vocabulary intelligence layer.

Each entry is the minimum viable record. The seed_thesaurus.py script enriches
these with Wordnik (EN) and Dicionário Aberto (PT) definitions and examples.

Fields:
    headword     — the word to flag
    language     — "en" or "pt"
    pt_equivalent — the PT headword that maps to this EN word (EN entries only)
    domain       — default "general"; override for domain-specific jargon
    why_avoid    — brief rationale (will be enriched by seed script)
    alternatives — list of {word, meaning_nuance, register, when_to_use}
"""

EN_WORDS = [
    {
        "headword": "leverage",
        "pt_equivalent": "alavancar",
        "domain": "general",
        "why_avoid": "Corporate jargon that distances the reader; AI writers reach for it reflexively.",
        "alternatives": [
            {"word": "use", "meaning_nuance": "Direct and clear", "register": "neutral", "when_to_use": "Default replacement in most contexts"},
            {"word": "draw on", "meaning_nuance": "Implies existing resource or relationship", "register": "neutral", "when_to_use": "When referring to existing assets or relationships"},
            {"word": "apply", "meaning_nuance": "More precise, implies skill", "register": "formal", "when_to_use": "When describing deliberate application of expertise"},
            {"word": "build on", "meaning_nuance": "Implies progressive development", "register": "neutral", "when_to_use": "When extending prior work"},
        ],
        "collocations": ["leverage resources", "leverage partnerships", "leverage expertise", "leverage synergies"],
        "example_bad": "We will leverage our networks to ensure maximum impact.",
        "example_good": "We will draw on our networks to extend the programme's reach.",
    },
    {
        "headword": "ensure",
        "pt_equivalent": "assegurar",
        "domain": "general",
        "why_avoid": "Weak AI filler that promises without specifying how. Overused in proposals.",
        "alternatives": [
            {"word": "guarantee", "meaning_nuance": "Stronger, implies binding commitment", "register": "formal", "when_to_use": "When a contractual commitment is implied"},
            {"word": "confirm", "meaning_nuance": "Verification-oriented", "register": "neutral", "when_to_use": "When verifying a state rather than creating it"},
            {"word": "secure", "meaning_nuance": "Implies active effort to obtain", "register": "neutral", "when_to_use": "When referring to obtaining approvals or resources"},
            {"word": "verify", "meaning_nuance": "Evidence-based confirmation", "register": "formal", "when_to_use": "Quality assurance contexts"},
        ],
        "collocations": ["ensure quality", "ensure compliance", "ensure sustainability", "ensure impact"],
        "example_bad": "We will ensure that all activities are implemented on time.",
        "example_good": "Activities will be tracked weekly against a Gantt chart to maintain schedule.",
    },
    {
        "headword": "robust",
        "pt_equivalent": "robusto",
        "domain": "general",
        "why_avoid": "Vague intensifier. Adds no information — what makes something 'robust' is never specified.",
        "alternatives": [
            {"word": "strong", "meaning_nuance": "Less jargon-heavy but still vague — use with specific evidence", "register": "neutral", "when_to_use": "When followed by evidence"},
            {"word": "rigorous", "meaning_nuance": "Implies methodological standards", "register": "formal", "when_to_use": "M&E and research contexts"},
            {"word": "comprehensive", "meaning_nuance": "Implies breadth of coverage", "register": "formal", "when_to_use": "When breadth is actually the point"},
            {"word": "tested", "meaning_nuance": "Evidence-based; implies prior validation", "register": "neutral", "when_to_use": "When describing approaches with prior evidence"},
        ],
        "collocations": ["robust framework", "robust system", "robust evidence", "robust approach"],
        "example_bad": "We will implement a robust monitoring framework.",
        "example_good": "The monitoring framework applies WHO's standard indicator set, validated in three prior projects.",
    },
    {
        "headword": "stakeholder",
        "pt_equivalent": "parte interessada",
        "domain": "governance",
        "why_avoid": "Bureaucratic filler. Almost always replaceable with the actual group being described.",
        "alternatives": [
            {"word": "communities", "meaning_nuance": "Centres the people affected", "register": "neutral", "when_to_use": "When referring to affected populations"},
            {"word": "partners", "meaning_nuance": "Implies active collaboration", "register": "neutral", "when_to_use": "When describing organisations working together"},
            {"word": "duty-bearers", "meaning_nuance": "Rights-based framing", "register": "formal", "when_to_use": "Human rights and accountability contexts"},
            {"word": "rights-holders", "meaning_nuance": "Rights-based framing for populations", "register": "formal", "when_to_use": "Human rights contexts"},
        ],
        "collocations": ["key stakeholders", "stakeholder engagement", "relevant stakeholders", "all stakeholders"],
        "example_bad": "We will engage key stakeholders throughout the project.",
        "example_good": "We will consult community leaders, health workers, and district officials at each project stage.",
    },
    {
        "headword": "holistic",
        "pt_equivalent": "holístico",
        "domain": "general",
        "why_avoid": "Signals AI or NGO-speak; usually vague. Specify what dimensions are actually included.",
        "alternatives": [
            {"word": "integrated", "meaning_nuance": "Implies components that work together", "register": "formal", "when_to_use": "When describing multi-component approaches"},
            {"word": "comprehensive", "meaning_nuance": "Implies breadth", "register": "formal", "when_to_use": "When breadth of coverage is actually demonstrated"},
            {"word": "multisectoral", "meaning_nuance": "Sector-specific, precise", "register": "academic", "when_to_use": "Health and development contexts"},
        ],
        "collocations": ["holistic approach", "holistic care", "holistic framework"],
        "example_bad": "We adopt a holistic approach to SRHR.",
        "example_good": "Our approach links clinical services, legal aid, and psychosocial support — addressing health, rights, and wellbeing together.",
    },
    {
        "headword": "foster",
        "pt_equivalent": "fomentar",
        "domain": "general",
        "why_avoid": "NGO and AI favourite. Abstract — never says how the fostering happens.",
        "alternatives": [
            {"word": "build", "meaning_nuance": "Active and concrete", "register": "neutral", "when_to_use": "When describing deliberate construction of relationships or capacity"},
            {"word": "strengthen", "meaning_nuance": "Implies something existing being improved", "register": "neutral", "when_to_use": "When building on existing structures"},
            {"word": "support", "meaning_nuance": "Less directive — implies enabling rather than leading", "register": "neutral", "when_to_use": "When the partner leads and you enable"},
            {"word": "cultivate", "meaning_nuance": "Long-term, relationship-building", "register": "formal", "when_to_use": "Relationships and trust-building contexts"},
        ],
        "collocations": ["foster collaboration", "foster trust", "foster partnerships", "foster innovation"],
        "example_bad": "The project will foster collaboration between health ministries and CSOs.",
        "example_good": "The project will convene quarterly joint planning sessions between health ministries and CSOs.",
    },
    {
        "headword": "facilitate",
        "pt_equivalent": "facilitar",
        "domain": "general",
        "why_avoid": "Passive-sounding. Avoid when you can say precisely what will happen.",
        "alternatives": [
            {"word": "run", "meaning_nuance": "Direct, clear", "register": "neutral", "when_to_use": "Workshops, sessions, meetings"},
            {"word": "convene", "meaning_nuance": "More formal; implies bringing people together", "register": "formal", "when_to_use": "Multi-party meetings and consultations"},
            {"word": "lead", "meaning_nuance": "Implies direction and accountability", "register": "neutral", "when_to_use": "When you are the primary actor"},
            {"word": "coordinate", "meaning_nuance": "Implies managing multiple parties", "register": "neutral", "when_to_use": "Multi-partner coordination"},
        ],
        "collocations": ["facilitate workshops", "facilitate access", "facilitate dialogue", "facilitate change"],
        "example_bad": "The team will facilitate capacity-building workshops.",
        "example_good": "The team will run three two-day capacity-building workshops for 60 community health workers.",
    },
    {
        "headword": "empower",
        "pt_equivalent": "capacitar",
        "domain": "srhr",
        "why_avoid": "Paternalistic when used to describe what outsiders do for communities. Often vague about mechanism.",
        "alternatives": [
            {"word": "support", "meaning_nuance": "Less directive", "register": "neutral", "when_to_use": "When communities lead their own change"},
            {"word": "train", "meaning_nuance": "Specific to skills transfer", "register": "neutral", "when_to_use": "Skills and capacity contexts"},
            {"word": "equip", "meaning_nuance": "Implies concrete tools or resources provided", "register": "neutral", "when_to_use": "When specific resources or skills are provided"},
        ],
        "collocations": ["empower women", "empower communities", "empower youth"],
        "example_bad": "The project will empower women to claim their rights.",
        "example_good": "The project will train 200 women paralegals to support others navigating the legal system.",
    },
    {
        "headword": "innovative",
        "pt_equivalent": "inovador",
        "domain": "general",
        "why_avoid": "Claims novelty without evidence. Almost always self-congratulatory filler.",
        "alternatives": [
            {"word": "new", "meaning_nuance": "Simple and honest", "register": "neutral", "when_to_use": "When something is genuinely new to the context"},
            {"word": "adapted", "meaning_nuance": "Implies contextualisation of existing approaches", "register": "neutral", "when_to_use": "When building on prior work"},
            {"word": "pilot", "meaning_nuance": "Implies first use and learning orientation", "register": "neutral", "when_to_use": "When testing an approach for the first time"},
        ],
        "collocations": ["innovative approach", "innovative solution", "innovative model"],
        "example_bad": "We will implement an innovative approach to community engagement.",
        "example_good": "We will test a peer-educator model not yet used in this district — with a learning review at six months.",
    },
    {
        "headword": "utilize",
        "pt_equivalent": "utilizar",
        "domain": "general",
        "why_avoid": "Pretentious form of 'use'. No added meaning.",
        "alternatives": [
            {"word": "use", "meaning_nuance": "Direct and always correct", "register": "neutral", "when_to_use": "Always"},
            {"word": "apply", "meaning_nuance": "When skill or method is implied", "register": "formal", "when_to_use": "When describing deliberate application"},
        ],
        "collocations": ["utilize resources", "utilize data", "utilize tools"],
        "example_bad": "We will utilize digital platforms to disseminate findings.",
        "example_good": "We will share findings via WhatsApp groups and community radio.",
    },
    {
        "headword": "synergy",
        "pt_equivalent": "sinergia",
        "domain": "general",
        "why_avoid": "Corporate buzzword. Almost always used without explaining what the synergy actually is.",
        "alternatives": [
            {"word": "coordination", "meaning_nuance": "Specific — implies planned joint action", "register": "neutral", "when_to_use": "When organisations are actively aligning"},
            {"word": "complementarity", "meaning_nuance": "Implies each party adds what the other lacks", "register": "formal", "when_to_use": "Partnership proposals"},
            {"word": "joint effort", "meaning_nuance": "Simple and clear", "register": "neutral", "when_to_use": "General use"},
        ],
        "collocations": ["create synergies", "leverage synergies", "achieve synergy"],
        "example_bad": "This project will create synergies between health and education sectors.",
        "example_good": "Health and education staff will co-deliver sessions, reducing duplication and sharing infrastructure.",
    },
    {
        "headword": "impactful",
        "pt_equivalent": "impactante",
        "domain": "general",
        "why_avoid": "Not a real word in formal English. Sounds AI-generated. Replace with what the impact actually is.",
        "alternatives": [
            {"word": "effective", "meaning_nuance": "Achieves its intended result", "register": "neutral", "when_to_use": "When results are demonstrated"},
            {"word": "meaningful", "meaning_nuance": "Implies significance to the people affected", "register": "neutral", "when_to_use": "Community-centred contexts"},
            {"word": "significant", "meaning_nuance": "Implies measurable scale", "register": "formal", "when_to_use": "When scale or data can be cited"},
        ],
        "collocations": ["impactful programme", "impactful approach", "impactful results"],
        "example_bad": "We deliver impactful programmes for key populations.",
        "example_good": "Our programmes reached 12,000 people in 2024, with 78% reporting improved access to services.",
    },
    {
        "headword": "comprehensive",
        "pt_equivalent": "abrangente",
        "domain": "general",
        "why_avoid": "Vague — every document claims to be comprehensive. Specify what is actually covered.",
        "alternatives": [
            {"word": "full", "meaning_nuance": "Complete within a defined scope", "register": "neutral", "when_to_use": "When scope is clear"},
            {"word": "detailed", "meaning_nuance": "Implies depth rather than breadth", "register": "neutral", "when_to_use": "When depth is the point"},
            {"word": "complete", "meaning_nuance": "Nothing omitted within the scope", "register": "neutral", "when_to_use": "When completeness within a defined scope is the claim"},
        ],
        "collocations": ["comprehensive approach", "comprehensive framework", "comprehensive review"],
        "example_bad": "This report provides a comprehensive analysis of the findings.",
        "example_good": "This report analyses all 847 survey responses across five districts.",
    },
    {
        "headword": "sustainable",
        "pt_equivalent": "sustentável",
        "domain": "general",
        "why_avoid": "Donor-speak filler unless followed by a specific mechanism explaining how sustainability is achieved.",
        "alternatives": [
            {"word": "lasting", "meaning_nuance": "Simple, time-oriented", "register": "neutral", "when_to_use": "When long-term continuation is the point"},
            {"word": "self-funded", "meaning_nuance": "Specific financial mechanism", "register": "neutral", "when_to_use": "Financial sustainability"},
            {"word": "institutionalised", "meaning_nuance": "Embedded in government or org structures", "register": "formal", "when_to_use": "When policy or institutional uptake is the mechanism"},
        ],
        "collocations": ["sustainable development", "sustainable impact", "sustainable model"],
        "example_bad": "The project will ensure sustainable impact beyond the funding period.",
        "example_good": "By Year 2, all trained paralegals will be absorbed into the Ministry's budget, continuing without project funding.",
    },
    {
        "headword": "capacity building",
        "pt_equivalent": "capacitação",
        "domain": "general",
        "why_avoid": "NGO jargon that hides the actual activity. Specify what is being built and how.",
        "alternatives": [
            {"word": "training", "meaning_nuance": "Specific to skills instruction", "register": "neutral", "when_to_use": "Formal skills instruction"},
            {"word": "mentoring", "meaning_nuance": "One-on-one guidance over time", "register": "neutral", "when_to_use": "Sustained individual development"},
            {"word": "coaching", "meaning_nuance": "Performance-focused, practical", "register": "neutral", "when_to_use": "Leadership and performance development"},
        ],
        "collocations": ["capacity building activities", "capacity building programme", "capacity building workshops"],
        "example_bad": "The project includes capacity building for 50 community health workers.",
        "example_good": "The project will train 50 community health workers in TB screening using a three-day clinical skills module.",
    },
    {
        "headword": "streamline",
        "pt_equivalent": "simplificar",
        "domain": "general",
        "why_avoid": "Business jargon. Means 'make simpler or more efficient' — just say that.",
        "alternatives": [
            {"word": "simplify", "meaning_nuance": "Reduce complexity", "register": "neutral", "when_to_use": "Processes made less complex"},
            {"word": "improve", "meaning_nuance": "General improvement", "register": "neutral", "when_to_use": "When efficiency is the goal"},
            {"word": "standardise", "meaning_nuance": "Make consistent across settings", "register": "formal", "when_to_use": "Multi-site or multi-partner contexts"},
        ],
        "collocations": ["streamline processes", "streamline operations", "streamline reporting"],
        "example_bad": "We will streamline reporting processes across all implementing partners.",
        "example_good": "We will introduce a single shared reporting template across all four implementing partners.",
    },
    {
        "headword": "actionable",
        "pt_equivalent": "accionável",
        "domain": "general",
        "why_avoid": "Management jargon. Just describe what the action is.",
        "alternatives": [
            {"word": "practical", "meaning_nuance": "Grounded in reality", "register": "neutral", "when_to_use": "Recommendations that can be implemented with available resources"},
            {"word": "specific", "meaning_nuance": "Precise and well-defined", "register": "neutral", "when_to_use": "Recommendations with clear steps"},
            {"word": "concrete", "meaning_nuance": "Tangible and unambiguous", "register": "neutral", "when_to_use": "Plans with real-world specificity"},
        ],
        "collocations": ["actionable recommendations", "actionable insights", "actionable steps"],
        "example_bad": "The study provides actionable recommendations for policymakers.",
        "example_good": "The study recommends three specific changes to the national testing protocol, each with a lead ministry and timeline.",
    },
    {
        "headword": "leverage synergies",
        "pt_equivalent": "alavancar sinergias",
        "domain": "general",
        "why_avoid": "Double jargon — two overused words combined. Almost meaningless.",
        "alternatives": [
            {"word": "coordinate jointly", "meaning_nuance": "Planned collaboration", "register": "neutral", "when_to_use": "Multi-partner coordination"},
            {"word": "share resources", "meaning_nuance": "Specific mechanism", "register": "neutral", "when_to_use": "When resource-sharing is the point"},
        ],
        "collocations": [],
        "example_bad": "Partners will leverage synergies to maximise impact.",
        "example_good": "Partners will share transport and data collection tools, reducing costs by an estimated 15%.",
    },
    {
        "headword": "cutting-edge",
        "pt_equivalent": "de ponta",
        "domain": "general",
        "why_avoid": "Cliché that claims novelty without evidence.",
        "alternatives": [
            {"word": "current", "meaning_nuance": "Up to date, without overclaiming", "register": "neutral", "when_to_use": "When currency is what matters"},
            {"word": "recent", "meaning_nuance": "Time-specific", "register": "neutral", "when_to_use": "When recency can be specified"},
            {"word": "advanced", "meaning_nuance": "Implies technical sophistication", "register": "formal", "when_to_use": "When technical level is the point and can be substantiated"},
        ],
        "collocations": ["cutting-edge technology", "cutting-edge approach", "cutting-edge research"],
        "example_bad": "We use cutting-edge technology to monitor outcomes.",
        "example_good": "We use CommCare, a validated mobile data platform used in 60+ countries.",
    },
    {
        "headword": "paradigm shift",
        "pt_equivalent": "mudança de paradigma",
        "domain": "general",
        "why_avoid": "Grandiose academic cliché; almost never justified.",
        "alternatives": [
            {"word": "change", "meaning_nuance": "Simple and honest", "register": "neutral", "when_to_use": "Default"},
            {"word": "shift", "meaning_nuance": "Directional change without overclaiming", "register": "neutral", "when_to_use": "When a trend or direction change is being described"},
            {"word": "rethinking", "meaning_nuance": "Implies deliberate reconsideration", "register": "formal", "when_to_use": "Policy or conceptual reform contexts"},
        ],
        "collocations": ["paradigm shift in", "represents a paradigm shift"],
        "example_bad": "This project represents a paradigm shift in community health delivery.",
        "example_good": "This project tests a model that moves decision-making from district level to community health committees.",
    },
]

PT_WORDS = [
    {
        "headword": "alavancar",
        "domain": "general",
        "why_avoid": "Anglicismo empresarial transferido directamente do inglês 'leverage'. Soa artificial em prosa portuguesa.",
        "alternatives": [
            {"word": "usar", "meaning_nuance": "Directo e claro", "register": "neutral", "when_to_use": "Substituição padrão na maioria dos contextos"},
            {"word": "aproveitar", "meaning_nuance": "Implica tirar partido de algo existente", "register": "neutral", "when_to_use": "Quando se refere a recursos ou relações existentes"},
            {"word": "mobilizar", "meaning_nuance": "Implica activação activa", "register": "formal", "when_to_use": "Recursos, parceiros, ou comunidades a serem activados"},
        ],
        "collocations": ["alavancar recursos", "alavancar parcerias"],
        "example_bad": "O projecto irá alavancar as redes existentes para garantir impacto.",
        "example_good": "O projecto irá mobilizar as redes comunitárias existentes para ampliar o alcance do programa.",
    },
    {
        "headword": "assegurar",
        "domain": "general",
        "why_avoid": "Promessa vaga sem especificar o mecanismo. Sobreusado em propostas.",
        "alternatives": [
            {"word": "garantir", "meaning_nuance": "Compromisso mais forte", "register": "formal", "when_to_use": "Quando existe compromisso contratual ou verificável"},
            {"word": "verificar", "meaning_nuance": "Orientado para a confirmação com evidência", "register": "neutral", "when_to_use": "Contextos de garantia de qualidade"},
            {"word": "confirmar", "meaning_nuance": "Implica validação de um estado existente", "register": "neutral", "when_to_use": "Quando se verifica algo já existente"},
        ],
        "collocations": ["assegurar qualidade", "assegurar conformidade", "assegurar sustentabilidade"],
        "example_bad": "Iremos assegurar que todas as actividades são implementadas atempadamente.",
        "example_good": "As actividades serão monitorizadas semanalmente através de um cronograma partilhado.",
    },
    {
        "headword": "robusto",
        "domain": "general",
        "why_avoid": "Intensificador vago que não acrescenta informação. Sobreusado em documentos institucionais.",
        "alternatives": [
            {"word": "rigoroso", "meaning_nuance": "Implica normas metodológicas", "register": "formal", "when_to_use": "Contextos de M&A e investigação"},
            {"word": "sólido", "meaning_nuance": "Implica base bem fundamentada", "register": "neutral", "when_to_use": "Quando se pode citar evidência de base"},
            {"word": "abrangente", "meaning_nuance": "Implica amplitude de cobertura", "register": "formal", "when_to_use": "Quando a abrangência é realmente demonstrada"},
        ],
        "collocations": ["quadro robusto", "sistema robusto", "abordagem robusta"],
        "example_bad": "Implementaremos um sistema de monitoria robusto.",
        "example_good": "O sistema de monitoria aplica o conjunto de indicadores padrão da OMS, validado em três projectos anteriores.",
    },
    {
        "headword": "holístico",
        "domain": "general",
        "why_avoid": "Anglicismo vago — sinaliza linguagem de ONG. Especifique as dimensões que são realmente abrangidas.",
        "alternatives": [
            {"word": "integrado", "meaning_nuance": "Implica componentes que funcionam em conjunto", "register": "formal", "when_to_use": "Abordagens com múltiplos componentes"},
            {"word": "abrangente", "meaning_nuance": "Implica amplitude", "register": "formal", "when_to_use": "Quando a amplitude é realmente demonstrada"},
            {"word": "multissectorial", "meaning_nuance": "Preciso e específico ao sector", "register": "academic", "when_to_use": "Contextos de saúde e desenvolvimento"},
        ],
        "collocations": ["abordagem holística", "cuidados holísticos"],
        "example_bad": "Adoptamos uma abordagem holística à SSRA.",
        "example_good": "A nossa abordagem articula serviços clínicos, apoio jurídico e suporte psicossocial.",
    },
    {
        "headword": "fomentar",
        "domain": "general",
        "why_avoid": "Abstracto — não especifica como o fomento acontece. Favorito de documentos de ONG.",
        "alternatives": [
            {"word": "construir", "meaning_nuance": "Activo e concreto", "register": "neutral", "when_to_use": "Construção deliberada de relações ou capacidades"},
            {"word": "fortalecer", "meaning_nuance": "Implica melhorar algo existente", "register": "neutral", "when_to_use": "Quando se trabalha sobre estruturas existentes"},
            {"word": "promover", "meaning_nuance": "Implica visibilidade e apoio activo", "register": "neutral", "when_to_use": "Sensibilização e advocacy"},
        ],
        "collocations": ["fomentar a colaboração", "fomentar a confiança", "fomentar parcerias"],
        "example_bad": "O projecto irá fomentar a colaboração entre ministérios.",
        "example_good": "O projecto realizará reuniões trimestrais conjuntas de planificação entre os ministérios da saúde e as OSC.",
    },
    {
        "headword": "inovador",
        "domain": "general",
        "why_avoid": "Reivindica novidade sem evidência. Quase sempre auto-congratulatório.",
        "alternatives": [
            {"word": "novo", "meaning_nuance": "Simples e honesto", "register": "neutral", "when_to_use": "Quando algo é genuinamente novo no contexto"},
            {"word": "adaptado", "meaning_nuance": "Implica contextualização de abordagens existentes", "register": "neutral", "when_to_use": "Quando se parte de trabalho anterior"},
            {"word": "piloto", "meaning_nuance": "Implica primeira utilização e orientação para a aprendizagem", "register": "neutral", "when_to_use": "Quando se testa uma abordagem pela primeira vez"},
        ],
        "collocations": ["abordagem inovadora", "solução inovadora", "modelo inovador"],
        "example_bad": "Implementaremos uma abordagem inovadora de envolvimento comunitário.",
        "example_good": "Testaremos um modelo de educadores de pares ainda não utilizado neste distrito.",
    },
    {
        "headword": "sustentável",
        "domain": "general",
        "why_avoid": "Linguagem de doador sem substância a menos que seja seguida de um mecanismo específico.",
        "alternatives": [
            {"word": "duradouro", "meaning_nuance": "Orientado para o tempo, simples", "register": "neutral", "when_to_use": "Quando a continuação a longo prazo é o ponto central"},
            {"word": "institucionalizado", "meaning_nuance": "Integrado em estruturas governamentais ou organizacionais", "register": "formal", "when_to_use": "Quando o mecanismo é a integração institucional"},
        ],
        "collocations": ["desenvolvimento sustentável", "impacto sustentável", "modelo sustentável"],
        "example_bad": "O projecto garantirá um impacto sustentável além do período de financiamento.",
        "example_good": "No Ano 2, todos os paralegais formados serão integrados no orçamento do Ministério, continuando sem financiamento do projecto.",
    },
    {
        "headword": "capacitação",
        "domain": "general",
        "why_avoid": "Jargão de ONG que esconde a actividade real. Especifique o que está a ser construído e como.",
        "alternatives": [
            {"word": "formação", "meaning_nuance": "Específico para instrução de competências", "register": "neutral", "when_to_use": "Instrução formal de competências"},
            {"word": "mentoria", "meaning_nuance": "Orientação individual ao longo do tempo", "register": "neutral", "when_to_use": "Desenvolvimento individual sustentado"},
        ],
        "collocations": ["actividades de capacitação", "programa de capacitação"],
        "example_bad": "O projecto inclui capacitação para 50 agentes de saúde comunitários.",
        "example_good": "O projecto formará 50 agentes de saúde comunitários em rastreio de tuberculose através de um módulo clínico de três dias.",
    },
]
```

- [ ] **Step 4.3: Commit**

```bash
git add scripts/data/__init__.py scripts/data/thesaurus_wordlist.py
git commit -m "feat: add curated AI-pattern word seed list (EN + PT)"
```

---

## Task 5: Create seed script

**Files:**
- Create: `scripts/seed_thesaurus.py`

The seed script enriches the word list with Wordnik (EN definitions) and Dicionário Aberto (PT synonyms), then imports everything into Qdrant.

- [ ] **Step 5.1: Create `scripts/seed_thesaurus.py`**

```python
"""
Seed script: populate writing_thesaurus collection.

Pipeline:
  1. Load curated word list from scripts/data/thesaurus_wordlist.py
  2. For EN entries: enrich definitions from Wordnik API (free, no key required)
  3. For PT entries: enrich synonyms from Dicionário Aberto XML (downloaded on first run)
  4. Import all entries via add_thesaurus_entry()

Usage:
    cd /path/to/mcp-writing-library
    source .venv/bin/activate
    python scripts/seed_thesaurus.py

    # Dry run (no Qdrant writes):
    python scripts/seed_thesaurus.py --dry-run

    # Skip API enrichment (use wordlist as-is):
    python scripts/seed_thesaurus.py --no-enrich

    # Only import PT words:
    python scripts/seed_thesaurus.py --language pt
"""
import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import requests
import structlog

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tools.collections import setup_collections
from src.tools.thesaurus import add_thesaurus_entry

logger = structlog.get_logger(__name__)

WORDNIK_BASE = "https://api.wordnik.com/v4/word.json/{word}/definitions"
WORDNIK_API_KEY = os.getenv("WORDNIK_API_KEY", "")  # Optional; raises rate limit without key

DICIONARIO_ABERTO_URL = "https://raw.githubusercontent.com/ambs/Dicionario-Aberto/master/dic.xml"
DICIONARIO_ABERTO_CACHE = Path(__file__).parent / "data" / "dicionario_aberto.xml"


# ── Wordnik enrichment ────────────────────────────────────────────────────────

def _wordnik_get(endpoint: str, params: dict) -> dict | list | None:
    """Call Wordnik API. Returns parsed JSON or None on failure."""
    url = f"{WORDNIK_BASE}/{endpoint}"
    if WORDNIK_API_KEY:
        params["api_key"] = WORDNIK_API_KEY
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Wordnik non-200", status=resp.status_code, url=url)
    except Exception as e:
        logger.warning("Wordnik request failed", error=str(e))
    return None


def enrich_en(entry: dict) -> dict:
    """Enrich an EN entry with Wordnik definitions."""
    headword = entry["headword"]
    # Skip multi-word phrases for Wordnik lookup
    if " " in headword:
        return entry

    encoded = quote(headword)
    data = _wordnik_get(f"{encoded}/definitions", {"limit": 3, "sourceDictionaries": "ahd-5,wordnet"})
    if data and isinstance(data, list):
        definitions = [d.get("text", "") for d in data if d.get("text")]
        if definitions and not entry.get("definition"):
            entry["definition"] = definitions[0]
        if not entry.get("part_of_speech") and data[0].get("partOfSpeech"):
            entry["part_of_speech"] = data[0]["partOfSpeech"].split("-")[0]  # "verb-transitive" → "verb"

    time.sleep(0.3)  # Wordnik rate limit: ~100 req/min without key
    return entry


# ── Dicionário Aberto enrichment ──────────────────────────────────────────────

def _download_dicionario_aberto() -> Path:
    """Download Dicionário Aberto XML to local cache if not present."""
    if DICIONARIO_ABERTO_CACHE.exists():
        print(f"  Using cached Dicionário Aberto: {DICIONARIO_ABERTO_CACHE}")
        return DICIONARIO_ABERTO_CACHE

    print(f"  Downloading Dicionário Aberto (~30MB)...")
    DICIONARIO_ABERTO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(DICIONARIO_ABERTO_URL, stream=True, timeout=120)
    resp.raise_for_status()
    with open(DICIONARIO_ABERTO_CACHE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Downloaded: {DICIONARIO_ABERTO_CACHE}")
    return DICIONARIO_ABERTO_CACHE


def _build_da_index(xml_path: Path) -> dict:
    """
    Parse Dicionário Aberto TEI XML and build a headword → {definition, synonyms} index.

    The TEI structure is:
      <entry>
        <form><orth>headword</orth></form>
        <sense><def>definition text</def></sense>
        <sense><usg type="syn">synonym1, synonym2</usg></sense>
      </entry>
    """
    print("  Parsing Dicionário Aberto XML (this may take 30-60s)...")
    index = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Strip namespace if present
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        for entry in root.iter(f"{ns}entry"):
            # Get headword
            orth = entry.find(f".//{ns}orth")
            if orth is None or not orth.text:
                continue
            headword = orth.text.strip().lower()

            # Get first definition
            definition = ""
            def_el = entry.find(f".//{ns}def")
            if def_el is not None and def_el.text:
                definition = def_el.text.strip()

            # Get synonyms from usg[@type='syn']
            synonyms = []
            for usg in entry.findall(f".//{ns}usg"):
                if usg.get("type") == "syn" and usg.text:
                    synonyms.extend([s.strip() for s in usg.text.split(",")])

            index[headword] = {"definition": definition, "synonyms": synonyms}

    except ET.ParseError as e:
        print(f"  Warning: XML parse error — {e}. PT enrichment will be skipped.")
        return {}

    print(f"  Indexed {len(index):,} entries from Dicionário Aberto.")
    return index


def enrich_pt(entry: dict, da_index: dict) -> dict:
    """Enrich a PT entry using Dicionário Aberto index."""
    headword = entry["headword"].lower()
    da_entry = da_index.get(headword)
    if not da_entry:
        return entry

    if not entry.get("definition") and da_entry.get("definition"):
        entry["definition"] = da_entry["definition"]

    # Add DA synonyms as alternatives if not already in list
    existing_words = {a["word"].lower() for a in entry.get("alternatives", [])}
    for syn in da_entry.get("synonyms", [])[:5]:
        if syn.lower() not in existing_words and syn.lower() != headword:
            entry.setdefault("alternatives", []).append({
                "word": syn,
                "meaning_nuance": "Sinónimo (Dicionário Aberto)",
                "register": "neutral",
                "when_to_use": "Consulte contexto — sinónimo de fonte lexicográfica",
            })
            existing_words.add(syn.lower())

    return entry


# ── Main seeding pipeline ─────────────────────────────────────────────────────

def seed(dry_run: bool = False, no_enrich: bool = False, language_filter: str | None = None):
    from scripts.data.thesaurus_wordlist import EN_WORDS, PT_WORDS

    print("Setting up collections...")
    if not dry_run:
        result = setup_collections()
        thesaurus_status = result.get("thesaurus", {}).get("status", "unknown")
        print(f"  writing_thesaurus: {thesaurus_status}")

    # Build Dicionário Aberto index (PT enrichment)
    da_index = {}
    if not no_enrich and (language_filter is None or language_filter == "pt"):
        try:
            xml_path = _download_dicionario_aberto()
            da_index = _build_da_index(xml_path)
        except Exception as e:
            print(f"  Warning: Dicionário Aberto download/parse failed — {e}. PT entries will use wordlist only.")

    all_entries = []
    if language_filter is None or language_filter == "en":
        for entry in EN_WORDS:
            e = {**entry, "language": "en", "source": "wordnik" if not no_enrich else "manual"}
            if not no_enrich:
                e = enrich_en(e)
            all_entries.append(e)

    if language_filter is None or language_filter == "pt":
        for entry in PT_WORDS:
            e = {**entry, "language": "pt", "source": "dicionario-aberto" if da_index else "manual"}
            if not no_enrich and da_index:
                e = enrich_pt(e, da_index)
            all_entries.append(e)

    print(f"\nSeeding {len(all_entries)} entries (dry_run={dry_run})...")
    succeeded = 0
    failed = 0
    skipped = 0

    for entry in all_entries:
        label = f"{entry['headword']} ({entry['language']})"
        if dry_run:
            print(f"  [DRY RUN] Would insert: {label}")
            succeeded += 1
            continue

        result = add_thesaurus_entry(
            headword=entry["headword"],
            language=entry["language"],
            domain=entry.get("domain", "general"),
            definition=entry.get("definition", ""),
            part_of_speech=entry.get("part_of_speech", "verb"),
            register=entry.get("register", "neutral"),
            alternatives=entry.get("alternatives", []),
            collocations=entry.get("collocations", []),
            why_avoid=entry.get("why_avoid", ""),
            example_bad=entry.get("example_bad", ""),
            example_good=entry.get("example_good", ""),
            source=entry.get("source", "manual"),
        )

        if result.get("success"):
            print(f"  ✓ {label}")
            succeeded += 1
        elif "already exists" in result.get("error", ""):
            print(f"  ~ {label} (already exists — skipped)")
            skipped += 1
        else:
            print(f"  ✗ {label}: {result.get('error')}")
            failed += 1

    print(f"\nDone. succeeded={succeeded} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed writing_thesaurus collection")
    parser.add_argument("--dry-run", action="store_true", help="Print entries without inserting")
    parser.add_argument("--no-enrich", action="store_true", help="Skip Wordnik and Dicionário Aberto enrichment")
    parser.add_argument("--language", choices=["en", "pt"], help="Only seed one language")
    args = parser.parse_args()
    seed(dry_run=args.dry_run, no_enrich=args.no_enrich, language_filter=args.language)
```

- [ ] **Step 5.2: Test the seed script dry run**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
source .venv/bin/activate
python scripts/seed_thesaurus.py --dry-run --no-enrich
```

Expected output:
```
Setting up collections...
  [DRY RUN] Would insert: leverage (en)
  [DRY RUN] Would insert: ensure (en)
  ...
  [DRY RUN] Would insert: capacitação (pt)
Done. succeeded=28 skipped=0 failed=0
```
(Exact count depends on wordlist length — expect ~20 EN + ~8 PT = ~28 entries)

- [ ] **Step 5.3: Commit**

```bash
git add scripts/seed_thesaurus.py
git commit -m "feat: add thesaurus seed script with Wordnik + Dicionário Aberto enrichment"
```

---

## Task 6: Run live seed and verify

- [ ] **Step 6.1: Run seed with Wordnik enrichment (EN only first)**

```bash
python scripts/seed_thesaurus.py --language en
```

Expected: All EN entries printed with `✓`. Any duplicates show `~ (already exists — skipped)`.

- [ ] **Step 6.2: Run seed for PT entries with Dicionário Aberto**

```bash
python scripts/seed_thesaurus.py --language pt
```

Expected: Downloads Dicionário Aberto XML (~30MB, one-time), parses, enriches PT entries, inserts. Cached at `scripts/data/dicionario_aberto.xml` for future runs.

- [ ] **Step 6.3: Verify via suggest_alternatives MCP tool**

```bash
python -c "
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from src.tools.thesaurus import suggest_alternatives
import json
result = suggest_alternatives(word='leverage', language='en', domain='general')
print(json.dumps(result, indent=2))
"
```

Expected:
```json
{
  "success": true,
  "found_in_thesaurus": true,
  "headword": "leverage",
  "language": "en",
  "alternatives": [
    {"word": "use", "meaning_nuance": "Direct and clear", ...},
    {"word": "draw on", ...},
    ...
  ],
  "why_avoid": "Corporate jargon...",
  ...
}
```

- [ ] **Step 6.4: Verify flag_vocabulary**

```bash
python -c "
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from src.tools.thesaurus import flag_vocabulary
import json
text = 'We will leverage our networks to ensure maximum impact through innovative approaches.'
result = flag_vocabulary(text=text, language='en', domain='general')
print(json.dumps(result, indent=2))
"
```

Expected: `flagged_count >= 2`, `leverage` and `ensure` in `flagged`, `verdict` is `review` or `ai-sounding`.

- [ ] **Step 6.5: Verify PT entry**

```bash
python -c "
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from src.tools.thesaurus import suggest_alternatives
import json
result = suggest_alternatives(word='alavancar', language='pt', domain='general')
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

Expected: `found_in_thesaurus: true`, `alternatives` list with `aproveitar`, `mobilizar`.

- [ ] **Step 6.6: Commit**

```bash
git add scripts/data/  # excludes dicionario_aberto.xml (add to .gitignore)
git commit -m "chore: verify live seed — thesaurus populated with EN+PT entries"
```

---

## Task 7: Update `.gitignore` and `CLAUDE.md`

**Files:**
- Modify: `.gitignore` (add Dicionário Aberto cache)
- Modify: `CLAUDE.md`

- [ ] **Step 7.1: Add Dicionário Aberto cache to `.gitignore`**

Check if `.gitignore` exists:

```bash
cat /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library/.gitignore
```

Add this line:

```
scripts/data/dicionario_aberto.xml
```

- [ ] **Step 7.2: Update `CLAUDE.md` — add thesaurus section**

In `CLAUDE.md`, add a new row to the Module Reference table:

```markdown
| `src/tools/thesaurus` | `add_thesaurus_entry`, `search_thesaurus`, `suggest_alternatives`, `flag_vocabulary` | Vocabulary intelligence: detect AI-pattern words, suggest rich alternatives with meaning/register/usage context |
```

Add a new pattern section after Pattern 8:

```markdown
## Pattern 9 — Vocabulary Intelligence

When reviewing a document for AI-pattern vocabulary, call `flag_vocabulary` to get lexical-level detection alongside `score_ai_patterns` (structural):

```python
lexical = flag_vocabulary(text="...", language="en", domain="governance")
# Returns: flagged_count, verdict, list of flagged headwords with alternatives_preview
```

For a specific word, get rich alternatives with meaning nuance:

```python
result = suggest_alternatives(word="leverage", language="en", domain="governance")
# Returns: definition, why_avoid, alternatives[{word, meaning_nuance, register, when_to_use}],
#          collocations, example_bad, example_good, found_in_thesaurus
```

To add a new AI-pattern word discovered during document review:

```python
result = add_thesaurus_entry(
    headword="leverage",
    language="en",
    domain="governance",
    definition="To use something to maximum advantage.",
    part_of_speech="verb",
    register="institutional",
    alternatives=[
        {"word": "use", "meaning_nuance": "Direct and clear", "register": "neutral", "when_to_use": "Default"},
    ],
    why_avoid="Corporate jargon overused in AI proposals.",
    example_bad="We will leverage our networks.",
    example_good="We will draw on our networks.",
    source="manual",
)
```
```

- [ ] **Step 7.3: Commit**

```bash
git add .gitignore CLAUDE.md
git commit -m "docs: update CLAUDE.md with thesaurus module and patterns"
```

---

## Task 8: Full test run and final verification

- [ ] **Step 8.1: Run full test suite**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
source .venv/bin/activate
python -m pytest tests/ -v
```

Expected: All tests pass. No regressions in existing tests.

- [ ] **Step 8.2: Run setup_collections to confirm thesaurus collection is registered**

```bash
python scripts/setup_collections.py
```

Expected output includes:
```
writing_thesaurus: already_exists  (point_count: N)
```

- [ ] **Step 8.3: Smoke test the MCP server starts cleanly**

```bash
timeout 5 python main.py 2>&1 || true
```

Expected: `🚀 Starting MCP Writing Library Server...` with no import errors.

- [ ] **Step 8.4: Final commit**

```bash
git add -A
git commit -m "feat: vocabulary intelligence layer — thesaurus collection, 4 MCP tools, EN+PT seed"
```

---

## Verification Checklist (from spec)

- [ ] `setup_collections.py` shows `writing_thesaurus` in output
- [ ] `seed_thesaurus.py` loads ~28 entries (EN + PT)
- [ ] `suggest_alternatives(word="leverage", language="en", domain="governance")` returns ≥3 alternatives with meaning nuance and register
- [ ] `flag_vocabulary` with a paragraph containing "leverage", "robust", "ensure" flags all three
- [ ] `suggest_alternatives(word="alavancar", language="pt")` returns PT alternatives
- [ ] `suggest_alternatives` for unknown word falls back to `search_terms` gracefully (no error)
- [ ] All existing tests still pass
