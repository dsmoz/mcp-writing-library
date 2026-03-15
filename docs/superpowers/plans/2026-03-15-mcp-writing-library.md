# MCP Writing Library Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated MCP server (`mcp-writing-library`) that stores and semantically searches exemplary writing passages and consultant terminology in Qdrant, usable by the copywriter agent and editorial-review skill.

**Architecture:** Thin FastMCP wrapper over `kbase-core` (shared library already used by mcp-knowledge-base), with two Qdrant collections — `writing_passages` for exemplary paragraphs/models and `writing_terms` for terminology dictionary entries. Tools are synchronous and purpose-built for editorial use cases. Seeded from existing copywriter markdown references.

**Tech Stack:** Python 3.11+, FastMCP, kbase-core (kbase.vector.sync_indexing + sync_search), Qdrant (existing local instance), uv for virtualenv

---

## Context for the Implementer

### Key Paths
- **New server:** `/Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library/`
- **kbase-core library:** `/Users/danilodasilva/Documents/Programming/mcp-servers/libraries/kbase-core/`
- **Reference pattern:** `/Users/danilodasilva/Documents/Programming/mcp-servers/mcp-knowledge-base/` — copy structure exactly
- **Claude config:** `/Users/danilodasilva/.claude.json` — add server registration
- **Seed sources:** `~/.claude/agents/references/copywriter/` — markdown files to import

### kbase-core Key Functions (already battle-tested)
```python
# Indexing
from kbase.vector.sync_indexing import index_document, ensure_collection, delete_document_vectors

# Search
from kbase.vector.sync_search import semantic_search

# index_document(collection_name, document_id, title, content, metadata) → List[str]
# semantic_search(collection_name, query, limit, filter_conditions) → List[Dict]
# ensure_collection(collection_name, vector_size=1536, hybrid=True) → bool
```

### MCP Pattern (from mcp-knowledge-base)
- `main.py` adds `kbase-core` path to `sys.path` explicitly (macOS UF_HIDDEN bug workaround)
- `FastMCP` from `mcp.server.fastmcp`
- `uv run --directory <path> python main.py` launch command
- Environment loaded from `.env` file

### Qdrant Collections to Create
| Collection | Purpose | Vector size |
|---|---|---|
| `writing_passages` | Exemplary paragraphs, before/after models | 1536 (text-embedding-3-small) |
| `writing_terms` | Terminology entries (preferred vs avoid) | 1536 |

### Passage Metadata Schema
```json
{
  "doc_type": "executive-summary|concept-note|policy-brief|report|email|general",
  "language": "en|pt",
  "domain": "srhr|governance|climate|general|m-and-e",
  "quality_notes": "what makes this passage work",
  "tags": ["discursive", "findings", "causation"],
  "source": "undp-hdr-2024|manual|copywriter-ref"
}
```

### Term Metadata Schema
```json
{
  "preferred": "rights-holder",
  "avoid": "victims",
  "domain": "srhr|governance|general",
  "language": "en|pt",
  "why": "deficit framing; UNDP standard",
  "example_bad": "The victims of the epidemic...",
  "example_good": "People living with HIV..."
}
```

### Decommission mcp-knowledge-base
- It was never registered in `~/.claude.json` (confirmed)
- Archive it: `mv mcp-knowledge-base _ARCHIVED_mcp-knowledge-base`
- No config changes needed (it was never wired up)

---

## Chunk 1: Project Scaffold and Dependencies

### Task 1: Create directory structure

**Files:**
- Create: `mcp-writing-library/pyproject.toml`
- Create: `mcp-writing-library/.env.example`
- Create: `mcp-writing-library/main.py`
- Create: `mcp-writing-library/src/__init__.py`
- Create: `mcp-writing-library/src/server.py`
- Create: `mcp-writing-library/src/tools/__init__.py`
- Create: `mcp-writing-library/src/tools/passages.py`
- Create: `mcp-writing-library/src/tools/terms.py`
- Create: `mcp-writing-library/src/tools/collections.py`
- Create: `mcp-writing-library/scripts/setup_collections.py`
- Create: `mcp-writing-library/scripts/seed_from_markdown.py`
- Create: `mcp-writing-library/tests/__init__.py`
- Create: `mcp-writing-library/tests/test_passages.py`
- Create: `mcp-writing-library/tests/test_terms.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "mcp-writing-library"
version = "1.0.0"
description = "MCP server for writing passages and terminology dictionary with hybrid semantic search"
readme = "README.md"
requires-python = ">=3.11"
authors = [
    { name = "Danilo da Silva" }
]

dependencies = [
    # Shared library (editable install)
    "kbase-core>=1.3.0",
    # MCP Framework
    "mcp>=1.13.1",
    # Core utilities
    "python-dotenv>=1.0.0",
    # Structured logging
    "structlog>=24.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[tool.uv.sources]
kbase-core = { path = "../libraries/kbase-core", editable = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Create .env.example**

```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# OpenAI Configuration (for embeddings)
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small

# Collection Names
COLLECTION_PASSAGES=writing_passages
COLLECTION_TERMS=writing_terms

# Logging
LOG_LEVEL=INFO
```

- [ ] **Step 3: Create .env by copying .env.example**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
cp .env.example .env
# Fill in OPENAI_API_KEY — same key used by mcp-knowledge-base
# Check: cat ../mcp-knowledge-base/.env | grep OPENAI
```

- [ ] **Step 4: Create src/__init__.py (empty)**

```python
```

- [ ] **Step 5: Create src/tools/__init__.py (empty)**

```python
```

- [ ] **Step 6: Initialize uv venv and install dependencies**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
uv venv
uv sync
```

Expected: venv created, kbase-core installed in editable mode, mcp installed.

- [ ] **Step 7: Verify kbase-core imports work**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
uv run python -c "
from kbase.vector.sync_indexing import ensure_collection, index_document
from kbase.vector.sync_search import semantic_search
print('kbase-core imports OK')
"
```

Expected: `kbase-core imports OK`

---

## Chunk 2: Core Tool Modules

### Task 2: Implement collections.py (collection setup)

**Files:**
- Create: `mcp-writing-library/src/tools/collections.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_collections.py`:

```python
"""Tests for collection management."""
import pytest
from unittest.mock import patch, MagicMock


def test_get_collection_names_returns_env_values(monkeypatch):
    """Collection names should come from env vars."""
    monkeypatch.setenv("COLLECTION_PASSAGES", "writing_passages")
    monkeypatch.setenv("COLLECTION_TERMS", "writing_terms")

    from src.tools.collections import get_collection_names
    names = get_collection_names()

    assert names["passages"] == "writing_passages"
    assert names["terms"] == "writing_terms"


def test_get_collection_names_uses_defaults(monkeypatch):
    """Collection names should default when env vars absent."""
    monkeypatch.delenv("COLLECTION_PASSAGES", raising=False)
    monkeypatch.delenv("COLLECTION_TERMS", raising=False)

    from src.tools.collections import get_collection_names
    names = get_collection_names()

    assert names["passages"] == "writing_passages"
    assert names["terms"] == "writing_terms"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
uv run pytest tests/test_collections.py -v
```

Expected: FAIL — `ImportError: cannot import name 'get_collection_names'`

- [ ] **Step 3: Implement collections.py**

```python
"""
Collection management for writing library Qdrant collections.
"""
import os
import structlog

logger = structlog.get_logger(__name__)

VECTOR_SIZE = 1536  # text-embedding-3-small


def get_collection_names() -> dict:
    """Return configured collection names from environment."""
    return {
        "passages": os.getenv("COLLECTION_PASSAGES", "writing_passages"),
        "terms": os.getenv("COLLECTION_TERMS", "writing_terms"),
    }


def setup_collections() -> dict:
    """
    Ensure both Qdrant collections exist with hybrid vector config.

    Returns:
        Dict with creation status for each collection.
    """
    from kbase.vector.sync_indexing import ensure_collection

    names = get_collection_names()
    results = {}

    for key, collection_name in names.items():
        try:
            created = ensure_collection(
                collection_name=collection_name,
                vector_size=VECTOR_SIZE,
                hybrid=True,
            )
            status = "created" if created else "already_exists"
            results[key] = {"collection": collection_name, "status": status}
            logger.info("Collection ready", collection=collection_name, status=status)
        except Exception as e:
            results[key] = {"collection": collection_name, "status": "error", "error": str(e)}
            logger.error("Collection setup failed", collection=collection_name, error=str(e))

    return results


def get_stats() -> dict:
    """Return point counts for both collections."""
    from kbase.vector.sync_search import get_collection_stats

    names = get_collection_names()
    stats = {}

    for key, collection_name in names.items():
        try:
            stats[key] = get_collection_stats(collection_name)
        except Exception as e:
            stats[key] = {"collection": collection_name, "error": str(e)}

    return stats
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_collections.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git init  # (if not already a git repo)
git add pyproject.toml .env.example src/ tests/test_collections.py
git commit -m "feat: scaffold mcp-writing-library with collections module"
```

---

### Task 3: Implement passages.py

**Files:**
- Create: `mcp-writing-library/src/tools/passages.py`
- Modify: `mcp-writing-library/tests/test_passages.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for passages tool."""
import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4


def test_add_passage_returns_document_id():
    """add_passage should return a document_id on success."""
    mock_point_ids = [str(uuid4()), str(uuid4())]

    with patch("src.tools.passages.index_document", return_value=mock_point_ids):
        from src.tools.passages import add_passage
        result = add_passage(
            text="The assessment reveals a familiar pattern: progress coexists with persistent gaps.",
            doc_type="executive-summary",
            language="en",
            domain="general",
            quality_notes="Good discursive opener with contrast",
            tags=["discursive", "contrast"],
            source="manual",
        )

    assert result["success"] is True
    assert "document_id" in result
    assert result["chunks_created"] == 2


def test_add_passage_validates_doc_type():
    """add_passage should reject invalid doc_type values."""
    from src.tools.passages import add_passage
    result = add_passage(
        text="Some text.",
        doc_type="invalid-type",
        language="en",
    )
    assert result["success"] is False
    assert "doc_type" in result["error"].lower()


def test_add_passage_validates_language():
    """add_passage should reject languages other than en or pt."""
    from src.tools.passages import add_passage
    result = add_passage(
        text="Some text.",
        doc_type="report",
        language="fr",
    )
    assert result["success"] is False
    assert "language" in result["error"].lower()


def test_search_passages_calls_semantic_search():
    """search_passages should call semantic_search with correct collection."""
    mock_results = [
        {
            "id": str(uuid4()),
            "score": 0.92,
            "document_id": str(uuid4()),
            "title": "Example passage",
            "text": "The assessment reveals...",
            "metadata": {"doc_type": "executive-summary", "language": "en"},
        }
    ]

    with patch("src.tools.passages.semantic_search", return_value=mock_results):
        from src.tools.passages import search_passages
        result = search_passages(query="implementation gaps", top_k=5)

    assert result["success"] is True
    assert len(result["results"]) == 1
    assert result["results"][0]["score"] == 0.92


def test_search_passages_filters_by_doc_type():
    """search_passages should pass doc_type filter to semantic_search."""
    with patch("src.tools.passages.semantic_search", return_value=[]) as mock_search:
        from src.tools.passages import search_passages
        search_passages(query="findings", doc_type="policy-brief", language="pt")

    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["filter_conditions"]["doc_type"] == "policy-brief"
    assert call_kwargs["filter_conditions"]["language"] == "pt"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_passages.py -v
```

Expected: FAIL — `ImportError: cannot import name 'add_passage'`

- [ ] **Step 3: Implement passages.py**

```python
"""
Writing passages tool: store and search exemplary writing passages.
"""
import os
from typing import Optional, List
from uuid import uuid4
import structlog

from src.tools.collections import get_collection_names

logger = structlog.get_logger(__name__)

VALID_DOC_TYPES = {
    "executive-summary", "concept-note", "policy-brief",
    "report", "email", "general"
}
VALID_LANGUAGES = {"en", "pt"}
VALID_DOMAINS = {"srhr", "governance", "climate", "general", "m-and-e"}


def add_passage(
    text: str,
    doc_type: str = "general",
    language: str = "en",
    domain: str = "general",
    quality_notes: str = "",
    tags: Optional[List[str]] = None,
    source: str = "manual",
) -> dict:
    """
    Store an exemplary writing passage in the writing_passages collection.

    Args:
        text: The passage text (paragraph or multiple sentences)
        doc_type: Document type context (executive-summary|concept-note|policy-brief|report|email|general)
        language: Language (en|pt)
        domain: Thematic domain (srhr|governance|climate|general|m-and-e)
        quality_notes: What makes this passage good (e.g. "strong discursive opener with contrast")
        tags: Labels for retrieval (e.g. ["discursive", "findings", "contrast"])
        source: Origin of passage (e.g. "undp-hdr-2024", "manual", "deliverable-xyz")

    Returns:
        Dict with success status, document_id, and chunks_created
    """
    # Validate inputs
    if doc_type not in VALID_DOC_TYPES:
        return {
            "success": False,
            "error": f"Invalid doc_type '{doc_type}'. Must be one of: {sorted(VALID_DOC_TYPES)}",
        }
    if language not in VALID_LANGUAGES:
        return {
            "success": False,
            "error": f"Invalid language '{language}'. Must be one of: {sorted(VALID_LANGUAGES)}",
        }
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    from kbase.vector.sync_indexing import index_document

    document_id = str(uuid4())
    collection = get_collection_names()["passages"]
    title = f"[{doc_type.upper()} | {language.upper()}] {text[:60]}..."

    metadata = {
        "doc_type": doc_type,
        "language": language,
        "domain": domain,
        "quality_notes": quality_notes,
        "tags": tags or [],
        "source": source,
        "entry_type": "passage",
    }

    try:
        point_ids = index_document(
            collection_name=collection,
            document_id=document_id,
            title=title,
            content=text,
            metadata=metadata,
            context_mode="metadata",  # Fast: no LLM for passage indexing
        )
        logger.info(
            "Passage added",
            document_id=document_id,
            doc_type=doc_type,
            language=language,
            chunks=len(point_ids),
        )
        return {
            "success": True,
            "document_id": document_id,
            "chunks_created": len(point_ids),
            "collection": collection,
        }
    except Exception as e:
        logger.error("Failed to add passage", error=str(e))
        return {"success": False, "error": str(e)}


def search_passages(
    query: str,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Search for exemplary writing passages by semantic similarity.

    Args:
        query: Natural language query (e.g. "implementation gaps health systems")
        doc_type: Filter by document type (optional)
        language: Filter by language en|pt (optional)
        domain: Filter by thematic domain (optional)
        top_k: Number of results to return (default: 5)

    Returns:
        Dict with success status and list of matching passages
    """
    from kbase.vector.sync_search import semantic_search

    collection = get_collection_names()["passages"]

    # Build optional metadata filters
    filter_conditions = {}
    if doc_type:
        filter_conditions["doc_type"] = doc_type
    if language:
        filter_conditions["language"] = language
    if domain:
        filter_conditions["domain"] = domain

    try:
        raw_results = semantic_search(
            collection_name=collection,
            query=query,
            limit=top_k,
            filter_conditions=filter_conditions if filter_conditions else None,
        )

        results = []
        for r in raw_results:
            results.append({
                "score": round(r["score"], 4),
                "text": r.get("text", ""),
                "title": r.get("title", ""),
                "doc_type": r.get("metadata", {}).get("doc_type"),
                "language": r.get("metadata", {}).get("language"),
                "domain": r.get("metadata", {}).get("domain"),
                "quality_notes": r.get("metadata", {}).get("quality_notes"),
                "tags": r.get("metadata", {}).get("tags", []),
                "source": r.get("metadata", {}).get("source"),
                "document_id": r.get("document_id"),
            })

        return {"success": True, "results": results, "total": len(results)}

    except Exception as e:
        logger.error("Passage search failed", error=str(e))
        return {"success": False, "error": str(e), "results": []}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_passages.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/passages.py tests/test_passages.py
git commit -m "feat: add passage indexing and search tools"
```

---

### Task 4: Implement terms.py

**Files:**
- Create: `mcp-writing-library/src/tools/terms.py`
- Modify: `mcp-writing-library/tests/test_terms.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for terminology dictionary tool."""
import pytest
from unittest.mock import patch
from uuid import uuid4


def test_add_term_returns_document_id():
    """add_term should return document_id on success."""
    mock_point_ids = [str(uuid4())]

    with patch("src.tools.terms.index_document", return_value=mock_point_ids):
        from src.tools.terms import add_term
        result = add_term(
            preferred="rights-holder",
            avoid="victim",
            domain="srhr",
            language="en",
            why="Deficit framing undermines agency. UNDP standard.",
            example_bad="HIV victims need services.",
            example_good="People living with HIV require access to services.",
        )

    assert result["success"] is True
    assert "document_id" in result


def test_add_term_requires_preferred():
    """add_term should fail if preferred term is empty."""
    from src.tools.terms import add_term
    result = add_term(preferred="", avoid="something", domain="general")
    assert result["success"] is False
    assert "preferred" in result["error"].lower()


def test_search_terms_returns_formatted_results():
    """search_terms should return terms with preferred/avoid fields."""
    mock_results = [
        {
            "id": str(uuid4()),
            "score": 0.95,
            "document_id": str(uuid4()),
            "title": "rights-holder",
            "text": "Preferred: rights-holder. Avoid: victim.",
            "metadata": {
                "preferred": "rights-holder",
                "avoid": "victim",
                "domain": "srhr",
                "language": "en",
                "why": "Deficit framing",
                "example_bad": "HIV victims...",
                "example_good": "People living with HIV...",
                "entry_type": "term",
            },
        }
    ]

    with patch("src.tools.terms.semantic_search", return_value=mock_results):
        from src.tools.terms import search_terms
        result = search_terms(query="person living with HIV language")

    assert result["success"] is True
    assert len(result["results"]) == 1
    assert result["results"][0]["preferred"] == "rights-holder"
    assert result["results"][0]["avoid"] == "victim"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_terms.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: Implement terms.py**

```python
"""
Terminology dictionary tool: store and search consultant vocabulary entries.
"""
from typing import Optional, List
from uuid import uuid4
import structlog

from src.tools.collections import get_collection_names

logger = structlog.get_logger(__name__)

VALID_DOMAINS = {"srhr", "governance", "climate", "general", "m-and-e"}
VALID_LANGUAGES = {"en", "pt", "both"}


def add_term(
    preferred: str,
    avoid: str = "",
    domain: str = "general",
    language: str = "en",
    why: str = "",
    example_bad: str = "",
    example_good: str = "",
) -> dict:
    """
    Store a terminology entry in the writing_terms collection.

    Args:
        preferred: The term to use (e.g. "rights-holder", "people living with HIV")
        avoid: Term(s) to avoid instead (e.g. "victim", "AIDS victim")
        domain: Thematic domain (srhr|governance|climate|general|m-and-e)
        language: Language (en|pt|both)
        why: Reason for the preference (e.g. "deficit framing; UNDP standard")
        example_bad: Example of poor usage
        example_good: Example of correct usage

    Returns:
        Dict with success status and document_id
    """
    if not preferred or not preferred.strip():
        return {"success": False, "error": "preferred term cannot be empty"}
    if domain not in VALID_DOMAINS:
        return {
            "success": False,
            "error": f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}",
        }

    from kbase.vector.sync_indexing import index_document

    document_id = str(uuid4())
    collection = get_collection_names()["terms"]

    # Build full text for semantic search (all fields combined for embedding)
    content_parts = [
        f"Preferred term: {preferred}",
        f"Avoid: {avoid}" if avoid else "",
        f"Why: {why}" if why else "",
        f"Bad example: {example_bad}" if example_bad else "",
        f"Good example: {example_good}" if example_good else "",
        f"Domain: {domain}",
        f"Language: {language}",
    ]
    content = "\n".join(p for p in content_parts if p)

    metadata = {
        "preferred": preferred,
        "avoid": avoid,
        "domain": domain,
        "language": language,
        "why": why,
        "example_bad": example_bad,
        "example_good": example_good,
        "entry_type": "term",
    }

    try:
        point_ids = index_document(
            collection_name=collection,
            document_id=document_id,
            title=preferred,
            content=content,
            metadata=metadata,
            context_mode="metadata",
        )
        logger.info("Term added", preferred=preferred, domain=domain)
        return {
            "success": True,
            "document_id": document_id,
            "chunks_created": len(point_ids),
            "collection": collection,
        }
    except Exception as e:
        logger.error("Failed to add term", error=str(e))
        return {"success": False, "error": str(e)}


def search_terms(
    query: str,
    domain: Optional[str] = None,
    language: Optional[str] = None,
    top_k: int = 8,
) -> dict:
    """
    Search the terminology dictionary for relevant entries.

    Args:
        query: What you're looking for (e.g. "person living with HIV language rights-based")
        domain: Filter by domain (optional)
        language: Filter by language (optional)
        top_k: Number of results (default: 8)

    Returns:
        Dict with success status and list of terminology entries
    """
    from kbase.vector.sync_search import semantic_search

    collection = get_collection_names()["terms"]

    filter_conditions = {}
    if domain:
        filter_conditions["domain"] = domain
    if language:
        filter_conditions["language"] = language

    try:
        raw_results = semantic_search(
            collection_name=collection,
            query=query,
            limit=top_k,
            filter_conditions=filter_conditions if filter_conditions else None,
        )

        results = []
        for r in raw_results:
            meta = r.get("metadata", {})
            results.append({
                "score": round(r["score"], 4),
                "preferred": meta.get("preferred", r.get("title", "")),
                "avoid": meta.get("avoid", ""),
                "domain": meta.get("domain"),
                "language": meta.get("language"),
                "why": meta.get("why", ""),
                "example_bad": meta.get("example_bad", ""),
                "example_good": meta.get("example_good", ""),
                "document_id": r.get("document_id"),
            })

        return {"success": True, "results": results, "total": len(results)}

    except Exception as e:
        logger.error("Term search failed", error=str(e))
        return {"success": False, "error": str(e), "results": []}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_terms.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/terms.py tests/test_terms.py
git commit -m "feat: add terminology dictionary indexing and search tools"
```

---

## Chunk 3: MCP Server and Registration

### Task 5: Implement server.py and main.py

**Files:**
- Create: `mcp-writing-library/src/server.py`
- Create: `mcp-writing-library/main.py`

- [ ] **Step 1: Implement src/server.py**

```python
"""
MCP Writing Library Server — FastMCP tool definitions.

Tools:
    search_passages   — semantic search for exemplary writing passages
    add_passage       — store a new exemplary passage
    search_terms      — semantic search in terminology dictionary
    add_term          — add a new terminology entry
    get_library_stats — collection point counts
    setup_collections — create/verify Qdrant collections (admin)
"""
from typing import Optional, List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("writing-library")


@mcp.tool()
def search_passages(
    query: str,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Search for exemplary writing passages by semantic similarity.

    Use this to find model paragraphs when drafting or reviewing documents.
    Returns passages ranked by relevance with quality notes explaining what makes each one effective.

    Args:
        query: What you need (e.g. "executive summary opening about health equity")
        doc_type: Filter by type: executive-summary|concept-note|policy-brief|report|email|general
        language: Filter by language: en|pt
        domain: Filter by domain: srhr|governance|climate|general|m-and-e
        top_k: Number of results (default 5, max 20)

    Returns:
        List of matching passages with scores, quality notes, and tags
    """
    from src.tools.passages import search_passages as _search
    return _search(query=query, doc_type=doc_type, language=language, domain=domain, top_k=top_k)


@mcp.tool()
def add_passage(
    text: str,
    doc_type: str = "general",
    language: str = "en",
    domain: str = "general",
    quality_notes: str = "",
    tags: Optional[List[str]] = None,
    source: str = "manual",
) -> dict:
    """
    Store an exemplary writing passage in the library.

    Use this when you produce a passage you are proud of, or to seed the library
    with models from reference documents (UNDP HDR, Global Fund reports, etc.).

    Args:
        text: The passage (one or more paragraphs)
        doc_type: Context: executive-summary|concept-note|policy-brief|report|email|general
        language: Language: en|pt
        domain: Thematic area: srhr|governance|climate|general|m-and-e
        quality_notes: What makes this passage good (helps future retrieval)
        tags: Labels for retrieval e.g. ["discursive", "findings", "contrast"]
        source: Where the passage came from (e.g. "undp-hdr-2024", "manual")

    Returns:
        document_id and chunks_created on success
    """
    from src.tools.passages import add_passage as _add
    return _add(
        text=text, doc_type=doc_type, language=language, domain=domain,
        quality_notes=quality_notes, tags=tags or [], source=source,
    )


@mcp.tool()
def search_terms(
    query: str,
    domain: Optional[str] = None,
    language: Optional[str] = None,
    top_k: int = 8,
) -> dict:
    """
    Search the terminology dictionary for preferred consultant vocabulary.

    Use this when choosing between terms, or when reviewing text for inappropriate language.
    Returns preferred terms, what to avoid, and why — with good/bad usage examples.

    Args:
        query: What you're looking for (e.g. "person living with HIV", "leverage", "unprecedented")
        domain: Filter by domain: srhr|governance|climate|general|m-and-e
        language: Filter by language: en|pt
        top_k: Number of results (default 8)

    Returns:
        List of terminology entries with preferred/avoid pairs and examples
    """
    from src.tools.terms import search_terms as _search
    return _search(query=query, domain=domain, language=language, top_k=top_k)


@mcp.tool()
def add_term(
    preferred: str,
    avoid: str = "",
    domain: str = "general",
    language: str = "en",
    why: str = "",
    example_bad: str = "",
    example_good: str = "",
) -> dict:
    """
    Add a terminology entry to the dictionary.

    Use this to codify vocabulary preferences — consultant language, UNDP standards,
    people-first terminology, or sector-specific expressions.

    Args:
        preferred: The term to use (e.g. "rights-holder", "key populations")
        avoid: Term to avoid (e.g. "victim", "vulnerable groups")
        domain: Thematic area: srhr|governance|climate|general|m-and-e
        language: Language: en|pt|both
        why: Reason for preference (e.g. "deficit framing; UNDP 2024 standard")
        example_bad: Example of poor usage
        example_good: Example of correct usage

    Returns:
        document_id on success
    """
    from src.tools.terms import add_term as _add
    return _add(
        preferred=preferred, avoid=avoid, domain=domain, language=language,
        why=why, example_bad=example_bad, example_good=example_good,
    )


@mcp.tool()
def get_library_stats() -> dict:
    """
    Return point counts for both Qdrant collections.

    Use this to verify the library is populated and working.

    Returns:
        Stats for writing_passages and writing_terms collections
    """
    from src.tools.collections import get_stats
    return get_stats()


@mcp.tool()
def setup_collections() -> dict:
    """
    Create or verify Qdrant collections for the writing library.

    Run this once on first setup, or to verify collections exist after a Qdrant restart.

    Returns:
        Status for each collection (created|already_exists|error)
    """
    from src.tools.collections import setup_collections as _setup
    return _setup()
```

- [ ] **Step 2: Implement main.py**

```python
#!/usr/bin/env python3
"""
MCP Writing Library Server

Stores and semantically searches exemplary writing passages and terminology
for use by the copywriter agent and editorial-review skill.

Collections:
    writing_passages — exemplary paragraphs by doc type, language, domain
    writing_terms    — terminology dictionary (preferred vs avoid)
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Fix for macOS UF_HIDDEN flag preventing .pth file processing in .venv directories.
# kbase-core editable install requires explicit sys.path entry.
kbase_core_path = project_root.parent / 'libraries' / 'kbase-core'
if kbase_core_path.exists():
    sys.path.insert(0, str(kbase_core_path))

# Load environment variables
env_file = os.getenv('ENV_FILE', str(project_root / '.env'))
if Path(env_file).exists():
    load_dotenv(env_file)
else:
    load_dotenv(project_root / '.env.example')
    print(f"⚠️  {env_file} not found, using .env.example", file=sys.stderr)


def main():
    """Main entry point for MCP server."""
    try:
        from src.server import mcp
        print("🚀 Starting MCP Writing Library Server...", file=sys.stderr)
        print(f"📍 Project: {project_root}", file=sys.stderr)
        mcp.run()
    except ImportError as e:
        print(f"❌ Failed to import MCP server: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test the server starts**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
# Start server and interrupt after 3 seconds to check startup
timeout 3 uv run python main.py 2>&1 || true
```

Expected output contains: `Starting MCP Writing Library Server...`

- [ ] **Step 4: Commit**

```bash
git add src/server.py main.py
git commit -m "feat: implement FastMCP server with 6 writing library tools"
```

---

### Task 6: Register server in Claude and decommission mcp-knowledge-base

**Files:**
- Modify: `/Users/danilodasilva/.claude.json` (add server registration)
- Rename: `mcp-knowledge-base/` → `_ARCHIVED_mcp-knowledge-base/` (decommission)

- [ ] **Step 1: Register mcp-writing-library in Claude config**

```bash
claude mcp add writing-library \
  --scope user \
  -- uv run \
  --directory /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library \
  python main.py
```

- [ ] **Step 2: Verify registration**

```bash
claude mcp list | grep writing-library
```

Expected: `writing-library` appears in the list.

- [ ] **Step 3: Archive mcp-knowledge-base**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers
mv mcp-knowledge-base _ARCHIVED_mcp-knowledge-base
```

Note: `mcp-knowledge-base` was never registered in `~/.claude.json` (confirmed), so no config change needed for removal.

- [ ] **Step 4: Commit**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
git add -A
git commit -m "chore: server registered in Claude; mcp-knowledge-base archived"
```

---

## Chunk 4: Collection Setup and Seeding

### Task 7: Setup Qdrant collections

**Files:**
- Create: `mcp-writing-library/scripts/setup_collections.py`

- [ ] **Step 1: Create setup script**

```python
#!/usr/bin/env python3
"""
Setup Qdrant collections for the writing library.

Run once before first use:
    uv run python scripts/setup_collections.py
"""
import sys
from pathlib import Path

# Ensure imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
kbase_core_path = project_root.parent / 'libraries' / 'kbase-core'
if kbase_core_path.exists():
    sys.path.insert(0, str(kbase_core_path))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.tools.collections import setup_collections, get_stats

if __name__ == "__main__":
    print("Setting up writing library collections...")
    results = setup_collections()

    for key, info in results.items():
        status = info.get("status", "unknown")
        collection = info.get("collection")
        if status == "error":
            print(f"  ❌ {key}: {collection} — {info.get('error')}")
        else:
            print(f"  ✅ {key}: {collection} — {status}")

    print("\nCollection stats:")
    stats = get_stats()
    for key, info in stats.items():
        print(f"  {key}: {info.get('points_count', 0)} points")
```

- [ ] **Step 2: Run setup script**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
uv run python scripts/setup_collections.py
```

Expected:
```
Setting up writing library collections...
  ✅ passages: writing_passages — created
  ✅ terms: writing_terms — created

Collection stats:
  passages: 0 points
  terms: 0 points
```

- [ ] **Step 3: Commit**

```bash
git add scripts/setup_collections.py
git commit -m "feat: add collection setup script; collections created in Qdrant"
```

---

### Task 8: Seed terminology from existing markdown references

**Files:**
- Create: `mcp-writing-library/scripts/seed_from_markdown.py`

- [ ] **Step 1: Create seed script**

The seed script imports the existing `overstated-language-alternatives.md` and the AI slop blocklist from `natural-writing-quick-reference.md` into the `writing_terms` collection.

```python
#!/usr/bin/env python3
"""
Seed writing_terms collection from existing copywriter reference files.

Sources:
    ~/.claude/agents/references/copywriter/data/overstated-language-alternatives.md
    (AI slop section of) natural-writing-quick-reference.md

Run:
    uv run python scripts/seed_from_markdown.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
kbase_core_path = project_root.parent / 'libraries' / 'kbase-core'
if kbase_core_path.exists():
    sys.path.insert(0, str(kbase_core_path))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.tools.terms import add_term
from src.tools.passages import add_passage

HOME = Path.home()
COPYWRITER_REF = HOME / '.claude/agents/references/copywriter/data'

# ----------------------------------------------------------------
# Terminology seed data
# ----------------------------------------------------------------
# Extracted from overstated-language-alternatives.md + AI slop blocklist
TERMS = [
    # Overstated language (consultancy context)
    dict(preferred="notable", avoid="unprecedented", domain="general", language="en",
         why="Superlative; undermines credibility in evidence-based writing",
         example_bad="This unprecedented initiative...", example_good="This notable initiative..."),
    dict(preferred="favourable", avoid="optimal", domain="general", language="en",
         why="Overstated; use measured language",
         example_bad="optimal conditions", example_good="favourable conditions"),
    dict(preferred="a priority", avoid="paramount", domain="general", language="en",
         why="Hyperbolic; plain language preferred",
         example_bad="It is paramount that...", example_good="It is a priority that..."),
    dict(preferred="significant", avoid="transformative", domain="general", language="en",
         why="Overpromising; use measured language",
         example_bad="transformative change", example_good="significant change"),
    dict(preferred="new", avoid="groundbreaking", domain="general", language="en",
         why="Hyperbole; UNDP standard prohibits superlatives",
         example_bad="groundbreaking research", example_good="new research"),
    dict(preferred="inconsistent", avoid="paradoxical", domain="general", language="en",
         why="Too abstract for evidence-based writing; suggests philosophical impossibility",
         example_bad="a paradoxical approach to rights", example_good="an inconsistent approach to rights"),
    # AI slop verbs
    dict(preferred="examine", avoid="delve into", domain="general", language="en",
         why="AI slop — never appears in sector documents",
         example_bad="Let us delve into the data...", example_good="The data reveals..."),
    dict(preferred="use", avoid="leverage", domain="general", language="en",
         why="Corporate jargon; plain language preferred",
         example_bad="leverage innovative approaches", example_good="apply community-led approaches"),
    dict(preferred="use", avoid="utilize", domain="general", language="en",
         why="Verbose; 'use' is always sufficient",
         example_bad="utilize available resources", example_good="use available resources"),
    dict(preferred="coordinate", avoid="synergize", domain="general", language="en",
         why="Management jargon; avoid in sector documents",
         example_bad="synergize stakeholder outcomes", example_good="coordinate stakeholder inputs"),
    dict(preferred="explain", avoid="unpack", domain="general", language="en",
         why="AI slop — informal; not in development sector writing",
         example_bad="Let us unpack this finding.", example_good="This finding reveals..."),
    # People-first language (SRHR)
    dict(preferred="people living with HIV", avoid="HIV victims", domain="srhr", language="en",
         why="Deficit framing undermines agency. UNDP/UNAIDS standard terminology.",
         example_bad="HIV victims need treatment.", example_good="People living with HIV require access to treatment."),
    dict(preferred="key populations", avoid="high-risk groups", domain="srhr", language="en",
         why="UNAIDS standard; 'high-risk' stigmatises rather than describes structural barriers",
         example_bad="high-risk groups face barriers", example_good="key populations face structural barriers"),
    dict(preferred="sex worker", avoid="prostitute", domain="srhr", language="en",
         why="Rights-based terminology; recognises labour rights",
         example_bad="prostitutes in the study area", example_good="sex workers in the study area"),
    dict(preferred="person who uses drugs", avoid="drug addict", domain="srhr", language="en",
         why="People-first language; avoids criminalising framing",
         example_bad="drug addicts in the programme", example_good="people who use drugs in the programme"),
    dict(preferred="people with disabilities", avoid="the disabled", domain="general", language="en",
         why="People-first language — CRPD standard",
         example_bad="services for the disabled", example_good="services for people with disabilities"),
    # Padding phrases
    dict(preferred="[state the point directly]", avoid="It is important to note that",
         domain="general", language="en",
         why="Padding — adds no meaning; direct statement preferred",
         example_bad="It is important to note that data shows...", example_good="Data shows..."),
    dict(preferred="[state the point directly]", avoid="It goes without saying",
         domain="general", language="en",
         why="If it goes without saying, don't say it",
         example_bad="It goes without saying that results matter.", example_good="Results matter."),
    # PT equivalents
    dict(preferred="pessoas vivendo com HIV", avoid="vítimas do HIV", domain="srhr", language="pt",
         why="Linguagem baseada em direitos. Padrão ONUSIDA/CNCS.",
         example_bad="As vítimas do HIV necessitam...", example_good="As pessoas vivendo com HIV necessitam..."),
    dict(preferred="populações-chave", avoid="grupos de alto risco", domain="srhr", language="pt",
         why="Terminologia padrão ONUSIDA; 'alto risco' estigmatiza em vez de descrever barreiras estruturais",
         example_bad="grupos de alto risco enfrentam barreiras", example_good="populações-chave enfrentam barreiras estruturais"),
]

# ----------------------------------------------------------------
# Passage seed data (before/after models from natural-writing-quick-reference.md)
# ----------------------------------------------------------------
PASSAGES = [
    dict(
        text="The assessment reveals a familiar pattern: progress coexists with persistent gaps. Service delivery has expanded, yet implementation challenges continue to undermine outcomes—particularly in coordination across sectors. What emerges clearly is the need for sustained investment, not as an option but as a prerequisite for consolidating gains.",
        doc_type="executive-summary", language="en", domain="general",
        quality_notes="Classic before/after rewrite. Replaces connector-only sequence with discursive opener and argumentative momentum. Shows contrast without forced 'However'.",
        tags=["discursive", "contrast", "argumentative-momentum"],
        source="natural-writing-quick-reference",
    ),
    dict(
        text="The legal framework has strengthened—reforms enacted since 2019 represent genuine progress. The challenge lies elsewhere: in the persistent gap between policy and practice. Capacity constraints explain part of this disconnect, but enforcement inconsistency points to deeper institutional factors. The question is not whether reforms are needed, but how to translate existing commitments into operational reality.",
        doc_type="report", language="en", domain="governance",
        quality_notes="Findings paragraph with argumentative flow. Short-medium-long rhythm. Uses em-dash for tight transition. Ends with a direct question that frames the analysis.",
        tags=["findings", "argumentative-flow", "rhythm-variation", "policy-practice-gap"],
        source="natural-writing-quick-reference",
    ),
    dict(
        text="O que a análise revela é um quadro de progressos reais mas frágeis. O enquadramento legal fortaleceu-se, os indicadores melhoraram—e no entanto, desafios estruturais persistem. O padrão é consistente: avanços formais que esbarram em limitações de implementação. A questão central não é se existem ganhos, mas se são sustentáveis sem intervenção adicional.",
        doc_type="report", language="pt", domain="governance",
        quality_notes="PT equivalent of the findings before/after. Uses discursive opener 'O que a análise revela'. Avoids em-dash intercalations (uses 'e no entanto' instead). Ends with a direct question.",
        tags=["findings", "discursive", "PT", "policy-practice-gap"],
        source="natural-writing-quick-reference",
    ),
    dict(
        text="Angola's legal framework is evolving. Recent reforms signal political will. However, implementation gaps persist, particularly in rural areas where service infrastructure remains underdeveloped. This urban-rural divide undermines national health targets.",
        doc_type="report", language="en", domain="governance",
        quality_notes="Demonstrates rhythm variation: 5-word emphasis sentence, 7-word sentence, 15-word medium, 8-word conclusion. Mix of short-medium sentences for impact.",
        tags=["rhythm-variation", "short-sentences", "urban-rural", "health"],
        source="natural-writing-quick-reference",
    ),
]


def seed_terms():
    print(f"\nSeeding {len(TERMS)} terminology entries...")
    ok, fail = 0, 0
    for term in TERMS:
        result = add_term(**term)
        if result["success"]:
            ok += 1
            print(f"  ✅ {term['preferred']} ({term['language']})")
        else:
            fail += 1
            print(f"  ❌ {term['preferred']}: {result.get('error')}")
    print(f"Terms: {ok} added, {fail} failed")


def seed_passages():
    print(f"\nSeeding {len(PASSAGES)} exemplary passages...")
    ok, fail = 0, 0
    for passage in PASSAGES:
        result = add_passage(**passage)
        if result["success"]:
            ok += 1
            print(f"  ✅ [{passage['doc_type']} | {passage['language']}] {passage['text'][:50]}...")
        else:
            fail += 1
            print(f"  ❌ {result.get('error')}")
    print(f"Passages: {ok} added, {fail} failed")


if __name__ == "__main__":
    seed_terms()
    seed_passages()
    print("\nDone. Run setup_collections.py stats to verify counts.")
```

- [ ] **Step 2: Run seed script**

```bash
cd /Users/danilodasilva/Documents/Programming/mcp-servers/mcp-writing-library
uv run python scripts/seed_from_markdown.py
```

Expected: All terms and passages seeded with ✅

- [ ] **Step 3: Verify counts**

```bash
uv run python scripts/setup_collections.py
```

Expected: `passages: N points`, `terms: N points` (N > 0)

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: seed script with 20 terms and 4 exemplary passages from copywriter references"
```

---

## Chunk 5: End-to-End Verification

### Task 9: Live test via MCP tools

- [ ] **Step 1: Restart Claude Code to pick up new MCP server**

Confirm `writing-library` appears in available MCP tools.

- [ ] **Step 2: Test search_passages**

Via Claude, run:
```
Search for exemplary passages about implementation gaps in governance
```

Expected: 1-2 results with score > 0.7, showing the findings paragraph.

- [ ] **Step 3: Test search_terms**

Via Claude, run:
```
Look up terminology for "leverage" in the writing library
```

Expected: Returns entry with `preferred: "use"`, `avoid: "leverage"`, with example.

- [ ] **Step 4: Test add_passage (new entry)**

Via Claude, run:
```
Add this passage to the writing library as an executive-summary in English, domain srhr:
"The data leaves little doubt: key populations bear a disproportionate burden while remaining systematically underserved. This is not a gap in knowledge—it is a gap in political commitment and resource allocation."
Quality notes: "Strong evidential opener. Short declarative sentence for impact. Frames problem as structural not informational."
```

Expected: Returns `success: true` with document_id.

- [ ] **Step 5: Verify new passage is searchable**

```
Search writing library for passages about structural barriers key populations
```

Expected: The just-added passage appears in results.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: end-to-end verified — writing library live and searchable"
```

---

## Summary

| Task | Deliverable |
|---|---|
| 1 | Project scaffold, pyproject.toml, .env, venv |
| 2 | `collections.py` — collection names + setup |
| 3 | `passages.py` — add/search passages |
| 4 | `terms.py` — add/search terminology |
| 5 | `server.py` + `main.py` — FastMCP server |
| 6 | Claude registration + mcp-knowledge-base archived |
| 7 | Qdrant collections created |
| 8 | Seed data imported (20 terms, 4 passages) |
| 9 | Live end-to-end verification |

**Post-build:** Update copywriter agent instructions and editorial-review skill to reference `mcp__writing-library__search_passages` and `mcp__writing-library__search_terms` when drafting or reviewing text.
