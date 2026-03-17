# CLAUDE.md — mcp-writing-library

MCP server for writing passages and terminology dictionary with hybrid semantic search backed by Qdrant.

## Module Reference

| Module | Functions | Description |
|--------|-----------|-------------|
| `src/tools/passages` | `add_passage`, `search_passages` | Store and retrieve exemplary writing passages |
| `src/tools/terms` | `add_term`, `search_terms` | Store and retrieve terminology dictionary entries |
| `src/tools/collections` | `get_collection_names`, `setup_collections`, `get_stats` | Manage Qdrant collections |
| `src/tools/styles` | `list_styles` | Writing style registry (14 labels across 4 categories) |
| `src/tools/plagiarism` | `check_internal_similarity`, `check_external_similarity`, `score_external_similarity` | Similarity detection against library and web |

## Pattern 1 — Vocabulary Review

When reviewing a document for terminology, call `search_terms` with the suspect
term as query. If a terminology entry exists, apply the preferred term and cite the
`why` field in the review comment.

```python
result = search_terms(query="leverage", domain="general")
# Returns preferred alternatives, what to avoid, and examples
```

## Pattern 2 — Seeding Model Passages

After producing an exceptional paragraph, store it for future retrieval:

```python
result = add_passage(
    text="...",
    doc_type="executive-summary",
    language="en",
    domain="srhr",
    quality_notes="Strong problem framing with HDR 2025 citation",
    tags=["problem-statement", "discursive"],
    source="lambda-proposal-2026"
)
```

## Pattern 3 — Similarity Check Before Adding

Before storing a new passage, verify it is not a duplicate of existing library content:

```python
result = check_internal_similarity(text="...", threshold=0.85)
# Returns verdict: "clean" or "flagged" with matching sentences and scores
# Only call add_passage if verdict == "clean"
```

## Common Pitfalls

- **Collections must exist before indexing.** Run `setup_collections` once before
  first use; the seed script also calls it automatically.
- **Qdrant env vars required.** Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env.local`.
- **`language` accepts `both` only for `add_term`.** For passages, use `en` or `pt`.
