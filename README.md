# mcp-writing-library

MCP server for writing passages and terminology dictionary with hybrid semantic search.

## Architecture

```mermaid
flowchart TD
    Claude["Claude Code\n(MCP Client)"] -->|run / list_modules| MCP["MCP Server\nmain.py"]

    MCP --> PASSAGES["Passages\npassages.py\nadd_passage · search_passages"]
    MCP --> TERMS["Terminology\nterms.py\nadd_term · search_terms"]
    MCP --> COLLECTIONS["Collections\ncollections.py\nsetup_collections · get_stats"]

    PASSAGES -.->|vector upsert/search| Qdrant1(("Qdrant Cloud\nwriting_passages"))
    TERMS -.->|vector upsert/search| Qdrant2(("Qdrant Cloud\nwriting_terms"))
    COLLECTIONS -.->|ensure collections| Qdrant1
    COLLECTIONS -.->|ensure collections| Qdrant2
```

## Tools

### Passages

| Tool | Function | Description |
|------|----------|-------------|
| `search_passages` | `search_passages(query, doc_type, language, domain, top_k)` | Semantic search over exemplary writing passages |
| `add_passage` | `add_passage(text, doc_type, language, domain, quality_notes, tags, source)` | Store an exemplary writing passage |

### Terminology

| Tool | Function | Description |
|------|----------|-------------|
| `search_terms` | `search_terms(query, domain, language, top_k)` | Search terminology dictionary for preferred vocabulary |
| `add_term` | `add_term(preferred, avoid, domain, language, why, example_bad, example_good)` | Add a terminology entry |

### Utility

| Tool | Function | Description |
|------|----------|-------------|
| `get_library_stats` | `get_library_stats()` | Return point counts for both collections |
| `setup_collections` | `setup_collections()` | Create or verify Qdrant collections |

## Setup

```bash
# Install dependencies
uv pip install -e .

# Copy and fill env vars
cp .env.example .env.local

# Create Qdrant collections (run once)
uv run python scripts/setup_collections.py

# Seed with initial data
uv run python scripts/seed_from_markdown.py

# Start server
uv run python main.py
```

## Valid Metadata Values

| Field | Values |
|-------|--------|
| `doc_type` | `executive-summary` · `concept-note` · `policy-brief` · `report` · `email` · `general` |
| `language` | `en` · `pt` |
| `domain` | `srhr` · `governance` · `climate` · `general` · `m-and-e` |

## Version History

| Version | Date | Summary |
|---------|------|---------|
| 1.0.0 | 2026-03-15 | Initial release: 6 tools, passages + terms modules, cloud Qdrant |
