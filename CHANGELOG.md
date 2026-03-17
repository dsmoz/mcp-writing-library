# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [1.1.0] - 2026-03-17

### Added

- `list_styles` — return all valid writing style labels grouped by category (structural, tonal, source, anti-pattern)
- `check_internal_similarity` — detect if a passage is too similar to content already in the library (sentence-level cosine similarity against writing_passages)
- `check_external_similarity` — check a passage against the web via Tavily API; returns verdict (clean|flagged) and web sources
- `score_external_similarity` — score pre-fetched Tavily search results for similarity; fallback for when TAVILY_API_KEY is absent
- `style` parameter on `add_passage` — tag passages with one or more style labels (structural, tonal, source, anti-pattern categories)
- `style` filter on `search_passages` — filter results by style label (post-filter)
- 12 style-tagged seed passages added to `scripts/seed_from_markdown.py`
- `src/tools/styles.py` — single source of truth for 14 valid style values

### Notes

- `check_external_similarity` requires `TAVILY_API_KEY` in env; gracefully degrades to manual fallback via `score_external_similarity`

## [1.0.0] - 2026-03-15

### Added

- `search_passages` — semantic search over exemplary writing passages with filters for doc_type, language, domain
- `add_passage` — store exemplary passages with metadata (doc_type, language, domain, quality_notes, tags, source)
- `search_terms` — semantic search over terminology dictionary; returns preferred terms, what to avoid, and why
- `add_term` — add terminology entries with preferred/avoid pairs, examples, and domain classification
- `get_library_stats` — returns point counts for both Qdrant collections
- `setup_collections` — creates or verifies Qdrant collections on first run
- Seed script with 20 terminology entries and 4 exemplary passages
- Hybrid vector search via `kbase-core` (text-embedding-nomic-embed-text-v1.5, VECTOR_SIZE=768)
- Cloud Qdrant backend

### Notes

- Requires Qdrant cloud instance (configured via env vars)
- Run `setup_collections` once before first use
