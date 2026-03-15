# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

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
