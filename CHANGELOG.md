# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [1.4.0] - 2026-04-03

### Added

- Multi-tenant collection isolation: each OAuth `client_id` gets its own Qdrant collection prefix (`{uid}_writing_passages`, `{uid}_writing_terms`, `{uid}_writing_style_profiles`); tools extract user identity from FastMCP `Context.client_id` and fall back to `"default"` in stdio mode
- `writing_contributions` and `writing_terms_shared` as new core shared collections
- Contribution and moderation system (`src/tools/contributions.py`): users submit entries to the shared pool via `contribute_term`, `contribute_thesaurus_entry`, `contribute_rubric`, `contribute_template`; admins approve/reject via `review_contribution`; non-admins can list only their own pending contributions
- `list_style_profiles(channel, limit)` — browse all saved style profiles, optionally filtered by publishing channel
- `channel` parameter on `save_style_profile`, `update_style_profile`, `search_style_profiles` — tag and filter profiles by publishing surface (linkedin|facebook|instagram|email|report|proposal|...)
- `VALID_CHANNELS` controlled vocabulary in `registry.py` (18 values); unknown values warn but do not block save
- `search_terms` now merges personal + shared (`writing_terms_shared`) results; personal terms win on deduplication; results re-sorted by score and capped at `top_k`

### Changed

- `collections.py` split into `get_user_collection_names(user_id)`, `get_core_collection_names()`, `get_collection_names(user_id)` (union view), and `setup_user_collections(user_id)`
- All per-user tools (`add_passage`, `search_passages`, `add_term`, `search_terms`, `save_style_profile`, `check_internal_similarity`, `score_semantic_ai_likelihood`, `export_library`, etc.) now accept `user_id` internally and read it from OAuth context in `server.py`

## [1.3.0] - 2026-04-03

### Added

- HTTP streaming transport via FastMCP `streamable-http` — server can run as a remote MCP endpoint (Railway, cloud)
- Bearer token authentication: `BearerTokenVerifier` class validates `Authorization: Bearer <token>` on every HTTP request; tokens configured via `API_TOKENS` env var (comma-separated)
- `TRANSPORT` env var: `stdio` (default, local Claude Code) or `http` (Railway/remote)
- `Dockerfile` and `.dockerignore` for Railway deployment; kbase-core vendored in `vendor/kbase/` to avoid Docker git clone issues
- `tests/test_transport.py`: 4 tests covering BearerTokenVerifier and transport mode detection

### Changed

- Embedding model migrated from local LM Studio nomic-embed-text (768D) to OpenAI `text-embedding-3-small` (1536D)
- `VECTOR_SIZE` in `collections.py` now reads `EMBEDDING_DIMENSIONS` env var (default 1536) instead of hardcoded 768
- `pyproject.toml`: kbase-core removed from dependencies (vendored); added `uvicorn>=0.27.0`, `openai>=1.12.0`, `tiktoken>=0.5.2`, `pydantic-settings>=2.1.0`
- Thesaurus expanded to 91 entries (~80 EN + ~28 PT) covering governance, advocacy, SRHR, M&E, and org management domains

## [1.2.2] - 2026-03-26

### Added

- Social media `doc_type` support: `facebook-post`, `linkedin-post`, `instagram-caption` added to `VALID_DOC_TYPES` in `registry.py`
- `score_ai_patterns` calibrated for social doc_types: paragraph limits (facebook: 2, linkedin: 3, instagram: 1) and discursive targets (all social: 0.0 or 0.5)
- `discursive_deficit` check skipped for social doc_types — short-form posts legitimately omit discursive connectors

## [1.2.1] - 2026-03-26

### Added

- `research_paths` parameter on `verify_claims`: accept a list of local file/directory paths (.md, .txt, .pdf); files are read at call time (never indexed) and searched first before Zotero/Cerebellum; a strong local match short-circuits remote searches for that claim
- `_read_research_files` — reads and chunks local research documents from paths/directories into 300-word chunks
- `_search_local_files` — keyword-overlap scoring of local chunks against a claim sentence (offline, no embedding required)

## [1.2.0] - 2026-03-26

### Added

- `score_ai_patterns` — detect AI writing patterns with `doc_type` calibration; 10 rule-based detectors (EN + PT); paragraph and discursive targets now vary by doc type
- `verify_claims` — evidence hallucination detection; extracts claim sentences and cross-references Zotero + Cerebellum; returns per-claim verdicts and ghost_stat flags; domain-aware claim patterns
- `score_evidence_density` — offline ratio of evidenced vs. bare assertions with domain-aware claim detection
- `score_against_rubric` — score a passage against stored donor evaluation criteria (USAID, UNDP, Global Fund, EU, general); optional `doc_context` parameter
- `check_structure` — detect present/missing/misplaced sections against stored proposal templates
- `add_rubric_criterion`, `list_rubric_donors` — build and list donor rubric criteria
- `add_template`, `list_templates` — store and list document structure templates
- `score_voice_consistency` — score multi-section documents for voice/style drift against a named style profile
- `detect_authorship_shift` — unsupervised detection of voice deviation across a document
- `save_style_profile`, `load_style_profile`, `search_style_profiles` — capture and retrieve writing style profiles from samples
- `update_passage`, `delete_passage`, `update_term`, `delete_term` — CRUD operations for passages and terms
- `batch_add_passages`, `batch_add_terms` — bulk insert from list of dicts
- `export_library` — export any collection to JSON or CSV
- `doc_type` parameter on `score_ai_patterns`: calibrates paragraph limits and discursive targets per document type (11 valid values)
- `domain` parameter on `verify_claims` and `score_evidence_density`: adds domain-specific claim patterns (health, finance, governance, climate, m-and-e, org)
- `doc_context` parameter on `score_against_rubric`: optional free-text context included in return dict
- New Qdrant collections: `writing_style_profiles`, `writing_rubrics`, `writing_templates`
- `scripts/seed_rubrics.py` — seeds 21 donor evaluation criteria (USAID, UNDP, Global Fund, EU + 3 general types)
- `scripts/seed_templates.py` — seeds 8 document structure templates including monitoring-report, assessment, tor, governance-review
- Calibration directives seeded into `writing_terms` and `writing_passages` collections (doc_type valid values, paragraph limits, discursive targets per doc type)

### Changed

- `setup_collections` now creates all 5 collections (added style_profiles, rubrics, templates)
- `_detect_paragraph_length` and `_detect_discursive_deficit` defaults now reference `_PARA_LIMITS` and `_DISCURSIVE_TARGETS` dicts (not hardcoded integers)
- Seed rubric data expanded from 12 to 21 criteria to cover non-proposal document types
- `writing-library-integration.md`, `copywriter.md`, `pre-delivery-final-check.md`, `editorial-review/SKILL.md`: doc_type inline lists replaced with pointer to canonical reference; SOTA tools documented

### Fixed

- `_detect_paragraph_length` and `_detect_discursive_deficit` function defaults previously hardcoded — would silently diverge if dict constants changed
- Duplicate `avaliação` in m-and-e domain claim pattern replaced with `supervisão`
- `org` domain patterns tightened to compound phrases only (bare `board`, `HR`, `management` caused false positives)

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
