# CLAUDE.md — mcp-writing-library

MCP server for writing quality, evidence verification, and document intelligence. Backed by Qdrant hybrid search.

## Multi-Tenant Architecture

The server uses **collection-prefix isolation (Option B)** to support multiple users on one Qdrant instance.

### Collection taxonomy

| Type | Collections | Isolation |
|------|-------------|-----------|
| **Per-user** | `{client_id}_writing_passages`, `{client_id}_writing_terms`, `{client_id}_writing_style_profiles` | Scoped to OAuth `client_id`; never visible to other users |
| **Core/shared** | `writing_thesaurus`, `writing_rubrics`, `writing_templates` | Global; read by all users; managed by seed scripts |

### How client_id flows

1. HTTP transport: `BearerAuthMiddleware` validates Bearer token against `API_TOKENS`, then extracts `client_id` from the gateway-injected `X-Client-ID` header and stores it in a `ContextVar`.
2. `_client_id(ctx)` in `server.py` reads `ctx.client_id` (MCP native) → falls back to `current_client_id` ContextVar (middleware-injected) → falls back to `"default"`.
3. `get_user_collection_names(client_id)` in `collections.py` prefixes with sanitised client_id → `{cid}_writing_passages` etc.
4. Per-user collections are created lazily via `setup_collections(client_id)` on first call.

### stdio mode / local dev

With `TRANSPORT=stdio`, there is no auth context. All tools operate on `default_writing_passages`, `default_writing_terms`, `default_writing_style_profiles`. Core collections are always shared. This is single-tenant by design.

### HTTP mode / multi-tenant

With `TRANSPORT=http` and `API_TOKENS` set, the `mcp-oauth-server` gateway authenticates the caller, resolves their `client_id`, and forwards it via the `X-Client-ID` HTTP header. `BearerAuthMiddleware` reads this header and sets the `current_client_id` ContextVar so all tool handlers receive the correct tenant identity automatically.

### Core collection access control

Write operations on core/shared collections (`add_rubric_criterion`, `add_template`, `add_thesaurus_entry`) are **admin-only** — they require `ADMIN_CLIENT_ID` to be set and the caller's `client_id` to match. Regular users should use `contribute_*()` tools which queue entries for admin review.

### client_id sanitisation

`_safe_client_id()` replaces any character outside `[a-zA-Z0-9_-]` with `_` to produce valid Qdrant collection name segments.

## Module Reference

| Module | Functions | Description |
|--------|-----------|-------------|
| `src/tools/passages` | `add_passage`, `search_passages`, `update_passage`, `delete_passage`, `batch_add_passages` | Store and retrieve exemplary writing passages (per-user) |
| `src/tools/terms` | `add_term`, `search_terms`, `update_term`, `delete_term`, `batch_add_terms` | Store and retrieve terminology dictionary entries (per-user) |
| `src/tools/collections` | `get_collection_names`, `get_user_collection_names`, `get_core_collection_names`, `setup_collections`, `setup_user_collections`, `get_stats` | Manage Qdrant collections; client_id-aware |
| `src/tools/contributions` | `contribute`, `contribute_term`, `contribute_thesaurus_entry`, `contribute_rubric`, `contribute_template`, `list_contributions`, `review_contribution` | Moderation queue for user contributions to shared collections |
| `src/tools/export` | `export_library` | Export any collection to JSON or CSV |
| `src/tools/styles` | `list_styles` | Writing style registry (14 labels across 4 categories) |
| `src/tools/plagiarism` | `check_internal_similarity`, `check_external_similarity`, `score_external_similarity` | Similarity detection against library and web |
| `src/tools/style_profiles` | `save_style_profile`, `load_style_profile`, `update_style_profile`, `search_style_profiles`, `list_style_profiles`, `harvest_corrections_to_profile` | Extract and retrieve writing style profiles from samples; channel-tagged |
| `src/tools/ai_patterns` | `score_ai_patterns` | Detect AI writing patterns; 10 rule-based detectors; calibrated by `doc_type` |
| `src/tools/thesaurus` | `add_thesaurus_entry`, `search_thesaurus`, `suggest_alternatives`, `flag_vocabulary` | Vocabulary intelligence: flag AI-pattern words, suggest naturalistic alternatives (EN + PT) |
| `src/tools/evidence` | `verify_claims`, `score_evidence_density` | Citation-based claim verification, ghost-stat detection, evidence density scoring |
| `src/tools/rubrics` | `add_rubric_criterion`, `score_against_rubric`, `list_rubric_donors` | Donor rubric alignment; USAID/UNDP/GF/EU/general criteria |
| `src/tools/templates` | `add_template`, `check_structure`, `list_templates` | Document structure templates; detect present/missing sections |
| `src/tools/consistency` | `score_voice_consistency`, `detect_authorship_shift` | Multi-author voice drift detection |

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

## Pattern 4 — Style Profile Extraction

When a user shares writing samples and wants their style captured:

1. Call `list_styles()` to see the 14 dimensions
2. Analyse the samples: score each dimension, extract rules, anti-patterns, and representative excerpts
3. Call `save_style_profile()` to persist the result

```python
result = save_style_profile(
    name="danilo-voice-pt",
    description="Analytical but direct. Southern African context. Rhythm variation.",
    style_scores={"narrative": 0.7, "argumentative": 0.8, "conversational": 0.6, "formal": 0.3},
    rules=["Varies sentence length deliberately", "Opens with a concrete observation"],
    anti_patterns=["'leverage'", "passive constructions", "Furthermore/Moreover"],
    sample_excerpts=["In Mozambique, the data tells only part of the story."],
    source_documents=["lambda-proposal-2026.docx", "linkedin-post-march-2026.txt"],
    channel="linkedin",  # optional — publishing surface this profile targets
)
```

Channel vocabulary (`channel` parameter) — controlled set, unknown values warn but save:

| Group | Values |
|-------|--------|
| Social media | `linkedin`, `facebook`, `instagram`, `twitter`, `whatsapp`, `tiktok` |
| Long-form digital | `blog`, `newsletter`, `substack` |
| Professional written | `email`, `report`, `proposal`, `executive-summary`, `tor`, `press-release`, `presentation` |
| Catch-all | `general` |

To retrieve profiles later:

```python
profile = load_style_profile(name="danilo-voice-pt")

# Search by similarity to a text sample (optionally filtered by channel)
matches = search_style_profiles(text="The evidence is clear, but the politics are not.", channel="linkedin")

# Browse all profiles, optionally by channel
all_profiles = list_style_profiles()
linkedin_profiles = list_style_profiles(channel="linkedin")
```

## Pattern 5 — AI Pattern Scoring with doc_type

Pass `doc_type` to calibrate paragraph-length and discursive-deficit thresholds:

```python
result = score_ai_patterns(
    text="...",
    language="auto",
    doc_type="monitoring-report",  # relaxes paragraph limit to 7, discursive target to 0.5
    # doc_type options: concept-note|full-proposal|eoi|executive-summary|general|
    #                   annual-report|monitoring-report|financial-report|assessment|tor|governance-review
)
```

## Pattern 6 — Evidence Verification

```python
result = verify_claims(
    text="...",
    domain="health",  # general|health|finance|governance|climate|m-and-e|org
)
# Returns: overall_evidence_score, verdict, per-claim verdicts, ghost_stat flags
# ghost_stat: True = number with no citation (always a blocker)
# Fully self-contained — checks for APA/numeric citation markers, no external KB
```

## Pattern 7 — Rubric Alignment

```python
result = score_against_rubric(
    text="...",
    donor="undp",       # usaid|undp|global-fund|eu|general
    section="results-framework",  # optional filter
    doc_context="annual report",  # optional context — not stored
)
# verdict: strong (≥0.7) | adequate (0.5–0.7) | weak (<0.5)
```

## Pattern 8 — Structure Check

```python
result = check_structure(
    text="...",
    donor="general",
    doc_type="tor",  # concept-note|full-proposal|eoi|monitoring-report|assessment|tor|governance-review
)
# Returns per-section status: present|partial|missing
```

## Pattern 9 — Vocabulary Flagging and Alternatives

When reviewing a document for AI-sounding vocabulary, scan the full text and retrieve alternatives:

```python
# Flag all AI-pattern words in a paragraph
result = flag_vocabulary(
    text="We will leverage our robust stakeholder network to ensure holistic outcomes.",
    language="en",
    domain="general",
)
# Returns: verdict ("clean" | "review" | "ai-sounding"), flagged_count, flagged[]
# Each flagged entry: headword, positions, alternatives preview

# Get rich alternatives for a specific word
result = suggest_alternatives(word="leverage", language="en", domain="governance")
# Returns: definition, alternatives (word + meaning_nuance + register + when_to_use),
#          collocations, why_avoid — enough for LLM to pick the right substitute

# PT equivalent
result = suggest_alternatives(word="alavancar", language="pt", domain="general")
```

`flag_vocabulary` complements `score_ai_patterns` (structural patterns) with lexical flagging.
`suggest_alternatives` falls back to `search_terms` when word is not in the thesaurus.

## Common Pitfalls

- **Collections must exist before indexing.** Run `setup_collections` once before
  first use; the seed script also calls it automatically.
- **Qdrant env vars required.** Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env.local`.
- **`language` accepts `both` only for `add_term`.** For passages, use `en` or `pt`.
- **`verify_claims` is self-contained.** It checks for citation markers in the text — no external knowledge bases required.
- **Seed scripts must be run manually** to populate rubrics and templates collections: `python scripts/seed_rubrics.py` and `python scripts/seed_templates.py`.
