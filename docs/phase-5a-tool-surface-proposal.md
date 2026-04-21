# Phase 5A — Tool Surface Reduction (Final)

**Date:** 2026-04-21
**Status:** Approved — in implementation
**Goal:** Reduce `src/server.py` MCP tool count from **40** to **20** via `manage_*` consolidation, `admin_add` merge, demotion of list registries to MCP resources, and removal of `setup_collections` from public surface.
**Decisions:** aggressive collapse; breaking changes OK; demote `list_*` registries.

## Final surface (20 tools)

### Scoring / search — kept as-is (12)
1. `search_passages`
2. `search_terms`
3. `check_internal_similarity`
4. `check_external_similarity`
5. `score_writing_patterns`
6. `verify_claims`
7. `score_evidence_density`
8. `score_against_rubric`
9. `check_structure`
10. `score_voice_consistency`
11. `detect_authorship_shift`
12. `flag_vocabulary`

### Merged CRUD / manage (6)
13. **`manage_passage(action, ...)`** — `add` (single or `items=list`), `update`, `delete`, `correction` (original/corrected pair). Replaces `add_passage` + `batch_add_passages` + `update_passage` + `delete_passage` + `record_correction`.
14. **`manage_term(action, ...)`** — `add` (single or `items=list`), `update`, `delete`. Replaces `add_term` + `batch_add_terms` + `update_term` + `delete_term`.
15. **`manage_style_profile(action, ...)`** — `save` (upsert), `load` (by name), `search` (by text/channel), `list`, `harvest-corrections`. Replaces `save_style_profile` + `update_style_profile` + `load_style_profile` + `search_style_profiles` + `list_style_profiles` + `harvest_corrections_to_profile`.
16. **`search_thesaurus(query, rich=False, ...)`** — unified; `rich=True` returns alternatives + collocations + why-avoid (former `suggest_alternatives`). Replaces `search_thesaurus` + `suggest_alternatives`.
17. **`manage_contributions(action, ...)`** — `list`, `review`. Replaces `list_contributions` + `review_contribution`.
18. **`manage_library(action, ...)`** — `stats`, `export`. Replaces `get_library_stats` + `export_library`. (`setup` is internal only.)

### Admin writes merged (1)
19. **`admin_add(kind, ...)`** — `kind ∈ {rubric, template, thesaurus}` with per-kind parameter schemas. Replaces `add_rubric_criterion` + `add_template` + `add_thesaurus_entry`. Admin-only; non-admin → contribution queue (unchanged semantics).

### Admin utility kept (1)
20. `harvest_corrections_to_profile` → folded into `manage_style_profile(action="harvest-corrections")` (already counted at #15). Slot 20 is reserved for reconsideration — likely retained at 20 total once verified against tests.

*(Actual count after folding: 19. Placeholder for any slot that resists merging during implementation.)*

### Demoted to MCP resources (3 removed from tool surface)
- `list_styles` → `writing-library://styles`
- `list_rubric_frameworks` → `writing-library://rubric-frameworks`
- `list_templates` → `writing-library://templates`

### Removed (1)
- `setup_collections` — kept as internal helper, called lazily on first write.

## Implementation phases

1. **Server refactor** — rewrite `src/server.py` tool registrations to the 19–20 new tools. Keep existing `src/tools/*` modules unchanged (merges happen only at the registration layer).
2. **Resources** — add FastMCP `@mcp.resource()` for the three list registries.
3. **Tests** — update `tests/test_phase1_surface.py` for new tool names and action-dispatch shapes. Add dispatch tests per `manage_*` action.
4. **Run suite** — expect 263-ish passing; fix any breakage introduced by the refactor (not the pre-existing `kbase` import failures).
5. **PR** — separate PR from Phase 4.
