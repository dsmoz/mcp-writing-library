# mcp-writing-library

MCP server for writing passages and terminology dictionary with hybrid semantic search.

## Architecture

```mermaid
flowchart TD
    Claude["Claude Code\n(MCP Client)"] -->|run / list_modules| MCP["MCP Server\nmain.py"]

    MCP --> PASSAGES["Passages\npassages.py\nadd · search · update · delete · batch · list_styles"]
    MCP --> TERMS["Terminology\nterms.py\nadd · search · update · delete · batch"]
    MCP --> COLLECTIONS["Collections\ncollections.py\nsetup · get_stats · export"]
    MCP --> PLAGIARISM["Similarity\nplagiarism.py\ncheck_internal · check_external · score_external"]
    MCP --> AI["AI Patterns\nai_patterns.py\nscore_ai_patterns"]
    MCP --> EVIDENCE["Evidence\nevidence.py\nverify_claims · score_evidence_density"]
    MCP --> RUBRICS["Rubrics\nrubrics.py\nscore_against_rubric · add_criterion · list_donors"]
    MCP --> TEMPLATES["Structure\ntemplates.py\ncheck_structure · add_template · list_templates"]
    MCP --> STYLES["Style Profiles\nstyle_profiles.py\nsave · load · search"]
    MCP --> CONSISTENCY["Consistency\nconsistency.py\nscore_voice_consistency · detect_authorship_shift"]

    PASSAGES -.->|vector upsert/search| Qdrant1(("Qdrant Cloud\nwriting_passages"))
    TERMS -.->|vector upsert/search| Qdrant2(("Qdrant Cloud\nwriting_terms"))
    STYLES -.->|vector upsert/search| Qdrant3(("Qdrant Cloud\nwriting_style_profiles"))
    RUBRICS -.->|vector upsert/search| Qdrant4(("Qdrant Cloud\nwriting_rubrics"))
    TEMPLATES -.->|vector upsert/search| Qdrant5(("Qdrant Cloud\nwriting_templates"))
    COLLECTIONS -.->|ensure all collections| Qdrant1
    PLAGIARISM -.->|sentence search| Qdrant1
    PLAGIARISM -.->|web search| Tavily(("Tavily API\noptional"))
    EVIDENCE -.->|claim search| Zotero(("mcp-zotero-qdrant\nZotero KB"))
    EVIDENCE -.->|claim search| Cerebellum(("mcp-cerebellum\nCerebellum KB"))
    CONSISTENCY -.->|profile match| Qdrant3
```

## Tools

### Passages

| Tool | Function | Description |
|------|----------|-------------|
| `search_passages` | `search_passages(query, doc_type, language, domain, style, top_k)` | Semantic search over exemplary writing passages |
| `add_passage` | `add_passage(text, doc_type, language, domain, quality_notes, tags, source, style)` | Store an exemplary writing passage |
| `list_styles` | `list_styles()` | List all valid writing style labels with descriptions |

### Terminology

| Tool | Function | Description |
|------|----------|-------------|
| `search_terms` | `search_terms(query, domain, language, top_k)` | Search terminology dictionary for preferred vocabulary |
| `add_term` | `add_term(preferred, avoid, domain, language, why, example_bad, example_good)` | Add a terminology entry |

### Plagiarism & Similarity

| Tool | Function | Description |
|------|----------|-------------|
| `check_internal_similarity` | `check_internal_similarity(text, threshold, top_k_per_sentence, verdict_threshold_pct)` | Detect similarity against the writing library |
| `check_external_similarity` | `check_external_similarity(text, threshold, max_sentences, verdict_threshold_pct)` | Check passage similarity against the web via Tavily |
| `score_external_similarity` | `score_external_similarity(text, search_results, threshold, verdict_threshold_pct)` | Score pre-fetched Tavily results for similarity |

### AI Pattern Scoring

| Tool | Function | Description |
|------|----------|-------------|
| `score_ai_patterns` | `score_ai_patterns(text, language, doc_type, threshold)` | Detect AI writing patterns; calibrated by doc_type |

### Evidence & Claims

| Tool | Function | Description |
|------|----------|-------------|
| `verify_claims` | `verify_claims(text, domain, top_k_per_claim)` | Hallucination detection via Zotero + Cerebellum |
| `score_evidence_density` | `score_evidence_density(text, domain)` | Offline ratio of evidenced vs. bare assertions |

### Donor Rubrics

| Tool | Function | Description |
|------|----------|-------------|
| `score_against_rubric` | `score_against_rubric(text, donor, section, doc_context)` | Score passage against donor evaluation criteria |
| `add_rubric_criterion` | `add_rubric_criterion(donor, section, criterion, weight, red_flags)` | Store a rubric evaluation criterion |
| `list_rubric_donors` | `list_rubric_donors()` | List all donors with stored criteria |

### Document Structure

| Tool | Function | Description |
|------|----------|-------------|
| `check_structure` | `check_structure(text, donor, doc_type)` | Detect present/missing/misplaced sections |
| `add_template` | `add_template(donor, doc_type, sections)` | Store document section template |
| `list_templates` | `list_templates()` | List all stored templates |

### Voice Consistency

| Tool | Function | Description |
|------|----------|-------------|
| `score_voice_consistency` | `score_voice_consistency(sections, profile_name)` | Score voice drift across document sections |
| `detect_authorship_shift` | `detect_authorship_shift(text)` | Unsupervised detection of voice deviation |

### Style Profiles

| Tool | Function | Description |
|------|----------|-------------|
| `save_style_profile` | `save_style_profile(name, description, style_scores, rules, anti_patterns, sample_excerpts, source_documents)` | Save a writing style profile |
| `load_style_profile` | `load_style_profile(name)` | Load a saved style profile by name |
| `search_style_profiles` | `search_style_profiles(text, top_k)` | Find best matching style profile for a text sample |

### Utility

| Tool | Function | Description |
|------|----------|-------------|
| `get_library_stats` | `get_library_stats()` | Return point counts for all collections |
| `setup_collections` | `setup_collections()` | Create or verify all Qdrant collections |
| `export_library` | `export_library(collection, format)` | Export a collection to JSON or CSV |

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
| `doc_type` (passages) | `executive-summary` · `concept-note` · `full-proposal` · `eoi` · `policy-brief` · `report` · `annual-report` · `monitoring-report` · `financial-report` · `assessment` · `tor` · `email` · `general` |
| `doc_type` (score_ai_patterns) | `concept-note` · `full-proposal` · `eoi` · `executive-summary` · `annual-report` · `monitoring-report` · `financial-report` · `assessment` · `tor` · `governance-review` · `general` |
| `language` | `en` · `pt` |
| `domain` | `srhr` · `governance` · `climate` · `general` · `m-and-e` · `finance` · `org` · `health` |

## Version History

| Version | Date | Summary |
|---------|------|---------|
| 1.2.0 | 2026-03-26 | 18 new tools: evidence, rubrics, templates, consistency, style profiles, CRUD, batch, export |
| 1.1.0 | 2026-03-17 | Styles system, plagiarism/similarity checks, 4 new tools |
| 1.0.0 | 2026-03-15 | Initial release: 6 tools, passages + terms modules, cloud Qdrant |
