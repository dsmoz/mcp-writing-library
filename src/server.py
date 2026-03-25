"""
MCP Writing Library Server — FastMCP tool definitions.

Tools:
    search_passages          — semantic search for exemplary writing passages
    add_passage              — store a new exemplary passage
    delete_passage           — delete a passage by document_id
    update_passage           — update fields of an existing passage
    list_styles              — list all valid writing style labels
    search_terms             — semantic search in terminology dictionary
    add_term                 — add a new terminology entry
    delete_term              — delete a term by document_id
    update_term              — update fields of an existing term
    get_library_stats        — collection point counts
    setup_collections        — create/verify Qdrant collections (admin)
    check_internal_similarity — detect similarity against library passages
    check_external_similarity — detect similarity against web content (Tavily)
    score_external_similarity — score pre-fetched search results for similarity
    score_ai_patterns        — score text against known AI writing patterns
    add_rubric_criterion     — store a donor evaluation criterion
    score_against_rubric     — score proposal text against stored criteria
    list_rubric_donors       — list donors with stored rubric criteria
    score_voice_consistency  — measure consistency of voice across sections
    detect_authorship_shift  — flag segments deviating stylistically from the majority
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
    style: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Search for exemplary writing passages by semantic similarity.

    Use this to find model paragraphs when drafting or reviewing documents.
    Returns passages ranked by relevance with quality notes explaining what makes each one effective.

    Args:
        query: What you need (e.g. "executive summary opening about health equity")
        doc_type: Filter by type: executive-summary|concept-note|policy-brief|report|annual-report|monitoring-report|financial-report|assessment|email|tor|general
        language: Filter by language: en|pt
        domain: Filter by domain: srhr|governance|climate|general|m-and-e
        style: Filter by style: narrative|data-driven|argumentative|minimalist|
               formal|conversational|donor-facing|advocacy|
               undp|global-fund|danilo-voice|
               ai-sounding|bureaucratic|jargon-heavy
               Call list_styles() to see descriptions.
        top_k: Number of results (default 5, max 20)

    Returns:
        List of matching passages with scores, quality notes, tags, and style labels
    """
    from src.tools.passages import search_passages as _search
    return _search(query=query, doc_type=doc_type, language=language, domain=domain, style=style, top_k=top_k)


@mcp.tool()
def add_passage(
    text: str,
    doc_type: str = "general",
    language: str = "en",
    domain: str = "general",
    quality_notes: str = "",
    tags: Optional[List[str]] = None,
    source: str = "manual",
    style: Optional[List[str]] = None,
) -> dict:
    """
    Store an exemplary writing passage in the library.

    Use this when you produce a passage you are proud of, or to seed the library
    with models from reference documents (UNDP HDR, Global Fund reports, M&E assessments, TORs, etc.).

    Args:
        text: The passage (one or more paragraphs)
        doc_type: Context: executive-summary|concept-note|policy-brief|report|email|tor|general
        language: Language: en|pt
        domain: Thematic area: srhr|governance|climate|general|m-and-e
        quality_notes: What makes this passage good (helps future retrieval)
        tags: Labels for retrieval e.g. ["discursive", "findings", "contrast"]
        source: Where the passage came from (e.g. "undp-hdr-2024", "manual")
        style: Style labels e.g. ["narrative", "donor-facing"].
               Call list_styles() to see all valid values.
               Unknown values are warned but do not block the save.

    Returns:
        document_id, chunks_created, and warnings on success
    """
    from src.tools.passages import add_passage as _add
    return _add(
        text=text, doc_type=doc_type, language=language, domain=domain,
        quality_notes=quality_notes, tags=tags or [], source=source,
        style=style or [],
    )


@mcp.tool()
def list_styles() -> dict:
    """
    Return all writing style labels with descriptions, grouped by category.

    Use before calling add_passage or search_passages with a style argument
    to see what values are valid.

    Categories:
        structural   — narrative, data-driven, argumentative, minimalist
        tonal        — formal, conversational, donor-facing, advocacy
        source       — undp, global-fund, danilo-voice
        anti-pattern — ai-sounding, bureaucratic, jargon-heavy

    Returns:
        styles grouped by category with descriptions, total count
    """
    from src.tools.styles import list_styles as _list
    return _list()


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
def delete_passage(document_id: str) -> dict:
    """
    Delete a passage from the writing library by document_id.

    Use this to remove an outdated, duplicated, or incorrect passage.
    The document_id is returned by add_passage or search_passages.

    Args:
        document_id: UUID of the passage to delete (from add_passage or search_passages)

    Returns:
        {success: True, document_id, deleted: True} on success,
        or {success: False, error} if not found or deletion fails
    """
    from src.tools.passages import delete_passage as _delete
    return _delete(document_id=document_id)


@mcp.tool()
def update_passage(
    document_id: str,
    text: Optional[str] = None,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    quality_notes: Optional[str] = None,
    tags: Optional[List[str]] = None,
    source: Optional[str] = None,
    style: Optional[List[str]] = None,
) -> dict:
    """
    Update one or more fields of an existing passage.

    Merges the provided fields with existing metadata and re-indexes.
    At least one field must be supplied. Fields not provided remain unchanged.

    Args:
        document_id: UUID of the passage to update
        text: Replacement passage text (re-embeds the document)
        doc_type: New document type: executive-summary|concept-note|policy-brief|report|annual-report|monitoring-report|financial-report|assessment|email|tor|general
        language: New language: en|pt
        domain: New domain: srhr|governance|climate|general|m-and-e
        quality_notes: Updated quality notes
        tags: Replacement tag list (replaces, does not append)
        source: Updated source reference
        style: Replacement style label list. Call list_styles() to see valid values.

    Returns:
        {success: True, document_id, updated_fields: [...], chunks_created, warnings} on success,
        or {success: False, error} on failure
    """
    from src.tools.passages import update_passage as _update
    return _update(
        document_id=document_id,
        text=text, doc_type=doc_type, language=language, domain=domain,
        quality_notes=quality_notes, tags=tags, source=source, style=style,
    )


@mcp.tool()
def delete_term(document_id: str) -> dict:
    """
    Delete a terminology entry from the writing library by document_id.

    Use this to remove a term that is no longer applicable or was added in error.
    The document_id is returned by add_term or search_terms.

    Args:
        document_id: UUID of the term to delete (from add_term or search_terms)

    Returns:
        {success: True, document_id, deleted: True} on success,
        or {success: False, error} if not found or deletion fails
    """
    from src.tools.terms import delete_term as _delete
    return _delete(document_id=document_id)


@mcp.tool()
def update_term(
    document_id: str,
    preferred: Optional[str] = None,
    avoid: Optional[str] = None,
    domain: Optional[str] = None,
    language: Optional[str] = None,
    why: Optional[str] = None,
    example_bad: Optional[str] = None,
    example_good: Optional[str] = None,
) -> dict:
    """
    Update one or more fields of an existing terminology entry.

    Merges the provided fields with existing metadata and re-indexes.
    At least one field must be supplied. Fields not provided remain unchanged.

    Args:
        document_id: UUID of the term to update
        preferred: New preferred term
        avoid: Updated term to avoid
        domain: New domain: srhr|governance|climate|general|m-and-e
        language: New language: en|pt|both
        why: Updated rationale for preference
        example_bad: Updated bad usage example
        example_good: Updated good usage example

    Returns:
        {success: True, document_id, updated_fields: [...], chunks_created} on success,
        or {success: False, error} on failure
    """
    from src.tools.terms import update_term as _update
    return _update(
        document_id=document_id,
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


@mcp.tool()
def check_internal_similarity(
    text: str,
    threshold: float = 0.85,
    top_k_per_sentence: int = 3,
    verdict_threshold_pct: float = 30.0,
) -> dict:
    """
    Check if a passage is too similar to content already in the writing library.

    Splits the text into sentences and searches each against the writing_passages
    collection. Useful for deduplication before adding new passages.

    Args:
        text: The passage to check
        threshold: Cosine similarity threshold to flag a sentence (default 0.85)
        top_k_per_sentence: Max matches to retrieve per sentence (default 3)
        verdict_threshold_pct: % of sentences flagged to trigger overall "flagged" verdict (default 30)

    Returns:
        overall_similarity_pct, verdict (clean|flagged), and list of flagged sentences
        with their matching library entries (document_id, score, excerpt, source)
    """
    from src.tools.plagiarism import check_internal_similarity as _check
    return _check(
        text=text,
        threshold=threshold,
        top_k_per_sentence=top_k_per_sentence,
        verdict_threshold_pct=verdict_threshold_pct,
    )


@mcp.tool()
def check_external_similarity(
    text: str,
    threshold: float = 0.75,
    max_sentences: int = 3,
    verdict_threshold_pct: float = 30.0,
) -> dict:
    """
    Search the web for content similar to the given text using the Tavily API.

    Requires TAVILY_API_KEY in the environment. If not set, returns key sentences
    and fallback instructions for manual search via mcp__MCP_DOCKER__tavily_search,
    followed by a call to score_external_similarity().

    Args:
        text: The passage to check
        threshold: Cosine similarity threshold to flag a match (default 0.75)
        max_sentences: Number of key sentences to search (default 3, reduces API usage)
        verdict_threshold_pct: % of sentences flagged to trigger overall "flagged" verdict (default 30)

    Returns:
        overall_similarity_pct, verdict (clean|flagged), and list of flagged sentences
        with their web sources (url, title, score, snippet).
        If TAVILY_API_KEY is missing: success=False with key_sentences and fallback_instructions.
    """
    from src.tools.plagiarism import check_external_similarity as _check
    return _check(
        text=text,
        threshold=threshold,
        max_sentences=max_sentences,
        verdict_threshold_pct=verdict_threshold_pct,
    )


@mcp.tool()
def save_style_profile(
    name: str,
    style_scores: dict,
    rules: list,
    anti_patterns: list,
    sample_excerpts: list,
    description: str = "",
    source_documents: list = [],
) -> dict:
    """
    Save a writing style profile extracted from writing samples.

    After analysing 2–5 writing samples against the 14 style dimensions,
    call this to persist the profile in Qdrant for future retrieval.

    Workflow:
        1. User shares 2–5 writing samples
        2. Claude analyses them against list_styles() dimensions
        3. Claude calls save_style_profile() with the resulting profile
        4. Use load_style_profile() or search_style_profiles() to retrieve later

    Args:
        name: Unique profile name (e.g. "danilo-voice-pt", "lambda-proposals")
        style_scores: Dict mapping style labels to scores 0.0–1.0
                      (e.g. {"narrative": 0.8, "conversational": 0.7})
        rules: Writing rules inferred from the samples
               (e.g. ["Uses em-dashes for asides", "Avoids passive voice"])
        anti_patterns: Patterns absent or contrary to this style
                       (e.g. ["'leverage'", "Excessive nominalisations"])
        sample_excerpts: Short representative quotes from the writing samples
        description: Human-readable summary of the style
        source_documents: Names or titles of the writing samples analysed

    Returns:
        {success, name, document_id, chunks_created, warnings}
    """
    from src.tools.style_profiles import save_style_profile as _save
    return _save(
        name=name,
        style_scores=style_scores,
        rules=rules,
        anti_patterns=anti_patterns,
        sample_excerpts=sample_excerpts,
        description=description,
        source_documents=source_documents,
    )


@mcp.tool()
def load_style_profile(name: str) -> dict:
    """
    Load a saved writing style profile by exact name.

    Use this to retrieve a profile previously saved with save_style_profile(),
    for example to guide writing generation or passage retrieval.

    Args:
        name: Profile name as used in save_style_profile (e.g. "danilo-voice-pt")

    Returns:
        {success, profile} with full profile payload, or {success: False, error}
    """
    from src.tools.style_profiles import load_style_profile as _load
    return _load(name=name)


@mcp.tool()
def search_style_profiles(text: str, top_k: int = 3) -> dict:
    """
    Find saved style profiles most similar to a writing sample.

    Use this to identify which saved writing style a piece of text most closely
    matches — for example, before tagging a passage or adapting a draft.

    Args:
        text: A writing sample to compare against saved profiles
        top_k: Number of results to return (default 3)

    Returns:
        {success, results: [{score, profile}], total}
    """
    from src.tools.style_profiles import search_style_profiles as _search
    return _search(text=text, top_k=top_k)


@mcp.tool()
def score_external_similarity(
    text: str,
    search_results: list,
    threshold: float = 0.75,
    verdict_threshold_pct: float = 30.0,
) -> dict:
    """
    Score similarity between text and pre-fetched web search results.

    Use this after manually searching with mcp__MCP_DOCKER__tavily_search
    when TAVILY_API_KEY is not configured. Pass search results directly.

    Args:
        text: The original text to check
        search_results: List of {url, content, title} dicts from Tavily search
        threshold: Cosine similarity threshold (default 0.75)
        verdict_threshold_pct: % of sentences flagged to trigger "flagged" verdict (default 30)

    Returns:
        overall_similarity_pct, verdict (clean|flagged), and flagged sentences with web sources
    """
    from src.tools.plagiarism import score_external_similarity as _score
    return _score(
        text=text,
        search_results=search_results,
        threshold=threshold,
        verdict_threshold_pct=verdict_threshold_pct,
    )


@mcp.tool()
def batch_add_passages(items: list[dict]) -> dict:
    """
    Add multiple writing passages in a single call.

    Calls add_passage() for each item. Never raises — errors are collected per item.
    Items missing the required 'text' field are recorded as failed.

    Args:
        items: List of passage dicts. Each dict supports the same fields as
               add_passage(): text (required), doc_type, language, domain,
               quality_notes, tags, source, style.

    Returns:
        {success: True, total, succeeded, failed, results: [per-item result with index]}
    """
    from src.tools.passages import batch_add_passages as _batch
    return _batch(items=items)


@mcp.tool()
def batch_add_terms(items: list[dict]) -> dict:
    """
    Add multiple terminology entries in a single call.

    Calls add_term() for each item. Never raises — errors are collected per item.
    Items missing the required 'preferred' field are recorded as failed.

    Args:
        items: List of term dicts. Each dict supports the same fields as
               add_term(): preferred (required), avoid, domain, language,
               why, example_bad, example_good.

    Returns:
        {success: True, total, succeeded, failed, results: [per-item result with index]}
    """
    from src.tools.terms import batch_add_terms as _batch
    return _batch(items=items)


@mcp.tool()
def export_library(collection: str, output_format: str = "json") -> dict:
    """
    Export all points from a writing library collection.

    Scrolls the entire Qdrant collection in batches of 1000 and returns
    all payloads as JSON or CSV. Useful for backups, audits, or seeding
    another environment.

    Args:
        collection: Logical alias — "passages", "terms", "style_profiles", or "rubrics" —
                    or the literal Qdrant collection name.
        output_format: Output format: "json" (default) or "csv".
                       CSV stringifies list/dict fields automatically.

    Returns:
        {success, collection, count, format, data} on success,
        where data is a list (json) or a CSV string (csv).
        On failure: {success: False, error}.
    """
    from src.tools.export import export_library as _export
    return _export(collection=collection, output_format=output_format)


@mcp.tool()
def verify_claims(
    text: str,
    domain: str = "general",
    top_k_per_claim: int = 3,
    corroboration_threshold: float = 0.65,
) -> dict:
    """
    Detect potential hallucinations by verifying claim-bearing sentences against
    Zotero and Cerebellum knowledge bases.

    Extracts sentences that contain statistics, causal assertions, epistemic
    verbs, citation placeholders, or country/prevalence keywords, then searches
    both knowledge bases for corroborating sources.

    Claim patterns detected:
        - Numbers and percentages (e.g. "12.5%", "45 percent")
        - Epistemic/causal verbs (shows that, indicates that, evidence suggests…)
        - APA citations (Author Year) or numeric citations [N]
        - Country/prevalence terms (in Mozambique, HIV prevalence, PLHIV…)

    Verdict thresholds:
        evidenced          — ≥80% of claims corroborated
        mixed              — 40–80% of claims corroborated
        unverified         — <40% of claims corroborated
        no_claims_detected — no claim-bearing sentences found; overall_evidence_score
                             is None and no verification was performed

    Ghost stats are unverified claim sentences that contain a number or
    percentage with no supporting source — high-risk hallucination candidates.

    Args:
        text: The passage or document section to analyse
        domain: Thematic domain for context (reserved for future use)
        top_k_per_claim: Sources to retrieve per claim sentence (default 3, max 10)
        corroboration_threshold: Minimum score to mark a claim as verified (default 0.65)

    Returns:
        overall_evidence_score (0–1 or None), verdict (evidenced|mixed|unverified|no_claims_detected),
        total_claims, verified_count, and per-claim results with source attribution
        and ghost_stat flag. Degrades gracefully if Zotero or Cerebellum is unavailable.
    """
    from src.tools.evidence import verify_claims as _verify
    return _verify(
        text=text,
        domain=domain,
        top_k_per_claim=top_k_per_claim,
        corroboration_threshold=corroboration_threshold,
    )


@mcp.tool()
def score_evidence_density(text: str) -> dict:
    """
    Analyse the ratio of cited claim sentences to total claim sentences,
    without calling any external search APIs.

    Use this for a fast, offline check of how well-evidenced a document is
    before running the more thorough verify_claims tool.

    Claim detection uses the same patterns as verify_claims:
        - Numbers and percentages
        - Epistemic verbs and causal assertions
        - APA or numeric citation markers
        - Country/prevalence keywords

    Citation detection looks for explicit markers:
        - APA style: (Author, YYYY) or (Author et al. YYYY)
        - Numeric style: [N]

    Verdict thresholds:
        well-evidenced        — cited/claims ≥ 0.6
        partially-evidenced   — cited/claims 0.3–0.6
        under-evidenced       — cited/claims < 0.3

    Args:
        text: The passage or document section to analyse

    Returns:
        total_sentences, claim_sentences, cited_sentences, evidence_density (0–1),
        verdict (well-evidenced|partially-evidenced|under-evidenced), and
        a recommendation string with actionable guidance.
    """
    from src.tools.evidence import score_evidence_density as _score
    return _score(text=text)


@mcp.tool()
def add_rubric_criterion(
    donor: str,
    section: str,
    criterion: str,
    weight: float = 1.0,
    red_flags: Optional[List[str]] = None,
) -> dict:
    """
    Store an evaluation criterion for a document in the rubric library.

    Use this to build up a library of evaluation criteria per donor so that
    documents can later be scored against them with score_against_rubric().

    Args:
        donor: Donor name — must be one of: usaid, undp, global-fund, eu, general
        section: Document section name (e.g. "technical-approach", "sustainability", "m-and-e")
        criterion: The criterion description (what evaluators look for)
        weight: Relative importance 0.1–2.0 (default 1.0). Higher = more important criterion.
        red_flags: Phrases or patterns that evaluators penalise (optional)

    Returns:
        {success, document_id, chunks_created, collection} on success,
        or {success: False, error} on invalid donor or empty criterion
    """
    from src.tools.rubrics import add_rubric_criterion as _add
    return _add(donor=donor, section=section, criterion=criterion, weight=weight, red_flags=red_flags)


@mcp.tool()
def score_against_rubric(
    text: str,
    donor: str,
    section: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Score a document section against stored evaluation criteria for a given donor or evaluator.

    Retrieves the most relevant criteria for the donor (and optionally section),
    computes a weighted semantic similarity score, and returns a verdict.

    Args:
        text: Document text to score (a section or full document)
        donor: Donor name to filter criteria — usaid|undp|global-fund|eu|general
        section: Optional section filter (e.g. "technical-approach"). If omitted, all sections are used.
        top_k: Number of criteria to match (default 5)

    Returns:
        {success, donor, section, text_length, criteria_matched, overall_score,
         verdict (strong|adequate|weak), criteria: [...]}
        Verdict: strong ≥0.7 | adequate 0.5–0.7 | weak <0.5
        Returns {success: False, error} if donor invalid or no criteria found.
    """
    from src.tools.rubrics import score_against_rubric as _score
    return _score(text=text, donor=donor, section=section, top_k=top_k)


@mcp.tool()
def list_rubric_donors() -> dict:
    """
    Return all donors that have at least one criterion stored in the rubric library.

    Use this to see which donors are ready for rubric scoring before calling
    score_against_rubric().

    Returns:
        {success, donors: [{donor, criterion_count}], total_donors, total_criteria}
        Donors are sorted alphabetically.
    """
    from src.tools.rubrics import list_rubric_donors as _list
    return _list()


@mcp.tool()
def add_template(
    donor: str,
    doc_type: str,
    sections: List[dict],
) -> dict:
    """
    Store a document template (list of required sections) for a donor and document type.

    Use this to define the expected structure of documents such as concept notes, proposals, EOIs,
    monitoring reports, or assessments for a specific donor. Once stored, use check_structure()
    to verify a draft covers all required sections.

    Args:
        donor: Donor name — must be one of: usaid, undp, global-fund, eu, general
        doc_type: Document type — must be one of: concept-note, full-proposal, eoi, annual-report, monitoring-report, financial-report, assessment, general
        sections: List of section dicts. Each must have:
                  - name (str): Section name (e.g. "Executive Summary")
                  - description (str): What this section should contain
                  - required (bool, optional): Whether mandatory (default True)
                  - order (int, optional): Expected position 1-based (default = list index + 1)

    Returns:
        {success, document_id, chunks_created, donor, doc_type, section_count} on success,
        or {success: False, error} on invalid input
    """
    from src.tools.templates import add_template as _add
    return _add(donor=donor, doc_type=doc_type, sections=sections)


@mcp.tool()
def check_structure(
    text: str,
    donor: str,
    doc_type: str,
) -> dict:
    """
    Check whether a document draft covers all required sections from the stored template.

    Retrieves the stored template for the donor+doc_type pair and evaluates each section's
    presence in the draft using semantic similarity (with keyword fallback).

    Args:
        text: The document draft text to check
        donor: Donor name — must be one of: usaid, undp, global-fund, eu, general
        doc_type: Document type — must be one of: concept-note, full-proposal, eoi, annual-report, monitoring-report, financial-report, assessment, general

    Returns:
        {success, donor, doc_type, template_document_id, total_sections, required_sections,
         present_count, partial_count, missing_count, verdict, sections, missing_required}
        verdict is "complete" (0 missing required) or "incomplete" (>0 missing required)
        Each section entry includes: name, required, status (present|partial|missing), coverage_score
    """
    from src.tools.templates import check_structure as _check
    return _check(text=text, donor=donor, doc_type=doc_type)


@mcp.tool()
def list_templates() -> dict:
    """
    Return all stored document templates.

    Use this to see which donor+doc_type combinations have templates stored,
    before calling check_structure().

    Returns:
        {success, templates: [{donor, doc_type, section_count, document_id}], total}
        Sorted alphabetically by donor then doc_type.
    """
    from src.tools.templates import list_templates as _list
    return _list()


@mcp.tool()
def score_voice_consistency(
    sections: List[str],
    profile_name: Optional[str] = None,
    top_k_profile: int = 1,
) -> dict:
    """
    Measure how consistently a list of text sections share a voice/style.

    Given 2–20 sections (e.g. proposal chapters written by different authors),
    computes pairwise similarity between sections and optionally scores each
    section against a saved style profile.

    Args:
        sections: List of text sections to compare (2–20 items required)
        profile_name: Saved style profile name to compare against (e.g. "danilo-voice-pt").
                      If omitted, sections are compared to each other only.
        top_k_profile: How many top profile matches to return when profile_name is None (default 1).

    Returns:
        {
            success, section_count,
            inter_section_consistency (0–1, higher = more consistent),
            consistency_verdict (consistent ≥0.7 | moderate 0.5–0.7 | inconsistent <0.5),
            profile_name, profile_consistency (None if no profile),
            profile_verdict (on-voice ≥0.65 | near-voice 0.45–0.65 | off-voice <0.45 | None),
            sections: [{index, preview, drift_score, profile_score}],
            highest_drift_section (index of most drifted section),
            scoring_method (embedding | fallback)
        }
    """
    from src.tools.consistency import score_voice_consistency as _score
    return _score(sections=sections, profile_name=profile_name, top_k_profile=top_k_profile)


@mcp.tool()
def detect_authorship_shift(
    text: str,
    min_segment_length: int = 100,
) -> dict:
    """
    Detect segments in a document that are stylistically different from the majority.

    Splits the text on double newlines, embeds each segment, and flags segments
    whose deviation from the centroid exceeds mean + 1.5 × std — indicating a
    possible change of author or voice.

    Requires at least 3 segments after filtering by min_segment_length.

    Args:
        text: Full document text to analyse
        min_segment_length: Minimum characters per segment (default 100)

    Returns:
        {
            success, total_segments, mean_deviation, std_deviation,
            shifted_segments: [{index, preview, deviation, z_score}],
            shift_detected (True if any shifted segments found),
            scoring_method (embedding | fallback)
        }
        Returns {success: False, error} if fewer than 3 segments found.

    Note: Shift detection becomes statistically unreliable with fewer than 5 segments.
    With exactly 3 segments, the 1.5×std threshold may exceed the maximum possible
    deviation, causing real shifts to go undetected. Prefer 5+ segments for reliable
    results.
    """
    from src.tools.consistency import detect_authorship_shift as _detect
    return _detect(text=text, min_segment_length=min_segment_length)


@mcp.tool()
def score_ai_patterns(
    text: str,
    language: str = "auto",
    threshold: float = 0.25,
    doc_type: str = "general",
) -> dict:
    """
    Score text against known AI writing patterns.

    Returns overall score (0.0=human, 1.0=AI), per-category findings, and a verdict.
    Use before delivery to identify and fix AI-sounding patterns.

    Categories scored:
        connector_repetition  — Overused connectors: Furthermore, Additionally, Moreover
        hollow_intensifiers   — "It is important to note that", "It is crucial that"
        grandiose_openers     — Dramatic paragraph openings typical of AI prose
        em_dash_intercalation — Paired em-dashes used as parenthetical inserts (AI pattern)
        sentence_monotony     — 3+ consecutive sentences of similar length (±3 words)
        passive_voice         — High passive voice density (>25% of sentences)
        paragraph_length      — Paragraphs exceeding configurable per doc_type (default: 5)
        discursive_deficit    — Fewer than configurable per doc_type (default: 1.0/page) discursive expressions
        mechanical_listing    — Paragraph openers: Firstly, Secondly, Thirdly, Finally
        generic_closings      — "In conclusion, this report has shown...", etc.

    Args:
        text: The text to score (full document or section)
        language: "en", "pt", or "auto" (default: auto-detect)
        threshold: Per-category score above which a category is flagged (default: 0.25)
        doc_type: Document type for threshold calibration. One of: concept-note, full-proposal,
                  eoi, executive-summary, general, annual-report, monitoring-report, financial-report,
                  assessment, tor, governance-review. Default: "general".

    Returns:
        dict with overall_score, verdict (clean|review|ai-sounding), per-category
        scores and findings with exact excerpts, summary, word_count, page_equivalent, doc_type.
        Verdict thresholds: clean <0.25 | review 0.25–0.55 | ai-sounding ≥0.55
    """
    from src.tools.ai_patterns import score_ai_patterns as _score
    return _score(text=text, language=language, threshold=threshold, doc_type=doc_type)
