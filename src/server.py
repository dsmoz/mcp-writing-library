"""
MCP Writing Library Server — FastMCP tool definitions.

Tools (see Patterns 1–9 in CLAUDE.md for chained workflows):

    Passages (per-user):
        search_passages, add_passage, update_passage, delete_passage, batch_add_passages
    Terminology (per-user; add_term routes share→library|queue|personal):
        search_terms, add_term, update_term, delete_term, batch_add_terms, record_correction
    Style profiles (per-user, channel-tagged):
        save_style_profile, load_style_profile, update_style_profile,
        search_style_profiles, list_style_profiles, harvest_corrections_to_profile
    Thesaurus (shared, admin-write; non-admin calls route to contribution queue):
        add_thesaurus_entry, search_thesaurus, suggest_alternatives, flag_vocabulary
    Rubrics & templates (shared, admin-write; non-admin calls route to queue):
        add_rubric_criterion, score_against_rubric, list_rubric_frameworks,
        add_template, check_structure, list_templates
    Similarity & craft:
        check_internal_similarity (library), check_external_similarity (web via Tavily),
        score_writing_patterns (mode ∈ {ai, semantic-ai, poetry, song, fiction})
    Evidence & voice:
        verify_claims, score_evidence_density,
        score_voice_consistency, detect_authorship_shift
    Infrastructure:
        list_styles, get_library_stats, setup_collections (admin),
        export_library, list_contributions, review_contribution (admin)
"""
import os
import sys
import requests
from contextvars import ContextVar
from typing import Optional, List
from mcp.server.fastmcp import FastMCP, Context

# ContextVar set by BearerAuthMiddleware from X-Client-ID header (gateway-injected)
current_client_id: ContextVar[str | None] = ContextVar("current_client_id", default=None)


def _client_id(ctx: Context) -> str:
    """Extract client_id from MCP context, middleware ContextVar, or fall back to 'default'."""
    ctx_cid = getattr(ctx, 'client_id', None) if ctx is not None else None
    cv = current_client_id.get()
    resolved = ctx_cid or cv or "default"
    print(f"_client_id: ctx.client_id={ctx_cid!r}, contextvar={cv!r}, resolved={resolved!r}", file=sys.stderr)
    if ctx_cid:
        return ctx_cid
    return cv if cv else "default"


def _require_admin(ctx: Context) -> Optional[str]:
    """Return None if caller is admin, else an error string."""
    admin_id = os.getenv("ADMIN_CLIENT_ID", "")
    if not admin_id:
        return "ADMIN_CLIENT_ID is not configured — admin tools are disabled"
    caller = _client_id(ctx)
    if caller != admin_id:
        return f"Admin access required. Caller '{caller}' is not the configured admin."
    return None


def _check_auth(token: str) -> bool:
    """Accept any token in the API_TOKENS comma-separated list. If unset, allow all."""
    api_tokens_env = os.getenv("API_TOKENS", "")
    if not api_tokens_env:
        return True
    valid = [t.strip() for t in api_tokens_env.split(",") if t.strip()]
    return token in valid


def _oauth_introspect_url() -> str:
    explicit = os.getenv("OAUTH_INTROSPECT_URL", "").strip()
    if explicit:
        return explicit
    issuer = os.getenv("OAUTH_ISSUER_URL", "").strip().rstrip("/")
    if issuer:
        return f"{issuer}/introspect"
    return ""


def _resolve_client_id_by_oauth_token(token: str) -> Optional[str]:
    """Resolve tenant key via mcp-oauth-server /introspect.

    Prefers ``user_id`` (user-level tenancy, post-multi-device migration) and
    falls back to ``client_id`` for legacy tokens issued before the migration.
    """
    introspect_url = _oauth_introspect_url()
    introspect_secret = os.getenv("INTROSPECT_SECRET", "").strip()
    if not token or not introspect_url or not introspect_secret:
        return None
    try:
        timeout_s = float(os.getenv("OAUTH_INTROSPECT_TIMEOUT", "3.0"))
        resp = requests.post(
            introspect_url,
            json={"token": token},
            headers={"x-introspect-secret": introspect_secret},
            timeout=timeout_s,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not payload.get("active"):
            return None
        user_id = (payload.get("user_id") or "").strip()
        if user_id:
            return user_id
        client_id = (payload.get("client_id") or "").strip()
        return client_id or None
    except Exception:
        return None


class BearerAuthMiddleware:
    """ASGI middleware — validates bearer token and extracts X-Client-ID from gateway."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Strip trailing slashes to prevent 307 redirects that break MCP clients
            path = scope.get("path", "")
            if path != "/" and path.endswith("/"):
                scope = dict(scope, path=path.rstrip("/"))
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            token = auth[7:] if auth.startswith("Bearer ") else ""
            oauth_client_id = _resolve_client_id_by_oauth_token(token)
            if not (_check_auth(token) or oauth_client_id is not None):
                import json as _json
                body = _json.dumps({"error": "invalid_token", "error_description": "Authentication required"}).encode()
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
            # Prefer X-User-ID (user-level tenancy), fall back to X-Client-ID
            # (legacy, pre-multi-device-migration) or the introspected token.
            user_id_hdr = headers.get(b"x-user-id", b"").decode().strip()
            client_id_hdr = headers.get(b"x-client-id", b"").decode().strip()
            client_id_val = user_id_hdr or client_id_hdr or oauth_client_id
            ctx_token = current_client_id.set(client_id_val)
            try:
                await self.app(scope, receive, send)
            finally:
                current_client_id.reset(ctx_token)
            return
        await self.app(scope, receive, send)


def _build_mcp() -> FastMCP:
    transport = os.getenv("TRANSPORT", "stdio")
    instructions = (
        "Writing-quality, evidence, and document-intelligence tools backed by Qdrant.\n\n"
        "Key workflows (see CLAUDE.md Patterns 1–9 for full examples):\n"
        "  1. Vocabulary review: search_terms → apply preferred term with `why` as rationale.\n"
        "  2. Seed model passages: check_internal_similarity (verdict='clean') → add_passage.\n"
        "  3. Similarity before adding: always check_internal_similarity first.\n"
        "  4. Style profile extraction: list_styles → save_style_profile (channel-tagged).\n"
        "  5. Craft scoring (unified): score_writing_patterns with mode∈{ai|semantic-ai|poetry|song|fiction}.\n"
        "  6. Evidence verification: verify_claims (ghost_stat=True always blocks) + score_evidence_density.\n"
        "  7. Rubric alignment: score_against_rubric (usaid|undp|global-fund|eu|general).\n"
        "  8. Structure check: check_structure (retrieves stored template, flags missing sections).\n"
        "  9. Vocabulary flagging: flag_vocabulary for lexical AI tells; suggest_alternatives for rich swaps.\n\n"
        "Tenancy: per-user collections ({client_id}_writing_*) are isolated; thesaurus/rubrics/templates/terms_shared "
        "are shared. add_term/add_thesaurus_entry/add_rubric_criterion/add_template route share→library|queue|personal "
        "based on ADMIN_CLIENT_ID. Use check_internal_similarity for your own library, check_external_similarity "
        "to scan the public web via Tavily."
    )
    if transport == "http":
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        return FastMCP("writing-library", host=host, port=port, instructions=instructions)
    return FastMCP("writing-library", instructions=instructions)

mcp = _build_mcp()


@mcp.tool()
def search_passages(
    query: str,
    ctx: Context,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    style: Optional[str] = None,
    rubric_section: Optional[str] = None,
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
        rubric_section: Filter by rubric section (e.g. "results-framework",
                        "technical-approach", "sustainability", "community-led").
                        Retrieves model passages for the specific section you are trying to strengthen.
        top_k: Number of results (default 5, max 20)

    Returns:
        List of matching passages with scores, quality notes, tags, style labels,
        and rubric_section where present
    """
    from src.tools.passages import search_passages as _search
    return _search(
        query=query, doc_type=doc_type, language=language, domain=domain,
        style=style, rubric_section=rubric_section, top_k=top_k,
        client_id=_client_id(ctx),
    )


@mcp.tool()
def add_passage(
    text: str,
    ctx: Context,
    doc_type: str = "general",
    language: str = "en",
    domain: str = "general",
    quality_notes: str = "",
    tags: Optional[List[str]] = None,
    source: str = "manual",
    style: Optional[List[str]] = None,
    rubric_section: Optional[str] = None,
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
        rubric_section: Optional rubric section this passage models
                        (e.g. "results-framework", "technical-approach", "sustainability").
                        Enables retrieval via search_passages(rubric_section=...) to find
                        model passages for a specific section you are trying to strengthen.

    Returns:
        document_id, chunks_created, and warnings on success
    """
    from src.tools.passages import add_passage as _add
    return _add(
        text=text, doc_type=doc_type, language=language, domain=domain,
        quality_notes=quality_notes, tags=tags or [], source=source,
        style=style or [], rubric_section=rubric_section,
        client_id=_client_id(ctx),
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
    ctx: Context,
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
    return _search(query=query, domain=domain, language=language, top_k=top_k, client_id=_client_id(ctx))


@mcp.tool()
def add_term(
    preferred: str,
    ctx: Context,
    avoid: str = "",
    domain: str = "general",
    language: str = "en",
    why: str = "",
    example_bad: str = "",
    example_good: str = "",
    share: bool = False,
    note: str = "",
) -> dict:
    """
    Add a terminology entry.

    By default (share=False), the entry is written to the caller's personal
    dictionary. Pass share=True to contribute to the shared library: admins
    write directly, non-admins submit to the moderation queue.

    Args:
        preferred: The term to use (e.g. "rights-holder", "key populations")
        avoid: Term to avoid (e.g. "victim", "vulnerable groups")
        domain: Thematic area: srhr|governance|climate|general|m-and-e
        language: Language: en|pt|both
        why: Reason for preference
        example_bad: Example of poor usage
        example_good: Example of correct usage
        share: If True, target the shared library (direct for admins, queue for others).
               If False (default), target the caller's personal dictionary.
        note: Optional note for moderators (only used for queued contributions).

    Returns:
        Includes routed_to: "personal" | "library" | "queue".
    """
    caller = _client_id(ctx)

    if not share:
        from src.tools.terms import add_term as _add
        result = _add(
            preferred=preferred, avoid=avoid, domain=domain, language=language,
            why=why, example_bad=example_bad, example_good=example_good,
            client_id=caller,
        )
        if result.get("success"):
            result["routed_to"] = "personal"
        return result

    # share=True
    if _require_admin(ctx) is None:
        from uuid import uuid4
        from src.tools.collections import get_core_collection_names
        from src.tools.registry import VALID_DOMAINS
        if domain not in VALID_DOMAINS:
            return {"success": False, "error": f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}"}
        if not preferred or not preferred.strip():
            return {"success": False, "error": "preferred term cannot be empty"}
        try:
            from kbase.vector.sync_indexing import index_document
        except ImportError:
            return {"success": False, "error": "kbase indexing unavailable"}
        collection = get_core_collection_names().get("terms_shared", "writing_terms_shared")
        document_id = str(uuid4())
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
            "client_id": "shared", "preferred": preferred, "avoid": avoid,
            "domain": domain, "language": language, "why": why,
            "example_bad": example_bad, "example_good": example_good,
            "entry_type": "term", "contributed_by": caller,
        }
        try:
            point_ids = index_document(
                collection_name=collection, document_id=document_id,
                title=preferred, content=content, metadata=metadata,
                context_mode="metadata",
            )
            return {
                "success": True, "document_id": document_id,
                "chunks_created": len(point_ids), "collection": collection,
                "routed_to": "library",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # share=True + non-admin → queue
    from src.tools.contributions import contribute_term as _contribute
    result = _contribute(
        preferred=preferred, contributed_by=caller, avoid=avoid, domain=domain,
        language=language, why=why, example_bad=example_bad,
        example_good=example_good, note=note,
    )
    if result.get("success"):
        _notify_contribution(result.get("contribution_id", ""), "terms", preferred, caller)
        result["routed_to"] = "queue"
    return result


@mcp.tool()
def record_correction(
    original: str,
    corrected: str,
    issue_type: str,
    ctx: Context,
    doc_type: str = "general",
    language: str = "en",
    domain: str = "general",
    source: str = "manual",
) -> dict:
    """
    Store a before/after correction pair so the library learns from real edits.

    Call this whenever an AI-generated or poor passage is rewritten. The original
    is stored tagged 'ai-corrected' (negative example); the corrected version is
    stored tagged 'human-corrected' (positive example). Both share a correction_id
    so they can be retrieved as a pair.

    Over time these pairs feed score_semantic_ai_likelihood() — the more corrections
    are stored, the stronger the semantic signal for what DS-MOZ prose looks like
    when fixed versus when it reads as AI-generated.

    Args:
        original: The original (poor/AI-sounding) passage
        corrected: The improved replacement
        issue_type: What was wrong — e.g. "ai-patterns", "passive-voice",
                    "hollow-intensifier", "deficit-framing", "missing-connector"
        doc_type: Document context: concept-note|report|tor|general|etc.
        language: Language: en|pt
        domain: Thematic area: srhr|governance|climate|general|m-and-e
        source: Origin of the correction (e.g. "danilo-correction", "editorial-review")

    Returns:
        {success, correction_id, collection, original: {document_id, chunks_created},
         corrected: {document_id, chunks_created}}
    """
    from src.tools.passages import record_correction as _record
    return _record(
        original=original, corrected=corrected, issue_type=issue_type,
        doc_type=doc_type, language=language, domain=domain, source=source,
        client_id=_client_id(ctx),
    )


@mcp.tool()
def delete_passage(document_id: str, ctx: Context) -> dict:
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
    return _delete(document_id=document_id, client_id=_client_id(ctx))


@mcp.tool()
def update_passage(
    document_id: str,
    ctx: Context,
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
        client_id=_client_id(ctx),
    )


@mcp.tool()
def delete_term(document_id: str, ctx: Context) -> dict:
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
    return _delete(document_id=document_id, client_id=_client_id(ctx))


@mcp.tool()
def update_term(
    document_id: str,
    ctx: Context,
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
        client_id=_client_id(ctx),
    )


@mcp.tool()
def get_library_stats(ctx: Context) -> dict:
    """
    Return point counts for both Qdrant collections.

    Use this to verify the library is populated and working.

    Returns:
        Stats for writing_passages and writing_terms collections
    """
    from src.tools.collections import get_stats
    return get_stats(client_id=_client_id(ctx))


@mcp.tool()
def setup_collections(ctx: Context) -> dict:
    """
    **Admin only.** Create or verify Qdrant collections for the writing library.

    Run this once on first setup, or to verify collections exist after a Qdrant restart.

    Returns:
        Status for each collection (created|already_exists|error)
    """
    from src.tools.collections import setup_collections as _setup
    return _setup(client_id=_client_id(ctx))


@mcp.tool()
def check_internal_similarity(
    text: str,
    ctx: Context,
    threshold: float = 0.85,
    top_k_per_sentence: int = 3,
    verdict_threshold_pct: float = 30.0,
) -> dict:
    """
    Check if a passage is too similar to content already in the writing library.

    Use this when deduplicating before adding a passage, or when auditing a draft
    for accidental self-plagiarism against previously-stored content. Splits the
    text into sentences and searches each against the writing_passages collection.

    Use `check_internal_similarity` for your own library; use
    `check_external_similarity` to scan the public web via Tavily.

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
        client_id=_client_id(ctx),
    )


@mcp.tool()
def check_external_similarity(
    text: str,
    threshold: float = 0.75,
    max_sentences: int = 3,
    verdict_threshold_pct: float = 30.0,
    search_results: Optional[list] = None,
) -> dict:
    """
    Check a passage against web content for similarity.

    Use `check_external_similarity` for web/public-internet comparisons; use
    `check_internal_similarity` to compare against your own library first.

    Default mode (search_results=None): fetches web results via the Tavily API
    (requires TAVILY_API_KEY) and scores them against the text. If the API key
    is missing, returns fallback instructions for manual search.

    Scoring mode (search_results provided): skips the web fetch and scores the
    supplied Tavily-style results directly — use this after manually calling
    mcp__MCP_DOCKER__tavily_search when no API key is configured.

    Args:
        text: The passage to check
        threshold: Cosine similarity threshold to flag a match (default 0.75)
        max_sentences: Number of key sentences to search (default 3)
        verdict_threshold_pct: % of sentences flagged to trigger "flagged" verdict (default 30)
        search_results: Optional pre-fetched list of {url, content, title} dicts.
                        If provided, web search is skipped.

    Returns:
        overall_similarity_pct, verdict (clean|flagged), flagged sentences with sources.
    """
    if search_results is not None:
        from src.tools.plagiarism import score_external_similarity as _score
        return _score(
            text=text,
            search_results=search_results,
            threshold=threshold,
            verdict_threshold_pct=verdict_threshold_pct,
        )
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
    ctx: Context,
    description: str = "",
    source_documents: list = [],
    channel: Optional[str] = None,
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
        channel: Publishing surface this profile targets
                 (linkedin|facebook|instagram|twitter|whatsapp|tiktok|
                  blog|newsletter|substack|email|report|proposal|
                  executive-summary|tor|press-release|presentation|general)

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
        channel=channel,
        client_id=_client_id(ctx),
    )


@mcp.tool()
def load_style_profile(name: str, ctx: Context) -> dict:
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
    return _load(name=name, client_id=_client_id(ctx))


@mcp.tool()
def update_style_profile(
    name: str,
    ctx: Context,
    new_style_scores: Optional[dict] = None,
    new_rules: Optional[List[str]] = None,
    new_anti_patterns: Optional[List[str]] = None,
    new_sample_excerpts: Optional[List[str]] = None,
    new_source_documents: Optional[List[str]] = None,
    description: Optional[str] = None,
    channel: Optional[str] = None,
    score_weight: float = 0.3,
) -> dict:
    """
    Merge new writing evidence into an existing style profile without overwriting it.

    Call this after analysing new writing samples from the same author — it blends
    the new scores with the existing profile rather than resetting it. Over time
    this builds a richer, more accurate fingerprint of the author's voice.

    Style scores are blended: updated = (1 - score_weight) * existing + score_weight * new.
    Rules and anti_patterns are unioned (no duplicates). Sample excerpts are appended
    and capped at 20.

    Args:
        name: Profile name to update (must already exist via save_style_profile)
        new_style_scores: New dimension scores to blend in (0.0–1.0 per dimension)
        new_rules: Additional writing rules to add (duplicates ignored)
        new_anti_patterns: Additional anti-patterns to add (duplicates ignored)
        new_sample_excerpts: New representative quotes to append
        new_source_documents: Additional source document names to record
        description: Replace the description if provided
        channel: Update the publishing channel tag
                 (linkedin|facebook|instagram|email|report|proposal|general|...)
        score_weight: How much weight to give new scores (default 0.3 = 30% new, 70% existing).
                      Use 0.5 when the new samples are equally representative.

    Returns:
        {success, name, document_id, updated_fields, chunks_created, score_weight_used, warnings}
    """
    from src.tools.style_profiles import update_style_profile as _update
    return _update(
        name=name,
        new_style_scores=new_style_scores,
        new_rules=new_rules,
        new_anti_patterns=new_anti_patterns,
        new_sample_excerpts=new_sample_excerpts,
        new_source_documents=new_source_documents,
        description=description,
        channel=channel,
        score_weight=score_weight,
        client_id=_client_id(ctx),
    )


@mcp.tool()
def harvest_corrections_to_profile(
    profile_name: str,
    ctx: Context,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    min_corrections: int = 3,
    top_k: int = 20,
) -> dict:
    """
    Scan the correction corpus and surface candidate rules for a style profile.

    Retrieves human-corrected passages (stored via record_correction()) and extracts
    the issue types that were most frequently corrected. Returns a list of candidate
    rules and anti-patterns for the agent to present to the user — the user approves
    or skips each one, then approved entries are merged via update_style_profile().

    Typical workflow:
        1. User has accumulated corrections via record_correction()
        2. Call harvest_corrections_to_profile(profile_name="danilo-voice-pt", language="pt")
        3. Present candidates to user: "Want to add these to your profile?"
        4. For approved ones: call update_style_profile(name, new_rules=[...], new_anti_patterns=[...])

    Args:
        profile_name: The style profile to enrich (must already exist)
        language: Filter to corrections in this language (en|pt). Omit for all.
        domain: Filter to corrections in this domain. Omit for all.
        min_corrections: Minimum human-corrected passages required before returning
                         candidates (default 3). Prevents noise from tiny corpora.
        top_k: Max corrections to analyse (default 20)

    Returns:
        {success, profile_name, corrections_found,
         candidates: [{type, text, source_issue_type, example_corrected}],
         insufficient_data, note}
    """
    from src.tools.style_profiles import harvest_corrections_to_profile as _harvest
    return _harvest(
        profile_name=profile_name,
        language=language,
        domain=domain,
        min_corrections=min_corrections,
        top_k=top_k,
        client_id=_client_id(ctx),
    )


@mcp.tool()
def search_style_profiles(
    text: str,
    ctx: Context,
    top_k: int = 3,
    channel: Optional[str] = None,
) -> dict:
    """
    Find saved style profiles most similar to a writing sample.

    Use this to identify which saved writing style a piece of text most closely
    matches — for example, before tagging a passage or adapting a draft.

    Args:
        text: A writing sample to compare against saved profiles
        top_k: Number of results to return (default 3)
        channel: Optional channel filter to narrow results
                 (linkedin|facebook|email|report|proposal|general|...)

    Returns:
        {success, results: [{score, profile}], total}
    """
    from src.tools.style_profiles import search_style_profiles as _search
    return _search(text=text, top_k=top_k, channel=channel, client_id=_client_id(ctx))


@mcp.tool()
def list_style_profiles(
    ctx: Context,
    channel: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """
    List all saved writing style profiles, optionally filtered by channel.

    Use this to browse available profiles before loading one, or to see
    which channels have profiles defined.

    Args:
        channel: Filter to profiles tagged with a specific publishing surface
                 (linkedin|facebook|instagram|twitter|whatsapp|tiktok|
                  blog|newsletter|substack|email|report|proposal|
                  executive-summary|tor|press-release|presentation|general)
        limit: Max profiles to return (default 50)

    Returns:
        {success, profiles: [{name, description, channel, style_scores, created_at, document_id}], total}
    """
    from src.tools.style_profiles import list_style_profiles as _list
    return _list(channel=channel, limit=limit, client_id=_client_id(ctx))


@mcp.tool()
def batch_add_passages(items: list[dict], ctx: Context) -> dict:
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
    return _batch(items=items, client_id=_client_id(ctx))


@mcp.tool()
def batch_add_terms(items: list[dict], ctx: Context) -> dict:
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
    return _batch(items=items, client_id=_client_id(ctx))


@mcp.tool()
def export_library(collection: str, ctx: Context, output_format: str = "json") -> dict:
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
    return _export(collection=collection, output_format=output_format, client_id=_client_id(ctx))


@mcp.tool()
def verify_claims(
    text: str,
    domain: str = "general",
) -> dict:
    """
    Detect potential hallucinations by checking claim-bearing sentences for
    explicit citation markers. Flags ghost stats (numbers without any source).

    Use this when auditing a draft before submission, or when triaging AI-generated
    prose for unsourced numeric claims. Ghost stats (verdict flag) should always
    block publication. Fully self-contained — no external knowledge-base dependencies.

    Extracts sentences that contain statistics, causal assertions, epistemic
    verbs, citation placeholders, or country/prevalence keywords, then checks
    each for APA or numeric citation markers.

    Claim patterns detected:
        - Numbers and percentages (e.g. "12.5%", "45 percent")
        - Epistemic/causal verbs (shows that, indicates that, evidence suggests…)
        - APA citations (Author Year) or numeric citations [N]
        - Country/prevalence terms (in Mozambique, HIV prevalence, PLHIV…)

    Verdict thresholds:
        evidenced          — ≥80% of claims have citations
        mixed              — 40–80% of claims have citations
        unverified         — <40% of claims have citations
        no_claims_detected — no claim-bearing sentences found; overall_evidence_score
                             is None and no verification was performed

    Ghost stats are uncited claim sentences that contain a number or
    percentage — high-risk hallucination candidates.

    Args:
        text: The passage or document section to analyse
        domain: Thematic domain for domain-specific claim pattern augmentation.
                Valid values: "general", "finance", "governance", "climate",
                "m-and-e", "org", "health". Unknown values fall back to general patterns.

    Returns:
        overall_evidence_score (0–1 or None), verdict (evidenced|mixed|unverified|no_claims_detected),
        total_claims, verified_count, and per-claim results with ghost_stat flag.
    """
    from src.tools.evidence import verify_claims as _verify
    return _verify(text=text, domain=domain)


@mcp.tool()
def score_evidence_density(text: str, domain: str = "general") -> dict:
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
        domain: Thematic domain for domain-specific claim pattern augmentation.
                Valid values: "general", "finance", "governance", "climate",
                "m-and-e", "org", "health". Unknown values fall back to general patterns.

    Returns:
        total_sentences, claim_sentences, cited_sentences, evidence_density (0–1),
        verdict (well-evidenced|partially-evidenced|under-evidenced), domain, and
        a recommendation string with actionable guidance.
    """
    from src.tools.evidence import score_evidence_density as _score
    return _score(text=text, domain=domain)


@mcp.tool()
def add_rubric_criterion(
    ctx: Context,
    framework: str,
    section: str,
    criterion: str,
    weight: float = 1.0,
    red_flags: Optional[List[str]] = None,
    note: str = "",
) -> dict:
    """
    **Admin only** for direct writes; non-admins are auto-routed to the review queue.
    Store an evaluation criterion in the rubric library.

    Admins write directly to the shared rubric collection (routed_to="library").
    Non-admin callers are routed to the moderation queue (routed_to="queue").

    Use this to build up evaluation criteria for any framework — donor proposals
    (usaid, undp, global-fund), client deliverables (lambda, oca-2025), or
    internal standards (ds-moz-editorial). Score later with score_against_rubric().

    Args:
        framework: Evaluation framework slug — any lowercase slug (e.g. "usaid", "undp", "lambda", "oca-2025")
        section: Document section name (e.g. "technical-approach", "sustainability", "m-and-e")
        criterion: The criterion description (what evaluators look for)
        weight: Relative importance 0.1–2.0 (default 1.0). Higher = more important criterion.
        red_flags: Phrases or patterns that evaluators penalise (optional)
        note: Optional free-text note stored with queued contributions (non-admin only)

    Returns:
        {success, routed_to: "library"|"queue", ...} — shape varies by route.
    """
    caller = _client_id(ctx)
    if _require_admin(ctx) is None:
        from src.tools.rubrics import add_rubric_criterion as _add
        result = _add(framework=framework, section=section, criterion=criterion, weight=weight, red_flags=red_flags)
        if result.get("success"):
            result["routed_to"] = "library"
        return result
    from src.tools.contributions import contribute_rubric as _contribute
    result = _contribute(
        framework=framework,
        section=section,
        criterion=criterion,
        contributed_by=caller,
        weight=weight,
        red_flags=red_flags,
        note=note,
    )
    if result.get("success"):
        _notify_contribution(result.get("contribution_id", ""), "rubrics", f"{framework}/{section}", caller)
        result["routed_to"] = "queue"
    return result


@mcp.tool()
def score_against_rubric(
    text: str,
    framework: str,
    section: Optional[str] = None,
    top_k: int = 5,
    doc_context: Optional[str] = None,
) -> dict:
    """
    Score a document section against stored evaluation criteria for a given framework.

    Use this when pre-screening a proposal or report against donor rubrics before
    submission (USAID, UNDP, Global Fund, EU, or custom frameworks). Retrieves the
    most relevant criteria, computes a weighted semantic similarity score, and
    returns a strong/adequate/weak verdict.

    Args:
        text: Document section to score
        framework: Evaluation framework slug to filter criteria (e.g. "usaid", "lambda", "oca-2025")
        section: Optional section filter (e.g. "technical-approach"). If omitted, all sections are used.
        top_k: Number of criteria to match (default 5)
        doc_context: Optional free-text context about the document type (e.g. "annual report"). Not stored — informational only.

    Returns:
        {success, framework, section, text_length, criteria_matched, overall_score,
         verdict (strong|adequate|weak), criteria: [...], doc_context}
        Verdict: strong ≥0.7 | adequate 0.5–0.7 | weak <0.5
        Returns {success: False, error} if framework is empty or no criteria found.
    """
    from src.tools.rubrics import score_against_rubric as _score
    return _score(text=text, framework=framework, section=section, top_k=top_k, doc_context=doc_context)


@mcp.tool()
def list_rubric_frameworks() -> dict:
    """
    Return all frameworks that have at least one criterion stored in the rubric library.

    Use this to see which frameworks are ready for rubric scoring before calling
    score_against_rubric().

    Returns:
        {success, frameworks: [{framework, criterion_count}], total_frameworks, total_criteria}
        Frameworks are sorted alphabetically.
    """
    from src.tools.rubrics import list_rubric_frameworks as _list
    return _list()


@mcp.tool()
def add_template(
    ctx: Context,
    framework: str,
    doc_type: str,
    sections: List[dict],
    note: str = "",
) -> dict:
    """
    **Admin only** for direct writes; non-admins are auto-routed to the review queue.
    Store a document template (list of required sections) for a framework and document type.

    Admins write directly to the shared template collection (routed_to="library").
    Non-admin callers are routed to the moderation queue (routed_to="queue").

    Use this to define the expected structure of documents for any framework — donor proposals
    (usaid, undp), client deliverables (lambda, oca-2025), or internal standards. Once stored,
    use check_structure() to verify a draft covers all required sections.

    Args:
        framework: Evaluation framework slug — any lowercase slug (e.g. "undp", "lambda", "ds-moz")
        doc_type: Document type — must be one of the valid doc_types (see registry)
        sections: List of section dicts. Each must have:
                  - name (str): Section name (e.g. "Executive Summary")
                  - description (str): What this section should contain
                  - required (bool, optional): Whether mandatory (default True)
                  - order (int, optional): Expected position 1-based (default = list index + 1)
        note: Optional free-text note stored with queued contributions (non-admin only)

    Returns:
        {success, routed_to: "library"|"queue", ...} — shape varies by route.
    """
    caller = _client_id(ctx)
    if _require_admin(ctx) is None:
        from src.tools.templates import add_template as _add
        result = _add(framework=framework, doc_type=doc_type, sections=sections)
        if result.get("success"):
            result["routed_to"] = "library"
        return result
    from src.tools.contributions import contribute_template as _contribute
    result = _contribute(
        framework=framework,
        doc_type=doc_type,
        sections=sections,
        contributed_by=caller,
        note=note,
    )
    if result.get("success"):
        _notify_contribution(result.get("contribution_id", ""), "templates", f"{framework}/{doc_type}", caller)
        result["routed_to"] = "queue"
    return result


@mcp.tool()
def check_structure(
    text: str,
    framework: str,
    doc_type: str,
) -> dict:
    """
    Check whether a document draft covers all required sections from the stored template.

    Use this when a draft needs a structural completeness check — before review,
    submission, or as a quick TOC audit. Retrieves the stored template for the
    framework+doc_type pair and evaluates each section's presence in the draft
    using semantic similarity (with keyword fallback).

    Args:
        text: The document draft text to check
        framework: Evaluation framework slug (e.g. "undp", "lambda", "ds-moz")
        doc_type: Document type — must be one of the valid doc_types (see registry)

    Returns:
        {success, framework, doc_type, template_document_id, total_sections, required_sections,
         present_count, partial_count, missing_count, verdict, sections, missing_required}
        verdict is "complete" (0 missing required) or "incomplete" (>0 missing required)
        Each section entry includes: name, required, status (present|partial|missing), coverage_score
    """
    from src.tools.templates import check_structure as _check
    return _check(text=text, framework=framework, doc_type=doc_type)


@mcp.tool()
def list_templates() -> dict:
    """
    Return all stored document templates.

    Use this to see which framework+doc_type combinations have templates stored,
    before calling check_structure().

    Returns:
        {success, templates: [{framework, doc_type, section_count, document_id}], total}
        Sorted alphabetically by framework then doc_type.
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

    Use this when reviewing a multi-author draft (proposal chapters, merged sections)
    for voice drift, or when validating that a ghost-written section matches a saved
    author profile.

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
def score_writing_patterns(
    text: str,
    mode: str,
    ctx: Context,
    language: str = "auto",
    doc_type: Optional[str] = None,
    threshold: float = 0.25,
    top_k: int = 10,
) -> dict:
    """
    Score text against craft patterns. Single entry point for all five scoring modes.

    Use this to detect AI-writing tells, semantic overlap with known-AI passages, or
    craft issues in poetry/song/fiction drafts — dispatch via `mode`.

    Modes:
        ai           — Known AI prose patterns (connectors, hollow intensifiers, em-dash intercalation,
                       passive voice, paragraph length, discursive deficit, etc.).
                       doc_type: concept-note|full-proposal|eoi|executive-summary|general|annual-report|
                                 monitoring-report|financial-report|assessment|tor|governance-review (default: general)
        semantic-ai  — Embedding-similarity to the user's ai-corrected vs human-corrected corpus.
                       Requires record_correction() calls to build the corpus first.
                       Uses top_k for neighbours per sub-corpus. Ignores doc_type/threshold/language.
        poetry       — Poem-specific craft (rhyme, meter, stanza consistency, line-ending clichés, etc.).
                       doc_type: haiku|sonnet|free-verse|villanelle|spoken-word (default: free-verse)
        song         — Lyric-specific craft (verse/chorus structure, hook repetition, singability, etc.).
                       doc_type: pop-song|ballad|rap-verse|hymn|jingle (default: pop-song)
        fiction      — Prose-fiction craft (show-vs-tell, dialogue tags, adverb overload, purple prose).
                       doc_type: novel-chapter|short-story|flash-fiction|screenplay|creative-nonfiction
                                 (default: short-story)

    Args:
        text: The text to score
        mode: ai|semantic-ai|poetry|song|fiction
        language: "en", "pt", or "auto" (ignored by semantic-ai)
        doc_type: Mode-specific document/form type (see per-mode defaults above)
        threshold: Per-category score above which a category is flagged (ignored by semantic-ai)
        top_k: Neighbours per sub-corpus (semantic-ai only)

    Returns:
        Mode-specific result dict. See the underlying helpers for shape.
        For ai/poetry/song/fiction: overall_score, verdict, per-category scores, counts, doc_type.
        For semantic-ai: likelihood, verdict, mean-similarity stats, sample counts, method.
    """
    if mode == "semantic-ai":
        from src.tools.ai_patterns import score_semantic_ai_likelihood as _score
        return _score(text=text, top_k=top_k, client_id=_client_id(ctx))
    if mode == "ai":
        from src.tools.ai_patterns import score_ai_patterns as _score
        return _score(text=text, language=language, threshold=threshold, doc_type=doc_type or "general")
    if mode == "poetry":
        from src.tools.poetry_patterns import score_poetry_patterns as _score
        return _score(text=text, doc_type=doc_type or "free-verse", language=language, threshold=threshold)
    if mode == "song":
        from src.tools.song_patterns import score_song_patterns as _score
        return _score(text=text, doc_type=doc_type or "pop-song", language=language, threshold=threshold)
    if mode == "fiction":
        from src.tools.fiction_patterns import score_fiction_patterns as _score
        return _score(text=text, doc_type=doc_type or "short-story", language=language, threshold=threshold)
    return {
        "success": False,
        "error": f"Invalid mode '{mode}'. Must be one of: ai, semantic-ai, poetry, song, fiction",
    }


@mcp.tool()
def suggest_alternatives(
    word: str,
    language: str = "en",
    domain: str = "general",
    context_sentence: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Look up a word in the vocabulary thesaurus and return rich alternatives.

    Returns definition, register, meaning nuances, collocations, and why the word sounds AI-generated.
    Falls back to search_terms if the word is not in the thesaurus.
    Use this when drafting or reviewing text and you want to replace an overused or AI-sounding word.

    Args:
        word: The word to look up (e.g. "leverage", "robust", "ensure")
        language: Language of the word: en|pt
        domain: Thematic domain: srhr|governance|climate|general|m-and-e|health|finance|org
        context_sentence: Optional sentence where the word appears (reserved for future semantic re-ranking)
        top_k: Maximum alternatives to return (default 5)

    Returns:
        definition, why_avoid, alternatives (with word, meaning_nuance, register, when_to_use),
        collocations, example_bad, example_good. found_in_thesaurus flag indicates source.
    """
    from src.tools.thesaurus import suggest_alternatives as _suggest
    return _suggest(word=word, language=language, domain=domain,
                    context_sentence=context_sentence, top_k=top_k)


@mcp.tool()
def add_thesaurus_entry(
    ctx: Context,
    headword: str,
    language: str = "en",
    domain: str = "general",
    definition: str = "",
    part_of_speech: str = "verb",
    register: str = "neutral",
    alternatives: Optional[List[dict]] = None,
    collocations: Optional[List[str]] = None,
    why_avoid: str = "",
    example_bad: str = "",
    example_good: str = "",
    source: str = "manual",
    note: str = "",
) -> dict:
    """
    **Admin only** for direct writes; non-admins are auto-routed to the review queue.
    Add an AI-pattern word to the vocabulary thesaurus.

    Admins write directly to the shared thesaurus (routed_to="library").
    Non-admin callers are routed to the moderation queue (routed_to="queue") and the
    entry becomes visible to all users only after an admin reviews it via
    review_contribution().

    Args:
        headword: The word to flag (e.g. "leverage")
        language: Language: en|pt
        domain: Thematic domain: srhr|governance|climate|general|m-and-e|health|finance|org
        definition: Concise definition of the headword
        part_of_speech: verb | noun | adjective | adverb | phrase
        register: formal | neutral | informal | institutional | academic
        alternatives: List of dicts: [{word, meaning_nuance, register, when_to_use}]
        collocations: Common collocations to flag (e.g. ["robust framework"])
        why_avoid: Why this word sounds AI-generated or overused
        example_bad: Sentence using the headword poorly
        example_good: Sentence using a preferred alternative
        source: Origin: manual|dicionario-aberto|wordnik|harvested (admin path only)
        note: Optional free-text note stored with queued contributions (non-admin only)

    Returns:
        {success, routed_to: "library"|"queue", ...} — shape varies by route.
    """
    caller = _client_id(ctx)
    if _require_admin(ctx) is None:
        from src.tools.thesaurus import add_thesaurus_entry as _add
        result = _add(headword=headword, language=language, domain=domain,
                      definition=definition, part_of_speech=part_of_speech,
                      register=register, alternatives=alternatives or [],
                      collocations=collocations or [], why_avoid=why_avoid,
                      example_bad=example_bad, example_good=example_good, source=source)
        if result.get("success"):
            result["routed_to"] = "library"
        return result
    from src.tools.contributions import contribute_thesaurus_entry as _contribute
    result = _contribute(
        headword=headword, language=language, domain=domain,
        definition=definition, part_of_speech=part_of_speech,
        register=register, alternatives=alternatives or [],
        collocations=collocations or [], why_avoid=why_avoid,
        example_bad=example_bad, example_good=example_good,
        contributed_by=caller, note=note,
    )
    if result.get("success"):
        _notify_contribution(result.get("contribution_id", ""), "thesaurus", headword, caller)
        result["routed_to"] = "queue"
    return result


@mcp.tool()
def search_thesaurus(
    query: str,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 8,
) -> dict:
    """
    Semantic search across thesaurus entries.

    Use to explore what AI-pattern words are stored, or to find entries
    related to a concept (e.g. "governance vocabulary", "verbs for action").

    Args:
        query: What you are searching for
        language: Filter by language: en|pt
        domain: Filter by domain: srhr|governance|climate|general|m-and-e|health|finance|org
        top_k: Number of results (default 8)

    Returns:
        List of matching entries with full metadata including alternatives
    """
    from src.tools.thesaurus import search_thesaurus as _search
    return _search(query=query, language=language, domain=domain, top_k=top_k)


@mcp.tool()
def flag_vocabulary(
    text: str,
    language: str = "en",
    domain: str = "general",
) -> dict:
    """
    Scan text for AI-pattern vocabulary headwords present in the thesaurus.

    Use alongside score_writing_patterns(mode="ai") (which catches structural patterns)
    to get lexical-level flagging. Returns flagged words with occurrence counts
    and a preview of alternatives.

    Args:
        text: The text to scan
        language: Language of the text: en|pt
        domain: Thematic domain: srhr|governance|climate|general|m-and-e|health|finance|org

    Returns:
        flagged_count, verdict (clean|review|ai-sounding), list of flagged entries
        with headword, occurrences, why_avoid, and alternatives_preview
    """
    from src.tools.thesaurus import flag_vocabulary as _flag
    return _flag(text=text, language=language, domain=domain)


# ---------------------------------------------------------------------------
# Contribution moderation (admin-facing)
#
# Note: contribute_* tools were removed. Non-admin callers of add_term/
# add_thesaurus_entry/add_rubric_criterion/add_template are routed to the
# moderation queue automatically (routed_to="queue"). Admins flow directly
# into the shared library (routed_to="library"). See those tools' docstrings.
# ---------------------------------------------------------------------------

@mcp.tool()
def list_contributions(
    ctx: Context,
    status: str = "pending",
    target: Optional[str] = None,
    mine: bool = False,
    limit: int = 50,
) -> dict:
    """
    List contributions from the moderation queue.

    Admins see all contributions. Regular users can only view their own (mine=True is enforced).

    Args:
        status: Filter by status — pending|published|rejected|all (default: pending)
        target: Filter by target — terms|thesaurus|rubrics|templates
        mine: If True, return only your own contributions (always True for non-admins)
        limit: Max results (default 50)

    Returns:
        {success, contributions: [...], total}
    """
    from src.tools.contributions import list_contributions as _list

    caller = _client_id(ctx)
    admin_id = os.getenv("ADMIN_CLIENT_ID", "")
    is_admin = bool(admin_id) and caller == admin_id

    # Non-admins can only see their own contributions
    contributed_by = caller if (mine or not is_admin) else None

    return _list(status=status, target=target, contributed_by=contributed_by, limit=limit)


@mcp.tool()
def review_contribution(
    contribution_id: str,
    action: str,
    ctx: Context,
    rejection_reason: str = "",
) -> dict:
    """
    **Admin only.** Publish or reject a pending contribution.

    On publish: copies the entry into the target shared collection (terms_shared,
    thesaurus, rubrics, or templates). On reject: marks as rejected with reason.

    Args:
        contribution_id: UUID of the contribution to review
        action: "publish" or "reject"
        rejection_reason: Required when action="reject"

    Returns:
        {success, contribution_id, action, target_collection, reviewed_at}
    """
    err = _require_admin(ctx)
    if err:
        return {"success": False, "error": err}

    from src.tools.contributions import review_contribution as _review
    return _review(
        contribution_id=contribution_id,
        action=action,
        reviewed_by=_client_id(ctx),
        rejection_reason=rejection_reason,
    )


# ---------------------------------------------------------------------------
# Internal: Telegram notification for new contributions
# ---------------------------------------------------------------------------

def _notify_contribution(contribution_id: str, target: str, label: str, contributed_by: str) -> None:
    """Fire-and-forget Telegram notification to admin on new contribution."""
    import asyncio

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_OWNER_CHAT_ID", "")
    if not token or not chat_id:
        return

    text = (
        f"📬 *New Contribution*\n\n"
        f"Type: `{target}`\n"
        f"Entry: `{label}`\n"
        f"From: `{contributed_by}`\n"
        f"ID: `{contribution_id}`\n\n"
        f"Review with `list_contributions()` → `review_contribution()`"
    )

    async def _send():
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                    timeout=5.0,
                )
        except Exception as e:
            logger.warning("Telegram contribution notification failed", error=str(e))

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_send())
        else:
            loop.run_until_complete(_send())
    except Exception:
        pass
