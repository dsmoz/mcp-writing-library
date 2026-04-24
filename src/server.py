"""
MCP Writing Library Server — FastMCP tool definitions (21-tool surface).

Scoring / search (12, unchanged):
    search_passages, search_terms, check_internal_similarity, check_external_similarity,
    score_writing_patterns, verify_claims, score_evidence_density, score_against_rubric,
    check_structure, score_voice_consistency, detect_authorship_shift, flag_vocabulary

Merged CRUD / manage (7):
    manage_passage(action ∈ {add, update, delete, correction})
    manage_term(action ∈ {add, update, delete})  — share-branching preserved
    manage_style_profile(action ∈ {save, load, search, list, delete, harvest-corrections, inject-context})
    search_thesaurus(query, rich=False, ...)  — rich=True = former suggest_alternatives
    manage_contributions(action ∈ {list, review})
    manage_library(action ∈ {stats, export})
    manage_patterns(action ∈ {list-files, list, add, remove, set-target, reset, my-overrides})
        — per-user overrides on top of core JSON pattern lists in data/patterns/

Admin writes merged (1):
    admin_add(kind ∈ {rubric, template, thesaurus})

Resources (3, demoted from list_* tools):
    writing-library://styles
    writing-library://rubric-frameworks
    writing-library://templates

Setup is internal: setup_collections is called lazily on first write; no longer a public tool.
"""
import os
import sys
import requests
from contextvars import ContextVar
from typing import Optional, List
from mcp.server.fastmcp import FastMCP, Context

from src.models import (
    PatternScoreResult,
    RubricScoreResult,
    SimilarityResult,
    StructureCheckResult,
    StyleProfileSearchResult,
    VerifyClaimsResult,
    VocabularyFlagResult,
)

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
        "20-tool surface (action-dispatch for CRUD, kind-dispatch for admin writes):\n"
        "  1. Vocabulary review: search_terms → apply preferred term with `why` as rationale.\n"
        "  2. Seed model passages: check_internal_similarity → manage_passage(action='add').\n"
        "  3. Corrections pair: manage_passage(action='correction', original=..., corrected=...).\n"
        "  4. Style profiles: manage_style_profile(action ∈ {save, load, search, list, delete, harvest-corrections}).\n"
        "  5. Craft scoring: score_writing_patterns(mode ∈ {ai|pt|semantic-ai|poetry|song|fiction}).\n"
        "  6. Evidence: verify_claims (ghost_stat=True always blocks) + score_evidence_density.\n"
        "  7. Rubric alignment: score_against_rubric; admins add criteria via admin_add(kind='rubric').\n"
        "  8. Structure check: check_structure; admins add templates via admin_add(kind='template').\n"
        "  9. Vocabulary: flag_vocabulary for lexical tells; search_thesaurus(rich=True) for swaps.\n"
        " 10. Contributions: manage_contributions(action ∈ {list, review}). Admins only for review.\n"
        " 11. Library ops: manage_library(action ∈ {stats, export}).\n\n"
        "Registries live as MCP resources: writing-library://styles, "
        "writing-library://rubric-frameworks, writing-library://templates.\n\n"
        "Tenancy: per-user collections ({client_id}_writing_*) are isolated; thesaurus/rubrics/"
        "templates/terms_shared are shared. manage_term(share=True)/admin_add route to library "
        "(admins) or the contribution queue (non-admins) based on ADMIN_CLIENT_ID. Use "
        "check_internal_similarity for your own library, check_external_similarity for the web via Tavily."
    )
    if transport == "http":
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        return FastMCP("writing-library", host=host, port=port, instructions=instructions)
    return FastMCP("writing-library", instructions=instructions)

mcp = _build_mcp()


# ===========================================================================
# 1. SCORING / SEARCH (12 tools, unchanged)
# ===========================================================================

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

    Args:
        query: What you need (e.g. "executive summary opening about health equity")
        doc_type: Filter by type: executive-summary|concept-note|policy-brief|report|email|tor|general
        language: Filter by language: en|pt
        domain: Filter by domain: srhr|governance|climate|general|m-and-e
        style: Filter by style label (see writing-library://styles resource)
        rubric_section: Filter by rubric section (e.g. "results-framework")
        top_k: Number of results (default 5, max 20)

    Returns:
        List of matching passages with scores, quality notes, tags, style labels, rubric_section
    """
    from src.tools.passages import search_passages as _search
    return _search(
        query=query, doc_type=doc_type, language=language, domain=domain,
        style=style, rubric_section=rubric_section, top_k=top_k,
        client_id=_client_id(ctx),
    )


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

    Args:
        query: What you're looking for (e.g. "person living with HIV", "leverage")
        domain: Filter by domain: srhr|governance|climate|general|m-and-e
        language: Filter by language: en|pt
        top_k: Number of results (default 8)

    Returns:
        List of terminology entries with preferred/avoid pairs and examples
    """
    from src.tools.terms import search_terms as _search
    return _search(query=query, domain=domain, language=language, top_k=top_k, client_id=_client_id(ctx))


@mcp.tool()
def check_internal_similarity(
    text: str,
    ctx: Context,
    threshold: float = 0.85,
    top_k_per_sentence: int = 3,
    verdict_threshold_pct: float = 30.0,
) -> SimilarityResult:
    """
    Check if a passage is too similar to content already in the writing library.

    Args:
        text: The passage to check
        threshold: Cosine similarity threshold to flag a sentence (default 0.85)
        top_k_per_sentence: Max matches per sentence (default 3)
        verdict_threshold_pct: % of sentences flagged to trigger overall "flagged" verdict (default 30)

    Returns:
        overall_similarity_pct, verdict (clean|flagged), list of flagged sentences with matches
    """
    from src.tools.plagiarism import check_internal_similarity as _check
    return _check(
        text=text, threshold=threshold, top_k_per_sentence=top_k_per_sentence,
        verdict_threshold_pct=verdict_threshold_pct, client_id=_client_id(ctx),
    )


@mcp.tool()
def check_external_similarity(
    text: str,
    threshold: float = 0.75,
    max_sentences: int = 3,
    verdict_threshold_pct: float = 30.0,
    search_results: Optional[list] = None,
) -> SimilarityResult:
    """
    Check a passage against web content for similarity.

    Default mode (search_results=None): fetches via Tavily API (TAVILY_API_KEY).
    Scoring mode (search_results provided): skips web fetch, scores provided results.

    Args:
        text: The passage to check
        threshold: Cosine similarity threshold (default 0.75)
        max_sentences: Number of key sentences to search (default 3)
        verdict_threshold_pct: % threshold for "flagged" verdict (default 30)
        search_results: Optional pre-fetched list of {url, content, title} dicts

    Returns:
        overall_similarity_pct, verdict (clean|flagged), flagged sentences with sources
    """
    if search_results is not None:
        from src.tools.plagiarism import score_external_similarity as _score
        return _score(
            text=text, search_results=search_results,
            threshold=threshold, verdict_threshold_pct=verdict_threshold_pct,
        )
    from src.tools.plagiarism import check_external_similarity as _check
    return _check(
        text=text, threshold=threshold, max_sentences=max_sentences,
        verdict_threshold_pct=verdict_threshold_pct,
    )


@mcp.tool()
def score_writing_patterns(
    text: str,
    mode: str,
    ctx: Context,
    language: str = "auto",
    doc_type: Optional[str] = None,
    threshold: float = 0.25,
    top_k: int = 10,
) -> PatternScoreResult:
    """
    Score text against craft patterns. Single entry point for all five scoring modes.

    Modes:
        ai           — Rule-based AI prose patterns. doc_type: concept-note|full-proposal|eoi|
                       executive-summary|general|annual-report|monitoring-report|financial-report|
                       assessment|tor|governance-review (default: general)
        semantic-ai  — Embedding similarity to user's ai-corrected vs human-corrected corpus.
                       Uses top_k. Ignores doc_type/threshold/language.
        poetry       — Poem craft. doc_type: haiku|sonnet|free-verse|villanelle|spoken-word
        song         — Lyric craft. doc_type: pop-song|ballad|rap-verse|hymn|jingle
        fiction      — Prose-fiction craft. doc_type: novel-chapter|short-story|flash-fiction|
                       screenplay|creative-nonfiction

    Args:
        text: The text to score
        mode: ai|pt|semantic-ai|poetry|song|fiction
        language: "en", "pt", or "auto" (ignored by semantic-ai)
        doc_type: Mode-specific document/form type
        threshold: Per-category flag threshold (ignored by semantic-ai)
        top_k: Neighbours per sub-corpus (semantic-ai only)

    Returns:
        Mode-specific dict. Invalid mode → {"success": False, "error": ...}.
    """
    if mode == "semantic-ai":
        from src.tools.ai_patterns import score_semantic_ai_likelihood as _score
        return _score(text=text, top_k=top_k, client_id=_client_id(ctx))
    if mode == "ai":
        from src.tools.ai_patterns import score_ai_patterns as _score
        return _score(text=text, language=language, threshold=threshold, doc_type=doc_type or "general", client_id=_client_id(ctx))
    if mode == "pt":
        from src.tools.pt_forensic import score_pt_forensic as _score
        return _score(text=text, language=language, threshold=threshold, doc_type=doc_type or "general", client_id=_client_id(ctx))
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
        "error": f"Invalid mode '{mode}'. Must be one of: ai, pt, semantic-ai, poetry, song, fiction",
    }


@mcp.tool()
def verify_claims(text: str, domain: str = "general") -> VerifyClaimsResult:
    """
    Detect potential hallucinations by checking claim-bearing sentences for citation markers.

    Flags ghost stats (numbers without any source) — always a blocker. Fully self-contained.

    Verdicts: evidenced (≥80% cited) | mixed (40–80%) | unverified (<40%) | no_claims_detected.

    Args:
        text: The passage to analyse
        domain: general|finance|governance|climate|m-and-e|org|health

    Returns:
        overall_evidence_score, verdict, total_claims, verified_count, per-claim ghost_stat flags.
    """
    from src.tools.evidence import verify_claims as _verify
    return _verify(text=text, domain=domain)


@mcp.tool()
def score_evidence_density(text: str, domain: str = "general") -> dict:
    """
    Offline ratio of cited claim sentences to total claim sentences.

    Verdicts: well-evidenced (≥0.6) | partially-evidenced (0.3–0.6) | under-evidenced (<0.3).

    Args:
        text: The passage to analyse
        domain: general|finance|governance|climate|m-and-e|org|health

    Returns:
        total_sentences, claim_sentences, cited_sentences, evidence_density, verdict, recommendation.
    """
    from src.tools.evidence import score_evidence_density as _score
    return _score(text=text, domain=domain)


@mcp.tool()
def score_against_rubric(
    text: str,
    framework: str,
    section: Optional[str] = None,
    top_k: int = 5,
    doc_context: Optional[str] = None,
) -> RubricScoreResult:
    """
    Score a document section against stored rubric criteria for a framework.

    Verdict: strong (≥0.7) | adequate (0.5–0.7) | weak (<0.5).

    Args:
        text: Document section to score
        framework: Framework slug (e.g. "usaid", "undp", "lambda")
        section: Optional section filter (e.g. "technical-approach")
        top_k: Number of criteria to match (default 5)
        doc_context: Optional free-text document-type context

    Returns:
        overall_score, verdict, matched criteria, doc_context.
    """
    from src.tools.rubrics import score_against_rubric as _score
    return _score(text=text, framework=framework, section=section, top_k=top_k, doc_context=doc_context)


@mcp.tool()
def check_structure(text: str, framework: str, doc_type: str) -> StructureCheckResult:
    """
    Check whether a document covers all required sections from the stored template.

    Args:
        text: The document text to check
        framework: Framework slug (e.g. "undp", "lambda", "ds-moz")
        doc_type: Document type (see registry)

    Returns:
        verdict (complete|incomplete), per-section status (present|partial|missing), counts.
    """
    from src.tools.templates import check_structure as _check
    return _check(text=text, framework=framework, doc_type=doc_type)


@mcp.tool()
def score_voice_consistency(
    sections: List[str],
    ctx: Context,
    profile_name: Optional[str] = None,
    top_k_profile: int = 1,
) -> dict:
    """
    Measure voice/style consistency across 2–20 text sections.

    Pairwise similarity + optional comparison to saved style profile.

    Args:
        sections: List of text sections to compare (2–20 items)
        profile_name: Saved profile name (e.g. "danilo-voice-pt"), optional
        top_k_profile: Top profile matches to return when profile_name=None (default 1)

    Returns:
        inter_section_consistency, consistency_verdict, profile_consistency, profile_verdict,
        per-section drift_score / profile_score, highest_drift_section.
    """
    from src.tools.consistency import score_voice_consistency as _score
    return _score(
        sections=sections,
        profile_name=profile_name,
        top_k_profile=top_k_profile,
        client_id=_client_id(ctx),
    )


@mcp.tool()
def detect_authorship_shift(text: str, min_segment_length: int = 100) -> dict:
    """
    Detect segments stylistically different from the majority (possible author change).

    Requires ≥3 segments after filtering. Prefer 5+ segments for statistical reliability.

    Args:
        text: Full document text
        min_segment_length: Minimum characters per segment (default 100)

    Returns:
        mean_deviation, std_deviation, shifted_segments (index, preview, z_score), shift_detected.
    """
    from src.tools.consistency import detect_authorship_shift as _detect
    return _detect(text=text, min_segment_length=min_segment_length)


@mcp.tool()
def flag_vocabulary(
    text: str,
    language: str = "en",
    domain: str = "general",
) -> VocabularyFlagResult:
    """
    Scan text for AI-pattern vocabulary headwords in the thesaurus.

    Args:
        text: The text to scan
        language: en|pt
        domain: srhr|governance|climate|general|m-and-e|health|finance|org

    Returns:
        verdict (clean|review|ai-sounding), flagged_count, flagged entries with alternatives_preview.
    """
    from src.tools.thesaurus import flag_vocabulary as _flag
    return _flag(text=text, language=language, domain=domain)


# ===========================================================================
# 2. MERGED CRUD / MANAGE (6 tools)
# ===========================================================================

@mcp.tool()
def manage_passage(
    action: str,
    ctx: Context,
    # add/update fields
    document_id: Optional[str] = None,
    text: Optional[str] = None,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    quality_notes: Optional[str] = None,
    tags: Optional[List[str]] = None,
    source: Optional[str] = None,
    style: Optional[List[str]] = None,
    rubric_section: Optional[str] = None,
    # batch add
    items: Optional[List[dict]] = None,
    # correction
    original: Optional[str] = None,
    corrected: Optional[str] = None,
    issue_type: Optional[str] = None,
) -> dict:
    """
    Manage exemplary writing passages (per-user collection).

    Actions:
        add        — Store a passage. Pass `text` (single) OR `items` (list of dicts).
                     Single fields: text, doc_type, language, domain, quality_notes, tags,
                     source, style, rubric_section.
        update     — Modify fields of an existing passage. Requires `document_id`.
        delete     — Remove a passage by `document_id`.
        correction — Store an original/corrected pair (both tagged for later semantic-ai scoring).
                     Requires: original, corrected, issue_type (e.g. "ai-patterns", "passive-voice").

    Args:
        action: add|update|delete|correction
        document_id: Required for update/delete
        text, doc_type, language, domain, quality_notes, tags, source, style, rubric_section:
            add/update fields (all optional on update; merged with existing metadata)
        items: For action="add" batch — list of passage dicts
        original, corrected, issue_type: For action="correction"

    Returns:
        Action-specific dict. See underlying passage tool for exact shape.
    """
    client_id = _client_id(ctx)

    if action == "add":
        if items is not None:
            from src.tools.passages import batch_add_passages as _batch
            return _batch(items=items, client_id=client_id)
        if text is None:
            return {"success": False, "error": "action='add' requires 'text' or 'items'"}
        from src.tools.passages import add_passage as _add
        return _add(
            text=text, doc_type=doc_type or "general", language=language or "en",
            domain=domain or "general", quality_notes=quality_notes or "",
            tags=tags or [], source=source or "manual",
            style=style or [], rubric_section=rubric_section,
            client_id=client_id,
        )

    if action == "update":
        if not document_id:
            return {"success": False, "error": "action='update' requires 'document_id'"}
        from src.tools.passages import update_passage as _update
        return _update(
            document_id=document_id,
            text=text, doc_type=doc_type, language=language, domain=domain,
            quality_notes=quality_notes, tags=tags, source=source, style=style,
            client_id=client_id,
        )

    if action == "delete":
        if not document_id:
            return {"success": False, "error": "action='delete' requires 'document_id'"}
        from src.tools.passages import delete_passage as _delete
        return _delete(document_id=document_id, client_id=client_id)

    if action == "correction":
        if not (original and corrected and issue_type):
            return {"success": False, "error": "action='correction' requires 'original', 'corrected', 'issue_type'"}
        from src.tools.passages import record_correction as _record
        return _record(
            original=original, corrected=corrected, issue_type=issue_type,
            doc_type=doc_type or "general", language=language or "en",
            domain=domain or "general", source=source or "manual",
            client_id=client_id,
        )

    return {"success": False, "error": f"Invalid action '{action}'. Must be one of: add, update, delete, correction"}


@mcp.tool()
def manage_term(
    action: str,
    ctx: Context,
    # add/update fields
    document_id: Optional[str] = None,
    preferred: Optional[str] = None,
    avoid: Optional[str] = None,
    domain: Optional[str] = None,
    language: Optional[str] = None,
    why: Optional[str] = None,
    example_bad: Optional[str] = None,
    example_good: Optional[str] = None,
    # share routing (add only)
    share: bool = False,
    note: str = "",
    # batch
    items: Optional[List[dict]] = None,
) -> dict:
    """
    Manage terminology entries. add routes share→library|queue|personal based on ADMIN_CLIENT_ID.

    Actions:
        add    — Add a term. Pass `preferred` (single) OR `items` (list).
                 share=False (default): writes to caller's personal dictionary.
                 share=True + admin: writes to shared library (routed_to="library").
                 share=True + non-admin: queues for moderation (routed_to="queue").
        update — Modify fields of an existing term. Requires `document_id`.
        delete — Remove a term by `document_id`.

    Args:
        action: add|update|delete
        document_id: Required for update/delete
        preferred, avoid, domain, language, why, example_bad, example_good: Term fields
        share: For action="add" — target shared library (admin) or queue (non-admin)
        note: Optional moderator note (queued contributions only)
        items: For action="add" batch — list of term dicts

    Returns:
        Action-specific dict. add includes routed_to: "personal"|"library"|"queue".
    """
    client_id = _client_id(ctx)

    if action == "add":
        if items is not None:
            from src.tools.terms import batch_add_terms as _batch
            return _batch(items=items, client_id=client_id)
        if not preferred:
            return {"success": False, "error": "action='add' requires 'preferred' or 'items'"}

        if not share:
            from src.tools.terms import add_term as _add
            result = _add(
                preferred=preferred, avoid=avoid or "", domain=domain or "general",
                language=language or "en", why=why or "",
                example_bad=example_bad or "", example_good=example_good or "",
                client_id=client_id,
            )
            if result.get("success"):
                result["routed_to"] = "personal"
            return result

        # share=True
        if _require_admin(ctx) is None:
            from uuid import uuid4
            from src.tools.collections import get_core_collection_names
            from src.tools.registry import VALID_DOMAINS
            dom = domain or "general"
            if dom not in VALID_DOMAINS:
                return {"success": False, "error": f"Invalid domain '{dom}'. Must be one of: {sorted(VALID_DOMAINS)}"}
            if not preferred or not preferred.strip():
                return {"success": False, "error": "preferred term cannot be empty"}
            try:
                from kbase.vector.sync_indexing import index_document
            except ImportError:
                return {"success": False, "error": "kbase indexing unavailable"}
            collection = get_core_collection_names().get("terms_shared", "writing_terms_shared")
            document_id_new = str(uuid4())
            content_parts = [
                f"Preferred term: {preferred}",
                f"Avoid: {avoid}" if avoid else "",
                f"Why: {why}" if why else "",
                f"Bad example: {example_bad}" if example_bad else "",
                f"Good example: {example_good}" if example_good else "",
                f"Domain: {dom}",
                f"Language: {language or 'en'}",
            ]
            content = "\n".join(p for p in content_parts if p)
            metadata = {
                "client_id": "shared", "preferred": preferred, "avoid": avoid or "",
                "domain": dom, "language": language or "en", "why": why or "",
                "example_bad": example_bad or "", "example_good": example_good or "",
                "entry_type": "term", "contributed_by": client_id,
            }
            try:
                point_ids = index_document(
                    collection_name=collection, document_id=document_id_new,
                    title=preferred, content=content, metadata=metadata,
                    context_mode="metadata",
                )
                return {
                    "success": True, "document_id": document_id_new,
                    "chunks_created": len(point_ids), "collection": collection,
                    "routed_to": "library",
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # share=True + non-admin → queue
        from src.tools.contributions import contribute_term as _contribute
        result = _contribute(
            preferred=preferred, contributed_by=client_id, avoid=avoid or "",
            domain=domain or "general", language=language or "en",
            why=why or "", example_bad=example_bad or "",
            example_good=example_good or "", note=note,
        )
        if result.get("success"):
            _notify_contribution(result.get("contribution_id", ""), "terms", preferred, client_id)
            result["routed_to"] = "queue"
        return result

    if action == "update":
        if not document_id:
            return {"success": False, "error": "action='update' requires 'document_id'"}
        from src.tools.terms import update_term as _update
        return _update(
            document_id=document_id,
            preferred=preferred, avoid=avoid, domain=domain, language=language,
            why=why, example_bad=example_bad, example_good=example_good,
            client_id=client_id,
        )

    if action == "delete":
        if not document_id:
            return {"success": False, "error": "action='delete' requires 'document_id'"}
        from src.tools.terms import delete_term as _delete
        return _delete(document_id=document_id, client_id=client_id)

    return {"success": False, "error": f"Invalid action '{action}'. Must be one of: add, update, delete"}


@mcp.tool()
def manage_style_profile(
    action: str,
    ctx: Context,
    # common
    name: Optional[str] = None,
    channel: Optional[str] = None,
    # save
    style_scores: Optional[dict] = None,
    rules: Optional[list] = None,
    anti_patterns: Optional[list] = None,
    sample_excerpts: Optional[list] = None,
    description: Optional[str] = None,
    source_documents: Optional[list] = None,
    # save: blend weight for upsert
    score_weight: float = 0.3,
    # search
    text: Optional[str] = None,
    top_k: int = 3,
    # list
    limit: int = 50,
    # harvest-corrections
    language: Optional[str] = None,
    domain: Optional[str] = None,
    min_corrections: int = 3,
    # delete
    document_id: Optional[str] = None,
) -> dict:
    """
    Manage writing style profiles (per-user, channel-tagged).

    Actions:
        save                 — Upsert a profile by `name`. If it exists, blends new evidence
                               into the existing profile (score_weight = new share); otherwise
                               creates it. Requires: name, style_scores, rules, anti_patterns,
                               sample_excerpts.
        load                 — Load by exact `name`.
        search               — Find profiles similar to `text`. Optional `channel` filter.
        list                 — List all profiles. Optional `channel` filter, `limit`.
        delete               — Remove all chunks for a profile by `document_id` from the
                               caller's per-user collection. Cross-user deletion is
                               blocked by collection-prefix isolation.
        harvest-corrections  — Scan correction corpus and surface candidate rules to merge
                               into profile `name`. Returns candidates only — agent must
                               re-save with manage_style_profile(action='save') after review.
        inject-context       — Format profile `name` as a few-shot prompt block (rules +
                               avoid list + example excerpts) ready to paste into a system
                               prompt. Research shows 3–5 excerpts give up to 23.5× style fidelity.

    Args:
        action: save|load|search|list|delete|harvest-corrections|inject-context
        name: Profile name (required by save/load/harvest-corrections)
        channel: Publishing surface (linkedin|email|report|proposal|general|...)
        style_scores, rules, anti_patterns, sample_excerpts, description, source_documents:
            save fields
        score_weight: save — blend weight for existing profile (default 0.3 = 30% new)
        text: search — sample to compare
        top_k: search — results count (default 3)
        limit: list — max profiles (default 50)
        language, domain, min_corrections: harvest-corrections filters
        document_id: delete — profile document UUID to remove (required)

    Returns:
        Action-specific dict. delete returns {success, document_id, chunks_deleted}.
    """
    client_id = _client_id(ctx)

    if action == "save":
        if not name:
            return {"success": False, "error": "action='save' requires 'name'"}
        # upsert: check if profile exists, dispatch to save vs update
        from src.tools.style_profiles import (
            load_style_profile as _load,
            save_style_profile as _save,
            update_style_profile as _update,
        )
        existing = _load(name=name, client_id=client_id)
        if existing.get("success"):
            return _update(
                name=name,
                new_style_scores=style_scores,
                new_rules=rules,
                new_anti_patterns=anti_patterns,
                new_sample_excerpts=sample_excerpts,
                new_source_documents=source_documents,
                description=description,
                channel=channel,
                score_weight=score_weight,
                client_id=client_id,
            )
        # create
        if style_scores is None or rules is None or anti_patterns is None or sample_excerpts is None:
            return {"success": False, "error": "action='save' on new profile requires style_scores, rules, anti_patterns, sample_excerpts"}
        return _save(
            name=name,
            style_scores=style_scores,
            rules=rules,
            anti_patterns=anti_patterns,
            sample_excerpts=sample_excerpts,
            description=description or "",
            source_documents=source_documents or [],
            channel=channel,
            client_id=client_id,
        )

    if action == "load":
        if not name:
            return {"success": False, "error": "action='load' requires 'name'"}
        from src.tools.style_profiles import load_style_profile as _load
        return _load(name=name, client_id=client_id)

    if action == "search":
        if not text:
            return {"success": False, "error": "action='search' requires 'text'"}
        from src.tools.style_profiles import search_style_profiles as _search
        return _search(text=text, top_k=top_k, channel=channel, client_id=client_id)

    if action == "list":
        from src.tools.style_profiles import list_style_profiles as _list
        return _list(channel=channel, limit=limit, client_id=client_id)

    if action == "harvest-corrections":
        if not name:
            return {"success": False, "error": "action='harvest-corrections' requires 'name' (profile_name)"}
        from src.tools.style_profiles import harvest_corrections_to_profile as _harvest
        return _harvest(
            profile_name=name,
            language=language,
            domain=domain,
            min_corrections=min_corrections,
            top_k=top_k if top_k != 3 else 20,
            client_id=client_id,
        )

    if action == "inject-context":
        if not name:
            return {"success": False, "error": "action='inject-context' requires 'name'"}
        from src.tools.style_profiles import get_style_injection_context as _inject
        return _inject(name=name, client_id=client_id)

    if action == "delete":
        if not document_id:
            return {"success": False, "error": "action='delete' requires 'document_id'"}
        from src.tools.style_profiles import delete_style_profile as _delete
        return _delete(document_id=document_id, client_id=client_id)

    return {"success": False, "error": f"Invalid action '{action}'. Must be one of: save, load, search, list, delete, harvest-corrections, inject-context"}


@mcp.tool()
def search_thesaurus(
    query: str,
    rich: bool = False,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 8,
    context_sentence: Optional[str] = None,
) -> dict:
    """
    Search the vocabulary thesaurus, optionally with rich alternatives lookup.

    rich=False (default): Semantic search across thesaurus entries. Returns list of
    matching entries with full metadata including alternatives.

    rich=True: Word-lookup mode (former suggest_alternatives). Returns definition,
    why-avoid, ranked alternatives with meaning_nuance/register/when_to_use, collocations.
    Falls back to search_terms if `query` is not in the thesaurus. Use when
    `query` is a single word you want to replace.

    Args:
        query: Search text, or the word to look up (rich=True)
        rich: If True, return detailed alternatives for `query` as a headword
        language: Filter: en|pt
        domain: Filter: srhr|governance|climate|general|m-and-e|health|finance|org
        top_k: Max results (default 8; alternatives default 5 in rich mode)
        context_sentence: Optional sentence context (rich=True, reserved for future re-ranking)

    Returns:
        rich=False: list of matching entries
        rich=True: definition, why_avoid, alternatives, collocations, found_in_thesaurus
    """
    if rich:
        from src.tools.thesaurus import suggest_alternatives as _suggest
        return _suggest(
            word=query,
            language=language or "en",
            domain=domain or "general",
            context_sentence=context_sentence,
            top_k=top_k if top_k != 8 else 5,
        )
    from src.tools.thesaurus import search_thesaurus as _search
    return _search(query=query, language=language, domain=domain, top_k=top_k)


@mcp.tool()
def manage_contributions(
    action: str,
    ctx: Context,
    # list
    status: str = "pending",
    target: Optional[str] = None,
    mine: bool = False,
    limit: int = 50,
    # review
    contribution_id: Optional[str] = None,
    review_action: Optional[str] = None,
    rejection_reason: str = "",
) -> dict:
    """
    Manage the contribution moderation queue.

    Actions:
        list   — List contributions. Admins see all; non-admins see only their own (mine enforced).
                 Filters: status ∈ {pending, published, rejected, all}, target ∈ {terms, thesaurus,
                 rubrics, templates}, mine (non-admins always True), limit.
        review — **Admin only.** Publish or reject a pending contribution.
                 Requires: contribution_id, review_action ∈ {publish, reject}.
                 rejection_reason required when review_action='reject'.

    Args:
        action: list|review
        status, target, mine, limit: list filters
        contribution_id, review_action, rejection_reason: review fields

    Returns:
        Action-specific dict.
    """
    caller = _client_id(ctx)
    admin_id = os.getenv("ADMIN_CLIENT_ID", "")
    is_admin = bool(admin_id) and caller == admin_id

    if action == "list":
        from src.tools.contributions import list_contributions as _list
        contributed_by = caller if (mine or not is_admin) else None
        return _list(status=status, target=target, contributed_by=contributed_by, limit=limit)

    if action == "review":
        err = _require_admin(ctx)
        if err:
            return {"success": False, "error": err}
        if not contribution_id or not review_action:
            return {"success": False, "error": "action='review' requires 'contribution_id' and 'review_action'"}
        from src.tools.contributions import review_contribution as _review
        return _review(
            contribution_id=contribution_id,
            action=review_action,
            reviewed_by=caller,
            rejection_reason=rejection_reason,
        )

    return {"success": False, "error": f"Invalid action '{action}'. Must be one of: list, review"}


@mcp.tool()
def manage_library(
    action: str,
    ctx: Context,
    # export
    collection: Optional[str] = None,
    output_format: str = "json",
) -> dict:
    """
    Library-level operations.

    Actions:
        stats   — Return point counts for the caller's Qdrant collections.
        export  — Export a collection to JSON or CSV. Requires `collection` alias
                  ("passages", "terms", "style_profiles", "rubrics") or literal name.

    Args:
        action: stats|export
        collection: For action="export" — collection alias or literal Qdrant name
        output_format: For action="export" — "json" (default) or "csv"

    Returns:
        Action-specific dict.
    """
    client_id = _client_id(ctx)

    if action == "stats":
        from src.tools.collections import get_stats
        return get_stats(client_id=client_id)

    if action == "export":
        if not collection:
            return {"success": False, "error": "action='export' requires 'collection'"}
        from src.tools.export import export_library as _export
        return _export(collection=collection, output_format=output_format, client_id=client_id)

    return {"success": False, "error": f"Invalid action '{action}'. Must be one of: stats, export"}


# ===========================================================================
# 3. ADMIN WRITES MERGED (1 tool)
# ===========================================================================

@mcp.tool()
def admin_add(
    kind: str,
    ctx: Context,
    # rubric
    framework: Optional[str] = None,
    section: Optional[str] = None,
    criterion: Optional[str] = None,
    weight: float = 1.0,
    red_flags: Optional[List[str]] = None,
    # template
    doc_type: Optional[str] = None,
    sections: Optional[List[dict]] = None,
    # thesaurus
    headword: Optional[str] = None,
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
    # common
    note: str = "",
) -> dict:
    """
    **Admin-only direct writes; non-admins auto-route to the moderation queue.**

    Add an entry to one of the three shared/core collections. `kind` selects the schema.

    Kinds:
        rubric     — Store a rubric criterion. Requires: framework, section, criterion.
                     Optional: weight (default 1.0), red_flags.
        template   — Store a document template. Requires: framework, doc_type, sections
                     (list of {name, description, required?, order?} dicts).
        thesaurus  — Store a thesaurus headword. Requires: headword.
                     Optional: language, domain, definition, part_of_speech, register,
                     alternatives, collocations, why_avoid, example_bad, example_good, source.

    Non-admin callers are routed to the moderation queue (routed_to="queue"); admins write
    directly (routed_to="library"). Pass `note` to leave a message for moderators when queued.

    Args:
        kind: rubric|template|thesaurus
        framework, section, criterion, weight, red_flags: rubric fields
        framework, doc_type, sections: template fields
        headword, language, domain, definition, part_of_speech, register, alternatives,
            collocations, why_avoid, example_bad, example_good, source: thesaurus fields
        note: Optional moderator note (queued contributions only)

    Returns:
        {success, routed_to: "library"|"queue", ...} — shape varies by kind and route.
    """
    caller = _client_id(ctx)
    is_admin = _require_admin(ctx) is None

    if kind == "rubric":
        if not (framework and section and criterion):
            return {"success": False, "error": "kind='rubric' requires 'framework', 'section', 'criterion'"}
        if is_admin:
            from src.tools.rubrics import add_rubric_criterion as _add
            result = _add(framework=framework, section=section, criterion=criterion,
                          weight=weight, red_flags=red_flags)
            if result.get("success"):
                result["routed_to"] = "library"
            return result
        from src.tools.contributions import contribute_rubric as _contribute
        result = _contribute(
            framework=framework, section=section, criterion=criterion,
            contributed_by=caller, weight=weight, red_flags=red_flags, note=note,
        )
        if result.get("success"):
            _notify_contribution(result.get("contribution_id", ""), "rubrics", f"{framework}/{section}", caller)
            result["routed_to"] = "queue"
        return result

    if kind == "template":
        if not (framework and doc_type and sections):
            return {"success": False, "error": "kind='template' requires 'framework', 'doc_type', 'sections'"}
        if is_admin:
            from src.tools.templates import add_template as _add
            result = _add(framework=framework, doc_type=doc_type, sections=sections)
            if result.get("success"):
                result["routed_to"] = "library"
            return result
        from src.tools.contributions import contribute_template as _contribute
        result = _contribute(
            framework=framework, doc_type=doc_type, sections=sections,
            contributed_by=caller, note=note,
        )
        if result.get("success"):
            _notify_contribution(result.get("contribution_id", ""), "templates", f"{framework}/{doc_type}", caller)
            result["routed_to"] = "queue"
        return result

    if kind == "thesaurus":
        if not headword:
            return {"success": False, "error": "kind='thesaurus' requires 'headword'"}
        if is_admin:
            from src.tools.thesaurus import add_thesaurus_entry as _add
            result = _add(
                headword=headword, language=language, domain=domain,
                definition=definition, part_of_speech=part_of_speech,
                register=register, alternatives=alternatives or [],
                collocations=collocations or [], why_avoid=why_avoid,
                example_bad=example_bad, example_good=example_good, source=source,
            )
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

    return {"success": False, "error": f"Invalid kind '{kind}'. Must be one of: rubric, template, thesaurus"}


# ===========================================================================
# 3b. PER-USER PATTERN MANAGEMENT (1 tool)
# ===========================================================================

@mcp.tool()
def manage_patterns(
    action: str,
    ctx: Context,
    file: Optional[str] = None,
    value: Optional[str] = None,
    doc_type: Optional[str] = None,
    target: Optional[float] = None,
) -> dict:
    """Manage per-user AI-pattern detection lists (runtime read/write).

    Reads return the effective merged data for the caller (core + per-user
    overrides). Writes only ever touch the per-user override layer — core
    defaults under data/patterns/ are read-only here.

    Actions:
        list-files    — Return all core pattern filenames (no params).
        list          — Return effective items/values for `file`.
        add           — Add `value` to caller's added list (items-style only).
        remove        — Drop from caller's added list, or mark as removed if
                        present in core (items-style only).
        set-target    — Upsert `doc_type` → `target` in caller's value overrides
                        (values-style only, e.g. para_limits, hedging_targets).
        reset         — Clear caller's override file for `file`.
        my-overrides  — Summary of all of caller's overrides (no params).

    Args:
        action: list-files|list|add|remove|set-target|reset|my-overrides
        file: pattern filename (without .json) — required by all actions except
              list-files and my-overrides
        value: the item to add/remove (add/remove only)
        doc_type: key for set-target (usually a doc_type like "concept-note")
        target: numeric value for set-target

    Returns:
        Action-specific dict. On mutation, returns the resulting effective
        items/values list.
    """
    from src.tools.pattern_store import (
        add_user_item,
        clear_cache,
        list_pattern_files,
        list_user_overrides,
        load_description,
        load_items,
        load_values,
        remove_user_item,
        reset_user_overrides,
        set_user_value,
    )

    client_id = _client_id(ctx)

    if action == "list-files":
        files = list_pattern_files()
        return {
            "success": True,
            "files": [
                {"file": f, "description": load_description(f)} for f in files
            ],
            "count": len(files),
        }

    if action == "my-overrides":
        return {"success": True, "client_id": client_id, "overrides": list_user_overrides(client_id)}

    if not file:
        return {"success": False, "error": f"action='{action}' requires 'file'"}

    if file not in list_pattern_files():
        return {"success": False, "error": f"Unknown pattern file '{file}'. Use action='list-files' to see available files."}

    try:
        if action == "list":
            from src.tools.pattern_store import _core_path, _read_json
            core = _read_json(_core_path(file)) or {}
            if "items" in core:
                items = load_items(file, client_id)
                return {"success": True, "file": file, "action": action, "items": items, "count": len(items), "source": "merged"}
            values = load_values(file, client_id)
            return {"success": True, "file": file, "action": action, "values": values, "source": "merged"}

        if action == "add":
            if value is None or value == "":
                return {"success": False, "error": "action='add' requires 'value'"}
            items = add_user_item(file, value, client_id)
            return {"success": True, "file": file, "action": action, "value": value, "items": items, "source": "user"}

        if action == "remove":
            if value is None or value == "":
                return {"success": False, "error": "action='remove' requires 'value'"}
            items = remove_user_item(file, value, client_id)
            return {"success": True, "file": file, "action": action, "value": value, "items": items, "source": "user"}

        if action == "set-target":
            if doc_type is None or target is None:
                return {"success": False, "error": "action='set-target' requires 'doc_type' and 'target'"}
            try:
                target_f = float(target)
            except (TypeError, ValueError):
                return {"success": False, "error": f"'target' must be numeric, got {target!r}"}
            values = set_user_value(file, doc_type, target_f, client_id)
            return {"success": True, "file": file, "action": action, "doc_type": doc_type, "target": target_f, "values": values, "source": "user"}

        if action == "reset":
            reset_user_overrides(file, client_id)
            clear_cache()
            return {"success": True, "file": file, "action": action, "note": "Overrides cleared. Reverted to core defaults."}

        return {"success": False, "error": f"Invalid action '{action}'. Must be one of: list-files, list, add, remove, set-target, reset, my-overrides"}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"manage_patterns failed: {e}"}


# ===========================================================================
# 4. REVIEW SESSION TOOLS (MCP Apps — interactive accept/reject UI)
# ===========================================================================

@mcp.tool()
def start_review_session(
    items: List[dict],
    ctx: Context,
    name: Optional[str] = None,
) -> dict:
    """
    Create an interactive review session and return a ui:// resource for in-chat rendering.

    The host (Claude.ai) renders the resource as a sandboxed iframe where the user can
    accept or reject each item. Decisions are submitted back via apply_review_decisions.

    Each item must have:
        type    — vocabulary_flag | passage_candidate | term_candidate
        label   — short description (e.g. "leverage → reinforce")
        context — surrounding sentence for context
        payload — full params for the apply tool (passed through unchanged on accept)
        id      — auto-generated if absent

    Args:
        items: List of ReviewItem dicts
        name: Optional session name (auto-generated from timestamp if absent)

    Returns:
        session_id, name, item_count, _meta.ui.resourceUri
    """
    from src.tools.review import start_review_session as _start
    return _start(items=items, client_id=_client_id(ctx), name=name)


@mcp.tool()
def apply_review_decisions(
    session_id: str,
    decisions: List[dict],
    ctx: Context,
) -> dict:
    """
    Execute accepted items from a review session and persist all decisions.

    Called by the review panel UI via postMessage. Each decision has:
        item_id — the ReviewItem id
        action  — "accept" | "reject"

    Accepted passage_candidate items → manage_passage(action='add').
    Accepted term_candidate items    → manage_term(action='add').
    Accepted vocabulary_flag items   → acknowledged (no library write).
    Rejected items are recorded and skipped.

    Args:
        session_id: ID returned by start_review_session
        decisions:  List of {item_id, action} dicts

    Returns:
        accepted_count, rejected_count, per-item results
    """
    from src.tools.review import apply_review_decisions as _apply
    return _apply(
        session_id=session_id,
        decisions_raw=decisions,
        client_id=_client_id(ctx),
    )


@mcp.tool()
def list_review_sessions(
    ctx: Context,
    status: str = "open",
) -> dict:
    """
    List review sessions for the caller.

    Args:
        status: open | completed | all (default: open)

    Returns:
        List of sessions: id, name, status, item_count, decision_count, created_at
    """
    from src.tools.review import list_review_sessions_tool as _list
    return _list(client_id=_client_id(ctx), status=status)


@mcp.resource("ui://review-sessions/{session_id}")
def resource_review_session(session_id: str, ctx: Context) -> str:
    """HTML review panel for a session — rendered as iframe by MCP Apps hosts."""
    from src.tools.review import get_review_session_html as _html
    return _html(session_id=session_id, client_id=_client_id(ctx))


# ===========================================================================
# 5. MCP RESOURCES (demoted from list_* tools)
# ===========================================================================

@mcp.resource("writing-library://styles")
def resource_styles() -> dict:
    """Writing style labels with descriptions, grouped by category (14 labels, 4 categories)."""
    from src.tools.styles import list_styles as _list
    return _list()


@mcp.resource("writing-library://rubric-frameworks")
def resource_rubric_frameworks() -> dict:
    """Rubric frameworks that have at least one criterion stored, with per-framework criterion counts."""
    from src.tools.rubrics import list_rubric_frameworks as _list
    return _list()


@mcp.resource("writing-library://templates")
def resource_templates() -> dict:
    """All stored document templates by framework + doc_type, with section counts."""
    from src.tools.templates import list_templates as _list
    return _list()


# ===========================================================================
# Internal: Telegram notification for new contributions
# ===========================================================================

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
        f"Review with `manage_contributions(action='list')` → `manage_contributions(action='review', ...)`"
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
        except Exception:
            pass

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_send())
        else:
            loop.run_until_complete(_send())
    except Exception:
        pass
