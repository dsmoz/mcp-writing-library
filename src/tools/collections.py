"""
Collection management for writing library Qdrant collections.

Multi-tenant architecture (Option B — collection prefix):

    Core collections (shared, read-only to users):
        writing_thesaurus
        writing_rubrics
        writing_templates

    Per-user collections (isolated by client_id prefix):
        {client_id}_writing_passages
        {client_id}_writing_terms
        {client_id}_writing_style_profiles

    In stdio mode (no auth), client_id defaults to "default", giving:
        default_writing_passages
        default_writing_terms
        default_writing_style_profiles

    Collections are created lazily on first use via setup_user_collections().
"""
import os
import sys
from pathlib import Path

import structlog

from src.sentry import capture_tool_error
from src.tools.qdrant_errors import handle_qdrant_error

logger = structlog.get_logger(__name__)

VECTOR_SIZE = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))  # text-embedding-3-small (OpenAI)

# Sanitise client_id to a safe Qdrant collection name segment.
# Qdrant collection names allow [a-zA-Z0-9_-]; replace anything else with _.
import re
def _safe_client_id(client_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", client_id)


# Keyword payload indexes required by the tools that filter on them. Qdrant
# returns HTTP 400 ("Index required but not found") when a filter hits an
# unindexed field, so we create them eagerly on first use per client_id.
_PAYLOAD_KEYWORD_INDEXES: dict[str, tuple[str, ...]] = {
    "passages": ("entry_type", "doc_type", "language", "domain", "rubric_section"),
    "style_profiles": ("name", "channel"),
}


def get_core_collection_names() -> dict:
    """Return names for shared/core collections (not user-scoped).

    writing_terms_shared — published terms contributed by users (read by all search_terms calls)
    writing_contributions — moderation queue (pending/published/rejected contributions)
    """
    return {
        "rubrics": os.getenv("COLLECTION_RUBRICS", "writing_rubrics"),
        "templates": os.getenv("COLLECTION_TEMPLATES", "writing_templates"),
        "thesaurus": os.getenv("COLLECTION_THESAURUS", "writing_thesaurus"),
        "terms_shared": os.getenv("COLLECTION_TERMS_SHARED", "writing_terms_shared"),
        "contributions": os.getenv("COLLECTION_CONTRIBUTIONS", "writing_contributions"),
    }


def get_user_collection_names(client_id: str = "default") -> dict:
    """Return per-user collection names prefixed with the client_id."""
    uid = _safe_client_id(client_id)
    return {
        "passages": f"{uid}_writing_passages",
        "terms": f"{uid}_writing_terms",
        "style_profiles": f"{uid}_writing_style_profiles",
    }


def get_collection_names(client_id: str = "default") -> dict:
    """Return all collection names — user-scoped + core — for a given user."""
    return {**get_user_collection_names(client_id), **get_core_collection_names()}


def setup_user_collections(client_id: str = "default") -> dict:
    """
    Ensure per-user Qdrant collections exist. Called lazily on first authenticated request.
    Core collections are NOT created here — seed scripts handle those.
    Returns dict with creation status for each per-user collection.
    """
    from kbase.vector.sync_indexing import ensure_collection

    names = get_user_collection_names(client_id)
    results = {}

    for key, collection_name in names.items():
        try:
            created = ensure_collection(
                collection_name=collection_name,
                vector_size=VECTOR_SIZE,
                hybrid=True,
            )
            status = "created" if created else "already_exists"
            results[key] = {"collection": collection_name, "status": status}
            logger.info("User collection ready", collection=collection_name, status=status)
        except Exception as e:
            qdrant_result = handle_qdrant_error(e, tool_name="setup_user_collections", collection=collection_name)
            if qdrant_result is not None:
                results[key] = {"collection": collection_name, "status": "error", "error": qdrant_result["error"], "error_type": qdrant_result.get("error_type")}
            else:
                results[key] = {"collection": collection_name, "status": "error", "error": str(e)}
                logger.error("User collection setup failed", collection=collection_name, error=str(e))
                capture_tool_error(e, tool_name="setup_user_collections", collection=collection_name)

    return results


def setup_collections(client_id: str = "default") -> dict:
    """
    Ensure all Qdrant collections exist — both core and per-user.
    Returns dict with creation status for each collection.
    """
    from kbase.vector.sync_indexing import ensure_collection

    all_names = get_collection_names(client_id)
    results = {}

    for key, collection_name in all_names.items():
        try:
            created = ensure_collection(
                collection_name=collection_name,
                vector_size=VECTOR_SIZE,
                hybrid=True,
            )
            status = "created" if created else "already_exists"
            results[key] = {"collection": collection_name, "status": status}
            logger.info("Collection ready", collection=collection_name, status=status)
        except Exception as e:
            qdrant_result = handle_qdrant_error(e, tool_name="setup_collections", collection=collection_name)
            if qdrant_result is not None:
                results[key] = {"collection": collection_name, "status": "error", "error": qdrant_result["error"], "error_type": qdrant_result.get("error_type")}
            else:
                results[key] = {"collection": collection_name, "status": "error", "error": str(e)}
                logger.error("Collection setup failed", collection=collection_name, error=str(e))
                capture_tool_error(e, tool_name="setup_collections", collection=collection_name)

    return results


_initialized_clients: set = set()


def _ensure_keyword_index(qc, collection: str, field: str) -> None:
    """Create a keyword payload index on *field*, tolerating 409 already-exists."""
    from qdrant_client.models import PayloadSchemaType

    try:
        qc.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Created keyword payload index", collection=collection, field=field)
    except Exception as e:
        msg = str(e).lower()
        if "already exists" in msg or "409" in msg:
            return
        logger.warning(
            "Could not create payload index",
            collection=collection,
            field=field,
            error=str(e),
        )


def ensure_user_collections_once(client_id: str = "default") -> None:
    """Create per-user Qdrant collections on first call per process per client_id.

    Idempotent: subsequent calls for the same client_id are no-ops.
    Also ensures the keyword payload indexes listed in ``_PAYLOAD_KEYWORD_INDEXES``
    exist so that filter-based queries do not fail with HTTP 400.
    """
    if client_id in _initialized_clients:
        return

    setup_user_collections(client_id)

    # Create keyword payload indexes for every field that tools filter on so
    # filter_conditions queries don't raise HTTP 400.
    try:
        from kbase.vector.sync_client import get_qdrant_client

        qc = get_qdrant_client()
        user_collections = get_user_collection_names(client_id)
        for collection_key, fields in _PAYLOAD_KEYWORD_INDEXES.items():
            collection = user_collections.get(collection_key)
            if not collection:
                continue
            for field in fields:
                _ensure_keyword_index(qc, collection, field)
    except Exception as e:
        logger.warning("Could not ensure keyword payload indexes", error=str(e))

    _initialized_clients.add(client_id)


def get_stats(client_id: str = "default") -> dict:
    """Return point counts for all collections (user + core)."""
    from kbase.vector.sync_search import get_collection_stats

    names = get_collection_names(client_id)
    stats = {}

    for key, collection_name in names.items():
        try:
            stats[key] = get_collection_stats(collection_name)
        except Exception as e:
            stats[key] = {"collection": collection_name, "error": str(e)}

    return stats
