"""
Collection management for writing library Qdrant collections.

Multi-tenant architecture (Option B — collection prefix):

    Core collections (shared, read-only to users):
        writing_thesaurus
        writing_rubrics
        writing_templates

    Per-user collections (isolated by client_id prefix):
        {user_id}_writing_passages
        {user_id}_writing_terms
        {user_id}_writing_style_profiles

    In stdio mode (no auth), user_id defaults to "default", giving:
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

logger = structlog.get_logger(__name__)

VECTOR_SIZE = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))  # text-embedding-3-small (OpenAI)

# Sanitise client_id to a safe Qdrant collection name segment.
# Qdrant collection names allow [a-zA-Z0-9_-]; replace anything else with _.
import re
def _safe_user_id(user_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)


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


def get_user_collection_names(user_id: str = "default") -> dict:
    """Return per-user collection names prefixed with the user_id."""
    uid = _safe_user_id(user_id)
    return {
        "passages": f"{uid}_writing_passages",
        "terms": f"{uid}_writing_terms",
        "style_profiles": f"{uid}_writing_style_profiles",
    }


def get_collection_names(user_id: str = "default") -> dict:
    """Return all collection names — user-scoped + core — for a given user."""
    return {**get_user_collection_names(user_id), **get_core_collection_names()}


def setup_user_collections(user_id: str = "default") -> dict:
    """
    Ensure per-user Qdrant collections exist. Called lazily on first authenticated request.
    Core collections are NOT created here — seed scripts handle those.
    Returns dict with creation status for each per-user collection.
    """
    from kbase.vector.sync_indexing import ensure_collection

    names = get_user_collection_names(user_id)
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
            results[key] = {"collection": collection_name, "status": "error", "error": str(e)}
            logger.error("User collection setup failed", collection=collection_name, error=str(e))
            capture_tool_error(e, tool_name="setup_user_collections", collection=collection_name)

    return results


def setup_collections(user_id: str = "default") -> dict:
    """
    Ensure all Qdrant collections exist — both core and per-user.
    Returns dict with creation status for each collection.
    """
    from kbase.vector.sync_indexing import ensure_collection

    all_names = get_collection_names(user_id)
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
            results[key] = {"collection": collection_name, "status": "error", "error": str(e)}
            logger.error("Collection setup failed", collection=collection_name, error=str(e))
            capture_tool_error(e, tool_name="setup_collections", collection=collection_name)

    return results


def get_stats(user_id: str = "default") -> dict:
    """Return point counts for all collections (user + core)."""
    from kbase.vector.sync_search import get_collection_stats

    names = get_collection_names(user_id)
    stats = {}

    for key, collection_name in names.items():
        try:
            stats[key] = get_collection_stats(collection_name)
        except Exception as e:
            stats[key] = {"collection": collection_name, "error": str(e)}

    return stats
