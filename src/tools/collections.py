"""
Collection management for writing library Qdrant collections.
"""
import os
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

VECTOR_SIZE = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))  # text-embedding-3-small (OpenAI)


def get_collection_names() -> dict:
    """Return configured collection names from environment."""
    return {
        "passages": os.getenv("COLLECTION_PASSAGES", "writing_passages"),
        "terms": os.getenv("COLLECTION_TERMS", "writing_terms"),
        "style_profiles": os.getenv("COLLECTION_STYLE_PROFILES", "writing_style_profiles"),
        "rubrics": os.getenv("COLLECTION_RUBRICS", "writing_rubrics"),
        "templates": os.getenv("COLLECTION_TEMPLATES", "writing_templates"),
        "thesaurus": os.getenv("COLLECTION_THESAURUS", "writing_thesaurus"),
    }


def setup_collections() -> dict:
    """
    Ensure all Qdrant collections exist with hybrid vector config.
    Returns dict with creation status for each collection.
    """
    from kbase.vector.sync_indexing import ensure_collection

    names = get_collection_names()
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
            logger.info("Collection ready", collection=collection_name, status=status)
        except Exception as e:
            results[key] = {"collection": collection_name, "status": "error", "error": str(e)}
            logger.error("Collection setup failed", collection=collection_name, error=str(e))

    return results


def get_stats() -> dict:
    """Return point counts for both collections."""
    from kbase.vector.sync_search import get_collection_stats

    names = get_collection_names()
    stats = {}

    for key, collection_name in names.items():
        try:
            stats[key] = get_collection_stats(collection_name)
        except Exception as e:
            stats[key] = {"collection": collection_name, "error": str(e)}

    return stats
