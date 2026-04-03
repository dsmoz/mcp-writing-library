"""
Synchronous Qdrant Client Singleton

This module provides a singleton QdrantClient instance for sync operations.
Based on the pattern from mcp-zotero-qdrant.

Performance Impact:
- Before: 10-50ms overhead per operation (new connection)
- After: 0ms overhead (reused connection)

Usage:
    from kbase.vector.sync_client import get_qdrant_client
    client = get_qdrant_client()
"""

import os
import sys
import threading
from typing import Optional

from qdrant_client import QdrantClient

import structlog

logger = structlog.get_logger(__name__)

# Thread lock for singleton initialization
_lock = threading.Lock()
_client: Optional[QdrantClient] = None


def get_qdrant_client(force_new: bool = False) -> QdrantClient:
    """
    Get the singleton QdrantClient instance.

    This function ensures only one QdrantClient is created and reused
    across all operations, eliminating connection overhead.

    Args:
        force_new: If True, creates a new client even if one exists.
                  Useful for reconnection after errors.

    Returns:
        QdrantClient: The singleton Qdrant client instance.

    Thread Safety:
        Uses a lock to ensure thread-safe singleton initialization.
    """
    global _client

    # Fast path: return existing client
    if _client is not None and not force_new:
        return _client

    # Slow path: create new client with lock
    with _lock:
        # Double-check after acquiring lock
        if _client is not None and not force_new:
            return _client

        # Get connection parameters from environment.
        # Supports both QDRANT_URL (preferred) and QDRANT_SERVER/QDRANT_PORT
        # (legacy, used by mcp-qualitative-research).
        url = os.getenv("QDRANT_URL")
        if not url:
            server = os.getenv("QDRANT_SERVER", "localhost")
            port = os.getenv("QDRANT_PORT", "6333")
            url = server if server.startswith("http") else f"http://{server}:{port}"
        api_key = os.getenv("QDRANT_API_KEY")

        # Create client with timeout settings
        _client = QdrantClient(
            url=url,
            api_key=api_key if api_key else None,
            timeout=30,  # Connection timeout in seconds
        )

        logger.info("QdrantClient singleton created", url=url)

        return _client


def reset_client():
    """
    Reset the singleton client.

    Use this to force a new connection, for example after a connection error.
    """
    global _client
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass  # Ignore close errors
            _client = None
            logger.info("QdrantClient singleton reset")


def ensure_payload_indexes(
    collection_name: str,
    keyword_fields: list = (),
    text_fields: list = (),
) -> None:
    """
    Create payload indexes on a Qdrant collection (idempotent).

    Checks existing schema before creating each index, so it is safe to call
    on every startup. Useful for filter-heavy collections where per-field
    indexes improve query performance.

    Args:
        collection_name: Name of the Qdrant collection
        keyword_fields:  Fields to index as KEYWORD (exact match, array contains)
        text_fields:     Fields to index as TEXT (full-text search via MatchText)

    Example:
        ensure_payload_indexes(
            "cerebellum",
            keyword_fields=["document_id", "collection_ids", "source_type", "tags"],
            text_fields=["chunk_text", "title"],
        )
    """
    from qdrant_client import models as qdrant_models

    client = get_qdrant_client()
    existing_info = client.get_collection(collection_name)
    existing_fields = set(existing_info.payload_schema.keys())

    for field in keyword_fields:
        if field not in existing_fields:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            logger.info("Created KEYWORD payload index", field=field, collection=collection_name)

    for field in text_fields:
        if field not in existing_fields:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=qdrant_models.PayloadSchemaType.TEXT,
            )
            logger.info("Created TEXT payload index", field=field, collection=collection_name)


def check_connection() -> bool:
    """
    Check if the Qdrant connection is healthy.

    Returns:
        bool: True if connection is healthy, False otherwise.
    """
    try:
        client = get_qdrant_client()
        # Simple health check - list collections
        client.get_collections()
        return True
    except Exception as e:
        logger.error("Qdrant connection check failed", error=str(e))
        return False
