"""
Shared Qdrant error handling utilities for mcp-writing-library tools.

Provides a consistent pattern for catching and categorising Qdrant
``UnexpectedResponse`` errors (404 collection/point not found, 400
malformed request, 5xx server errors) so that every tool returns a clear
``{"success": False, "error": ...}`` dict instead of crashing with an
unhandled exception.

Usage in tool modules::

    from src.tools.qdrant_errors import handle_qdrant_error

    try:
        # ... Qdrant operation ...
    except Exception as e:
        result = handle_qdrant_error(e, tool_name="search_passages", collection=col)
        if result is not None:
            return result
        # Fall through to generic handling
        ...
"""

import structlog

from src.sentry import capture_tool_error

logger = structlog.get_logger(__name__)

# Import the Qdrant exception type; guard for environments where the
# client is not installed.
try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    UnexpectedResponse = None  # type: ignore


def handle_qdrant_error(
    exc: Exception,
    tool_name: str,
    collection: str = "",
    **sentry_context,
) -> dict | None:
    """Inspect *exc* and return a structured error dict if it is a known Qdrant error.

    Returns ``None`` when *exc* is not a Qdrant ``UnexpectedResponse``,
    signalling the caller to fall through to its generic ``except Exception``
    handler.

    Args:
        exc: The caught exception.
        tool_name: MCP tool name for Sentry tagging (e.g. ``"search_passages"``).
        collection: Qdrant collection name involved (for the error message).
        **sentry_context: Extra key/value pairs forwarded to ``capture_tool_error``.

    Returns:
        A ``{"success": False, "error": ..., "error_type": ...}`` dict for
        recognised Qdrant errors, or ``None`` if *exc* is not an
        ``UnexpectedResponse``.

    Example:
        >>> from src.tools.qdrant_errors import handle_qdrant_error
        >>> result = handle_qdrant_error(exc, tool_name="add_passage", collection="default_writing_passages")
        >>> if result is not None:
        ...     return result

    Raises:
        Never raises; always returns a dict or None.
    """
    if UnexpectedResponse is None or not isinstance(exc, UnexpectedResponse):
        return None

    status = getattr(exc, "status_code", None)
    content = getattr(exc, "content", b"")
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    col_hint = f" (collection: {collection})" if collection else ""

    if status == 404:
        # Collection or point not found
        logger.warning(
            "Qdrant 404: resource not found",
            tool=tool_name,
            collection=collection,
            detail=content[:300],
        )
        capture_tool_error(
            exc,
            tool_name=tool_name,
            error_type="qdrant_not_found",
            collection=collection,
            **sentry_context,
        )
        return {
            "success": False,
            "error": (
                f"Qdrant resource not found{col_hint}. "
                f"The collection may not exist yet — run setup_collections() to create it. "
                f"Detail: {content[:200]}"
            ),
            "error_type": "qdrant_not_found",
        }

    if status is not None and 400 <= status < 500:
        # Client error (400 bad request, 409 conflict, etc.)
        logger.warning(
            "Qdrant client error",
            tool=tool_name,
            status=status,
            collection=collection,
            detail=content[:300],
        )
        capture_tool_error(
            exc,
            tool_name=tool_name,
            error_type="qdrant_client_error",
            status_code=status,
            collection=collection,
            **sentry_context,
        )
        return {
            "success": False,
            "error": (
                f"Qdrant rejected the request (HTTP {status}){col_hint}. "
                f"This usually indicates a malformed payload or invalid filter. "
                f"Detail: {content[:200]}"
            ),
            "error_type": "qdrant_client_error",
        }

    if status is not None and status >= 500:
        # Server-side error
        logger.error(
            "Qdrant server error",
            tool=tool_name,
            status=status,
            collection=collection,
            detail=content[:300],
        )
        capture_tool_error(
            exc,
            tool_name=tool_name,
            error_type="qdrant_server_error",
            status_code=status,
            collection=collection,
            **sentry_context,
        )
        return {
            "success": False,
            "error": (
                f"Qdrant server error (HTTP {status}){col_hint}. "
                f"The Qdrant instance may be overloaded or unavailable. "
                f"Detail: {content[:200]}"
            ),
            "error_type": "qdrant_server_error",
        }

    # UnexpectedResponse with an unknown/missing status code — still handle it
    logger.error(
        "Qdrant unexpected response",
        tool=tool_name,
        status=status,
        collection=collection,
        detail=content[:300],
    )
    capture_tool_error(
        exc,
        tool_name=tool_name,
        error_type="qdrant_unexpected",
        status_code=status,
        collection=collection,
        **sentry_context,
    )
    return {
        "success": False,
        "error": f"Unexpected Qdrant response{col_hint}: {content[:200]}",
        "error_type": "qdrant_unexpected",
    }
