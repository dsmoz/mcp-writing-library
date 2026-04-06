"""
Tests for the shared Qdrant error handling utility.

Covers:
    - 404 (collection/point not found) → structured error with setup_collections hint
    - 400 (malformed request) → structured error with payload hint
    - 5xx (server error) → structured error with availability hint
    - Non-Qdrant exceptions → returns None (passthrough)
    - UnexpectedResponse unavailable (import guard) → returns None
"""

import pytest
from unittest.mock import patch, MagicMock

from src.tools.qdrant_errors import handle_qdrant_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unexpected_response(status_code: int, content: bytes = b"test error detail"):
    """Create a mock UnexpectedResponse with the expected attributes."""
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
        import httpx
        return UnexpectedResponse(
            status_code=status_code,
            reason_phrase="Error",
            content=content,
            headers=httpx.Headers(),
        )
    except ImportError:
        pytest.skip("qdrant_client not installed")


# ---------------------------------------------------------------------------
# 404 — collection or point not found
# ---------------------------------------------------------------------------

class TestQdrant404:
    def test_returns_structured_error_for_404(self):
        exc = _make_unexpected_response(404, b"Not found - Collection `missing_col` doesn't exist!")
        result = handle_qdrant_error(exc, tool_name="search_passages", collection="missing_col")
        assert result is not None
        assert result["success"] is False
        assert result["error_type"] == "qdrant_not_found"
        assert "setup_collections()" in result["error"]
        assert "missing_col" in result["error"]

    def test_includes_collection_in_message(self):
        exc = _make_unexpected_response(404)
        result = handle_qdrant_error(exc, tool_name="add_passage", collection="default_writing_passages")
        assert "default_writing_passages" in result["error"]

    def test_reports_to_sentry(self):
        exc = _make_unexpected_response(404)
        with patch("src.tools.qdrant_errors.capture_tool_error") as mock_sentry:
            handle_qdrant_error(exc, tool_name="search_passages", collection="test_col")
            mock_sentry.assert_called_once()
            call_kwargs = mock_sentry.call_args
            assert call_kwargs[1]["error_type"] == "qdrant_not_found"


# ---------------------------------------------------------------------------
# 400 — malformed request
# ---------------------------------------------------------------------------

class TestQdrant400:
    def test_returns_structured_error_for_400(self):
        exc = _make_unexpected_response(400, b'{"status":{"error":"Wrong input: wrong field name"}}')
        result = handle_qdrant_error(exc, tool_name="add_passage", collection="test_col")
        assert result is not None
        assert result["success"] is False
        assert result["error_type"] == "qdrant_client_error"
        assert "400" in result["error"]
        assert "malformed" in result["error"].lower()

    def test_other_4xx_handled(self):
        exc = _make_unexpected_response(409, b"conflict")
        result = handle_qdrant_error(exc, tool_name="add_passage", collection="test_col")
        assert result is not None
        assert result["error_type"] == "qdrant_client_error"
        assert "409" in result["error"]


# ---------------------------------------------------------------------------
# 5xx — server errors
# ---------------------------------------------------------------------------

class TestQdrant5xx:
    def test_returns_structured_error_for_500(self):
        exc = _make_unexpected_response(500, b"Internal server error")
        result = handle_qdrant_error(exc, tool_name="search_terms", collection="test_col")
        assert result is not None
        assert result["success"] is False
        assert result["error_type"] == "qdrant_server_error"
        assert "500" in result["error"]
        assert "overloaded" in result["error"].lower() or "unavailable" in result["error"].lower()


# ---------------------------------------------------------------------------
# Non-Qdrant exceptions — passthrough
# ---------------------------------------------------------------------------

class TestPassthrough:
    def test_returns_none_for_regular_exception(self):
        result = handle_qdrant_error(ValueError("bad value"), tool_name="test_tool")
        assert result is None

    def test_returns_none_for_runtime_error(self):
        result = handle_qdrant_error(RuntimeError("boom"), tool_name="test_tool")
        assert result is None

    def test_returns_none_for_connection_error(self):
        result = handle_qdrant_error(ConnectionError("refused"), tool_name="test_tool")
        assert result is None


# ---------------------------------------------------------------------------
# Import guard — UnexpectedResponse not available
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_returns_none_when_qdrant_not_installed(self):
        with patch("src.tools.qdrant_errors.UnexpectedResponse", None):
            result = handle_qdrant_error(Exception("anything"), tool_name="test_tool")
            assert result is None
