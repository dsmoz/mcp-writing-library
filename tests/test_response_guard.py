"""Tests for ResponseGuardMiddleware ASGI middleware."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_response_guard_allows_single_response():
    """Single response (start + body) passes through unmodified."""
    from src.server import ResponseGuardMiddleware

    messages_received = []

    async def mock_send(message):
        messages_received.append(message)

    async def mock_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"OK"})

    middleware = ResponseGuardMiddleware(mock_app)
    scope = {"type": "http", "path": "/test"}
    receive = AsyncMock()

    await middleware(scope, receive, mock_send)

    assert len(messages_received) == 2
    assert messages_received[0]["type"] == "http.response.start"
    assert messages_received[1]["type"] == "http.response.body"


@pytest.mark.asyncio
async def test_response_guard_drops_duplicate_start():
    """Duplicate http.response.start is dropped, body is suppressed."""
    from src.server import ResponseGuardMiddleware

    messages_received = []

    async def mock_send(message):
        messages_received.append(message)

    async def mock_app(scope, receive, send):
        # Send initial response
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"OK"})
        # Send duplicate (as SDK exception handler would)
        await send({"type": "http.response.start", "status": 500, "headers": []})
        await send({"type": "http.response.body", "body": b"Error"})

    middleware = ResponseGuardMiddleware(mock_app)
    scope = {"type": "http", "path": "/test"}
    receive = AsyncMock()

    await middleware(scope, receive, mock_send)

    # Should only have the first response (start + body)
    # The duplicate start should be dropped, and its body should be suppressed
    assert len(messages_received) == 2
    assert messages_received[0]["type"] == "http.response.start"
    assert messages_received[0]["status"] == 200
    assert messages_received[1]["type"] == "http.response.body"
    assert messages_received[1]["body"] == b"OK"


@pytest.mark.asyncio
async def test_response_guard_per_request_isolation():
    """Two sequential requests don't interfere (per-request state isolation)."""
    from src.server import ResponseGuardMiddleware

    all_messages = []

    async def mock_send(message):
        all_messages.append(message)

    async def mock_app_1(scope, receive, send):
        # First request: normal response
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"Request1"})

    async def mock_app_2(scope, receive, send):
        # Second request: normal response
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"Request2"})

    middleware1 = ResponseGuardMiddleware(mock_app_1)
    middleware2 = ResponseGuardMiddleware(mock_app_2)

    scope = {"type": "http", "path": "/test"}
    receive = AsyncMock()

    # First request
    await middleware1(scope, receive, mock_send)
    first_request_count = len(all_messages)

    # Second request
    await middleware2(scope, receive, mock_send)

    # Should have 4 messages total (2 per request)
    assert len(all_messages) == 4
    assert all_messages[0]["type"] == "http.response.start"
    assert all_messages[1]["type"] == "http.response.body"
    assert all_messages[1]["body"] == b"Request1"
    assert all_messages[2]["type"] == "http.response.start"
    assert all_messages[3]["type"] == "http.response.body"
    assert all_messages[3]["body"] == b"Request2"


@pytest.mark.asyncio
async def test_response_guard_passes_through_non_http():
    """Non-HTTP requests pass through unmodified."""
    from src.server import ResponseGuardMiddleware

    app_called = False

    async def mock_app(scope, receive, send):
        nonlocal app_called
        app_called = True

    middleware = ResponseGuardMiddleware(mock_app)
    scope = {"type": "websocket"}
    receive = AsyncMock()
    send = AsyncMock()

    await middleware(scope, receive, send)

    assert app_called
