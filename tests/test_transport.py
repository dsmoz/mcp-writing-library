"""Tests for dual-mode transport configuration and bearer auth.

Auth moved off the FastMCP ``token_verifier`` onto an ASGI
``BearerAuthMiddleware`` (src/server.py). These tests pin the current surface:
``_check_auth`` token validation and the middleware's 401-vs-pass-through gate.
The admin-flag resolution lives in test_admin_header.py.
"""
import asyncio

import pytest

from src import server as srv


def _drive_middleware(headers):
    """Run BearerAuthMiddleware over a fake ASGI request; capture whether the
    downstream app ran and the response status the client received."""
    captured: dict = {"app_ran": False}

    async def app(scope, receive, send):
        captured["app_ran"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "path": "/mcp", "headers": headers}
    asyncio.run(srv.BearerAuthMiddleware(app)(scope, receive, send))
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    captured["status"] = status
    return captured


def test_stdio_mode_builds_mcp_without_auth(monkeypatch):
    """With TRANSPORT=stdio and no API_TOKENS, FastMCP builds; auth is the
    ASGI middleware, never a FastMCP token_verifier (the old class is gone)."""
    monkeypatch.setenv("TRANSPORT", "stdio")
    monkeypatch.delenv("API_TOKENS", raising=False)
    import importlib
    import src.server as mod
    importlib.reload(mod)
    assert mod.mcp is not None
    # The legacy FastMCP token_verifier was removed in favour of BearerAuthMiddleware.
    assert not hasattr(mod, "BearerTokenVerifier")
    assert getattr(mod.mcp, "_token_verifier", None) is None


def test_check_auth_accepts_valid_token(monkeypatch):
    """_check_auth accepts a token present in API_TOKENS."""
    monkeypatch.setenv("API_TOKENS", "secret123,secret456")
    assert srv._check_auth("secret123") is True
    assert srv._check_auth("secret456") is True


def test_check_auth_rejects_invalid_token(monkeypatch):
    """_check_auth rejects a token absent from API_TOKENS."""
    monkeypatch.setenv("API_TOKENS", "secret123")
    assert srv._check_auth("wrong-token") is False


def test_check_auth_allows_all_when_no_tokens_set(monkeypatch):
    """With API_TOKENS unset/empty the server is open (single-tenant/local)."""
    monkeypatch.setenv("API_TOKENS", "")
    assert srv._check_auth("anything") is True


def test_middleware_rejects_invalid_token(monkeypatch):
    """When API_TOKENS is set, an unknown bearer token gets a 401 and the
    downstream app never runs."""
    monkeypatch.setenv("API_TOKENS", "secret123")
    monkeypatch.delenv("OAUTH_INTROSPECT_URL", raising=False)
    monkeypatch.delenv("OAUTH_ISSUER_URL", raising=False)
    monkeypatch.delenv("INTROSPECT_SECRET", raising=False)
    captured = _drive_middleware([(b"authorization", b"Bearer wrong-token")])
    assert captured["status"] == 401
    assert captured["app_ran"] is False


def test_middleware_accepts_valid_token(monkeypatch):
    """A bearer token in API_TOKENS passes the gate and the app runs."""
    monkeypatch.setenv("API_TOKENS", "secret123")
    monkeypatch.delenv("OAUTH_INTROSPECT_URL", raising=False)
    monkeypatch.delenv("OAUTH_ISSUER_URL", raising=False)
    monkeypatch.delenv("INTROSPECT_SECRET", raising=False)
    captured = _drive_middleware([(b"authorization", b"Bearer secret123")])
    assert captured["status"] == 200
    assert captured["app_ran"] is True
