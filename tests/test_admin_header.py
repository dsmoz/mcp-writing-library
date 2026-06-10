"""BearerAuthMiddleware admin-flag resolution.

Gateway-routed mode: the Authorization bearer is the shared upstream API key,
not the user's token, so introspecting it never yields the caller's admin flag.
The gateway resolves admin status and forwards X-Is-Admin (mirroring the
X-User-ID tenancy header). These tests pin that the middleware honours the
header and falls back safely when it is absent.
"""
import asyncio

import pytest

from src import server as srv


def _drive_middleware(headers):
    """Run BearerAuthMiddleware over a fake ASGI request, capturing the
    ContextVar values visible to the downstream app (they reset on exit)."""
    captured: dict = {}

    async def app(scope, receive, send):
        captured["is_admin"] = srv.current_is_admin.get()
        captured["client_id"] = srv.current_client_id.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "path": "/mcp", "headers": headers}
    asyncio.run(srv.BearerAuthMiddleware(app)(scope, receive, send))
    return captured, sent


@pytest.fixture(autouse=True)
def _no_auth_env(monkeypatch):
    # API_TOKENS unset → allow-all; introspect unset → no network, oauth admin False.
    monkeypatch.delenv("API_TOKENS", raising=False)
    monkeypatch.delenv("OAUTH_INTROSPECT_URL", raising=False)
    monkeypatch.delenv("OAUTH_ISSUER_URL", raising=False)
    monkeypatch.delenv("INTROSPECT_SECRET", raising=False)


def test_x_is_admin_true_grants_admin():
    headers = [
        (b"authorization", b"Bearer gateway-shared-key"),
        (b"x-user-id", b"usr_93f07c15894b4877"),
        (b"x-is-admin", b"true"),
    ]
    captured, _ = _drive_middleware(headers)
    assert captured["is_admin"] is True
    # Tenancy still resolves from X-User-ID, independent of the admin flag.
    assert captured["client_id"] == "usr_93f07c15894b4877"


def test_missing_header_defaults_non_admin():
    headers = [
        (b"authorization", b"Bearer gateway-shared-key"),
        (b"x-user-id", b"usr_93f07c15894b4877"),
    ]
    captured, _ = _drive_middleware(headers)
    assert captured["is_admin"] is False


def test_x_is_admin_false_is_non_admin():
    headers = [
        (b"authorization", b"Bearer gateway-shared-key"),
        (b"x-user-id", b"usr_someone"),
        (b"x-is-admin", b"false"),
    ]
    captured, _ = _drive_middleware(headers)
    assert captured["is_admin"] is False


def test_admin_flag_resets_after_request():
    """ContextVar must not leak across requests."""
    _drive_middleware([
        (b"authorization", b"Bearer k"),
        (b"x-user-id", b"usr_admin"),
        (b"x-is-admin", b"true"),
    ])
    # Outside any request, the default is non-admin.
    assert srv.current_is_admin.get() is False
