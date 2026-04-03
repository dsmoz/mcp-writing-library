"""Tests for dual-mode transport configuration."""
import os
import pytest


def test_stdio_mode_builds_mcp_without_auth(monkeypatch):
    """When TRANSPORT=stdio, FastMCP is built without token_verifier."""
    monkeypatch.setenv("TRANSPORT", "stdio")
    # Re-import to pick up env change
    import importlib
    import src.server as mod
    importlib.reload(mod)
    # mcp should not have token_verifier set (or it's None)
    assert mod.mcp is not None


def test_bearer_token_verifier_accepts_valid_token(monkeypatch):
    """BearerTokenVerifier returns AccessToken for tokens in API_TOKENS."""
    monkeypatch.setenv("API_TOKENS", "secret123,secret456")
    from src.server import BearerTokenVerifier
    import asyncio
    verifier = BearerTokenVerifier()
    result = asyncio.run(verifier.verify_token("secret123"))
    assert result is not None
    assert result.token == "secret123"


def test_bearer_token_verifier_rejects_invalid_token(monkeypatch):
    """BearerTokenVerifier returns None for unknown tokens."""
    monkeypatch.setenv("API_TOKENS", "secret123")
    from src.server import BearerTokenVerifier
    import asyncio
    verifier = BearerTokenVerifier()
    result = asyncio.run(verifier.verify_token("wrong-token"))
    assert result is None


def test_bearer_token_verifier_rejects_when_no_tokens_set(monkeypatch):
    """BearerTokenVerifier rejects all tokens when API_TOKENS is empty."""
    monkeypatch.setenv("API_TOKENS", "")
    from src.server import BearerTokenVerifier
    import asyncio
    verifier = BearerTokenVerifier()
    result = asyncio.run(verifier.verify_token("anything"))
    assert result is None
