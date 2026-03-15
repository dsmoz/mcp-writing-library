"""Tests for collection management."""
import pytest
from unittest.mock import patch, MagicMock


def test_get_collection_names_returns_env_values(monkeypatch):
    monkeypatch.setenv("COLLECTION_PASSAGES", "writing_passages")
    monkeypatch.setenv("COLLECTION_TERMS", "writing_terms")
    from src.tools.collections import get_collection_names
    names = get_collection_names()
    assert names["passages"] == "writing_passages"
    assert names["terms"] == "writing_terms"


def test_get_collection_names_uses_defaults(monkeypatch):
    monkeypatch.delenv("COLLECTION_PASSAGES", raising=False)
    monkeypatch.delenv("COLLECTION_TERMS", raising=False)
    from src.tools.collections import get_collection_names
    names = get_collection_names()
    assert names["passages"] == "writing_passages"
    assert names["terms"] == "writing_terms"
