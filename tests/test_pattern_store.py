"""Tests for two-layer pattern loader (core + per-user overrides)."""
from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture
def ps(tmp_path, monkeypatch):
    """Reload pattern_store with a temporary PATTERNS_USER_DIR, fresh cache."""
    monkeypatch.setenv("PATTERNS_USER_DIR", str(tmp_path / "users"))
    import src.tools.pattern_store as _ps
    importlib.reload(_ps)
    yield _ps
    _ps.clear_cache()


def test_list_pattern_files_includes_core_and_new(ps):
    files = ps.list_pattern_files()
    assert "connectors_en" in files
    assert "hedging_words_en" in files
    assert "juridiques_terms" in files
    assert "config" in files


def test_core_only_matches_json(ps):
    items = ps.load_items("connectors_en", "default")
    assert "furthermore" in items
    assert "additionally" in items


def test_per_user_add_is_isolated(ps):
    ps.add_user_item("connectors_en", "zzfoo", "alice")
    assert "zzfoo" in ps.load_items("connectors_en", "alice")
    assert "zzfoo" not in ps.load_items("connectors_en", "bob")


def test_per_user_remove_hides_core_term_for_that_user(ps):
    ps.remove_user_item("connectors_en", "furthermore", "alice")
    assert "furthermore" not in ps.load_items("connectors_en", "alice")
    assert "furthermore" in ps.load_items("connectors_en", "bob")


def test_remove_of_user_added_term_drops_it(ps):
    ps.add_user_item("connectors_en", "mytest", "alice")
    assert "mytest" in ps.load_items("connectors_en", "alice")
    ps.remove_user_item("connectors_en", "mytest", "alice")
    assert "mytest" not in ps.load_items("connectors_en", "alice")


def test_value_overrides_are_per_user(ps):
    ps.set_user_value("para_limits", "concept-note", 9, "alice")
    assert ps.load_values("para_limits", "alice")["concept-note"] == 9
    assert ps.load_values("para_limits", "bob")["concept-note"] == 4


def test_reset_clears_user_overrides(ps):
    ps.add_user_item("connectors_en", "zzfoo", "alice")
    ps.reset_user_overrides("connectors_en", "alice")
    ps.clear_cache()
    assert "zzfoo" not in ps.load_items("connectors_en", "alice")
    assert "furthermore" in ps.load_items("connectors_en", "alice")


def test_cache_invalidates_on_user_file_change(ps):
    first = ps.load_items("connectors_en", "alice")
    ps.add_user_item("connectors_en", "zzcache", "alice")
    second = ps.load_items("connectors_en", "alice")
    assert "zzcache" not in first
    assert "zzcache" in second


def test_add_rejects_values_style_file(ps):
    with pytest.raises(ValueError):
        ps.add_user_item("para_limits", "foo", "alice")


def test_set_value_rejects_items_style_file(ps):
    with pytest.raises(ValueError):
        ps.set_user_value("connectors_en", "key", 1.0, "alice")


def test_safe_client_id_prevents_path_injection(ps, tmp_path):
    ps.add_user_item("connectors_en", "foo", "../evil")
    # The override must land under the sanitised dir, not escape the user dir.
    udir = tmp_path / "users"
    assert udir.exists()
    sanitised_dirs = [p.name for p in udir.iterdir() if p.is_dir()]
    assert all(".." not in d and "/" not in d for d in sanitised_dirs)


def test_my_overrides_lists_per_client(ps):
    ps.add_user_item("connectors_en", "zzfoo", "alice")
    ps.set_user_value("para_limits", "tor", 10, "alice")
    out = ps.list_user_overrides("alice")
    assert "connectors_en" in out
    assert "para_limits" in out
    assert ps.list_user_overrides("bob") == {}
