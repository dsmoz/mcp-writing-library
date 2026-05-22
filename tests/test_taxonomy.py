"""Tests for taxonomy introspection and zero-result hints."""
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import src.tools.taxonomy  # noqa: F401  — register submodule for mock.patch path resolution


def _fake_point(payload):
    return SimpleNamespace(payload=payload)


def _fake_qdrant_client(points):
    client = MagicMock()
    # Single scroll call returns the points and offset=None to terminate the loop.
    client.scroll.return_value = (points, None)
    return client


# ---------------------------------------------------------------------------
# get_distinct_values
# ---------------------------------------------------------------------------

def test_get_distinct_values_flattens_lists():
    points = [
        _fake_point({"domain": "srhr", "tags": ["hiv", "kp"]}),
        _fake_point({"domain": "governance", "tags": ["kp"]}),
        _fake_point({"domain": "srhr", "tags": []}),
    ]
    with patch("src.tools.taxonomy.get_qdrant_client", return_value=_fake_qdrant_client(points)):
        from src.tools.taxonomy import get_distinct_values
        out = get_distinct_values("coll", ["domain", "tags"])
    assert out["domain"] == ["governance", "srhr"]
    assert out["tags"] == ["hiv", "kp"]


def test_get_distinct_values_returns_empty_on_client_error():
    client = MagicMock()
    client.scroll.side_effect = RuntimeError("boom")
    with patch("src.tools.taxonomy.get_qdrant_client", return_value=client):
        from src.tools.taxonomy import get_distinct_values
        assert get_distinct_values("coll", ["domain"]) == {}


# ---------------------------------------------------------------------------
# build_zero_result_hint
# ---------------------------------------------------------------------------

def test_hint_flags_mismatched_filter_value():
    distinct = {"domain": ["srhr", "governance"], "doc_type": ["report"]}
    with patch("src.tools.taxonomy.get_distinct_values", return_value=distinct):
        from src.tools.taxonomy import build_zero_result_hint
        hint = build_zero_result_hint(
            "coll", {"domain": "health"}, ["domain", "doc_type"]
        )
    assert hint is not None
    assert "filter_mismatches" in hint
    assert hint["filter_mismatches"]["domain"]["requested"] == "health"
    assert hint["filter_mismatches"]["domain"]["available"] == ["srhr", "governance"]
    assert hint["available_values"] == distinct


def test_hint_message_when_filters_valid_but_no_combined_match():
    distinct = {"domain": ["srhr"], "doc_type": ["report"]}
    with patch("src.tools.taxonomy.get_distinct_values", return_value=distinct):
        from src.tools.taxonomy import build_zero_result_hint
        hint = build_zero_result_hint(
            "coll", {"domain": "srhr", "doc_type": "report"}, ["domain", "doc_type"]
        )
    assert hint is not None
    assert "filter_mismatches" not in hint
    assert "combination" in hint["message"]


def test_hint_empty_corpus():
    distinct = {"domain": [], "doc_type": []}
    with patch("src.tools.taxonomy.get_distinct_values", return_value=distinct):
        from src.tools.taxonomy import build_zero_result_hint
        hint = build_zero_result_hint("coll", {"domain": "srhr"}, ["domain", "doc_type"])
    assert hint["available_values"] == {}
    assert "empty" in hint["message"].lower()


def test_hint_returns_none_when_introspection_fails():
    with patch("src.tools.taxonomy.get_distinct_values", return_value={}):
        from src.tools.taxonomy import build_zero_result_hint
        assert build_zero_result_hint("coll", {"domain": "srhr"}, ["domain"]) is None


def test_hint_returns_none_when_no_filters_requested():
    from src.tools.taxonomy import build_zero_result_hint
    assert build_zero_result_hint("coll", {}, ["domain"]) is None


# ---------------------------------------------------------------------------
# search_passages — zero-result hint integration
# ---------------------------------------------------------------------------

def test_search_passages_emits_hint_on_zero_results():
    fake_hint = {"message": "No entries matched domain='health'.", "filter_mismatches": {}}
    with patch("src.tools.passages.semantic_search", return_value=[]), \
         patch("src.tools.taxonomy.build_zero_result_hint", return_value=fake_hint):
        from src.tools.passages import search_passages
        result = search_passages(query="hiv", domain="health")
    assert result["success"] is True
    assert result["total"] == 0
    assert result["hint"] == fake_hint


def test_search_passages_no_hint_when_no_filters():
    with patch("src.tools.passages.semantic_search", return_value=[]):
        from src.tools.passages import search_passages
        result = search_passages(query="hiv")
    assert "hint" not in result


def test_search_passages_no_hint_when_results_present():
    mock_results = [{
        "id": "1", "score": 0.9, "document_id": "d1",
        "title": "t", "text": "x",
        "metadata": {"doc_type": "report", "language": "en"},
    }]
    with patch("src.tools.passages.semantic_search", return_value=mock_results):
        from src.tools.passages import search_passages
        result = search_passages(query="hiv", domain="srhr")
    assert "hint" not in result


# ---------------------------------------------------------------------------
# search_terms — zero-result hint integration
# ---------------------------------------------------------------------------

def test_search_terms_emits_hint_on_zero_results():
    fake_hint = {"message": "No entries matched domain='health'."}
    with patch("src.tools.terms.semantic_search", return_value=[]), \
         patch("src.tools.taxonomy.build_zero_result_hint", return_value=fake_hint):
        from src.tools.terms import search_terms
        result = search_terms(query="kp", domain="health")
    assert result["success"] is True
    assert result["total"] == 0
    assert result["hint"] == fake_hint


def test_search_terms_no_hint_without_filters():
    with patch("src.tools.terms.semantic_search", return_value=[]):
        from src.tools.terms import search_terms
        result = search_terms(query="kp")
    assert "hint" not in result


# ---------------------------------------------------------------------------
# list_taxonomy tool
# ---------------------------------------------------------------------------

def test_list_taxonomy_returns_per_collection_facets():
    def fake_distinct(collection, fields, limit=2000):
        if "passages" in collection:
            return {"doc_type": ["report"], "domain": ["srhr"]}
        if "terms" in collection:
            return {"domain": ["general"], "language": ["en"]}
        return {"channel": ["linkedin"]}

    with patch("src.tools.taxonomy.get_distinct_values", side_effect=fake_distinct):
        from src.tools.taxonomy import list_taxonomy
        out = list_taxonomy(client_id="alice")
    assert out["success"] is True
    assert out["client_id"] == "alice"
    assert "alice_writing_passages" in out["passages"]["collection"]
    assert out["passages"]["values"]["doc_type"] == ["report"]
    assert out["terms"]["values"]["language"] == ["en"]
    assert out["style_profiles"]["values"]["channel"] == ["linkedin"]
