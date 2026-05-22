"""Tests for filter aliasing in search_passages and search_terms."""
from unittest.mock import patch, MagicMock

import src.tools.aliases  # noqa: F401
import src.tools.passages  # noqa: F401
import src.tools.terms  # noqa: F401


# ---------------------------------------------------------------------------
# expand()
# ---------------------------------------------------------------------------

def test_expand_returns_original_value_first():
    from src.tools.aliases import expand
    assert expand("domain", "health")[0] == "health"
    assert "srhr" in expand("domain", "health")


def test_expand_unknown_value_returns_singleton():
    from src.tools.aliases import expand
    assert expand("domain", "srhr") == ["srhr"]


def test_expand_unknown_field_returns_singleton():
    from src.tools.aliases import expand
    assert expand("nonsense", "value") == ["value"]


def test_expand_none_returns_empty():
    from src.tools.aliases import expand
    assert expand("domain", None) == []


def test_doc_type_annual_report_expands():
    from src.tools.aliases import expand
    out = expand("doc_type", "annual-report")
    assert out[0] == "annual-report"
    assert "report" in out
    assert "monitoring-report" in out


# ---------------------------------------------------------------------------
# search_passages — alias fan-out
# ---------------------------------------------------------------------------

def test_search_passages_health_fans_out_to_srhr():
    """When caller asks domain='health', search runs once per alias and merges."""
    calls = []

    def fake_search(collection_name, query, limit, filter_conditions):
        calls.append(filter_conditions)
        domain = (filter_conditions or {}).get("domain")
        if domain == "srhr":
            return [{
                "id": "p1", "document_id": "p1", "score": 0.9,
                "title": "HIV passage", "text": "Key populations...",
                "metadata": {"domain": "srhr", "doc_type": "report", "language": "en"},
            }]
        return []

    with patch("src.tools.passages.semantic_search", side_effect=fake_search):
        from src.tools.passages import search_passages
        result = search_passages(query="hiv key populations", domain="health")

    filters_seen = [c.get("domain") for c in calls if c]
    assert "health" in filters_seen
    assert "srhr" in filters_seen
    assert result["total"] == 1
    assert result["results"][0]["domain"] == "srhr"


def test_search_passages_annual_report_fans_out():
    calls = []

    def fake_search(collection_name, query, limit, filter_conditions):
        calls.append(filter_conditions)
        dt = (filter_conditions or {}).get("doc_type")
        if dt == "report":
            return [{
                "id": "r1", "document_id": "r1", "score": 0.8,
                "title": "Report", "text": "...",
                "metadata": {"doc_type": "report", "language": "en"},
            }]
        return []

    with patch("src.tools.passages.semantic_search", side_effect=fake_search):
        from src.tools.passages import search_passages
        result = search_passages(query="annual report", doc_type="annual-report")

    doc_types_queried = [c.get("doc_type") for c in calls if c]
    assert {"annual-report", "report", "monitoring-report"} <= set(doc_types_queried)
    assert result["total"] == 1


def test_search_passages_deduplicates_across_aliases_keeps_max_score():
    """Same document returned under multiple aliases keeps highest score."""
    def fake_search(collection_name, query, limit, filter_conditions):
        domain = (filter_conditions or {}).get("domain")
        score = 0.7 if domain == "health" else 0.95
        return [{
            "id": "p1", "document_id": "p1", "score": score,
            "title": "P", "text": "x",
            "metadata": {"domain": "srhr", "doc_type": "report"},
        }]

    with patch("src.tools.passages.semantic_search", side_effect=fake_search):
        from src.tools.passages import search_passages
        result = search_passages(query="q", domain="health")
    assert result["total"] == 1
    assert result["results"][0]["score"] == 0.95


def test_search_passages_no_alias_runs_once():
    calls = []

    def fake_search(collection_name, query, limit, filter_conditions):
        calls.append(filter_conditions)
        return []

    with patch("src.tools.passages.semantic_search", side_effect=fake_search):
        from src.tools.passages import search_passages
        search_passages(query="q", domain="srhr")

    assert len(calls) == 1
    assert calls[0] == {"domain": "srhr"}


def test_search_passages_combines_aliases_cartesian():
    """domain=health + doc_type=annual-report -> 2 * 3 = 6 search variants."""
    calls = []

    def fake_search(collection_name, query, limit, filter_conditions):
        calls.append(filter_conditions)
        return []

    with patch("src.tools.passages.semantic_search", side_effect=fake_search):
        from src.tools.passages import search_passages
        search_passages(query="q", domain="health", doc_type="annual-report")

    assert len(calls) == 6
    domains_seen = {c["domain"] for c in calls}
    doc_types_seen = {c["doc_type"] for c in calls}
    assert domains_seen == {"health", "srhr"}
    assert doc_types_seen == {"annual-report", "report", "monitoring-report"}


# ---------------------------------------------------------------------------
# search_terms — alias fan-out
# ---------------------------------------------------------------------------

def test_search_terms_health_fans_out():
    calls = []

    def fake_search(collection_name, query, limit, filter_conditions):
        calls.append(filter_conditions)
        domain = (filter_conditions or {}).get("domain")
        if domain == "srhr":
            return [{
                "id": "t1", "document_id": "t1", "score": 0.85,
                "title": "KP", "text": "key populations",
                "metadata": {"preferred": "key populations", "avoid": "MSM", "domain": "srhr", "language": "en"},
            }]
        return []

    with patch("src.tools.terms.semantic_search", side_effect=fake_search):
        from src.tools.terms import search_terms
        result = search_terms(query="kp", domain="health", include_shared=False)

    domains_queried = [c.get("domain") for c in calls if c]
    assert "health" in domains_queried
    assert "srhr" in domains_queried
    assert result["total"] == 1
    assert result["results"][0]["preferred"] == "key populations"
