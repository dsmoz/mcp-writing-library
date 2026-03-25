"""Tests for terminology dictionary tool."""
import pytest
from unittest.mock import patch
from uuid import uuid4


def test_add_term_returns_document_id():
    mock_point_ids = [str(uuid4())]
    with patch("src.tools.terms.index_document", return_value=mock_point_ids):
        from src.tools.terms import add_term
        result = add_term(
            preferred="rights-holder",
            avoid="victim",
            domain="srhr",
            language="en",
            why="Deficit framing undermines agency. UNDP standard.",
            example_bad="HIV victims need services.",
            example_good="People living with HIV require access to services.",
        )
    assert result["success"] is True
    assert "document_id" in result


def test_add_term_requires_preferred():
    from src.tools.terms import add_term
    result = add_term(preferred="", avoid="something", domain="general")
    assert result["success"] is False
    assert "preferred" in result["error"].lower()


def test_search_terms_returns_formatted_results():
    mock_results = [{
        "id": str(uuid4()), "score": 0.95, "document_id": str(uuid4()),
        "title": "rights-holder", "text": "Preferred: rights-holder. Avoid: victim.",
        "metadata": {
            "preferred": "rights-holder", "avoid": "victim",
            "domain": "srhr", "language": "en",
            "why": "Deficit framing",
            "example_bad": "HIV victims...", "example_good": "People living with HIV...",
            "entry_type": "term",
        },
    }]
    with patch("src.tools.terms.semantic_search", return_value=mock_results):
        from src.tools.terms import search_terms
        result = search_terms(query="person living with HIV language")
    assert result["success"] is True
    assert len(result["results"]) == 1
    assert result["results"][0]["preferred"] == "rights-holder"
    assert result["results"][0]["avoid"] == "victim"


# --- delete_term tests ---

def test_delete_term_success():
    doc_id = str(uuid4())
    with patch("src.tools.terms.check_document_indexed", return_value={"indexed": True, "chunk_count": 1}), \
         patch("src.tools.terms.delete_document_vectors", return_value=1):
        from src.tools.terms import delete_term
        result = delete_term(document_id=doc_id)
    assert result["success"] is True
    assert result["document_id"] == doc_id
    assert result["deleted"] is True


def test_delete_term_not_found():
    doc_id = str(uuid4())
    with patch("src.tools.terms.check_document_indexed", return_value={"indexed": False, "chunk_count": 0}):
        from src.tools.terms import delete_term
        result = delete_term(document_id=doc_id)
    assert result["success"] is False
    assert doc_id in result["error"]
    assert "No term found" in result["error"]


def test_delete_term_propagates_exception():
    doc_id = str(uuid4())
    with patch("src.tools.terms.check_document_indexed", side_effect=Exception("connection timeout")):
        from src.tools.terms import delete_term
        result = delete_term(document_id=doc_id)
    assert result["success"] is False
    assert "connection timeout" in result["error"]


# --- update_term tests ---

def _make_mock_point(payload: dict):
    from unittest.mock import MagicMock
    point = MagicMock()
    point.payload = payload
    return point


def _make_mock_qdrant_client(scroll_result):
    from unittest.mock import MagicMock
    client = MagicMock()
    client.scroll.return_value = scroll_result
    return client


def test_update_term_requires_at_least_one_field():
    from src.tools.terms import update_term
    result = update_term(document_id=str(uuid4()))
    assert result["success"] is False
    assert "At least one field" in result["error"]


def test_update_term_validates_domain():
    from src.tools.terms import update_term
    result = update_term(document_id=str(uuid4()), domain="invalid-domain")
    assert result["success"] is False
    assert "domain" in result["error"].lower()


def test_update_term_validates_language():
    from src.tools.terms import update_term
    result = update_term(document_id=str(uuid4()), language="de")
    assert result["success"] is False
    assert "language" in result["error"].lower()


def test_update_term_not_found():
    doc_id = str(uuid4())
    mock_client = _make_mock_qdrant_client(scroll_result=([], None))
    with patch("src.tools.terms.get_qdrant_client", return_value=mock_client):
        from src.tools.terms import update_term
        result = update_term(document_id=doc_id, why="Updated rationale")
    assert result["success"] is False
    assert "No term found" in result["error"]


def test_update_term_success():
    doc_id = str(uuid4())
    existing_payload = {
        "preferred": "rights-holder",
        "avoid": "victim",
        "domain": "srhr",
        "language": "en",
        "why": "old rationale",
        "example_bad": "old bad example",
        "example_good": "old good example",
    }
    mock_point = _make_mock_point(existing_payload)
    mock_client = _make_mock_qdrant_client(scroll_result=([mock_point], None))
    mock_point_ids = [str(uuid4())]

    with patch("src.tools.terms.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.terms.delete_document_vectors", return_value=1), \
         patch("src.tools.terms.index_document", return_value=mock_point_ids):
        from src.tools.terms import update_term
        result = update_term(document_id=doc_id, why="New rationale aligned with UNDP 2024", domain="governance")

    assert result["success"] is True
    assert result["document_id"] == doc_id
    assert "why" in result["updated_fields"]
    assert "domain" in result["updated_fields"]
    assert result["chunks_created"] == 1
