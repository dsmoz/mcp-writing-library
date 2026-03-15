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
