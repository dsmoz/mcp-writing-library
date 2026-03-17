"""Tests for passages tool."""
import pytest
from unittest.mock import patch
from uuid import uuid4


def test_add_passage_returns_document_id():
    mock_point_ids = [str(uuid4()), str(uuid4())]
    with patch("src.tools.passages.index_document", return_value=mock_point_ids):
        from src.tools.passages import add_passage
        result = add_passage(
            text="The assessment reveals a familiar pattern: progress coexists with persistent gaps.",
            doc_type="executive-summary",
            language="en",
            domain="general",
            quality_notes="Good discursive opener with contrast",
            tags=["discursive", "contrast"],
            source="manual",
        )
    assert result["success"] is True
    assert "document_id" in result
    assert result["chunks_created"] == 2


def test_add_passage_validates_doc_type():
    from src.tools.passages import add_passage
    result = add_passage(text="Some text.", doc_type="invalid-type", language="en")
    assert result["success"] is False
    assert "doc_type" in result["error"].lower()


def test_add_passage_validates_language():
    from src.tools.passages import add_passage
    result = add_passage(text="Some text.", doc_type="report", language="fr")
    assert result["success"] is False
    assert "language" in result["error"].lower()


def test_search_passages_calls_semantic_search():
    mock_results = [{
        "id": str(uuid4()), "score": 0.92, "document_id": str(uuid4()),
        "title": "Example", "text": "The assessment reveals...",
        "metadata": {"doc_type": "executive-summary", "language": "en"},
    }]
    with patch("src.tools.passages.semantic_search", return_value=mock_results):
        from src.tools.passages import search_passages
        result = search_passages(query="implementation gaps", top_k=5)
    assert result["success"] is True
    assert len(result["results"]) == 1
    assert result["results"][0]["score"] == 0.92


def test_search_passages_filters_by_doc_type():
    with patch("src.tools.passages.semantic_search", return_value=[]) as mock_search:
        from src.tools.passages import search_passages
        search_passages(query="findings", doc_type="policy-brief", language="pt")
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["filter_conditions"]["doc_type"] == "policy-brief"
    assert call_kwargs["filter_conditions"]["language"] == "pt"


def test_add_passage_accepts_valid_style():
    mock_point_ids = [str(uuid4())]
    with patch("src.tools.passages.index_document", return_value=mock_point_ids):
        from src.tools.passages import add_passage
        result = add_passage(
            text="The assessment reveals a familiar pattern.",
            doc_type="report",
            language="en",
            style=["narrative", "donor-facing"],
        )
    assert result["success"] is True
    assert result["warnings"] == []


def test_add_passage_warns_on_unknown_style_but_succeeds():
    mock_point_ids = [str(uuid4())]
    with patch("src.tools.passages.index_document", return_value=mock_point_ids):
        from src.tools.passages import add_passage
        result = add_passage(
            text="The assessment reveals a familiar pattern.",
            doc_type="report",
            language="en",
            style=["narrative", "made-up-style"],
        )
    assert result["success"] is True
    assert len(result["warnings"]) == 1
    assert "made-up-style" in result["warnings"][0]


def test_search_passages_post_filters_by_style():
    mock_results = [
        {
            "id": "1", "score": 0.95, "document_id": "doc1",
            "title": "T1", "text": "Narrative passage",
            "metadata": {
                "doc_type": "report", "language": "en",
                "style": ["narrative", "danilo-voice"],
            },
        },
        {
            "id": "2", "score": 0.88, "document_id": "doc2",
            "title": "T2", "text": "Formal passage",
            "metadata": {
                "doc_type": "report", "language": "en",
                "style": ["formal", "donor-facing"],
            },
        },
    ]
    with patch("src.tools.passages.semantic_search", return_value=mock_results):
        from src.tools.passages import search_passages
        result = search_passages(query="governance report", style="narrative")
    assert result["success"] is True
    assert len(result["results"]) == 1
    assert "narrative" in result["results"][0]["style"]
