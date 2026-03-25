"""Tests for passages tool."""
from unittest.mock import patch
from uuid import uuid4

from tests.conftest import _make_mock_point, _make_mock_qdrant_client


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


# --- delete_passage tests ---

def test_delete_passage_success():
    doc_id = str(uuid4())
    with patch("src.tools.passages.check_document_indexed", return_value={"indexed": True, "chunk_count": 2}), \
         patch("src.tools.passages.delete_document_vectors", return_value=2):
        from src.tools.passages import delete_passage
        result = delete_passage(document_id=doc_id)
    assert result["success"] is True
    assert result["document_id"] == doc_id
    assert result["deleted"] is True


def test_delete_passage_not_found():
    doc_id = str(uuid4())
    with patch("src.tools.passages.check_document_indexed", return_value={"indexed": False, "chunk_count": 0}):
        from src.tools.passages import delete_passage
        result = delete_passage(document_id=doc_id)
    assert result["success"] is False
    assert doc_id in result["error"]
    assert "No passage found" in result["error"]


def test_delete_passage_propagates_exception():
    doc_id = str(uuid4())
    with patch("src.tools.passages.check_document_indexed", side_effect=Exception("Qdrant connection refused")):
        from src.tools.passages import delete_passage
        result = delete_passage(document_id=doc_id)
    assert result["success"] is False
    assert "Qdrant connection refused" in result["error"]


# --- update_passage tests ---

def test_update_passage_requires_at_least_one_field():
    from src.tools.passages import update_passage
    result = update_passage(document_id=str(uuid4()))
    assert result["success"] is False
    assert "At least one field" in result["error"]


def test_update_passage_validates_doc_type():
    from src.tools.passages import update_passage
    result = update_passage(document_id=str(uuid4()), doc_type="invalid-type")
    assert result["success"] is False
    assert "doc_type" in result["error"].lower()


def test_update_passage_validates_language():
    from src.tools.passages import update_passage
    result = update_passage(document_id=str(uuid4()), language="de")
    assert result["success"] is False
    assert "language" in result["error"].lower()


def test_update_passage_validates_domain():
    from src.tools.passages import update_passage
    result = update_passage(document_id=str(uuid4()), domain="invalid-domain")
    assert result["success"] is False
    assert "domain" in result["error"].lower()


def test_update_passage_not_found():
    doc_id = str(uuid4())
    mock_client = _make_mock_qdrant_client(scroll_result=([], None))
    with patch("src.tools.passages.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.passages.Filter"), \
         patch("src.tools.passages.FieldCondition"), \
         patch("src.tools.passages.MatchValue"):
        from src.tools.passages import update_passage
        result = update_passage(document_id=doc_id, quality_notes="updated notes")
    assert result["success"] is False
    assert "No passage found" in result["error"]


def test_update_passage_success():
    doc_id = str(uuid4())
    existing_payload = {
        "text": "Original text about the assessment.",
        "doc_type": "report",
        "language": "en",
        "domain": "general",
        "quality_notes": "old notes",
        "tags": ["old-tag"],
        "source": "manual",
        "style": ["narrative"],
    }
    mock_point = _make_mock_point(existing_payload)
    mock_client = _make_mock_qdrant_client(scroll_result=([mock_point], None))
    mock_point_ids = [str(uuid4())]

    with patch("src.tools.passages.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.passages.delete_document_vectors", return_value=2), \
         patch("src.tools.passages.index_document", return_value=mock_point_ids), \
         patch("src.tools.passages.Filter"), \
         patch("src.tools.passages.FieldCondition"), \
         patch("src.tools.passages.MatchValue"):
        from src.tools.passages import update_passage
        result = update_passage(document_id=doc_id, quality_notes="better notes", domain="srhr")

    assert result["success"] is True
    assert result["document_id"] == doc_id
    assert "quality_notes" in result["updated_fields"]
    assert "domain" in result["updated_fields"]
    assert result["chunks_created"] == 1


def test_update_passage_warns_on_unknown_style():
    doc_id = str(uuid4())
    existing_payload = {
        "text": "Some text.", "doc_type": "report", "language": "en",
        "domain": "general", "quality_notes": "", "tags": [], "source": "manual", "style": [],
    }
    mock_point = _make_mock_point(existing_payload)
    mock_client = _make_mock_qdrant_client(scroll_result=([mock_point], None))
    mock_point_ids = [str(uuid4())]

    with patch("src.tools.passages.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.passages.delete_document_vectors", return_value=1), \
         patch("src.tools.passages.index_document", return_value=mock_point_ids), \
         patch("src.tools.passages.Filter"), \
         patch("src.tools.passages.FieldCondition"), \
         patch("src.tools.passages.MatchValue"):
        from src.tools.passages import update_passage
        result = update_passage(document_id=doc_id, style=["narrative", "unknown-style"])

    assert result["success"] is True
    assert len(result["warnings"]) == 1
    assert "unknown-style" in result["warnings"][0]


# --- batch_add_passages tests ---

def test_batch_add_passages_all_succeed():
    mock_point_ids = [str(uuid4())]
    items = [
        {"text": "First passage about governance."},
        {"text": "Second passage about climate.", "domain": "climate", "language": "en"},
    ]
    with patch("src.tools.passages.index_document", return_value=mock_point_ids):
        from src.tools.passages import batch_add_passages
        result = batch_add_passages(items=items)
    assert result["success"] is True
    assert result["total"] == 2
    assert result["succeeded"] == 2
    assert result["failed"] == 0
    assert len(result["results"]) == 2
    assert all(r["success"] for r in result["results"])
    assert result["results"][0]["index"] == 0
    assert result["results"][1]["index"] == 1


def test_batch_add_passages_partial_failure():
    mock_point_ids = [str(uuid4())]
    items = [
        {"text": "Valid passage."},
        {"doc_type": "report"},  # missing text — should fail
        {"text": "Another valid passage."},
    ]
    with patch("src.tools.passages.index_document", return_value=mock_point_ids):
        from src.tools.passages import batch_add_passages
        result = batch_add_passages(items=items)
    assert result["success"] is True
    assert result["total"] == 3
    assert result["succeeded"] == 2
    assert result["failed"] == 1
    failed = [r for r in result["results"] if not r["success"]]
    assert len(failed) == 1
    assert failed[0]["index"] == 1
    assert "text" in failed[0]["error"]


def test_batch_add_passages_empty_list():
    from src.tools.passages import batch_add_passages
    result = batch_add_passages(items=[])
    assert result["success"] is True
    assert result["total"] == 0
    assert result["succeeded"] == 0
    assert result["failed"] == 0
    assert result["results"] == []


def test_batch_add_passages_non_dict_item():
    from src.tools.passages import batch_add_passages
    result = batch_add_passages(items=["not a dict", None])
    assert result["success"] is True
    assert result["total"] == 2
    assert result["failed"] == 2
    for r in result["results"]:
        assert r["success"] is False


def test_batch_add_passages_per_item_validation_error_does_not_raise():
    """An invalid doc_type inside an item should produce a per-item failure, not raise."""
    items = [{"text": "Good text.", "doc_type": "invalid-type"}]
    from src.tools.passages import batch_add_passages
    result = batch_add_passages(items=items)
    assert result["success"] is True
    assert result["failed"] == 1
    assert "doc_type" in result["results"][0]["error"].lower()
