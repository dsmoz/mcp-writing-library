"""Tests for the export_library tool."""
from unittest.mock import patch, MagicMock

from tests.conftest import _make_mock_point


def _make_scroll_client(pages: list[list[dict]]) -> MagicMock:
    """
    Build a mock Qdrant client whose .scroll() returns successive pages.

    pages: list of lists of payload dicts. The last page returns next_offset=None.
    Example: [[{"text": "a"}], [{"text": "b"}]] → two pages, two payloads.
    """
    client = MagicMock()
    scroll_calls = []
    for i, page_payloads in enumerate(pages):
        points = [_make_mock_point(p) for p in page_payloads]
        next_offset = i + 1 if i < len(pages) - 1 else None
        scroll_calls.append((points, next_offset))

    client.scroll.side_effect = scroll_calls
    return client


# ---------------------------------------------------------------------------
# export_library — JSON format
# ---------------------------------------------------------------------------

def test_export_library_json_passages():
    payloads = [
        {"text": "First passage.", "doc_type": "report", "language": "en"},
        {"text": "Second passage.", "doc_type": "email", "language": "pt"},
    ]
    mock_client = _make_scroll_client([payloads])

    with patch("src.tools.export.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.export.get_collection_names", return_value={
             "passages": "writing_passages",
             "terms": "writing_terms",
             "style_profiles": "writing_style_profiles",
         }):
        from src.tools.export import export_library
        result = export_library(collection="passages", format="json")

    assert result["success"] is True
    assert result["count"] == 2
    assert result["format"] == "json"
    assert isinstance(result["data"], list)
    assert result["data"][0]["text"] == "First passage."


def test_export_library_json_terms():
    payloads = [{"preferred": "rights-holder", "avoid": "victim", "domain": "srhr"}]
    mock_client = _make_scroll_client([payloads])

    with patch("src.tools.export.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.export.get_collection_names", return_value={
             "passages": "writing_passages",
             "terms": "writing_terms",
             "style_profiles": "writing_style_profiles",
         }):
        from src.tools.export import export_library
        result = export_library(collection="terms", format="json")

    assert result["success"] is True
    assert result["count"] == 1
    assert result["data"][0]["preferred"] == "rights-holder"


def test_export_library_json_style_profiles():
    payloads = [{"name": "danilo-voice-pt", "description": "Analytical."}]
    mock_client = _make_scroll_client([payloads])

    with patch("src.tools.export.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.export.get_collection_names", return_value={
             "passages": "writing_passages",
             "terms": "writing_terms",
             "style_profiles": "writing_style_profiles",
         }):
        from src.tools.export import export_library
        result = export_library(collection="style_profiles", format="json")

    assert result["success"] is True
    assert result["collection"] == "writing_style_profiles"


def test_export_library_accepts_literal_collection_name():
    payloads = [{"text": "A passage."}]
    mock_client = _make_scroll_client([payloads])

    with patch("src.tools.export.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.export.get_collection_names", return_value={
             "passages": "writing_passages",
             "terms": "writing_terms",
             "style_profiles": "writing_style_profiles",
         }):
        from src.tools.export import export_library
        result = export_library(collection="writing_passages", format="json")

    assert result["success"] is True
    assert result["collection"] == "writing_passages"


def test_export_library_paginates_multiple_pages():
    page1 = [{"text": f"Passage {i}."} for i in range(3)]
    page2 = [{"text": f"Passage {i}."} for i in range(3, 5)]
    mock_client = _make_scroll_client([page1, page2])

    with patch("src.tools.export.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.export.get_collection_names", return_value={
             "passages": "writing_passages",
             "terms": "writing_terms",
             "style_profiles": "writing_style_profiles",
         }):
        from src.tools.export import export_library
        result = export_library(collection="passages", format="json")

    assert result["success"] is True
    assert result["count"] == 5


# ---------------------------------------------------------------------------
# export_library — CSV format
# ---------------------------------------------------------------------------

def test_export_library_csv_returns_string():
    payloads = [
        {"text": "First.", "doc_type": "report", "tags": ["a", "b"]},
        {"text": "Second.", "doc_type": "email", "tags": []},
    ]
    mock_client = _make_scroll_client([payloads])

    with patch("src.tools.export.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.export.get_collection_names", return_value={
             "passages": "writing_passages",
             "terms": "writing_terms",
             "style_profiles": "writing_style_profiles",
         }):
        from src.tools.export import export_library
        result = export_library(collection="passages", format="csv")

    assert result["success"] is True
    assert result["format"] == "csv"
    assert isinstance(result["data"], str)
    # CSV header must include field names
    assert "text" in result["data"]
    assert "doc_type" in result["data"]
    # Values present
    assert "First." in result["data"]
    assert "Second." in result["data"]


def test_export_library_csv_empty_collection():
    mock_client = _make_scroll_client([[]])

    with patch("src.tools.export.get_qdrant_client", return_value=mock_client), \
         patch("src.tools.export.get_collection_names", return_value={
             "passages": "writing_passages",
             "terms": "writing_terms",
             "style_profiles": "writing_style_profiles",
         }):
        from src.tools.export import export_library
        result = export_library(collection="passages", format="csv")

    assert result["success"] is True
    assert result["count"] == 0
    assert result["data"] == ""


# ---------------------------------------------------------------------------
# export_library — error handling
# ---------------------------------------------------------------------------

def test_export_library_unknown_collection():
    with patch("src.tools.export.get_collection_names", return_value={
        "passages": "writing_passages",
        "terms": "writing_terms",
        "style_profiles": "writing_style_profiles",
    }):
        from src.tools.export import export_library
        result = export_library(collection="nonexistent_collection")

    assert result["success"] is False
    assert "nonexistent_collection" in result["error"]
    assert "Valid aliases" in result["error"]


def test_export_library_invalid_format():
    with patch("src.tools.export.get_collection_names", return_value={
        "passages": "writing_passages",
        "terms": "writing_terms",
        "style_profiles": "writing_style_profiles",
    }):
        from src.tools.export import export_library
        result = export_library(collection="passages", format="xml")

    assert result["success"] is False
    assert "xml" in result["error"]


def test_export_library_scroll_exception_returns_error():
    client = MagicMock()
    client.scroll.side_effect = Exception("Qdrant connection refused")

    with patch("src.tools.export.get_qdrant_client", return_value=client), \
         patch("src.tools.export.get_collection_names", return_value={
             "passages": "writing_passages",
             "terms": "writing_terms",
             "style_profiles": "writing_style_profiles",
         }):
        from src.tools.export import export_library
        result = export_library(collection="passages", format="json")

    assert result["success"] is False
    assert "Qdrant connection refused" in result["error"]
