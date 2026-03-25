"""Tests for templates tool."""
from unittest.mock import patch, MagicMock
from uuid import uuid4

from tests.conftest import _make_mock_point, _make_mock_qdrant_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_sections(count=3):
    return [
        {
            "name": f"Section {i}",
            "description": f"Description for section {i} with relevant content",
            "required": True,
            "order": i,
        }
        for i in range(1, count + 1)
    ]


# ---------------------------------------------------------------------------
# add_template
# ---------------------------------------------------------------------------

def test_add_template_success():
    mock_point_ids = [str(uuid4()), str(uuid4())]
    with patch("src.tools.templates.index_document", return_value=mock_point_ids):
        from src.tools.templates import add_template
        result = add_template(
            donor="undp",
            doc_type="concept-note",
            sections=_valid_sections(3),
        )
    assert result["success"] is True
    assert "document_id" in result
    assert result["chunks_created"] == 2
    assert result["donor"] == "undp"
    assert result["doc_type"] == "concept-note"
    assert result["section_count"] == 3


def test_add_template_invalid_donor():
    from src.tools.templates import add_template
    result = add_template(
        donor="worldbank",
        doc_type="concept-note",
        sections=_valid_sections(2),
    )
    assert result["success"] is False
    assert "donor" in result["error"].lower()


def test_add_template_invalid_doc_type():
    from src.tools.templates import add_template
    result = add_template(
        donor="undp",
        doc_type="grant-application",
        sections=_valid_sections(2),
    )
    assert result["success"] is False
    assert "doc_type" in result["error"].lower()


def test_add_template_empty_sections():
    from src.tools.templates import add_template
    result = add_template(donor="usaid", doc_type="full-proposal", sections=[])
    assert result["success"] is False
    assert "section" in result["error"].lower()


def test_add_template_section_missing_name():
    from src.tools.templates import add_template
    result = add_template(
        donor="eu",
        doc_type="eoi",
        sections=[{"description": "A description without name", "required": True}],
    )
    assert result["success"] is False
    assert "name" in result["error"].lower()


def test_add_template_section_missing_description():
    from src.tools.templates import add_template
    result = add_template(
        donor="eu",
        doc_type="eoi",
        sections=[{"name": "Relevance"}],
    )
    assert result["success"] is False
    assert "description" in result["error"].lower()


def test_add_template_donor_case_insensitive():
    mock_point_ids = [str(uuid4())]
    with patch("src.tools.templates.index_document", return_value=mock_point_ids):
        from src.tools.templates import add_template
        result = add_template(
            donor="UNDP",
            doc_type="concept-note",
            sections=_valid_sections(2),
        )
    assert result["success"] is True
    assert result["donor"] == "undp"


def test_add_template_section_defaults_applied():
    """required defaults to True and order defaults to list index + 1."""
    mock_point_ids = [str(uuid4())]
    captured = {}

    def fake_index(collection_name, document_id, title, content, metadata, context_mode):
        captured.update(metadata)
        return mock_point_ids

    with patch("src.tools.templates.index_document", side_effect=fake_index):
        from src.tools.templates import add_template
        add_template(
            donor="usaid",
            doc_type="full-proposal",
            sections=[
                {"name": "Technical Approach", "description": "Methodology"},
                {"name": "Personnel", "description": "Key staff"},
            ],
        )

    sections = captured["sections"]
    assert sections[0]["required"] is True
    assert sections[0]["order"] == 1
    assert sections[1]["required"] is True
    assert sections[1]["order"] == 2


def test_add_template_kbase_unavailable():
    from src.tools.templates import add_template
    with patch("src.tools.templates.index_document", None):
        result = add_template(donor="undp", doc_type="concept-note", sections=_valid_sections(2))
    assert result["success"] is False
    assert "kbase" in result["error"].lower()


# ---------------------------------------------------------------------------
# check_structure
# ---------------------------------------------------------------------------

def _make_template_search_result(donor, doc_type, sections):
    doc_id = str(uuid4())
    return {
        "id": str(uuid4()),
        "score": 0.9,
        "document_id": doc_id,
        "title": f"[{donor.upper()} | {doc_type}] Template",
        "text": " ".join(f"{s['name']}: {s['description']}" for s in sections),
        "metadata": {
            "donor": donor,
            "doc_type": doc_type,
            "sections": sections,
            "section_count": len(sections),
            "entry_type": "template",
        },
    }


def test_check_structure_no_template_found():
    with patch("src.tools.templates.semantic_search", return_value=[]):
        from src.tools.templates import check_structure
        result = check_structure(
            text="Some proposal text.",
            donor="usaid",
            doc_type="concept-note",
        )
    assert result["success"] is False
    assert "usaid" in result["error"]
    assert "concept-note" in result["error"]


def test_check_structure_invalid_donor():
    from src.tools.templates import check_structure
    result = check_structure(text="Some text.", donor="idb", doc_type="concept-note")
    assert result["success"] is False
    assert "donor" in result["error"].lower()


def test_check_structure_invalid_doc_type():
    from src.tools.templates import check_structure
    result = check_structure(text="Some text.", donor="undp", doc_type="brochure")
    assert result["success"] is False
    assert "doc_type" in result["error"].lower()


def test_check_structure_complete_verdict():
    """
    All sections detected via keyword fallback — verdict should be 'complete'.
    Use sections whose names appear verbatim in the text so keyword matching succeeds.
    """
    sections = [
        {"name": "Executive Summary", "description": "overview objectives budget", "required": True, "order": 1},
        {"name": "Problem Statement", "description": "development problem populations", "required": True, "order": 2},
    ]
    template_result = _make_template_search_result("undp", "concept-note", sections)

    # Draft text contains enough words to trigger keyword match
    draft = (
        "Executive Summary\n\nThis project addresses a critical need.\n\n"
        "Problem Statement\n\nThe development problem affects many populations."
    )

    with patch("src.tools.templates.semantic_search", return_value=[template_result]):
        with patch("src.tools.templates._embedding_available", False):
            with patch("src.tools.templates._generate_embedding", None):
                from src.tools.templates import check_structure
                result = check_structure(text=draft, donor="undp", doc_type="concept-note")

    assert result["success"] is True
    assert result["donor"] == "undp"
    assert result["doc_type"] == "concept-note"
    assert result["total_sections"] == 2
    assert result["required_sections"] == 2
    assert result["verdict"] == "complete"
    assert result["missing_count"] == 0
    assert result["missing_required"] == []
    # Both sections should be present or partial
    statuses = {s["name"]: s["status"] for s in result["sections"]}
    assert statuses["Executive Summary"] in ("present", "partial")
    assert statuses["Problem Statement"] in ("present", "partial")


def test_check_structure_incomplete_verdict():
    """
    One required section is completely absent — verdict should be 'incomplete'.
    """
    sections = [
        {"name": "Executive Summary", "description": "overview objectives budget", "required": True, "order": 1},
        {"name": "Zanzibar Methodology", "description": "zanzibar implementation uniqueterm12345", "required": True, "order": 2},
    ]
    template_result = _make_template_search_result("undp", "concept-note", sections)

    # Draft text covers first section but not the second
    draft = "Executive Summary\n\nThis is an executive overview with objectives and budget details."

    with patch("src.tools.templates.semantic_search", return_value=[template_result]):
        with patch("src.tools.templates._embedding_available", False):
            with patch("src.tools.templates._generate_embedding", None):
                from src.tools.templates import check_structure
                result = check_structure(text=draft, donor="undp", doc_type="concept-note")

    assert result["success"] is True
    assert result["verdict"] == "incomplete"
    assert "Zanzibar Methodology" in result["missing_required"]
    assert result["missing_count"] >= 1


def test_check_structure_returns_section_detail():
    """Result must include per-section coverage_score, status, and scoring_method fields."""
    sections = [
        {"name": "Technical Approach", "description": "methodology theory of change", "required": True, "order": 1},
    ]
    template_result = _make_template_search_result("usaid", "full-proposal", sections)
    draft = "Technical Approach\n\nOur methodology is evidence-based."

    with patch("src.tools.templates.semantic_search", return_value=[template_result]):
        with patch("src.tools.templates._embedding_available", False):
            with patch("src.tools.templates._generate_embedding", None):
                from src.tools.templates import check_structure
                result = check_structure(text=draft, donor="usaid", doc_type="full-proposal")

    assert result["success"] is True
    assert len(result["sections"]) == 1
    sec = result["sections"][0]
    assert sec["name"] == "Technical Approach"
    assert sec["required"] is True
    assert "status" in sec
    assert "coverage_score" in sec
    assert isinstance(sec["coverage_score"], float)
    assert "scoring_method" in sec
    assert sec["scoring_method"] == "keyword"
    # Top-level scoring_method should be present
    assert result["scoring_method"] == "keyword"


def test_check_structure_with_embedding_success():
    """When embedding is available, cosine similarity path is used."""
    sections = [
        {"name": "Budget", "description": "financial cost categories", "required": True, "order": 1},
    ]
    template_result = _make_template_search_result("eu", "eoi", sections)
    draft = "Budget\n\nDetailed financial breakdown with cost categories."

    # Return a fake embedding vector (unit vector)
    fake_embedding = [1.0] + [0.0] * 767

    with patch("src.tools.templates.semantic_search", return_value=[template_result]):
        with patch("src.tools.templates._embedding_available", True):
            with patch("src.tools.templates._generate_embedding", return_value=fake_embedding):
                from src.tools.templates import check_structure
                result = check_structure(text=draft, donor="eu", doc_type="eoi")

    assert result["success"] is True
    # With identical embeddings for all texts, cosine sim = 1.0 → "present"
    assert result["sections"][0]["status"] == "present"
    assert result["sections"][0]["scoring_method"] == "embedding"
    assert result["scoring_method"] == "embedding"


def test_check_structure_kbase_unavailable():
    from src.tools.templates import check_structure
    with patch("src.tools.templates.semantic_search", None):
        result = check_structure(text="Some text.", donor="undp", doc_type="concept-note")
    assert result["success"] is False
    assert "kbase" in result["error"].lower()


def test_check_structure_empty_text():
    """Empty or whitespace-only text should return early with an error."""
    from src.tools.templates import check_structure
    result = check_structure(text="", donor="undp", doc_type="concept-note")
    assert result["success"] is False
    assert "empty" in result["error"].lower()

    result_ws = check_structure(text="   \n\n  ", donor="undp", doc_type="concept-note")
    assert result_ws["success"] is False
    assert "empty" in result_ws["error"].lower()


def test_check_structure_per_section_embedding_fallback():
    """
    When embedding fails for one section but succeeds for another,
    that section uses keyword scoring while the other uses embedding.
    The top-level scoring_method should be 'mixed'.
    """
    sections = [
        {"name": "Executive Summary", "description": "overview objectives budget", "required": True, "order": 1},
        {"name": "Problem Statement", "description": "development problem populations", "required": True, "order": 2},
    ]
    template_result = _make_template_search_result("undp", "concept-note", sections)
    draft = (
        "Executive Summary\n\nThis project addresses a critical need.\n\n"
        "Problem Statement\n\nThe development problem affects many populations."
    )

    fake_embedding = [1.0] + [0.0] * 767

    def embedding_side_effect(text):
        # Fail specifically when generating the embedding for the second section query
        if "Problem Statement" in text and "development problem populations" in text:
            raise RuntimeError("simulated embedding failure")
        return fake_embedding

    with patch("src.tools.templates.semantic_search", return_value=[template_result]):
        with patch("src.tools.templates._embedding_available", True):
            with patch("src.tools.templates._generate_embedding", side_effect=embedding_side_effect):
                from src.tools.templates import check_structure
                result = check_structure(text=draft, donor="undp", doc_type="concept-note")

    assert result["success"] is True
    # Top-level scoring_method must be 'mixed' (one embedding, one keyword)
    assert result["scoring_method"] == "mixed"
    methods = {s["name"]: s["scoring_method"] for s in result["sections"]}
    assert methods["Executive Summary"] == "embedding"
    assert methods["Problem Statement"] == "keyword"


def test_check_structure_keyword_thresholds():
    """
    Keyword scoring should use lower thresholds so matching text is not classified as missing.
    """
    sections = [
        {
            "name": "Budget Overview",
            "description": "budget financial cost categories",
            "required": True,
            "order": 1,
        },
    ]
    template_result = _make_template_search_result("eu", "eoi", sections)
    # Text has a few matching words — should score above KW_THRESHOLD_PRESENT (0.15)
    draft = "Budget Overview\n\nThe budget includes financial cost categories for all activities."

    with patch("src.tools.templates.semantic_search", return_value=[template_result]):
        with patch("src.tools.templates._embedding_available", False):
            with patch("src.tools.templates._generate_embedding", None):
                from src.tools.templates import check_structure
                result = check_structure(text=draft, donor="eu", doc_type="eoi")

    assert result["success"] is True
    sec = result["sections"][0]
    assert sec["scoring_method"] == "keyword"
    # With several matching words (budget, financial, cost, categories), should NOT be missing
    assert sec["status"] in ("present", "partial")


# ---------------------------------------------------------------------------
# list_templates
# ---------------------------------------------------------------------------

def test_list_templates_returns_sorted_list():
    points = [
        _make_mock_point({
            "entry_type": "template",
            "donor": "usaid",
            "doc_type": "full-proposal",
            "section_count": 5,
            "document_id": str(uuid4()),
        }),
        _make_mock_point({
            "entry_type": "template",
            "donor": "eu",
            "doc_type": "eoi",
            "section_count": 4,
            "document_id": str(uuid4()),
        }),
        _make_mock_point({
            "entry_type": "template",
            "donor": "undp",
            "doc_type": "concept-note",
            "section_count": 6,
            "document_id": str(uuid4()),
        }),
    ]
    mock_client = _make_mock_qdrant_client((points, None))

    with patch("src.tools.templates.get_qdrant_client", return_value=mock_client):
        from src.tools.templates import list_templates
        result = list_templates()

    assert result["success"] is True
    assert result["total"] == 3
    donors = [t["donor"] for t in result["templates"]]
    assert donors == sorted(donors)


def test_list_templates_empty_collection():
    mock_client = _make_mock_qdrant_client(([], None))

    with patch("src.tools.templates.get_qdrant_client", return_value=mock_client):
        from src.tools.templates import list_templates
        result = list_templates()

    assert result["success"] is True
    assert result["total"] == 0
    assert result["templates"] == []


def test_list_templates_filters_non_template_entries():
    """Points without entry_type='template' should be excluded."""
    points = [
        _make_mock_point({
            "entry_type": "template",
            "donor": "undp",
            "doc_type": "concept-note",
            "section_count": 6,
            "document_id": str(uuid4()),
        }),
        _make_mock_point({
            "entry_type": "rubric_criterion",
            "donor": "undp",
            "section": "results",
        }),
    ]
    mock_client = _make_mock_qdrant_client((points, None))

    with patch("src.tools.templates.get_qdrant_client", return_value=mock_client):
        from src.tools.templates import list_templates
        result = list_templates()

    assert result["success"] is True
    assert result["total"] == 1
    assert result["templates"][0]["donor"] == "undp"


def test_list_templates_kbase_unavailable():
    from src.tools.templates import list_templates
    with patch("src.tools.templates.get_qdrant_client", None):
        result = list_templates()
    assert result["success"] is False
    assert "kbase" in result["error"].lower()
