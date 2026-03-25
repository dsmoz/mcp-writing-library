"""Tests for rubrics tool."""
from unittest.mock import patch, MagicMock
from uuid import uuid4

from tests.conftest import _make_mock_point, _make_mock_qdrant_client


# ---------------------------------------------------------------------------
# add_rubric_criterion
# ---------------------------------------------------------------------------

def test_add_rubric_criterion_success():
    mock_point_ids = [str(uuid4()), str(uuid4())]
    with patch("src.tools.rubrics.index_document", return_value=mock_point_ids):
        from src.tools.rubrics import add_rubric_criterion
        result = add_rubric_criterion(
            donor="usaid",
            section="technical-approach",
            criterion="The proposal clearly articulates a theory of change linking activities to outcomes.",
            weight=1.5,
            red_flags=["vague", "unclear linkage"],
        )
    assert result["success"] is True
    assert "document_id" in result
    assert result["chunks_created"] == 2
    assert "collection" in result


def test_add_rubric_criterion_invalid_donor():
    from src.tools.rubrics import add_rubric_criterion
    result = add_rubric_criterion(
        donor="worldbank",
        section="technical-approach",
        criterion="Some criterion text.",
    )
    assert result["success"] is False
    assert "donor" in result["error"].lower()


def test_add_rubric_criterion_weight_clamping_below():
    mock_point_ids = [str(uuid4())]
    captured_metadata = {}

    def fake_index_document(collection_name, document_id, title, content, metadata, context_mode):
        captured_metadata.update(metadata)
        return mock_point_ids

    with patch("src.tools.rubrics.index_document", side_effect=fake_index_document):
        from src.tools.rubrics import add_rubric_criterion
        result = add_rubric_criterion(
            donor="undp",
            section="results-framework",
            criterion="SMART indicators are defined with baselines and targets.",
            weight=-5.0,  # should be clamped to 0.1
        )
    assert result["success"] is True
    assert captured_metadata["weight"] == 0.1


def test_add_rubric_criterion_weight_clamping_above():
    mock_point_ids = [str(uuid4())]
    captured_metadata = {}

    def fake_index_document(collection_name, document_id, title, content, metadata, context_mode):
        captured_metadata.update(metadata)
        return mock_point_ids

    with patch("src.tools.rubrics.index_document", side_effect=fake_index_document):
        from src.tools.rubrics import add_rubric_criterion
        result = add_rubric_criterion(
            donor="eu",
            section="relevance",
            criterion="The proposal demonstrates added value over existing interventions.",
            weight=99.0,  # should be clamped to 2.0
        )
    assert result["success"] is True
    assert captured_metadata["weight"] == 2.0


def test_add_rubric_criterion_empty_criterion():
    from src.tools.rubrics import add_rubric_criterion
    result = add_rubric_criterion(donor="usaid", section="technical-approach", criterion="")
    assert result["success"] is False
    assert "criterion" in result["error"].lower()


def test_add_rubric_criterion_donor_case_insensitive():
    mock_point_ids = [str(uuid4())]
    with patch("src.tools.rubrics.index_document", return_value=mock_point_ids):
        from src.tools.rubrics import add_rubric_criterion
        result = add_rubric_criterion(
            donor="USAID",  # uppercase — should be normalised
            section="technical-approach",
            criterion="Theory of change is clearly articulated.",
        )
    assert result["success"] is True


# ---------------------------------------------------------------------------
# score_against_rubric
# ---------------------------------------------------------------------------

def _make_search_result(criterion_text, score, weight, section="technical-approach"):
    return {
        "id": str(uuid4()),
        "score": score,
        "document_id": str(uuid4()),
        "title": f"[USAID | {section}] {criterion_text[:60]}",
        "text": criterion_text,
        "metadata": {
            "donor": "usaid",
            "section": section,
            "weight": weight,
            "red_flags": ["vague"],
            "entry_type": "rubric_criterion",
        },
    }


def test_score_against_rubric_success_with_weighted_scores():
    mock_results = [
        _make_search_result("Theory of change linking activities to outcomes.", score=0.8, weight=1.5),
        _make_search_result("Past performance evidence with specific results.", score=0.6, weight=1.0),
    ]
    with patch("src.tools.rubrics.semantic_search", return_value=mock_results):
        from src.tools.rubrics import score_against_rubric
        result = score_against_rubric(
            text="Our theory of change is clearly articulated with measurable outcomes.",
            donor="usaid",
            section="technical-approach",
        )
    assert result["success"] is True
    assert result["donor"] == "usaid"
    assert result["criteria_matched"] == 2
    # weighted average: (0.8*1.5 + 0.6*1.0) / (1.5 + 1.0) = 1.8 / 2.5 = 0.72
    assert result["overall_score"] == round((0.8 * 1.5 + 0.6 * 1.0) / (1.5 + 1.0), 4)
    assert result["verdict"] == "strong"
    assert len(result["criteria"]) == 2
    assert result["criteria"][0]["weighted_score"] == round(0.8 * 1.5, 4)


def test_score_against_rubric_verdict_adequate():
    mock_results = [
        _make_search_result("Criterion A.", score=0.55, weight=1.0),
        _make_search_result("Criterion B.", score=0.65, weight=1.0),
    ]
    with patch("src.tools.rubrics.semantic_search", return_value=mock_results):
        from src.tools.rubrics import score_against_rubric
        result = score_against_rubric(text="Some proposal text.", donor="eu")
    assert result["success"] is True
    assert result["verdict"] == "adequate"


def test_score_against_rubric_verdict_weak():
    mock_results = [
        _make_search_result("Criterion A.", score=0.3, weight=1.0),
    ]
    with patch("src.tools.rubrics.semantic_search", return_value=mock_results):
        from src.tools.rubrics import score_against_rubric
        result = score_against_rubric(text="Vague proposal text.", donor="undp")
    assert result["success"] is True
    assert result["verdict"] == "weak"


def test_score_against_rubric_invalid_donor():
    from src.tools.rubrics import score_against_rubric
    result = score_against_rubric(text="Some text.", donor="idb")
    assert result["success"] is False
    assert "donor" in result["error"].lower()


def test_score_against_rubric_no_criteria_found():
    with patch("src.tools.rubrics.semantic_search", return_value=[]):
        from src.tools.rubrics import score_against_rubric
        result = score_against_rubric(text="Proposal text.", donor="global-fund", section="governance")
    assert result["success"] is False
    assert "global-fund" in result["error"]


def test_score_against_rubric_section_filter_passed_to_search():
    with patch("src.tools.rubrics.semantic_search", return_value=[]) as mock_search:
        from src.tools.rubrics import score_against_rubric
        score_against_rubric(text="Proposal text.", donor="usaid", section="sustainability")
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["filter_conditions"]["donor"] == "usaid"
    assert call_kwargs["filter_conditions"]["section"] == "sustainability"


def test_score_against_rubric_no_section_filter_omits_section_key():
    with patch("src.tools.rubrics.semantic_search", return_value=[]) as mock_search:
        from src.tools.rubrics import score_against_rubric
        score_against_rubric(text="Proposal text.", donor="usaid")
    call_kwargs = mock_search.call_args[1]
    assert "section" not in call_kwargs["filter_conditions"]


def test_score_against_rubric_doc_context_included_in_return():
    mock_results = [
        _make_search_result("Theory of change linking activities to outcomes.", score=0.8, weight=1.0),
    ]
    with patch("src.tools.rubrics.semantic_search", return_value=mock_results):
        from src.tools.rubrics import score_against_rubric
        result = score_against_rubric(
            text="Our annual report narrative.",
            donor="general",
            doc_context="annual report",
        )
    assert result["success"] is True
    assert "doc_context" in result
    assert result["doc_context"] == "annual report"


# ---------------------------------------------------------------------------
# list_rubric_donors
# ---------------------------------------------------------------------------

def test_list_rubric_donors_returns_correct_counts():
    points = [
        _make_mock_point({"donor": "usaid", "entry_type": "rubric_criterion"}),
        _make_mock_point({"donor": "usaid", "entry_type": "rubric_criterion"}),
        _make_mock_point({"donor": "undp", "entry_type": "rubric_criterion"}),
        _make_mock_point({"donor": "eu", "entry_type": "rubric_criterion"}),
    ]
    mock_client = _make_mock_qdrant_client((points, None))

    with patch("src.tools.rubrics.get_qdrant_client", return_value=mock_client):
        from src.tools.rubrics import list_rubric_donors
        result = list_rubric_donors()

    assert result["success"] is True
    assert result["total_donors"] == 3
    assert result["total_criteria"] == 4

    donor_map = {d["donor"]: d["criterion_count"] for d in result["donors"]}
    assert donor_map["usaid"] == 2
    assert donor_map["undp"] == 1
    assert donor_map["eu"] == 1


def test_list_rubric_donors_sorted_alphabetically():
    points = [
        _make_mock_point({"donor": "usaid"}),
        _make_mock_point({"donor": "eu"}),
        _make_mock_point({"donor": "undp"}),
        _make_mock_point({"donor": "global-fund"}),
    ]
    mock_client = _make_mock_qdrant_client((points, None))

    with patch("src.tools.rubrics.get_qdrant_client", return_value=mock_client):
        from src.tools.rubrics import list_rubric_donors
        result = list_rubric_donors()

    assert result["success"] is True
    names = [d["donor"] for d in result["donors"]]
    assert names == sorted(names)


def test_list_rubric_donors_empty_collection():
    mock_client = _make_mock_qdrant_client(([], None))

    with patch("src.tools.rubrics.get_qdrant_client", return_value=mock_client):
        from src.tools.rubrics import list_rubric_donors
        result = list_rubric_donors()

    assert result["success"] is True
    assert result["total_donors"] == 0
    assert result["total_criteria"] == 0
    assert result["donors"] == []
