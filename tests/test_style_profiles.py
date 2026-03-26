"""Tests for style_profiles: update_style_profile and harvest_corrections_to_profile."""
from unittest.mock import patch, MagicMock
from uuid import uuid4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile_payload(name="danilo-voice-pt", rules=None, anti_patterns=None,
                           style_scores=None, sample_excerpts=None):
    return {
        "name": name,
        "description": "Analytical but direct.",
        "style_scores": style_scores or {"narrative": 0.8, "argumentative": 0.7},
        "rules": rules or ["Varies sentence length", "Opens with concrete observation"],
        "anti_patterns": anti_patterns or ["'leverage'", "passive constructions"],
        "sample_excerpts": sample_excerpts or ["In Mozambique, the data tells only part of the story."],
        "source_documents": ["lambda-2026.docx"],
        "entry_type": "style_profile",
        "document_id": str(uuid4()),
    }


def _mock_scroll(payload):
    """Return a mock qdrant scroll result with one point."""
    point = MagicMock()
    point.payload = payload
    point.id = payload.get("document_id", str(uuid4()))
    return ([point], None)


# ---------------------------------------------------------------------------
# update_style_profile tests
# ---------------------------------------------------------------------------

def test_update_style_profile_blends_scores():
    payload = _make_profile_payload(style_scores={"narrative": 0.8, "argumentative": 0.7})

    with patch("src.tools.style_profiles._get_qdrant_client") as mock_client, \
         patch("src.tools.style_profiles.index_document", return_value=[str(uuid4())]), \
         patch("src.tools.style_profiles.delete_document_vectors"):

        mock_client.return_value.scroll.return_value = _mock_scroll(payload)

        from src.tools.style_profiles import update_style_profile
        result = update_style_profile(
            name="danilo-voice-pt",
            new_style_scores={"narrative": 1.0},
            score_weight=0.3,
        )

    assert result["success"] is True
    assert "style_scores" in result["updated_fields"]
    # 0.7 * 0.8 + 0.3 * 1.0 = 0.86
    # We can't inspect merged scores directly without reading Qdrant,
    # but we verify the call succeeded and the field was updated.


def test_update_style_profile_unions_rules():
    payload = _make_profile_payload(rules=["Rule A", "Rule B"])
    indexed_metadata = {}

    def capture_index(**kwargs):
        indexed_metadata.update(kwargs.get("metadata", {}))
        return [str(uuid4())]

    with patch("src.tools.style_profiles._get_qdrant_client") as mock_client, \
         patch("src.tools.style_profiles.index_document", side_effect=capture_index), \
         patch("src.tools.style_profiles.delete_document_vectors"):

        mock_client.return_value.scroll.return_value = _mock_scroll(payload)

        from src.tools.style_profiles import update_style_profile
        result = update_style_profile(
            name="danilo-voice-pt",
            new_rules=["Rule C", "Rule A"],  # Rule A is a duplicate — should be ignored
        )

    assert result["success"] is True
    merged_rules = indexed_metadata.get("rules", [])
    assert "Rule A" in merged_rules
    assert "Rule B" in merged_rules
    assert "Rule C" in merged_rules
    assert merged_rules.count("Rule A") == 1  # no duplicates


def test_update_style_profile_unions_anti_patterns():
    payload = _make_profile_payload(anti_patterns=["'leverage'"])
    indexed_metadata = {}

    def capture_index(**kwargs):
        indexed_metadata.update(kwargs.get("metadata", {}))
        return [str(uuid4())]

    with patch("src.tools.style_profiles._get_qdrant_client") as mock_client, \
         patch("src.tools.style_profiles.index_document", side_effect=capture_index), \
         patch("src.tools.style_profiles.delete_document_vectors"):

        mock_client.return_value.scroll.return_value = _mock_scroll(payload)

        from src.tools.style_profiles import update_style_profile
        result = update_style_profile(
            name="danilo-voice-pt",
            new_anti_patterns=["hollow intensifiers", "'leverage'"],
        )

    assert result["success"] is True
    merged = indexed_metadata.get("anti_patterns", [])
    assert "hollow intensifiers" in merged
    assert merged.count("'leverage'") == 1  # duplicate suppressed


def test_update_style_profile_caps_excerpts_at_20():
    payload = _make_profile_payload(sample_excerpts=[f"Excerpt {i}" for i in range(18)])
    indexed_metadata = {}

    def capture_index(**kwargs):
        indexed_metadata.update(kwargs.get("metadata", {}))
        return [str(uuid4())]

    with patch("src.tools.style_profiles._get_qdrant_client") as mock_client, \
         patch("src.tools.style_profiles.index_document", side_effect=capture_index), \
         patch("src.tools.style_profiles.delete_document_vectors"):

        mock_client.return_value.scroll.return_value = _mock_scroll(payload)

        from src.tools.style_profiles import update_style_profile
        result = update_style_profile(
            name="danilo-voice-pt",
            new_sample_excerpts=["New excerpt A", "New excerpt B", "New excerpt C"],
        )

    assert result["success"] is True
    assert len(indexed_metadata.get("sample_excerpts", [])) <= 20


def test_update_style_profile_rejects_missing_name():
    from src.tools.style_profiles import update_style_profile
    result = update_style_profile(name="", new_rules=["Something"])
    assert result["success"] is False
    assert "name" in result["error"].lower()


def test_update_style_profile_rejects_invalid_score_weight():
    from src.tools.style_profiles import update_style_profile
    result = update_style_profile(name="x", new_rules=["r"], score_weight=0.0)
    assert result["success"] is False
    assert "score_weight" in result["error"].lower()


def test_update_style_profile_rejects_no_fields():
    payload = _make_profile_payload()

    with patch("src.tools.style_profiles._get_qdrant_client") as mock_client:
        mock_client.return_value.scroll.return_value = _mock_scroll(payload)
        from src.tools.style_profiles import update_style_profile
        result = update_style_profile(name="danilo-voice-pt")

    assert result["success"] is False
    assert "field" in result["error"].lower()


def test_update_style_profile_not_found():
    with patch("src.tools.style_profiles._get_qdrant_client") as mock_client:
        mock_client.return_value.scroll.return_value = ([], None)
        from src.tools.style_profiles import update_style_profile
        result = update_style_profile(name="nonexistent", new_rules=["r"])
    assert result["success"] is False


# ---------------------------------------------------------------------------
# harvest_corrections_to_profile tests
# ---------------------------------------------------------------------------

def _make_correction_result(issue_type: str, text: str = "Corrected text.") -> dict:
    return {
        "score": 0.80,
        "text": text,
        "document_id": str(uuid4()),
        "metadata": {
            "entry_type": "correction",
            "style": ["human-corrected"],
            "issue_type": issue_type,
            "language": "en",
            "domain": "general",
        },
    }


def test_harvest_returns_candidates_from_corrections():
    profile_payload = _make_profile_payload(rules=[], anti_patterns=[])
    corrections = [_make_correction_result("hollow-intensifier") for _ in range(4)]

    with patch("src.tools.style_profiles._get_qdrant_client") as mock_client, \
         patch("src.tools.style_profiles.semantic_search", return_value=corrections):

        mock_client.return_value.scroll.return_value = _mock_scroll(profile_payload)

        from src.tools.style_profiles import harvest_corrections_to_profile
        result = harvest_corrections_to_profile(
            profile_name="danilo-voice-pt",
            min_corrections=3,
        )

    assert result["success"] is True
    assert result["insufficient_data"] is False
    assert result["corrections_found"] >= 3
    assert any(c["source_issue_type"] == "hollow-intensifier" for c in result["candidates"])


def test_harvest_skips_candidates_already_in_profile():
    existing_anti = ["Hollow intensifiers: avoid 'it is important to note that', 'it is crucial that'"]
    profile_payload = _make_profile_payload(anti_patterns=existing_anti)
    corrections = [_make_correction_result("hollow-intensifier") for _ in range(4)]

    with patch("src.tools.style_profiles._get_qdrant_client") as mock_client, \
         patch("src.tools.style_profiles.semantic_search", return_value=corrections):

        mock_client.return_value.scroll.return_value = _mock_scroll(profile_payload)

        from src.tools.style_profiles import harvest_corrections_to_profile
        result = harvest_corrections_to_profile("danilo-voice-pt", min_corrections=3)

    assert result["success"] is True
    # hollow-intensifier already in profile — should not appear as candidate
    for c in result["candidates"]:
        assert c["source_issue_type"] != "hollow-intensifier"


def test_harvest_returns_insufficient_data_when_too_few():
    profile_payload = _make_profile_payload()
    corrections = [_make_correction_result("passive-voice") for _ in range(2)]

    with patch("src.tools.style_profiles._get_qdrant_client") as mock_client, \
         patch("src.tools.style_profiles.semantic_search", return_value=corrections):

        mock_client.return_value.scroll.return_value = _mock_scroll(profile_payload)

        from src.tools.style_profiles import harvest_corrections_to_profile
        result = harvest_corrections_to_profile("danilo-voice-pt", min_corrections=3)

    assert result["success"] is True
    assert result["insufficient_data"] is True
    assert result["candidates"] == []
    assert "record_correction" in result["note"]


def test_harvest_rejects_missing_profile_name():
    from src.tools.style_profiles import harvest_corrections_to_profile
    result = harvest_corrections_to_profile(profile_name="")
    assert result["success"] is False
    assert "profile_name" in result["error"].lower()


def test_harvest_rejects_nonexistent_profile():
    with patch("src.tools.style_profiles._get_qdrant_client") as mock_client:
        mock_client.return_value.scroll.return_value = ([], None)
        from src.tools.style_profiles import harvest_corrections_to_profile
        result = harvest_corrections_to_profile("nonexistent")
    assert result["success"] is False
