"""Tests for the writing styles registry."""


def test_list_styles_returns_all_categories():
    from src.tools.styles import list_styles
    result = list_styles()
    assert result["success"] is True
    assert "structural" in result["styles"]
    assert "tonal" in result["styles"]
    assert "source" in result["styles"]
    assert "anti-pattern" in result["styles"]
    assert result["total"] == 14


def test_valid_styles_set_matches_styles_dict():
    from src.tools.styles import STYLES, VALID_STYLES
    assert VALID_STYLES == set(STYLES.keys())


def test_all_styles_have_required_keys():
    from src.tools.styles import STYLES
    for name, info in STYLES.items():
        assert "category" in info, f"{name} missing 'category'"
        assert "description" in info, f"{name} missing 'description'"


def test_structural_styles_present():
    from src.tools.styles import VALID_STYLES
    for s in ["narrative", "data-driven", "argumentative", "minimalist"]:
        assert s in VALID_STYLES


def test_tonal_styles_present():
    from src.tools.styles import VALID_STYLES
    for s in ["formal", "conversational", "donor-facing", "advocacy"]:
        assert s in VALID_STYLES


def test_source_styles_present():
    from src.tools.styles import VALID_STYLES
    for s in ["undp", "global-fund", "danilo-voice"]:
        assert s in VALID_STYLES


def test_antipattern_styles_present():
    from src.tools.styles import VALID_STYLES
    for s in ["ai-sounding", "bureaucratic", "jargon-heavy"]:
        assert s in VALID_STYLES
