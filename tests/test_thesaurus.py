"""Tests for the writing thesaurus collection and tooling."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock


def test_get_collection_names_includes_thesaurus():
    from src.tools.collections import get_collection_names
    names = get_collection_names()
    assert "thesaurus" in names
    assert names["thesaurus"] == "writing_thesaurus"


def test_add_thesaurus_entry_returns_document_id():
    with patch("src.tools.thesaurus.index_document") as mock_index, \
         patch("src.tools.thesaurus.semantic_search") as mock_search:
        mock_index.return_value = ["point-1"]
        mock_search.return_value = []
        from src.tools.thesaurus import add_thesaurus_entry
        result = add_thesaurus_entry(
            headword="leverage",
            language="en",
            domain="general",
            definition="To use something to maximum advantage.",
            part_of_speech="verb",
            register="institutional",
            alternatives=[
                {"word": "use", "meaning_nuance": "Direct and clear", "register": "neutral", "when_to_use": "Default replacement in most contexts"},
            ],
            collocations=["leverage resources", "leverage partnerships"],
            why_avoid="Overused in AI-generated donor proposals; sounds mechanical.",
            example_bad="We will leverage our networks to ensure impact.",
            example_good="We will draw on our networks to extend reach.",
            source="manual",
        )
        assert result["success"] is True
        assert "document_id" in result


def test_add_thesaurus_entry_rejects_invalid_domain():
    from src.tools.thesaurus import add_thesaurus_entry
    result = add_thesaurus_entry(
        headword="leverage",
        language="en",
        domain="invalid-domain",
        definition="To use something to maximum advantage.",
        part_of_speech="verb",
        register="institutional",
        alternatives=[],
    )
    assert result["success"] is False
    assert "domain" in result["error"]


def test_add_thesaurus_entry_rejects_invalid_language():
    from src.tools.thesaurus import add_thesaurus_entry
    result = add_thesaurus_entry(
        headword="leverage",
        language="fr",
        domain="general",
        definition="...",
        part_of_speech="verb",
        register="neutral",
        alternatives=[],
    )
    assert result["success"] is False
    assert "language" in result["error"]


def test_add_thesaurus_entry_rejects_invalid_part_of_speech():
    from src.tools.thesaurus import add_thesaurus_entry
    result = add_thesaurus_entry(
        headword="leverage",
        language="en",
        domain="general",
        definition="To use something to maximum advantage.",
        part_of_speech="gerund",
        register="neutral",
        alternatives=[],
    )
    assert result["success"] is False
    assert "part_of_speech" in result["error"]


def test_add_thesaurus_entry_rejects_invalid_register():
    from src.tools.thesaurus import add_thesaurus_entry
    result = add_thesaurus_entry(
        headword="leverage",
        language="en",
        domain="general",
        definition="To use something to maximum advantage.",
        part_of_speech="verb",
        register="slang",
        alternatives=[],
    )
    assert result["success"] is False
    assert "register" in result["error"]


def test_add_thesaurus_entry_rejects_duplicate(monkeypatch):
    """If headword+language already exists, return error."""
    import src.tools.thesaurus as mod
    monkeypatch.setattr(mod, "semantic_search", lambda **kw: [
        {"metadata": {"headword": "leverage", "language": "en"}, "score": 0.99, "document_id": "abc", "title": "leverage"}
    ])
    result = mod.add_thesaurus_entry(
        headword="leverage",
        language="en",
        domain="general",
        definition="...",
        part_of_speech="verb",
        register="neutral",
        alternatives=[],
    )
    assert result["success"] is False
    assert "already exists" in result["error"]


def test_search_thesaurus_returns_results(monkeypatch):
    import src.tools.thesaurus as mod
    import json
    monkeypatch.setattr(mod, "semantic_search", lambda **kw: [
        {
            "score": 0.92,
            "document_id": "abc-123",
            "title": "leverage",
            "metadata": {
                "headword": "leverage",
                "language": "en",
                "domain": "general",
                "definition": "To use something to maximum advantage.",
                "part_of_speech": "verb",
                "register": "institutional",
                "alternatives": json.dumps([{"word": "use", "meaning_nuance": "Direct", "register": "neutral", "when_to_use": "Default"}]),
                "collocations": json.dumps(["leverage resources"]),
                "why_avoid": "Overused in AI text.",
                "example_bad": "We will leverage our networks.",
                "example_good": "We will draw on our networks.",
                "source": "manual",
                "entry_type": "thesaurus",
            },
        }
    ])
    result = mod.search_thesaurus(query="leverage")
    assert result["success"] is True
    assert len(result["results"]) == 1
    entry = result["results"][0]
    assert entry["headword"] == "leverage"
    assert isinstance(entry["alternatives"], list)
    assert entry["alternatives"][0]["word"] == "use"


def test_search_thesaurus_empty_query():
    from src.tools.thesaurus import search_thesaurus
    result = search_thesaurus(query="")
    assert result["success"] is False
    assert "query" in result["error"]


def test_suggest_alternatives_found(monkeypatch):
    import src.tools.thesaurus as mod
    import json
    monkeypatch.setattr(mod, "semantic_search", lambda **kw: [
        {
            "score": 0.97,
            "document_id": "abc-123",
            "title": "leverage",
            "metadata": {
                "headword": "leverage",
                "language": "en",
                "domain": "general",
                "definition": "To use something to maximum advantage.",
                "part_of_speech": "verb",
                "register": "institutional",
                "alternatives": json.dumps([
                    {"word": "use", "meaning_nuance": "Direct and clear", "register": "neutral", "when_to_use": "Default"},
                    {"word": "draw on", "meaning_nuance": "Implies existing resource", "register": "neutral", "when_to_use": "When referring to existing assets"},
                    {"word": "apply", "meaning_nuance": "More technical", "register": "formal", "when_to_use": "When precision matters"},
                ]),
                "collocations": json.dumps(["leverage resources", "leverage partnerships"]),
                "why_avoid": "Overused in AI text.",
                "example_bad": "We will leverage our networks.",
                "example_good": "We will draw on our networks.",
                "source": "manual",
                "entry_type": "thesaurus",
            },
        }
    ])
    monkeypatch.setattr(mod, "_search_terms_fallback", lambda word, language: [])
    result = mod.suggest_alternatives(word="leverage", language="en", domain="general")
    assert result["success"] is True
    assert result["found_in_thesaurus"] is True
    assert len(result["alternatives"]) == 3
    assert result["alternatives"][0]["word"] == "use"
    assert "why_avoid" in result
    assert "definition" in result


def test_suggest_alternatives_not_found_falls_back(monkeypatch):
    """When word not in thesaurus, fall back to search_terms."""
    import src.tools.thesaurus as mod
    monkeypatch.setattr(mod, "semantic_search", lambda **kw: [])
    monkeypatch.setattr(mod, "_search_terms_fallback", lambda word, language: [
        {"preferred": "use", "avoid": "leverage", "why": "More direct"}
    ])
    result = mod.suggest_alternatives(word="leverage", language="en", domain="general")
    assert result["success"] is True
    assert result["found_in_thesaurus"] is False
    assert len(result["alternatives"]) >= 1


def test_suggest_alternatives_empty_word():
    from src.tools.thesaurus import suggest_alternatives
    result = suggest_alternatives(word="", language="en", domain="general")
    assert result["success"] is False
    assert "word" in result["error"]


def test_flag_vocabulary_detects_headwords(monkeypatch):
    import src.tools.thesaurus as mod
    import json

    def mock_search(collection_name, query, limit, filter_conditions=None):
        entries = {
            "leverage": {
                "headword": "leverage",
                "language": "en",
                "domain": "general",
                "why_avoid": "AI overuse.",
                "alternatives": json.dumps([{"word": "use", "meaning_nuance": "Direct", "register": "neutral", "when_to_use": "Default"}]),
                "collocations": "[]",
                "definition": "...", "part_of_speech": "verb", "register": "institutional",
                "example_bad": "", "example_good": "", "source": "manual", "entry_type": "thesaurus",
            },
            "ensure": {
                "headword": "ensure",
                "language": "en",
                "domain": "general",
                "why_avoid": "Weak AI filler.",
                "alternatives": json.dumps([{"word": "guarantee", "meaning_nuance": "Stronger commitment", "register": "formal", "when_to_use": "High-stakes commitments"}]),
                "collocations": "[]",
                "definition": "...", "part_of_speech": "verb", "register": "institutional",
                "example_bad": "", "example_good": "", "source": "manual", "entry_type": "thesaurus",
            },
        }
        hits = []
        for headword, meta in entries.items():
            if headword in query.lower():
                hits.append({"score": 0.95, "document_id": f"id-{headword}", "title": headword, "metadata": meta})
        return hits

    monkeypatch.setattr(mod, "semantic_search", mock_search)
    text = "We will leverage our networks to ensure maximum impact."
    result = mod.flag_vocabulary(text=text, language="en", domain="general")
    assert result["success"] is True
    flagged = [f["headword"] for f in result["flagged"]]
    assert "leverage" in flagged
    assert "ensure" in flagged
    assert result["flagged_count"] == 2


def test_flag_vocabulary_empty_text():
    from src.tools.thesaurus import flag_vocabulary
    result = flag_vocabulary(text="", language="en", domain="general")
    assert result["success"] is False
    assert "text" in result["error"]


def test_flag_vocabulary_clean_text(monkeypatch):
    import src.tools.thesaurus as mod
    monkeypatch.setattr(mod, "semantic_search", lambda **kw: [])
    result = mod.flag_vocabulary(text="The team met on Tuesday to discuss the findings.", language="en", domain="general")
    assert result["success"] is True
    assert result["flagged_count"] == 0
    assert result["verdict"] == "clean"
