"""Regression tests for the Phase 5A tool surface (20 tools).

Covers:
    - manage_term add share/admin branching (personal | library | queue)
    - admin_add routing for kind ∈ {thesaurus, rubric, template} (admin | non-admin)
    - score_writing_patterns mode dispatch (ai | semantic-ai | poetry | song | fiction) + invalid
    - check_external_similarity search_results branch
    - manage_* action dispatch (passage, style_profile, contributions, library)
    - search_thesaurus rich=True path
"""
import sys
import types
from unittest.mock import MagicMock, patch

import src.server as server


def _install_fake_kbase_indexer(return_value=None):
    """Inject a fake kbase.vector.sync_indexing module so server can import it."""
    idx = MagicMock(return_value=return_value if return_value is not None else ["p1"])
    kbase = sys.modules.setdefault("kbase", types.ModuleType("kbase"))
    vector = sys.modules.setdefault("kbase.vector", types.ModuleType("kbase.vector"))
    sync = types.ModuleType("kbase.vector.sync_indexing")
    sync.index_document = idx
    sys.modules["kbase.vector.sync_indexing"] = sync
    kbase.vector = vector
    vector.sync_indexing = sync
    return idx


def _ok(**extra):
    base = {"success": True, "document_id": "doc-xyz", "chunks_created": 1}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# manage_term(action="add") — share=False | share=True+admin | share=True+non-admin
# ---------------------------------------------------------------------------

def test_manage_term_add_personal_when_share_false():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.terms.add_term", return_value=_ok()) as personal:
        result = server.manage_term(
            action="add", preferred="rights-holder", ctx=None, share=False,
        )
    assert result["success"] is True
    assert result["routed_to"] == "personal"
    personal.assert_called_once()
    assert personal.call_args.kwargs["client_id"] == "alice"


def test_manage_term_add_library_when_share_true_and_admin():
    idx = _install_fake_kbase_indexer(return_value=["p1"])
    with patch.object(server, "_client_id", return_value="admin-1"), \
         patch.object(server, "_require_admin", return_value=None):
        result = server.manage_term(
            action="add", preferred="rights-holder", ctx=None,
            domain="srhr", share=True,
        )
    assert result["success"] is True
    assert result["routed_to"] == "library"
    assert idx.call_args.kwargs["collection_name"].endswith("writing_terms_shared")
    assert idx.call_args.kwargs["metadata"]["contributed_by"] == "admin-1"


def test_manage_term_add_queue_when_share_true_and_non_admin():
    contribute_result = {"success": True, "contribution_id": "c-1"}
    with patch.object(server, "_client_id", return_value="bob"), \
         patch.object(server, "_require_admin", return_value="Admin access required."), \
         patch.object(server, "_notify_contribution"), \
         patch("src.tools.contributions.contribute_term", return_value=contribute_result) as contrib:
        result = server.manage_term(
            action="add", preferred="rights-holder", ctx=None,
            share=True, note="please review",
        )
    assert result["success"] is True
    assert result["routed_to"] == "queue"
    contrib.assert_called_once()
    assert contrib.call_args.kwargs["contributed_by"] == "bob"


def test_manage_term_update_requires_document_id():
    out = server.manage_term(action="update", ctx=None)
    assert out["success"] is False
    assert "document_id" in out["error"]


def test_manage_term_delete_dispatches():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.terms.delete_term", return_value={"success": True}) as deleter:
        result = server.manage_term(action="delete", ctx=None, document_id="t-1")
    assert result["success"] is True
    deleter.assert_called_once()
    assert deleter.call_args.kwargs["document_id"] == "t-1"


def test_manage_term_invalid_action():
    out = server.manage_term(action="bogus", ctx=None)
    assert out["success"] is False
    assert "Invalid action" in out["error"]


# ---------------------------------------------------------------------------
# admin_add(kind="thesaurus")
# ---------------------------------------------------------------------------

def test_admin_add_thesaurus_admin_routes_library():
    with patch.object(server, "_client_id", return_value="admin-1"), \
         patch.object(server, "_require_admin", return_value=None), \
         patch("src.tools.thesaurus.add_thesaurus_entry", return_value=_ok()) as direct:
        result = server.admin_add(kind="thesaurus", ctx=None, headword="leverage")
    assert result["routed_to"] == "library"
    direct.assert_called_once()


def test_admin_add_thesaurus_non_admin_routes_queue():
    with patch.object(server, "_client_id", return_value="bob"), \
         patch.object(server, "_require_admin", return_value="no"), \
         patch.object(server, "_notify_contribution"), \
         patch("src.tools.contributions.contribute_thesaurus_entry",
               return_value={"success": True, "contribution_id": "c-2"}) as contrib:
        result = server.admin_add(kind="thesaurus", ctx=None, headword="leverage", note="ai-ish")
    assert result["routed_to"] == "queue"
    contrib.assert_called_once()
    assert contrib.call_args.kwargs["contributed_by"] == "bob"


# ---------------------------------------------------------------------------
# admin_add(kind="rubric")
# ---------------------------------------------------------------------------

def test_admin_add_rubric_admin_routes_library():
    with patch.object(server, "_client_id", return_value="admin-1"), \
         patch.object(server, "_require_admin", return_value=None), \
         patch("src.tools.rubrics.add_rubric_criterion", return_value=_ok()) as direct:
        result = server.admin_add(
            kind="rubric", ctx=None, framework="usaid",
            section="technical-approach", criterion="Clear theory of change",
        )
    assert result["routed_to"] == "library"
    direct.assert_called_once()


def test_admin_add_rubric_non_admin_routes_queue():
    with patch.object(server, "_client_id", return_value="bob"), \
         patch.object(server, "_require_admin", return_value="no"), \
         patch.object(server, "_notify_contribution"), \
         patch("src.tools.contributions.contribute_rubric",
               return_value={"success": True, "contribution_id": "c-3"}) as contrib:
        result = server.admin_add(
            kind="rubric", ctx=None, framework="usaid",
            section="sustainability", criterion="x",
        )
    assert result["routed_to"] == "queue"
    contrib.assert_called_once()


def test_admin_add_rubric_missing_fields():
    out = server.admin_add(kind="rubric", ctx=None, framework="usaid")
    assert out["success"] is False
    assert "framework" in out["error"] or "section" in out["error"]


# ---------------------------------------------------------------------------
# admin_add(kind="template")
# ---------------------------------------------------------------------------

def test_admin_add_template_admin_routes_library():
    sections = [{"name": "Executive Summary", "description": "summary"}]
    with patch.object(server, "_client_id", return_value="admin-1"), \
         patch.object(server, "_require_admin", return_value=None), \
         patch("src.tools.templates.add_template", return_value=_ok()) as direct:
        result = server.admin_add(
            kind="template", ctx=None, framework="undp",
            doc_type="concept-note", sections=sections,
        )
    assert result["routed_to"] == "library"
    direct.assert_called_once()


def test_admin_add_template_non_admin_routes_queue():
    sections = [{"name": "Executive Summary", "description": "summary"}]
    with patch.object(server, "_client_id", return_value="bob"), \
         patch.object(server, "_require_admin", return_value="no"), \
         patch.object(server, "_notify_contribution"), \
         patch("src.tools.contributions.contribute_template",
               return_value={"success": True, "contribution_id": "c-4"}) as contrib:
        result = server.admin_add(
            kind="template", ctx=None, framework="undp",
            doc_type="concept-note", sections=sections,
        )
    assert result["routed_to"] == "queue"
    contrib.assert_called_once()


def test_admin_add_invalid_kind():
    out = server.admin_add(kind="bogus", ctx=None)
    assert out["success"] is False
    assert "Invalid kind" in out["error"]


# ---------------------------------------------------------------------------
# score_writing_patterns — dispatch across the five modes + invalid mode
# ---------------------------------------------------------------------------

def test_score_writing_patterns_ai_mode():
    sentinel = {"success": True, "mode": "ai"}
    with patch("src.tools.ai_patterns.score_ai_patterns", return_value=sentinel) as scorer:
        out = server.score_writing_patterns(text="t", mode="ai", ctx=None)
    assert out is sentinel
    scorer.assert_called_once()


def test_score_writing_patterns_semantic_ai_mode():
    sentinel = {"success": True, "mode": "semantic-ai"}
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.ai_patterns.score_semantic_ai_likelihood", return_value=sentinel) as scorer:
        out = server.score_writing_patterns(text="t", mode="semantic-ai", ctx=None, top_k=5)
    assert out is sentinel
    scorer.assert_called_once_with(text="t", top_k=5, client_id="alice")


def test_score_writing_patterns_poetry_mode():
    sentinel = {"success": True, "mode": "poetry"}
    with patch("src.tools.poetry_patterns.score_poetry_patterns", return_value=sentinel) as scorer:
        out = server.score_writing_patterns(text="t", mode="poetry", ctx=None)
    assert out is sentinel
    scorer.assert_called_once()


def test_score_writing_patterns_song_mode():
    sentinel = {"success": True, "mode": "song"}
    with patch("src.tools.song_patterns.score_song_patterns", return_value=sentinel) as scorer:
        out = server.score_writing_patterns(text="t", mode="song", ctx=None)
    assert out is sentinel
    scorer.assert_called_once()


def test_score_writing_patterns_fiction_mode():
    sentinel = {"success": True, "mode": "fiction"}
    with patch("src.tools.fiction_patterns.score_fiction_patterns", return_value=sentinel) as scorer:
        out = server.score_writing_patterns(text="t", mode="fiction", ctx=None)
    assert out is sentinel
    scorer.assert_called_once()


def test_score_writing_patterns_invalid_mode_returns_error():
    out = server.score_writing_patterns(text="t", mode="bogus", ctx=None)
    assert out["success"] is False
    assert "Invalid mode" in out["error"]
    assert "ai, semantic-ai, poetry, song, fiction" in out["error"]


# ---------------------------------------------------------------------------
# check_external_similarity — search_results branch
# ---------------------------------------------------------------------------

def test_check_external_similarity_delegates_to_scorer_when_results_provided():
    sentinel = {"success": True, "verdict": "clean"}
    results = [{"url": "https://example.org", "content": "hi", "title": "t"}]
    with patch("src.tools.plagiarism.score_external_similarity", return_value=sentinel) as scorer, \
         patch("src.tools.plagiarism.check_external_similarity") as fetcher:
        out = server.check_external_similarity(text="t", search_results=results)
    assert out is sentinel
    scorer.assert_called_once()
    fetcher.assert_not_called()


def test_check_external_similarity_fetches_web_when_results_none():
    sentinel = {"success": True, "verdict": "clean"}
    with patch("src.tools.plagiarism.check_external_similarity", return_value=sentinel) as fetcher, \
         patch("src.tools.plagiarism.score_external_similarity") as scorer:
        out = server.check_external_similarity(text="t")
    assert out is sentinel
    fetcher.assert_called_once()
    scorer.assert_not_called()


# ---------------------------------------------------------------------------
# manage_passage — action dispatch
# ---------------------------------------------------------------------------

def test_manage_passage_add_single():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.passages.add_passage", return_value=_ok()) as adder:
        out = server.manage_passage(action="add", ctx=None, text="x", doc_type="report")
    assert out["success"] is True
    adder.assert_called_once()
    assert adder.call_args.kwargs["client_id"] == "alice"


def test_manage_passage_add_batch_uses_items():
    items = [{"text": "a", "doc_type": "r"}, {"text": "b", "doc_type": "r"}]
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.passages.batch_add_passages", return_value={"success": True, "added": 2}) as batcher:
        out = server.manage_passage(action="add", ctx=None, items=items)
    assert out["success"] is True
    batcher.assert_called_once()


def test_manage_passage_update_requires_document_id():
    out = server.manage_passage(action="update", ctx=None)
    assert out["success"] is False
    assert "document_id" in out["error"]


def test_manage_passage_correction_action():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.passages.record_correction", return_value=_ok()) as recorder:
        out = server.manage_passage(
            action="correction", ctx=None,
            original="bad", corrected="good", issue_type="ai-patterns",
        )
    assert out["success"] is True
    recorder.assert_called_once()


def test_manage_passage_invalid_action():
    out = server.manage_passage(action="bogus", ctx=None)
    assert out["success"] is False
    assert "Invalid action" in out["error"]


# ---------------------------------------------------------------------------
# manage_style_profile — upsert, load, search, list, harvest-corrections
# ---------------------------------------------------------------------------

def test_manage_style_profile_save_new_profile_creates():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.style_profiles.load_style_profile",
               return_value={"success": False}), \
         patch("src.tools.style_profiles.save_style_profile", return_value=_ok()) as saver, \
         patch("src.tools.style_profiles.update_style_profile") as updater:
        out = server.manage_style_profile(
            action="save", ctx=None, name="danilo",
            description="desc", style_scores={"narrative": 0.7},
            rules=["r1"], anti_patterns=["a1"], sample_excerpts=["s1"],
        )
    assert out["success"] is True
    saver.assert_called_once()
    updater.assert_not_called()


def test_manage_style_profile_save_existing_profile_updates():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.style_profiles.load_style_profile",
               return_value={"success": True, "profile": {"name": "danilo"}}), \
         patch("src.tools.style_profiles.save_style_profile") as saver, \
         patch("src.tools.style_profiles.update_style_profile", return_value=_ok()) as updater:
        out = server.manage_style_profile(
            action="save", ctx=None, name="danilo",
            rules=["new rule"],
        )
    assert out["success"] is True
    updater.assert_called_once()
    saver.assert_not_called()


def test_manage_style_profile_load():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.style_profiles.load_style_profile",
               return_value={"success": True}) as loader:
        out = server.manage_style_profile(action="load", ctx=None, name="danilo")
    assert out["success"] is True
    loader.assert_called_once()


def test_manage_style_profile_search():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.style_profiles.search_style_profiles",
               return_value={"success": True, "matches": []}) as searcher:
        out = server.manage_style_profile(
            action="search", ctx=None, text="sample", channel="linkedin",
        )
    assert out["success"] is True
    searcher.assert_called_once()


def test_manage_style_profile_list():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.style_profiles.list_style_profiles",
               return_value={"success": True, "profiles": []}) as lister:
        out = server.manage_style_profile(action="list", ctx=None)
    assert out["success"] is True
    lister.assert_called_once()


def test_manage_style_profile_harvest_corrections():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.style_profiles.harvest_corrections_to_profile",
               return_value={"success": True}) as harvester:
        out = server.manage_style_profile(
            action="harvest-corrections", ctx=None, name="danilo",
        )
    assert out["success"] is True
    harvester.assert_called_once()


def test_manage_style_profile_invalid_action():
    out = server.manage_style_profile(action="bogus", ctx=None)
    assert out["success"] is False
    assert "Invalid action" in out["error"]


# ---------------------------------------------------------------------------
# search_thesaurus — rich=False normal, rich=True delegates to suggest_alternatives
# ---------------------------------------------------------------------------

def test_search_thesaurus_default_path():
    with patch("src.tools.thesaurus.search_thesaurus",
               return_value={"success": True, "results": []}) as searcher, \
         patch("src.tools.thesaurus.suggest_alternatives") as rich:
        out = server.search_thesaurus(query="leverage")
    assert out["success"] is True
    searcher.assert_called_once()
    rich.assert_not_called()


def test_search_thesaurus_rich_uses_suggest_alternatives():
    with patch("src.tools.thesaurus.search_thesaurus") as searcher, \
         patch("src.tools.thesaurus.suggest_alternatives",
               return_value={"success": True, "alternatives": []}) as rich:
        out = server.search_thesaurus(query="leverage", rich=True)
    assert out["success"] is True
    rich.assert_called_once()
    searcher.assert_not_called()


# ---------------------------------------------------------------------------
# manage_contributions — list, review (review requires admin)
# ---------------------------------------------------------------------------

def test_manage_contributions_list():
    with patch("src.tools.contributions.list_contributions",
               return_value={"success": True, "contributions": []}) as lister:
        out = server.manage_contributions(action="list", ctx=None)
    assert out["success"] is True
    lister.assert_called_once()


def test_manage_contributions_review_requires_admin():
    with patch.object(server, "_require_admin", return_value="Admin required."):
        out = server.manage_contributions(
            action="review", ctx=None, contribution_id="c-1", review_action="publish",
        )
    assert out["success"] is False
    assert "Admin" in out["error"] or "admin" in out["error"]


def test_manage_contributions_review_admin_dispatches():
    with patch.object(server, "_client_id", return_value="admin-1"), \
         patch.object(server, "_require_admin", return_value=None), \
         patch("src.tools.contributions.review_contribution",
               return_value={"success": True}) as reviewer:
        out = server.manage_contributions(
            action="review", ctx=None, contribution_id="c-1", review_action="publish",
        )
    assert out["success"] is True
    reviewer.assert_called_once()


def test_manage_contributions_invalid_action():
    out = server.manage_contributions(action="bogus", ctx=None)
    assert out["success"] is False
    assert "Invalid action" in out["error"]


# ---------------------------------------------------------------------------
# manage_library — stats, export
# ---------------------------------------------------------------------------

def test_manage_library_stats():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.collections.get_stats",
               return_value={"success": True, "stats": {}}) as stats:
        out = server.manage_library(action="stats", ctx=None)
    assert out["success"] is True
    stats.assert_called_once()


def test_manage_library_export():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.export.export_library",
               return_value={"success": True, "format": "json"}) as exporter:
        out = server.manage_library(
            action="export", ctx=None, collection="passages", output_format="json",
        )
    assert out["success"] is True
    exporter.assert_called_once()


def test_manage_library_invalid_action():
    out = server.manage_library(action="bogus", ctx=None)
    assert out["success"] is False
    assert "Invalid action" in out["error"]
