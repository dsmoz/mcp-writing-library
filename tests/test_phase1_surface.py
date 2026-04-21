"""Regression tests for Phase 1 tool-surface changes.

Covers:
    - add_term share/admin branching (personal | library | queue)
    - add_thesaurus_entry, add_rubric_criterion, add_template admin/non-admin routing
    - score_writing_patterns mode dispatch (ai | semantic-ai | poetry | song | fiction) + invalid
    - check_external_similarity search_results branch
"""
import sys
import types
from unittest.mock import MagicMock, patch

import src.server as server


def _install_fake_kbase_indexer(return_value=None):
    """Inject a fake kbase.vector.sync_indexing module so server.add_term can import it."""
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
# add_term — share=False | share=True+admin | share=True+non-admin
# ---------------------------------------------------------------------------

def test_add_term_personal_when_share_false():
    with patch.object(server, "_client_id", return_value="alice"), \
         patch("src.tools.terms.add_term", return_value=_ok()) as personal:
        result = server.add_term(preferred="rights-holder", ctx=None, share=False)
    assert result["success"] is True
    assert result["routed_to"] == "personal"
    personal.assert_called_once()
    assert personal.call_args.kwargs["client_id"] == "alice"


def test_add_term_library_when_share_true_and_admin():
    idx = _install_fake_kbase_indexer(return_value=["p1"])
    with patch.object(server, "_client_id", return_value="admin-1"), \
         patch.object(server, "_require_admin", return_value=None):
        result = server.add_term(
            preferred="rights-holder", ctx=None, domain="srhr", share=True,
        )
    assert result["success"] is True
    assert result["routed_to"] == "library"
    assert idx.call_args.kwargs["collection_name"].endswith("writing_terms_shared")
    assert idx.call_args.kwargs["metadata"]["contributed_by"] == "admin-1"


def test_add_term_queue_when_share_true_and_non_admin():
    contribute_result = {"success": True, "contribution_id": "c-1"}
    with patch.object(server, "_client_id", return_value="bob"), \
         patch.object(server, "_require_admin", return_value="Admin access required."), \
         patch.object(server, "_notify_contribution"), \
         patch("src.tools.contributions.contribute_term", return_value=contribute_result) as contrib:
        result = server.add_term(
            preferred="rights-holder", ctx=None, share=True, note="please review",
        )
    assert result["success"] is True
    assert result["routed_to"] == "queue"
    contrib.assert_called_once()
    assert contrib.call_args.kwargs["contributed_by"] == "bob"


# ---------------------------------------------------------------------------
# add_thesaurus_entry
# ---------------------------------------------------------------------------

def test_add_thesaurus_entry_admin_routes_library():
    with patch.object(server, "_client_id", return_value="admin-1"), \
         patch.object(server, "_require_admin", return_value=None), \
         patch("src.tools.thesaurus.add_thesaurus_entry", return_value=_ok()) as direct:
        result = server.add_thesaurus_entry(ctx=None, headword="leverage")
    assert result["routed_to"] == "library"
    direct.assert_called_once()


def test_add_thesaurus_entry_non_admin_routes_queue():
    with patch.object(server, "_client_id", return_value="bob"), \
         patch.object(server, "_require_admin", return_value="no"), \
         patch.object(server, "_notify_contribution"), \
         patch("src.tools.contributions.contribute_thesaurus_entry",
               return_value={"success": True, "contribution_id": "c-2"}) as contrib:
        result = server.add_thesaurus_entry(ctx=None, headword="leverage", note="ai-ish")
    assert result["routed_to"] == "queue"
    contrib.assert_called_once()
    assert contrib.call_args.kwargs["contributed_by"] == "bob"


# ---------------------------------------------------------------------------
# add_rubric_criterion
# ---------------------------------------------------------------------------

def test_add_rubric_criterion_admin_routes_library():
    with patch.object(server, "_client_id", return_value="admin-1"), \
         patch.object(server, "_require_admin", return_value=None), \
         patch("src.tools.rubrics.add_rubric_criterion", return_value=_ok()) as direct:
        result = server.add_rubric_criterion(
            ctx=None, framework="usaid", section="technical-approach",
            criterion="Clear theory of change",
        )
    assert result["routed_to"] == "library"
    direct.assert_called_once()


def test_add_rubric_criterion_non_admin_routes_queue():
    with patch.object(server, "_client_id", return_value="bob"), \
         patch.object(server, "_require_admin", return_value="no"), \
         patch.object(server, "_notify_contribution"), \
         patch("src.tools.contributions.contribute_rubric",
               return_value={"success": True, "contribution_id": "c-3"}) as contrib:
        result = server.add_rubric_criterion(
            ctx=None, framework="usaid", section="sustainability", criterion="x",
        )
    assert result["routed_to"] == "queue"
    contrib.assert_called_once()


# ---------------------------------------------------------------------------
# add_template
# ---------------------------------------------------------------------------

def test_add_template_admin_routes_library():
    sections = [{"name": "Executive Summary", "description": "summary"}]
    with patch.object(server, "_client_id", return_value="admin-1"), \
         patch.object(server, "_require_admin", return_value=None), \
         patch("src.tools.templates.add_template", return_value=_ok()) as direct:
        result = server.add_template(
            ctx=None, framework="undp", doc_type="concept-note", sections=sections,
        )
    assert result["routed_to"] == "library"
    direct.assert_called_once()


def test_add_template_non_admin_routes_queue():
    sections = [{"name": "Executive Summary", "description": "summary"}]
    with patch.object(server, "_client_id", return_value="bob"), \
         patch.object(server, "_require_admin", return_value="no"), \
         patch.object(server, "_notify_contribution"), \
         patch("src.tools.contributions.contribute_template",
               return_value={"success": True, "contribution_id": "c-4"}) as contrib:
        result = server.add_template(
            ctx=None, framework="undp", doc_type="concept-note", sections=sections,
        )
    assert result["routed_to"] == "queue"
    contrib.assert_called_once()


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
