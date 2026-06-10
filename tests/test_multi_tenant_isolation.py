"""
Multi-tenant isolation tests for rubrics, templates, thesaurus (per-user collections).

These tests verify:
1. Per-user collection naming with client_id prefix
2. Direct write to per-user collections (no moderation queue)
3. Isolation between users — entries in user A's collection not visible to user B
4. Both admin and non-admin users write to their own collections
"""
import sys
import types
from unittest.mock import MagicMock, patch, call

import src.server as server
import src.tools.contributions as contributions
import src.tools.rubrics as rubrics
import src.tools.templates as templates
import src.tools.thesaurus as thesaurus


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
# Test 1: Rubric collection naming with client_id prefix
# ---------------------------------------------------------------------------

def test_rubric_collection_name_includes_client_id():
    """Verify collection name includes sanitized client_id prefix."""
    from src.tools.collections import get_collection_names

    names_alice = get_collection_names("alice")
    assert names_alice["rubrics"] == "alice_writing_rubrics"

    names_bob = get_collection_names("bob")
    assert names_bob["rubrics"] == "bob_writing_rubrics"

    # Default client
    names_default = get_collection_names("default")
    assert names_default["rubrics"] == "default_writing_rubrics"


def test_template_collection_name_includes_client_id():
    """Verify collection name includes sanitized client_id prefix."""
    from src.tools.collections import get_collection_names

    names_alice = get_collection_names("alice")
    assert names_alice["templates"] == "alice_writing_templates"

    names_bob = get_collection_names("bob")
    assert names_bob["templates"] == "bob_writing_templates"


def test_thesaurus_collection_name_includes_client_id():
    """Verify collection name includes sanitized client_id prefix."""
    from src.tools.collections import get_collection_names

    names_alice = get_collection_names("alice")
    assert names_alice["thesaurus"] == "alice_writing_thesaurus"

    names_bob = get_collection_names("bob")
    assert names_bob["thesaurus"] == "bob_writing_thesaurus"


# ---------------------------------------------------------------------------
# Test 3: admin_add() routes both admin and non-admin to caller collection
#
# admin_add is the sole create-path for rubric/template/thesaurus. It calls the
# per-user add_* functions directly (no moderation queue) with client_id=caller.
# Patches target the add_* source-module attributes; admin_add imports them
# inside the function body, so the patched attribute is bound at call time.
# ---------------------------------------------------------------------------

def test_admin_add_rubric_admin_user_to_own_collection():
    """admin_add(kind='rubric') writes to caller's per-user collection via add_rubric_criterion."""
    with patch.object(server, "_client_id", return_value="admin-alice"), \
         patch("src.tools.rubrics.add_rubric_criterion",
               return_value={"success": True, "document_id": "doc-1"}) as mock_add:
        result = server.admin_add(
            kind="rubric",
            ctx=None,
            framework="lambda",
            section="sustainability",
            criterion="Demonstrates long-term viability",
        )

    assert result["success"] is True
    assert result["routed_to"] == "caller_collection"
    mock_add.assert_called_once()
    assert mock_add.call_args.kwargs["client_id"] == "admin-alice"


def test_admin_add_template_non_admin_user_to_own_collection():
    """admin_add(kind='template') writes to caller's per-user collection via add_template (not queue)."""
    sections = [{"name": "Background", "description": "Context", "required": True}]

    with patch.object(server, "_client_id", return_value="analyst-bob"), \
         patch("src.tools.templates.add_template",
               return_value={"success": True, "document_id": "doc-2"}) as mock_add:
        result = server.admin_add(
            kind="template",
            ctx=None,
            framework="gf",
            doc_type="monitoring-report",
            sections=sections,
        )

    assert result["success"] is True
    assert result["routed_to"] == "caller_collection"
    mock_add.assert_called_once()
    assert mock_add.call_args.kwargs["client_id"] == "analyst-bob"


def test_admin_add_thesaurus_different_users_separate_entries():
    """Two users adding thesaurus entries write to their own separate per-user collections."""
    with patch.object(server, "_client_id", return_value="user-x"), \
         patch("src.tools.thesaurus.add_thesaurus_entry",
               return_value={"success": True, "document_id": "doc-x"}) as mock_add_x:
        result_x = server.admin_add(
            kind="thesaurus",
            ctx=None,
            headword="robust",
            definition="Strong and durable",
        )

    with patch.object(server, "_client_id", return_value="user-y"), \
         patch("src.tools.thesaurus.add_thesaurus_entry",
               return_value={"success": True, "document_id": "doc-y"}) as mock_add_y:
        result_y = server.admin_add(
            kind="thesaurus",
            ctx=None,
            headword="robust",
            definition="Solidly built (alternative sense)",
        )

    assert result_x["success"] is True
    assert result_y["success"] is True
    assert mock_add_x.call_args.kwargs["client_id"] == "user-x"
    assert mock_add_y.call_args.kwargs["client_id"] == "user-y"


# ---------------------------------------------------------------------------
# Test 4: Ensure per-user functions call ensure_user_collections_once()
# ---------------------------------------------------------------------------

def test_add_rubric_criterion_calls_ensure_user_collections():
    """add_rubric_criterion() ensures per-user collections exist before write."""
    _install_fake_kbase_indexer()

    with patch("src.tools.rubrics.ensure_user_collections_once") as mock_ensure, \
         patch("src.tools.rubrics.index_document"):
        rubrics.add_rubric_criterion(
            framework="usaid",
            section="results",
            criterion="Test criterion",
            client_id="alice",
        )

    mock_ensure.assert_called_once_with("alice")


def test_add_template_calls_ensure_user_collections():
    """add_template() ensures per-user collections exist before write."""
    _install_fake_kbase_indexer()

    with patch("src.tools.templates.ensure_user_collections_once") as mock_ensure, \
         patch("src.tools.templates.index_document"):
        templates.add_template(
            framework="undp",
            doc_type="eoi",
            sections=[{"name": "Test", "description": "Test section"}],
            client_id="bob",
        )

    mock_ensure.assert_called_once_with("bob")


def test_add_thesaurus_entry_calls_ensure_user_collections():
    """add_thesaurus_entry() ensures per-user collections exist before write."""
    _install_fake_kbase_indexer()

    with patch("src.tools.thesaurus.ensure_user_collections_once") as mock_ensure, \
         patch("src.tools.thesaurus.index_document"):
        thesaurus.add_thesaurus_entry(
            headword="test",
            language="en",
            domain="general",
            client_id="charlie",
        )

    mock_ensure.assert_called_once_with("charlie")


# ---------------------------------------------------------------------------
# Test 5: Default client_id fallback
# ---------------------------------------------------------------------------

def test_add_functions_default_to_default_client():
    """add_* functions default to client_id='default' when not specified."""
    _install_fake_kbase_indexer()

    with patch("src.tools.thesaurus.ensure_user_collections_once") as mock_ensure, \
         patch("src.tools.thesaurus.index_document"):
        thesaurus.add_thesaurus_entry(
            headword="test",
            # client_id not specified, should default to "default"
        )

    mock_ensure.assert_called_once_with("default")
