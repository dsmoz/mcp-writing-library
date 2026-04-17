"""Tests for collection management."""
import pytest
from unittest.mock import patch, MagicMock


def test_get_collection_names_default_client():
    """Default client_id produces 'default_' prefixed per-user collections."""
    from src.tools.collections import get_collection_names
    names = get_collection_names()
    assert names["passages"] == "default_writing_passages"
    assert names["terms"] == "default_writing_terms"
    assert names["style_profiles"] == "default_writing_style_profiles"
    # Core collections are not prefixed
    assert names["rubrics"] == "writing_rubrics"
    assert names["thesaurus"] == "writing_thesaurus"


def test_get_collection_names_custom_client():
    """Custom client_id produces correctly prefixed per-user collections."""
    from src.tools.collections import get_collection_names
    names = get_collection_names("acme-corp")
    assert names["passages"] == "acme-corp_writing_passages"
    assert names["terms"] == "acme-corp_writing_terms"
    assert names["style_profiles"] == "acme-corp_writing_style_profiles"
    # Core collections unchanged
    assert names["rubrics"] == "writing_rubrics"


def test_ensure_user_collections_once_creates_all_filter_indexes():
    """Regression for Sentry MCP-WRITING-5 and MCP-WRITING-1.

    search_passages filters on doc_type/language/domain/rubric_section and
    load_style_profile filters on name — Qdrant returns HTTP 400 when the
    keyword index is missing. Every filtered field must have an index
    created on first use.
    """
    from src.tools import collections as collections_mod

    collections_mod._initialized_clients.discard("idxtest")

    mock_client = MagicMock()

    with patch.object(collections_mod, "setup_user_collections") as setup:
        setup.return_value = {}
        with patch("kbase.vector.sync_client.get_qdrant_client", return_value=mock_client):
            collections_mod.ensure_user_collections_once("idxtest")

    indexed = {
        (call.kwargs["collection_name"], call.kwargs["field_name"])
        for call in mock_client.create_payload_index.call_args_list
    }

    passages = "idxtest_writing_passages"
    profiles = "idxtest_writing_style_profiles"

    # MCP-WRITING-5
    assert (passages, "doc_type") in indexed
    assert (passages, "language") in indexed
    assert (passages, "domain") in indexed
    assert (passages, "rubric_section") in indexed
    # Existing entry_type index must still be created
    assert (passages, "entry_type") in indexed

    # MCP-WRITING-1
    assert (profiles, "name") in indexed
    assert (profiles, "channel") in indexed


def test_ensure_user_collections_once_tolerates_existing_indexes():
    """Creating an index that already exists must not raise."""
    from src.tools import collections as collections_mod

    collections_mod._initialized_clients.discard("dup")

    mock_client = MagicMock()
    mock_client.create_payload_index.side_effect = Exception("Index already exists (409)")

    with patch.object(collections_mod, "setup_user_collections", return_value={}):
        with patch("kbase.vector.sync_client.get_qdrant_client", return_value=mock_client):
            collections_mod.ensure_user_collections_once("dup")  # must not raise
