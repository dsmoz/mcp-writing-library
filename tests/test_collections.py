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
