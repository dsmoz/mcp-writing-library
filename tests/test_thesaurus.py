import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock


def test_get_collection_names_includes_thesaurus():
    from src.tools.collections import get_collection_names
    names = get_collection_names()
    assert "thesaurus" in names
    assert names["thesaurus"] == "writing_thesaurus"
