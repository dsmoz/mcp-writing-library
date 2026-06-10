"""Shared test helpers for writing-library test suite."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mirror main.py: make the project root and vendored kbase importable so tests
# can import kbase.* directly (e.g. kbase.vector.sync_embeddings).
_project_root = Path(__file__).resolve().parent.parent
for _p in (_project_root, _project_root / "vendor"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _make_mock_point(payload: dict):
    """Return a minimal object mimicking a Qdrant ScrollResult point."""
    point = MagicMock()
    point.payload = payload
    return point


def _make_mock_qdrant_client(scroll_result):
    """Helper to produce a mock Qdrant client with a configured scroll response."""
    client = MagicMock()
    client.scroll.return_value = scroll_result
    return client
