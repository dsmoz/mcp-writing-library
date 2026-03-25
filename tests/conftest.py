"""Shared test helpers for writing-library test suite."""
from unittest.mock import MagicMock


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
