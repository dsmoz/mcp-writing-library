"""Test Sentry before_send filter for benign stream exceptions."""

import sys
from pathlib import Path

# Add parent directory to path so imports work in pytest
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_before_send_filters_sdk_noise_loggers():
    """before_send drops SDK noise from known loggers."""
    _SDK_NOISE_LOGGERS = ("mcp.server", "mcp.server.streamable_http_manager")
    _SDK_NOISE_MESSAGES = ("ClientDisconnect", "Received exception from stream:")

    def _before_send(event, hint):
        logger = event.get("logger", "")
        if logger in _SDK_NOISE_LOGGERS:
            msg = event.get("message") or ""
            exc_values = event.get("exception", {}).get("values", [])
            exc_type = exc_values[0].get("type", "") if exc_values else ""
            if any(m in msg for m in _SDK_NOISE_MESSAGES) or exc_type in ("ClientDisconnect", "ClosedResourceError", "BrokenResourceError", "EndOfStream", "ConnectionResetError"):
                return None

        exc_values = event.get("exception", {}).get("values", [])
        benign_exc_types = {"ClosedResourceError", "BrokenResourceError", "EndOfStream", "ConnectionResetError"}
        if any(v.get("type") in benign_exc_types for v in exc_values):
            return None

        return event

    event = {
        "logger": "mcp.server.streamable_http_manager",
        "message": "Received exception from stream: Connection closed"
    }
    result = _before_send(event, {})
    assert result is None


def test_before_send_filters_closed_resource_error():
    """before_send drops ClosedResourceError outside SDK loggers."""
    _SDK_NOISE_LOGGERS = ("mcp.server", "mcp.server.streamable_http_manager")
    _SDK_NOISE_MESSAGES = ("ClientDisconnect", "Received exception from stream:")

    def _before_send(event, hint):
        logger = event.get("logger", "")
        if logger in _SDK_NOISE_LOGGERS:
            msg = event.get("message") or ""
            exc_values = event.get("exception", {}).get("values", [])
            exc_type = exc_values[0].get("type", "") if exc_values else ""
            if any(m in msg for m in _SDK_NOISE_MESSAGES) or exc_type in ("ClientDisconnect", "ClosedResourceError", "BrokenResourceError", "EndOfStream", "ConnectionResetError"):
                return None

        exc_values = event.get("exception", {}).get("values", [])
        benign_exc_types = {"ClosedResourceError", "BrokenResourceError", "EndOfStream", "ConnectionResetError"}
        if any(v.get("type") in benign_exc_types for v in exc_values):
            return None

        return event

    event = {
        "logger": "some.other.logger",
        "exception": {"values": [{"type": "ClosedResourceError"}]}
    }
    result = _before_send(event, {})
    assert result is None


def test_before_send_filters_end_of_stream():
    """before_send drops EndOfStream exceptions."""
    _SDK_NOISE_LOGGERS = ("mcp.server", "mcp.server.streamable_http_manager")
    _SDK_NOISE_MESSAGES = ("ClientDisconnect", "Received exception from stream:")

    def _before_send(event, hint):
        logger = event.get("logger", "")
        if logger in _SDK_NOISE_LOGGERS:
            msg = event.get("message") or ""
            exc_values = event.get("exception", {}).get("values", [])
            exc_type = exc_values[0].get("type", "") if exc_values else ""
            if any(m in msg for m in _SDK_NOISE_MESSAGES) or exc_type in ("ClientDisconnect", "ClosedResourceError", "BrokenResourceError", "EndOfStream", "ConnectionResetError"):
                return None

        exc_values = event.get("exception", {}).get("values", [])
        benign_exc_types = {"ClosedResourceError", "BrokenResourceError", "EndOfStream", "ConnectionResetError"}
        if any(v.get("type") in benign_exc_types for v in exc_values):
            return None

        return event

    event = {
        "exception": {"values": [{"type": "EndOfStream"}]}
    }
    result = _before_send(event, {})
    assert result is None


def test_before_send_preserves_real_errors():
    """before_send preserves real application errors."""
    _SDK_NOISE_LOGGERS = ("mcp.server", "mcp.server.streamable_http_manager")
    _SDK_NOISE_MESSAGES = ("ClientDisconnect", "Received exception from stream:")

    def _before_send(event, hint):
        logger = event.get("logger", "")
        if logger in _SDK_NOISE_LOGGERS:
            msg = event.get("message") or ""
            exc_values = event.get("exception", {}).get("values", [])
            exc_type = exc_values[0].get("type", "") if exc_values else ""
            if any(m in msg for m in _SDK_NOISE_MESSAGES) or exc_type in ("ClientDisconnect", "ClosedResourceError", "BrokenResourceError", "EndOfStream", "ConnectionResetError"):
                return None

        exc_values = event.get("exception", {}).get("values", [])
        benign_exc_types = {"ClosedResourceError", "BrokenResourceError", "EndOfStream", "ConnectionResetError"}
        if any(v.get("type") in benign_exc_types for v in exc_values):
            return None

        return event

    event = {
        "logger": "myapp.handler",
        "exception": {"values": [{"type": "ValueError", "value": "Invalid input"}]}
    }
    result = _before_send(event, {})
    assert result is not None
    assert result == event
