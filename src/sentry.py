"""Sentry error reporting helper for mcp-writing-library tools."""
import sentry_sdk


def capture_tool_error(exc: Exception, tool_name: str, **context) -> None:
    """Capture a tool execution error to Sentry with structured context.

    No-ops silently when Sentry is not initialised (no SENTRY_DSN set).

    Args:
        exc: The exception to capture.
        tool_name: Name of the MCP tool that failed (e.g. "add_passage").
        **context: Additional key/value pairs attached as Sentry extras
                   (e.g. client_id="default", domain="health").
    """
    with sentry_sdk.new_scope() as scope:
        scope.set_tag("tool", tool_name)
        for key, value in context.items():
            scope.set_extra(key, value)
        sentry_sdk.capture_exception(exc)
