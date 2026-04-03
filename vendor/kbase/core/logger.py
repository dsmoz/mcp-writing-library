"""
Structured logging utility for kbase-core.

Provides a configured structlog logger with consistent formatting
across all kbase modules.
"""

import logging
import sys
from typing import Optional

import structlog


def configure_structlog(
    log_level: int = logging.INFO,
    output_file: Optional[str] = None,
    json_format: bool = False,
) -> None:
    """
    Configure structlog with standard processors.

    Args:
        log_level: Logging level (default: INFO)
        output_file: Optional file path for log output
        json_format: If True, output JSON format; otherwise use console renderer
    """
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Determine output target
    if output_file:
        log_output = open(output_file, "a")
    else:
        log_output = sys.stderr

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=log_output),
        cache_logger_on_first_use=True,
    )


# Configure structlog on module import with defaults
configure_structlog()


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured structlog logger for the given module name.

    Args:
        name: The name of the module (typically __name__)

    Returns:
        A configured structlog BoundLogger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing document", doc_id="123")
    """
    return structlog.get_logger(name)


__all__ = ["get_logger", "configure_structlog"]
