"""
Retry mechanism with exponential backoff for handling transient failures.

Provides configurable retry logic with categorization of errors,
exponential backoff with jitter, and both sync and async support.

Based on patterns from mcp-scholar but generalized for reuse.
"""

import time
import random
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retriable_patterns: List[str] = field(default_factory=list)


# Default retry configurations by error category
DEFAULT_RETRY_CONFIGS: Dict[str, RetryConfig] = {
    "api_failures": RetryConfig(
        max_attempts=5,
        base_delay=2.0,
        max_delay=60.0,
        backoff_factor=2.0,
        jitter=True,
        retriable_patterns=[
            "http 5",  # 5xx server errors
            "connectionerror",
            "timeout",
            "requestexception",
            "connection",
            "broken pipe",
            "errno 32",
        ]
    ),
    "memory_failures": RetryConfig(
        max_attempts=3,
        base_delay=30.0,
        max_delay=300.0,
        backoff_factor=2.0,
        jitter=False,
        retriable_patterns=[
            "critical memory usage",
            "memory",
            "out of memory",
            "cuda out of memory",
            "mps out of memory"
        ]
    ),
    "file_failures": RetryConfig(
        max_attempts=4,
        base_delay=5.0,
        max_delay=120.0,
        backoff_factor=1.5,
        jitter=True,
        retriable_patterns=[
            "no such file or directory",
            "errno 2",
            "file not found",
            "permission denied",
            "errno 13"
        ]
    ),
    "processing_failures": RetryConfig(
        max_attempts=3,
        base_delay=10.0,
        max_delay=180.0,
        backoff_factor=2.0,
        jitter=True,
        retriable_patterns=[
            "conversion failed",
            "list index out of range",
            "indexerror",
            "processing error",
        ]
    ),
}

# Non-retriable error patterns - permanent failures that should never be retried
NON_RETRIABLE_PATTERNS = [
    # API schema validation errors
    "is not a valid field",
    "unsupportedparamserror",
    "invalid field",
    # Client errors (4xx except timeout-related)
    "400",  # Bad Request
    "403",  # Forbidden
    "404",  # Not Found
    "409",  # Conflict
    "412",  # Precondition Failed
    # Data/logic errors
    "item does not exist",
    "no such item",
    "invalid item key",
    "item not found",
    # Document errors that won't resolve with retry
    "no pdf attachment found",
    "not a pdf file",
    "corrupted pdf",
    "invalid pdf",
    "document not found",
]


class RetryableError(Exception):
    """Exception for errors that should be retried."""
    def __init__(self, message: str, error_category: str = "general"):
        super().__init__(message)
        self.error_category = error_category


class NonRetryableError(Exception):
    """Exception for errors that should NOT be retried."""
    pass


def is_non_retriable_error(error_message: str) -> bool:
    """
    Check if an error should NOT be retried (permanent failure).

    Args:
        error_message: The error message to check

    Returns:
        True if error is non-retriable, False otherwise
    """
    error_lower = error_message.lower()
    for pattern in NON_RETRIABLE_PATTERNS:
        if pattern.lower() in error_lower:
            return True
    return False


def categorize_error(
    error_message: str,
    configs: Optional[Dict[str, RetryConfig]] = None
) -> Optional[str]:
    """
    Categorize an error message to determine retry strategy.

    Args:
        error_message: The error message to categorize
        configs: Custom retry configurations (uses defaults if None)

    Returns:
        Error category name or None if not retriable
    """
    # Check non-retriable patterns first
    if is_non_retriable_error(error_message):
        return None

    error_lower = error_message.lower()
    configs = configs or DEFAULT_RETRY_CONFIGS

    for category, config in configs.items():
        for pattern in config.retriable_patterns:
            if pattern.lower() in error_lower:
                return category

    return None


def should_retry_error(
    error_message: str,
    configs: Optional[Dict[str, RetryConfig]] = None
) -> bool:
    """Check if an error should be retried."""
    return categorize_error(error_message, configs) is not None


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for retry attempt using exponential backoff.

    Args:
        attempt: Current attempt number (0-based)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    delay = config.base_delay * (config.backoff_factor ** attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        jitter = delay * 0.1 * random.random()  # Up to 10% jitter
        delay += jitter

    return delay


class RetryManager:
    """Manages retry logic for different types of operations."""

    def __init__(
        self,
        configs: Optional[Dict[str, RetryConfig]] = None,
    ):
        """
        Initialize retry manager.

        Args:
            configs: Custom retry configurations (uses defaults if None)
        """
        self.configs = configs or DEFAULT_RETRY_CONFIGS
        self.stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_after_retries": 0,
            "categories": {}
        }

    async def retry_async(
        self,
        func: Callable,
        *args,
        error_context: str = "",
        custom_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> Any:
        """
        Retry an async function with exponential backoff.

        Args:
            func: Async function to retry
            *args: Arguments for the function
            error_context: Context string for logging
            custom_config: Custom retry configuration
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            NonRetryableError: If error is non-retriable
            Exception: The last exception if all retries fail
        """
        last_exception = None
        error_category = None

        # Try once without retry first
        try:
            self.stats["total_attempts"] += 1
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            last_exception = e
            error_message = str(e)
            error_category = categorize_error(error_message, self.configs)

            if not error_category:
                logger.warning(
                    "Non-retriable error",
                    error_context=error_context,
                    error=error_message,
                )
                raise NonRetryableError(f"Non-retriable error: {error_message}")

        # Get retry configuration
        config = custom_config or self.configs.get(
            error_category,
            self.configs.get("api_failures", DEFAULT_RETRY_CONFIGS["api_failures"])
        )

        logger.info(
            "Starting retry sequence",
            error_context=error_context,
            category=error_category,
        )

        # Track category stats
        if error_category not in self.stats["categories"]:
            self.stats["categories"][error_category] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0
            }

        category_stats = self.stats["categories"][error_category]

        # Retry loop
        for attempt in range(1, config.max_attempts):
            delay = calculate_delay(attempt - 1, config)

            logger.info(
                "Retry attempt",
                attempt=attempt,
                max_attempts=config.max_attempts - 1,
                delay=round(delay, 1),
                error_context=error_context,
            )
            await asyncio.sleep(delay)

            try:
                self.stats["total_attempts"] += 1
                category_stats["attempts"] += 1

                result = await func(*args, **kwargs)

                # Success after retry
                self.stats["successful_retries"] += 1
                category_stats["successes"] += 1
                logger.info(
                    "Retry successful",
                    attempt=attempt + 1,
                    error_context=error_context,
                )
                return result

            except Exception as e:
                last_exception = e
                error_message = str(e)

                # Check if error category changed
                new_category = categorize_error(error_message, self.configs)
                if new_category != error_category:
                    logger.warning(
                        "Error category changed",
                        old_category=error_category,
                        new_category=new_category,
                    )
                    if not new_category:
                        break  # Became non-retriable

                logger.warning(
                    "Retry failed",
                    attempt=attempt,
                    error=error_message,
                    error_context=error_context,
                )

        # All retries exhausted
        self.stats["failed_after_retries"] += 1
        category_stats["failures"] += 1

        logger.error(
            "All retries exhausted",
            error_context=error_context,
            final_error=str(last_exception),
        )
        raise last_exception

    def retry_sync(
        self,
        func: Callable,
        *args,
        error_context: str = "",
        custom_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> Any:
        """
        Retry a synchronous function with exponential backoff.

        Args:
            func: Function to retry
            *args: Arguments for the function
            error_context: Context string for logging
            custom_config: Custom retry configuration
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            NonRetryableError: If error is non-retriable
            Exception: The last exception if all retries fail
        """
        last_exception = None
        error_category = None

        # Try once without retry first
        try:
            self.stats["total_attempts"] += 1
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            last_exception = e
            error_message = str(e)
            error_category = categorize_error(error_message, self.configs)

            if not error_category:
                logger.warning(
                    "Non-retriable error",
                    error_context=error_context,
                    error=error_message,
                )
                raise NonRetryableError(f"Non-retriable error: {error_message}")

        # Get retry configuration
        config = custom_config or self.configs.get(
            error_category,
            self.configs.get("api_failures", DEFAULT_RETRY_CONFIGS["api_failures"])
        )

        logger.info(
            "Starting retry sequence",
            error_context=error_context,
            category=error_category,
        )

        # Track category stats
        if error_category not in self.stats["categories"]:
            self.stats["categories"][error_category] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0
            }

        category_stats = self.stats["categories"][error_category]

        # Retry loop
        for attempt in range(1, config.max_attempts):
            delay = calculate_delay(attempt - 1, config)

            logger.info(
                "Retry attempt",
                attempt=attempt,
                max_attempts=config.max_attempts - 1,
                delay=round(delay, 1),
                error_context=error_context,
            )
            time.sleep(delay)

            try:
                self.stats["total_attempts"] += 1
                category_stats["attempts"] += 1

                result = func(*args, **kwargs)

                # Success after retry
                self.stats["successful_retries"] += 1
                category_stats["successes"] += 1
                logger.info(
                    "Retry successful",
                    attempt=attempt + 1,
                    error_context=error_context,
                )
                return result

            except Exception as e:
                last_exception = e
                error_message = str(e)

                # Check if error category changed
                new_category = categorize_error(error_message, self.configs)
                if new_category != error_category:
                    logger.warning(
                        "Error category changed",
                        old_category=error_category,
                        new_category=new_category,
                    )
                    if not new_category:
                        break  # Became non-retriable

                logger.warning(
                    "Retry failed",
                    attempt=attempt,
                    error=error_message,
                    error_context=error_context,
                )

        # All retries exhausted
        self.stats["failed_after_retries"] += 1
        category_stats["failures"] += 1

        logger.error(
            "All retries exhausted",
            error_context=error_context,
            final_error=str(last_exception),
        )
        raise last_exception

    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            "stats": self.stats.copy(),
            "success_rate": (
                self.stats["successful_retries"] /
                max(1, self.stats["total_attempts"])
            ) * 100,
            "timestamp": datetime.now().isoformat()
        }


def retryable(
    error_context: str = "",
    custom_config: Optional[RetryConfig] = None,
):
    """
    Decorator for making functions retryable.

    Args:
        error_context: Context string for logging
        custom_config: Custom retry configuration
    """
    def decorator(func):
        retry_manager = RetryManager()

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                context = error_context or f"{func.__name__}"
                return await retry_manager.retry_async(
                    func, *args,
                    error_context=context,
                    custom_config=custom_config,
                    **kwargs
                )
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                context = error_context or f"{func.__name__}"
                return retry_manager.retry_sync(
                    func, *args,
                    error_context=context,
                    custom_config=custom_config,
                    **kwargs
                )
            return sync_wrapper

    return decorator


# Global retry manager instance
_global_retry_manager: Optional[RetryManager] = None


def get_retry_manager() -> RetryManager:
    """Get the global retry manager instance."""
    global _global_retry_manager
    if _global_retry_manager is None:
        _global_retry_manager = RetryManager()
    return _global_retry_manager
