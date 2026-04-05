"""
Enhanced Retry Logic with Chunked Processing Integration.

This module extends the retry system with intelligent retry strategies
specifically designed for chunked processing failures. It implements:

1. Automatic retry with reduced chunk sizes on memory failures
2. Adaptive chunk sizing based on failure patterns
3. Pre-emptive memory monitoring to prevent failures
4. Graceful degradation for problematic documents

Based on mcp-scholar's enhanced_retry.py.
"""

import asyncio
import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple

import structlog

from kbase.jobs.retry import RetryManager, NonRetryableError
from kbase.jobs.memory import MemoryMonitor, MemoryConfig, get_memory_monitor

logger = structlog.get_logger(__name__)


class PermanentProcessingFailure(Exception):
    """
    Exception raised when a document permanently fails after all retry attempts.

    This indicates graceful degradation - the document should be skipped
    and marked as permanently failed rather than blocking the queue.

    Attributes:
        item_id: Unique identifier for the item
        attempts: Number of retry attempts made
        chunk_sizes_tried: List of chunk sizes attempted
        last_error: The final error message
        file_size_mb: Size of the file in MB
    """

    def __init__(
        self,
        item_id: str,
        attempts: int,
        chunk_sizes_tried: List[int],
        last_error: str,
        file_size_mb: float = 0.0
    ):
        self.item_id = item_id
        self.attempts = attempts
        self.chunk_sizes_tried = chunk_sizes_tried
        self.last_error = last_error
        self.file_size_mb = file_size_mb

        message = (
            f"Document {item_id} ({file_size_mb:.1f}MB) permanently failed after {attempts} attempts. "
            f"Chunk sizes tried: {chunk_sizes_tried}MB. Last error: {last_error}"
        )
        super().__init__(message)


@dataclass
class ChunkRetryConfig:
    """Configuration for chunked processing retry strategies."""

    # Initial chunk size (MB)
    initial_chunk_size_mb: int = 15

    # Minimum chunk size before giving up (MB)
    min_chunk_size_mb: int = 2

    # Chunk size reduction factor on retry (e.g., 0.5 = reduce by half)
    reduction_factor: float = 0.5

    # Maximum retry attempts with reduced chunk size
    max_chunk_retries: int = 3

    # Memory threshold for pre-emptive action (% of available memory)
    memory_warning_threshold: float = 75.0

    # Memory threshold for critical action (% of available memory)
    memory_critical_threshold: float = 85.0

    # Enable aggressive garbage collection
    enable_aggressive_gc: bool = True

    # Enable pre-emptive memory monitoring
    enable_preemptive_monitoring: bool = True

    # Enable graceful degradation (skip on permanent failure)
    enable_graceful_degradation: bool = True

    # Delay between retries (seconds)
    retry_delay_seconds: float = 5.0


@dataclass
class RetryContext:
    """Context information for retry operations."""

    item_id: str
    attempt_number: int
    chunk_size_mb: int
    file_size_mb: float
    error_message: str
    memory_usage_percent: float
    previous_attempts: List[Dict] = field(default_factory=list)


class ChunkedProcessingRetry:
    """Enhanced retry logic for chunked processing with adaptive chunk sizing."""

    def __init__(self, config: Optional[ChunkRetryConfig] = None):
        """
        Initialize chunked processing retry manager.

        Args:
            config: Retry configuration
        """
        self.config = config or ChunkRetryConfig()
        self.retry_manager = RetryManager()

        # Create memory config from chunk retry config
        memory_config = MemoryConfig(
            warning_threshold_percent=self.config.memory_warning_threshold,
            critical_threshold_percent=self.config.memory_critical_threshold,
            enable_aggressive_gc=self.config.enable_aggressive_gc,
            enable_preemptive_monitoring=self.config.enable_preemptive_monitoring,
        )
        self.memory_monitor = MemoryMonitor(memory_config)

        # Track retry history for learning
        self.retry_history: Dict[str, List[Dict]] = {}

    def _is_memory_error(self, error: Exception) -> bool:
        """Check if error is memory-related."""
        error_str = str(error).lower()
        memory_keywords = [
            "memory", "memoryerror", "out of memory", "oom",
            "allocation failed", "cannot allocate", "mps out of memory",
            "cuda out of memory"
        ]

        return any(keyword in error_str for keyword in memory_keywords)

    def _calculate_reduced_chunk_size(self, current_size_mb: int, attempt: int) -> int:
        """
        Calculate reduced chunk size for retry attempt.

        Args:
            current_size_mb: Current chunk size in MB
            attempt: Retry attempt number (1-based)

        Returns:
            Reduced chunk size in MB
        """
        # Reduce chunk size exponentially
        reduced_size = int(current_size_mb * (self.config.reduction_factor ** attempt))

        # Ensure minimum chunk size
        reduced_size = max(reduced_size, self.config.min_chunk_size_mb)

        return reduced_size

    def _should_retry_with_smaller_chunks(
        self,
        error: Exception,
        current_chunk_size_mb: int,
        attempt: int
    ) -> bool:
        """
        Determine if we should retry with smaller chunks.

        Args:
            error: The exception that occurred
            current_chunk_size_mb: Current chunk size
            attempt: Current retry attempt number

        Returns:
            True if should retry with smaller chunks
        """
        # Check if it's a memory error
        if not self._is_memory_error(error):
            return False

        # Check if we've exceeded max retries
        if attempt >= self.config.max_chunk_retries:
            return False

        # Check if chunk size is already at minimum
        if current_chunk_size_mb <= self.config.min_chunk_size_mb:
            return False

        return True

    def _get_adaptive_chunk_size(self, item_id: str, file_size_mb: float) -> int:
        """
        Get adaptive chunk size based on previous retry history.

        Args:
            item_id: Unique item identifier
            file_size_mb: File size in MB

        Returns:
            Recommended chunk size in MB
        """
        # Check if we have retry history for this item
        if item_id in self.retry_history:
            # Find the smallest successful chunk size
            successful_sizes = [
                entry["chunk_size_mb"]
                for entry in self.retry_history[item_id]
                if entry.get("success", False)
            ]

            if successful_sizes:
                # Use the smallest successful size as starting point
                recommended = min(successful_sizes)
                # Ensure chunk size isn't larger than file
                return min(recommended, int(file_size_mb * 0.8))

        # Default logic based on file size
        if file_size_mb > 50:
            return 10  # Large files: 10MB chunks
        elif file_size_mb > 25:
            return 15  # Medium files: 15MB chunks
        elif file_size_mb > 10:
            return int(file_size_mb * 0.8)  # 80% of file size for small files
        else:
            # For very small files, use half the file size or minimum, whichever is larger
            return max(int(file_size_mb * 0.5), self.config.min_chunk_size_mb)

    def _record_retry_attempt(
        self,
        item_id: str,
        chunk_size_mb: int,
        success: bool,
        error_message: Optional[str] = None
    ):
        """Record retry attempt in history for future learning."""
        if item_id not in self.retry_history:
            self.retry_history[item_id] = []

        self.retry_history[item_id].append({
            "timestamp": time.time(),
            "chunk_size_mb": chunk_size_mb,
            "success": success,
            "error_message": error_message
        })

    async def process_with_chunked_retry(
        self,
        processor_func: Callable,
        item_id: str,
        file_path: Optional[str] = None,
        file_size_mb: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Process with automatic retry using reduced chunk sizes on failure.

        Args:
            processor_func: Async function to process chunks.
                           Signature: async (item_id, chunk_size_mb, **kwargs) -> result
            item_id: Unique item identifier
            file_path: Optional path to file (for size calculation)
            file_size_mb: Optional file size in MB (if file_path not provided)
            **kwargs: Additional arguments for processor

        Returns:
            Processing result

        Raises:
            NonRetryableError: If error is not retriable
            PermanentProcessingFailure: If all retries exhausted (graceful degradation)
            Exception: If all retries fail and graceful degradation disabled
        """
        # Get file size
        if file_size_mb is None and file_path:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        elif file_size_mb is None:
            file_size_mb = 0.0

        # Get adaptive initial chunk size
        initial_chunk_size = self._get_adaptive_chunk_size(item_id, file_size_mb)

        logger.info(
            "Starting chunked processing",
            item_id=item_id,
            file_size_mb=round(file_size_mb, 1),
            initial_chunk_size_mb=initial_chunk_size
        )

        # Set memory baseline
        if self.config.enable_preemptive_monitoring:
            self.memory_monitor.set_baseline()

        attempt = 0
        current_chunk_size = initial_chunk_size
        last_error = None

        while attempt < self.config.max_chunk_retries:
            try:
                # Pre-emptive memory check
                if self.config.enable_preemptive_monitoring:
                    # Estimate memory usage (rough: 5x chunk size for overhead)
                    estimated_memory = current_chunk_size * 5

                    can_proceed, mem_msg = self.memory_monitor.check_memory_before_processing(
                        estimated_memory
                    )

                    if not can_proceed:
                        logger.warning("Pre-emptive memory check failed", message=mem_msg)
                        # Trigger GC and try with smaller chunks
                        self.memory_monitor.trigger_garbage_collection()

                        if current_chunk_size > self.config.min_chunk_size_mb:
                            attempt += 1
                            current_chunk_size = self._calculate_reduced_chunk_size(
                                initial_chunk_size, attempt
                            )
                            logger.info(
                                "Reducing chunk size due to memory constraints",
                                new_chunk_size_mb=current_chunk_size
                            )
                            continue
                        else:
                            raise NonRetryableError(
                                f"Insufficient memory even with minimum chunk size: {mem_msg}"
                            )

                    if "Warning" in mem_msg:
                        logger.warning("Memory warning", message=mem_msg)
                        self.memory_monitor.trigger_garbage_collection()

                logger.info(
                    "Processing attempt",
                    attempt=attempt + 1,
                    chunk_size_mb=current_chunk_size
                )

                # Process with current chunk size
                result = await processor_func(
                    item_id=item_id,
                    chunk_size_mb=current_chunk_size,
                    **kwargs
                )

                # Success!
                logger.info(
                    "Successfully processed",
                    item_id=item_id,
                    chunk_size_mb=current_chunk_size
                )

                # Record successful attempt
                self._record_retry_attempt(item_id, current_chunk_size, success=True)

                return result

            except NonRetryableError:
                # Don't retry non-retriable errors
                raise

            except Exception as e:
                last_error = e
                error_str = str(e)

                logger.warning(
                    "Attempt failed",
                    attempt=attempt + 1,
                    error=error_str
                )

                # Record failed attempt
                self._record_retry_attempt(
                    item_id, current_chunk_size, success=False, error_message=error_str
                )

                # Check if we should retry with smaller chunks
                if self._should_retry_with_smaller_chunks(e, current_chunk_size, attempt + 1):
                    attempt += 1

                    # Calculate new chunk size
                    new_chunk_size = self._calculate_reduced_chunk_size(
                        initial_chunk_size, attempt
                    )

                    logger.info(
                        "Memory error detected, reducing chunk size",
                        old_chunk_size_mb=current_chunk_size,
                        new_chunk_size_mb=new_chunk_size
                    )

                    # Trigger garbage collection
                    self.memory_monitor.trigger_garbage_collection()

                    # Update chunk size for next attempt
                    current_chunk_size = new_chunk_size

                    # Wait a bit before retry
                    await asyncio.sleep(self.config.retry_delay_seconds)

                else:
                    # Not a retriable error or exhausted retries
                    logger.warning(
                        "Cannot retry",
                        attempt=attempt,
                        chunk_size_mb=current_chunk_size,
                        is_memory_error=self._is_memory_error(e)
                    )

                    # Check if we exhausted retries vs non-retriable error
                    if self._is_memory_error(e) and (attempt + 1) >= self.config.max_chunk_retries:
                        # Exhausted all retry attempts
                        if self.config.enable_graceful_degradation:
                            # Get all chunk sizes tried
                            chunk_sizes_tried = []
                            if item_id in self.retry_history:
                                chunk_sizes_tried = [
                                    entry["chunk_size_mb"]
                                    for entry in self.retry_history[item_id]
                                ]

                            logger.warning(
                                "Graceful degradation: skipping item",
                                item_id=item_id,
                                attempts=attempt + 1
                            )

                            raise PermanentProcessingFailure(
                                item_id=item_id,
                                attempts=attempt + 1,
                                chunk_sizes_tried=chunk_sizes_tried,
                                last_error=str(last_error),
                                file_size_mb=file_size_mb
                            )
                        else:
                            # No graceful degradation: raise generic exception
                            error_msg = f"Failed after {attempt + 1} chunked processing retries. Last error: {last_error}"
                            logger.error(error_msg)
                            raise Exception(error_msg)
                    else:
                        # Non-retriable error, re-raise original
                        raise

        # All retries exhausted (shouldn't reach here, but just in case)
        if self.config.enable_graceful_degradation:
            chunk_sizes_tried = []
            if item_id in self.retry_history:
                chunk_sizes_tried = [
                    entry["chunk_size_mb"]
                    for entry in self.retry_history[item_id]
                ]

            logger.warning(
                "Graceful degradation: skipping item after exhausting retries",
                item_id=item_id,
                attempts=attempt
            )

            raise PermanentProcessingFailure(
                item_id=item_id,
                attempts=attempt,
                chunk_sizes_tried=chunk_sizes_tried,
                last_error=str(last_error) if last_error else "Unknown error",
                file_size_mb=file_size_mb
            )
        else:
            error_msg = f"Failed after {attempt} chunked processing retries. Last error: {last_error}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def get_retry_stats(self, item_id: Optional[str] = None) -> Dict:
        """
        Get retry statistics.

        Args:
            item_id: Optional item ID to filter by

        Returns:
            Dict with retry statistics
        """
        if item_id:
            history = self.retry_history.get(item_id, [])

            return {
                "item_id": item_id,
                "total_attempts": len(history),
                "successful_attempts": sum(1 for h in history if h["success"]),
                "failed_attempts": sum(1 for h in history if not h["success"]),
                "chunk_sizes_tried": list(set(h["chunk_size_mb"] for h in history)),
                "history": history
            }
        else:
            # Overall stats
            all_attempts = sum(len(h) for h in self.retry_history.values())
            all_successes = sum(
                sum(1 for attempt in h if attempt["success"])
                for h in self.retry_history.values()
            )

            return {
                "total_items": len(self.retry_history),
                "total_attempts": all_attempts,
                "successful_attempts": all_successes,
                "failed_attempts": all_attempts - all_successes,
                "success_rate": (all_successes / all_attempts * 100) if all_attempts > 0 else 0
            }

    def clear_history(self, item_id: Optional[str] = None) -> None:
        """
        Clear retry history.

        Args:
            item_id: Optional item ID to clear (clears all if None)
        """
        if item_id:
            self.retry_history.pop(item_id, None)
        else:
            self.retry_history.clear()


# Global instance for convenience
_enhanced_retry: Optional[ChunkedProcessingRetry] = None


def get_chunked_retry(config: Optional[ChunkRetryConfig] = None) -> ChunkedProcessingRetry:
    """
    Get the global chunked retry instance.

    Args:
        config: Optional configuration (only used if creating new instance)

    Returns:
        ChunkedProcessingRetry instance
    """
    global _enhanced_retry

    if _enhanced_retry is None:
        _enhanced_retry = ChunkedProcessingRetry(config)

    return _enhanced_retry


# Helper functions for easy integration
async def process_with_adaptive_chunking(
    processor_func: Callable,
    item_id: str,
    file_path: Optional[str] = None,
    file_size_mb: Optional[float] = None,
    **kwargs
) -> Any:
    """
    Convenience function for processing with adaptive chunking and retry.

    Args:
        processor_func: Processing function
        item_id: Unique item identifier
        file_path: Path to file (optional)
        file_size_mb: File size in MB (optional)
        **kwargs: Additional processing arguments

    Returns:
        Processing result
    """
    retry = get_chunked_retry()
    return await retry.process_with_chunked_retry(
        processor_func,
        item_id,
        file_path=file_path,
        file_size_mb=file_size_mb,
        **kwargs
    )


def get_retry_statistics(item_id: Optional[str] = None) -> Dict:
    """
    Get retry statistics.

    Args:
        item_id: Optional item ID to filter stats

    Returns:
        Retry statistics dictionary
    """
    retry = get_chunked_retry()
    return retry.get_retry_stats(item_id)
