"""
Compatibility Layer for mcp-scholar

This module provides compatibility imports for projects migrating from
mcp-scholar's job modules to the unified kbase.jobs module.

Usage in mcp-scholar:
    # Instead of:
    # from src.jobs.retry_manager import RetryManager, categorize_error
    # from src.jobs.processor_pause import ProcessorPauseManager

    # Use:
    from kbase.jobs.compat import RetryManager, categorize_error
    from kbase.jobs.compat import ProcessorPauseManager

This allows gradual migration while keeping mcp-scholar's specialized
queue implementation.
"""

# Re-export retry components
from kbase.jobs.retry import (
    RetryManager,
    RetryConfig,
    RetryableError,
    NonRetryableError,
    categorize_error,
    should_retry_error,
    is_non_retriable_error,
    calculate_delay,
    retryable,
    get_retry_manager,
    DEFAULT_RETRY_CONFIGS,
    NON_RETRIABLE_PATTERNS,
)

# Re-export queue components
from kbase.jobs.queue import (
    JobQueue,
    JobStatus,
    Job,
    get_job_queue,
)

# Re-export pause manager
from kbase.jobs.pause import (
    ProcessorPauseManager,
    get_pause_manager,
)

# Re-export memory monitoring
from kbase.jobs.memory import (
    MemoryMonitor,
    MemoryConfig,
    get_memory_monitor,
    check_memory_available,
    trigger_gc,
)

# Re-export cleanup utilities
from kbase.jobs.cleanup import (
    CleanupConfig,
    JSONLCleanup,
    ProcessMonitorCleanup,
    JobQueueCleanup,
    cleanup_all,
    get_all_statistics,
)

# Re-export enhanced retry
from kbase.jobs.enhanced_retry import (
    ChunkedProcessingRetry,
    ChunkRetryConfig,
    PermanentProcessingFailure,
    RetryContext,
    get_chunked_retry,
    process_with_adaptive_chunking,
    get_retry_statistics,
)

__all__ = [
    # Retry
    "RetryManager",
    "RetryConfig",
    "RetryableError",
    "NonRetryableError",
    "categorize_error",
    "should_retry_error",
    "is_non_retriable_error",
    "calculate_delay",
    "retryable",
    "get_retry_manager",
    "DEFAULT_RETRY_CONFIGS",
    "NON_RETRIABLE_PATTERNS",
    # Queue
    "JobQueue",
    "JobStatus",
    "Job",
    "get_job_queue",
    # Pause
    "ProcessorPauseManager",
    "get_pause_manager",
    # Memory
    "MemoryMonitor",
    "MemoryConfig",
    "get_memory_monitor",
    "check_memory_available",
    "trigger_gc",
    # Cleanup
    "CleanupConfig",
    "JSONLCleanup",
    "ProcessMonitorCleanup",
    "JobQueueCleanup",
    "cleanup_all",
    "get_all_statistics",
    # Enhanced Retry
    "ChunkedProcessingRetry",
    "ChunkRetryConfig",
    "PermanentProcessingFailure",
    "RetryContext",
    "get_chunked_retry",
    "process_with_adaptive_chunking",
    "get_retry_statistics",
]
