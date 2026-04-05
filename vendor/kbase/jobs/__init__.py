"""
Unified Job Processing System

Provides file-based job queue management with retry logic, timeout handling,
memory management, and cleanup utilities. Works with both async and sync
processing patterns.

This module is designed to be used by multiple MCP servers (mcp-cerebellum,
mcp-scholar, etc.) while maintaining backwards compatibility.

Usage:
    from kbase.jobs import JobQueue, BaseJobProcessor, RetryManager

    # Create queue with custom directory
    queue = JobQueue(queue_dir=".jobs")

    # Add a job
    job_id = queue.add_job(
        job_type="process_document",
        item_key="doc-123",
        collection="my_collection",
        custom_data={"key": "value"}
    )

    # Process jobs with custom handler
    class MyProcessor(BaseJobProcessor):
        async def handle_job(self, job):
            if job["type"] == "process_document":
                return await self.process_document(job)

    processor = MyProcessor(queue)
    await processor.run()

Advanced Features:
    # Pause/resume processing
    from kbase.jobs import ProcessorPauseManager
    pause_mgr = ProcessorPauseManager()
    pause_mgr.pause_processor(minutes=30)

    # Memory monitoring
    from kbase.jobs import MemoryMonitor, check_memory_available
    can_proceed, msg = check_memory_available(estimated_usage_mb=500)

    # Cleanup old records
    from kbase.jobs import cleanup_all
    cleanup_all(queue_dir=".jobs", force=True)

    # Chunked processing with adaptive retry
    from kbase.jobs import ChunkedProcessingRetry, process_with_adaptive_chunking
    result = await process_with_adaptive_chunking(process_func, item_id, file_path)
"""

from kbase.jobs.queue import (
    JobQueue,
    JobStatus,
    Job,
    get_job_queue,
)

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

from kbase.jobs.processor import (
    BaseJobProcessor,
    JobHandler,
    ProcessorConfig,
    SimpleJobProcessor,
    DEFAULT_JOB_TIMEOUTS,
    DEFAULT_MEMORY_LIMITS,
)

from kbase.jobs.pause import (
    ProcessorPauseManager,
    get_pause_manager,
)

from kbase.jobs.memory import (
    MemoryMonitor,
    MemoryConfig,
    get_memory_monitor,
    check_memory_available,
    trigger_gc,
)

from kbase.jobs.cleanup import (
    CleanupConfig,
    JSONLCleanup,
    ProcessMonitorCleanup,
    JobQueueCleanup,
    cleanup_all,
    get_all_statistics,
    # Compatibility functions for mcp-scholar
    perform_process_monitor_cleanup,
    get_process_monitor_statistics,
    restore_process_monitor_backup,
)

from kbase.jobs.enhanced_retry import (
    ChunkedProcessingRetry,
    ChunkRetryConfig,
    PermanentProcessingFailure,
    RetryContext,
    get_chunked_retry,
    process_with_adaptive_chunking,
    get_retry_statistics,
)

# Performance Optimizer (Phase 3)
from kbase.jobs.performance_optimizer import (
    OptimizationConfig,
    MemoryPool,
    PerformanceOptimizer,
    create_performance_optimizer,
)

__all__ = [
    # Queue
    "JobQueue",
    "JobStatus",
    "Job",
    "get_job_queue",
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
    # Processor
    "BaseJobProcessor",
    "JobHandler",
    "ProcessorConfig",
    "SimpleJobProcessor",
    "DEFAULT_JOB_TIMEOUTS",
    "DEFAULT_MEMORY_LIMITS",
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
    "perform_process_monitor_cleanup",
    "get_process_monitor_statistics",
    "restore_process_monitor_backup",
    # Enhanced Retry
    "ChunkedProcessingRetry",
    "ChunkRetryConfig",
    "PermanentProcessingFailure",
    "RetryContext",
    "get_chunked_retry",
    "process_with_adaptive_chunking",
    "get_retry_statistics",
    # Performance Optimizer (Phase 3)
    "OptimizationConfig",
    "MemoryPool",
    "PerformanceOptimizer",
    "create_performance_optimizer",
]
