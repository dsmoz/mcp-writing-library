"""
Performance Optimizer for Job Processing

This module provides performance optimizations for job queue processing:

1. Parallel Processing: Process independent documents concurrently with controlled concurrency
2. Memory Pooling: Reuse frequently allocated objects to reduce GC pressure
3. Configurable Memory Limits: Per-document-type memory limits to prevent OOM errors

Features:
- Automatic detection of independent jobs for parallel execution
- Memory pool for embeddings, vectors, and temporary buffers
- Dynamic concurrency adjustment based on memory pressure
- Per-job-type memory budgets
- Resource tracking and statistics

Usage:
    from kbase.jobs import PerformanceOptimizer, OptimizationConfig

    # Create optimizer with custom config
    config = OptimizationConfig(
        max_parallel_jobs=3,
        memory_pool_size_mb=500,
        enable_parallel_processing=True
    )

    optimizer = PerformanceOptimizer(config)

    # Process jobs in parallel
    results = await optimizer.process_jobs_parallel(jobs, processor_func)

    # Get statistics
    stats = optimizer.get_statistics()
"""

import asyncio
import gc
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""

    # Parallel Processing
    max_parallel_jobs: int = 3  # Maximum concurrent jobs
    min_parallel_jobs: int = 1  # Minimum concurrent jobs (when memory-constrained)
    enable_parallel_processing: bool = True  # Enable/disable parallel processing

    # Memory Management
    memory_pool_size_mb: int = 500  # Size of reusable memory pool
    memory_warning_threshold: float = 75.0  # % memory to trigger warnings
    memory_critical_threshold: float = 85.0  # % memory to reduce parallelism
    emergency_threshold: float = 92.0  # % memory to pause processing

    # Per-Document-Type Memory Limits (MB)
    memory_limit_small_doc: int = 100  # < 10MB files
    memory_limit_medium_doc: int = 300  # 10-50MB files
    memory_limit_large_doc: int = 800  # > 50MB files
    memory_limit_pdf: int = 500  # PDF processing
    memory_limit_metadata: int = 50  # Metadata updates

    # Resource Monitoring
    monitor_interval_seconds: float = 5.0  # How often to check memory
    enable_resource_tracking: bool = True  # Track per-job resource usage

    # Memory Pool Configuration
    pool_preallocate: bool = True  # Pre-allocate pool on startup
    pool_max_item_age_seconds: float = 300.0  # Max age before eviction (5 min)
    pool_cleanup_interval_seconds: float = 60.0  # How often to clean pool


@dataclass
class MemoryPoolStats:
    """Statistics for the memory pool."""

    pool_sizes: Dict[str, int]
    hits: int
    misses: int
    evictions: int
    allocations: int
    hit_rate: float


class MemoryPool:
    """
    Memory pool for frequently allocated objects.

    Reduces memory allocation overhead and GC pressure by reusing objects.
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize the memory pool.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.pools: Dict[str, List[Any]] = {
            "embeddings": [],  # Embedding vectors
            "buffers": [],  # Text buffers
            "arrays": [],  # NumPy arrays
        }
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "allocations": 0,
        }
        self._item_metadata: Dict[int, float] = {}  # Track allocation time
        self._last_cleanup = time.time()

        if config.pool_preallocate:
            self._preallocate()

    def _preallocate(self) -> None:
        """Pre-allocate pool items to reduce initial allocation overhead."""
        print("Pre-allocating memory pool...", file=sys.stderr)

        # Pre-allocate embedding vectors (768D for text-embedding-3-small)
        for _ in range(10):
            self.pools["embeddings"].append(np.zeros(768, dtype=np.float32))

        # Pre-allocate text buffers
        for _ in range(20):
            self.pools["buffers"].append(bytearray(1024 * 1024))  # 1MB buffers

        # Pre-allocate arrays
        for _ in range(10):
            self.pools["arrays"].append(np.zeros(1000, dtype=np.float32))

        print(
            f"Memory pool pre-allocated: {len(self.pools['embeddings'])} embeddings, "
            f"{len(self.pools['buffers'])} buffers, {len(self.pools['arrays'])} arrays",
            file=sys.stderr,
        )

    def get_embedding_vector(self, size: int = 768) -> np.ndarray:
        """
        Get an embedding vector from pool or allocate new one.

        Args:
            size: Vector size (default: 768 for text-embedding-3-small)

        Returns:
            NumPy array for embedding
        """
        pool = self.pools["embeddings"]

        # Try to reuse from pool
        for i, vec in enumerate(pool):
            if len(vec) == size:
                self._stats["hits"] += 1
                return pool.pop(i)

        # Not in pool, allocate new
        self._stats["misses"] += 1
        self._stats["allocations"] += 1
        return np.zeros(size, dtype=np.float32)

    def return_embedding_vector(self, vec: np.ndarray) -> None:
        """
        Return embedding vector to pool for reuse.

        Args:
            vec: Vector to return to pool
        """
        if len(self.pools["embeddings"]) < 50:  # Max 50 cached vectors
            # Reset to zeros for reuse
            vec.fill(0)
            self.pools["embeddings"].append(vec)
            self._item_metadata[id(vec)] = time.time()
        else:
            # Pool full, allow GC
            self._stats["evictions"] += 1

    def get_buffer(self, size: int = 1024 * 1024) -> bytearray:
        """
        Get a text buffer from pool or allocate new one.

        Args:
            size: Buffer size in bytes (default: 1MB)

        Returns:
            Bytearray buffer
        """
        pool = self.pools["buffers"]

        # Try to reuse from pool (allow 10% size variance)
        for i, buf in enumerate(pool):
            if abs(len(buf) - size) / size < 0.1:
                self._stats["hits"] += 1
                return pool.pop(i)

        # Not in pool, allocate new
        self._stats["misses"] += 1
        self._stats["allocations"] += 1
        return bytearray(size)

    def return_buffer(self, buf: bytearray) -> None:
        """
        Return buffer to pool for reuse.

        Args:
            buf: Buffer to return to pool
        """
        if len(self.pools["buffers"]) < 30:  # Max 30 cached buffers
            # Clear buffer for reuse
            buf[:] = b"\x00" * len(buf)
            self.pools["buffers"].append(buf)
            self._item_metadata[id(buf)] = time.time()
        else:
            self._stats["evictions"] += 1

    def cleanup_expired_items(self) -> None:
        """Remove items older than max_age from pool."""
        current_time = time.time()
        max_age = self.config.pool_max_item_age_seconds

        for pool_name, pool in self.pools.items():
            expired_indices = []
            for i, item in enumerate(pool):
                item_id = id(item)
                if item_id in self._item_metadata:
                    age = current_time - self._item_metadata[item_id]
                    if age > max_age:
                        expired_indices.append(i)

            # Remove expired items (reverse order to maintain indices)
            for i in reversed(expired_indices):
                pool.pop(i)
                self._stats["evictions"] += 1

        self._last_cleanup = current_time

    def get_stats(self) -> MemoryPoolStats:
        """Get pool statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / max(1, total_requests)

        return MemoryPoolStats(
            pool_sizes={name: len(pool) for name, pool in self.pools.items()},
            hits=self._stats["hits"],
            misses=self._stats["misses"],
            evictions=self._stats["evictions"],
            allocations=self._stats["allocations"],
            hit_rate=hit_rate,
        )

    def clear(self) -> None:
        """Clear all pools."""
        for pool in self.pools.values():
            pool.clear()
        self._item_metadata.clear()


class PerformanceOptimizer:
    """
    Performance optimizer for job queue processing.

    Provides:
    - Parallel processing of independent jobs
    - Memory pooling for frequent allocations
    - Configurable memory limits per job type
    - Dynamic concurrency adjustment
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize the performance optimizer.

        Args:
            config: Optional optimization configuration (uses defaults if not provided)
        """
        self.config = config or OptimizationConfig()
        self.memory_pool = MemoryPool(self.config)

        # Parallelism state
        self.current_parallelism = self.config.max_parallel_jobs
        self.active_jobs = 0
        self.parallelism_history: List[Dict[str, Any]] = []

        # Resource tracking
        self.job_stats = {
            "total_processed": 0,
            "parallel_processed": 0,
            "sequential_processed": 0,
            "memory_constrained_count": 0,
            "errors": 0,
        }

        self.resource_usage_history: List[Dict[str, Any]] = []

        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False

    def get_memory_limit_for_job(self, job: Dict[str, Any]) -> int:
        """
        Get memory limit (MB) for a specific job type.

        Args:
            job: Job dictionary with type and metadata

        Returns:
            Memory limit in MB
        """
        job_type = job.get("type", "")

        # For PDF chunk jobs, use PDF limit
        if job_type == "add_to_qdrant_chunk" or job.get("is_chunk"):
            return self.config.memory_limit_pdf

        # For metadata updates, use metadata limit
        if job_type == "update_metadata":
            return self.config.memory_limit_metadata

        # For main indexing jobs, determine by file size if available
        if job_type in ("add_to_qdrant", "add_document", "index_document"):
            # Try to estimate file size from kwargs
            file_size_mb = job.get("kwargs", {}).get("estimated_size_mb", 0)

            if file_size_mb > 50:
                return self.config.memory_limit_large_doc
            elif file_size_mb > 10:
                return self.config.memory_limit_medium_doc
            else:
                return self.config.memory_limit_small_doc

        # Default to medium doc limit
        return self.config.memory_limit_medium_doc

    def get_current_memory_usage(self) -> Tuple[float, float]:
        """
        Get current system memory usage.

        Returns:
            Tuple of (used_percent, available_gb)
        """
        memory = psutil.virtual_memory()
        return memory.percent, memory.available / (1024**3)

    def can_start_job(self, job: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if system has enough memory to start a job.

        Args:
            job: Job to check

        Returns:
            Tuple of (can_start, reason)
        """
        memory_percent, available_gb = self.get_current_memory_usage()
        memory_limit_mb = self.get_memory_limit_for_job(job)
        memory_limit_gb = memory_limit_mb / 1024

        # Check emergency threshold
        if memory_percent >= self.config.emergency_threshold:
            return (
                False,
                f"Emergency: Memory at {memory_percent:.1f}% (>= {self.config.emergency_threshold}%)",
            )

        # Check if we have enough available memory
        if available_gb < memory_limit_gb:
            return (
                False,
                f"Insufficient memory: {available_gb:.2f}GB available, need {memory_limit_gb:.2f}GB",
            )

        # Check critical threshold
        if memory_percent >= self.config.memory_critical_threshold:
            # Only allow if we have significant headroom
            if available_gb < memory_limit_gb * 1.5:
                return (
                    False,
                    f"Critical: Memory at {memory_percent:.1f}%, need {memory_limit_gb * 1.5:.2f}GB headroom",
                )

        return True, "OK"

    def adjust_parallelism(self) -> None:
        """Dynamically adjust parallelism based on memory pressure."""
        memory_percent, available_gb = self.get_current_memory_usage()

        old_parallelism = self.current_parallelism
        reason = "stable"

        # Emergency: reduce to 1
        if memory_percent >= self.config.emergency_threshold:
            self.current_parallelism = 1
            reason = "emergency"

        # Critical: reduce by half
        elif memory_percent >= self.config.memory_critical_threshold:
            self.current_parallelism = max(1, self.current_parallelism // 2)
            reason = "critical"
            self.job_stats["memory_constrained_count"] += 1

        # Warning: reduce by 1
        elif memory_percent >= self.config.memory_warning_threshold:
            self.current_parallelism = max(1, self.current_parallelism - 1)
            reason = "warning"

        # Normal: gradually increase back to max
        else:
            if self.current_parallelism < self.config.max_parallel_jobs:
                self.current_parallelism = min(
                    self.config.max_parallel_jobs,
                    self.current_parallelism + 1,
                )
                reason = "normal"

        if old_parallelism != self.current_parallelism:
            print(
                f"Parallelism adjusted: {old_parallelism} -> {self.current_parallelism} "
                f"(memory: {memory_percent:.1f}%, reason: {reason})",
                file=sys.stderr,
            )

            self.parallelism_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "old_value": old_parallelism,
                    "new_value": self.current_parallelism,
                    "memory_percent": memory_percent,
                    "reason": reason,
                }
            )

    def can_process_parallel(self, jobs: List[Dict[str, Any]]) -> bool:
        """
        Check if jobs can be processed in parallel.

        Jobs are independent if:
        - They operate on different item keys
        - None are chunk jobs with same parent

        Args:
            jobs: List of jobs to check

        Returns:
            True if jobs are independent
        """
        if not self.config.enable_parallel_processing:
            return False

        if len(jobs) < 2:
            return False

        # Check if all jobs operate on different items
        item_keys: set = set()
        parent_keys: set = set()

        for job in jobs:
            # Get actual item key
            item_key = job.get("item_key")
            if item_key in item_keys:
                return False  # Duplicate item
            item_keys.add(item_key)

            # Check for chunk jobs with same parent
            parent_key = job.get("kwargs", {}).get("parent_item_key")
            if parent_key:
                if parent_key in parent_keys:
                    return False  # Chunks from same parent
                parent_keys.add(parent_key)

        return True

    async def process_jobs_parallel(
        self,
        jobs: List[Dict[str, Any]],
        processor_func: Callable,
        **processor_kwargs: Any,
    ) -> List[Any]:
        """
        Process multiple independent jobs in parallel.

        Args:
            jobs: List of independent jobs
            processor_func: Async function to process each job
            **processor_kwargs: Additional kwargs for processor

        Returns:
            List of results for each job
        """
        if not self.can_process_parallel(jobs):
            print(
                "Jobs not suitable for parallel processing, falling back to sequential",
                file=sys.stderr,
            )
            results = []
            for job in jobs:
                result = await processor_func(job, **processor_kwargs)
                results.append(result)
                self.job_stats["sequential_processed"] += 1
            return results

        # Adjust parallelism based on current memory
        self.adjust_parallelism()

        # Process in batches of current_parallelism
        results: List[Any] = []
        batch_size = self.current_parallelism

        print(
            f"Processing {len(jobs)} jobs in parallel (batch size: {batch_size})",
            file=sys.stderr,
        )

        for i in range(0, len(jobs), batch_size):
            batch = jobs[i : i + batch_size]

            # Check memory before each batch
            memory_ok = all(self.can_start_job(job)[0] for job in batch)

            if not memory_ok:
                # Memory pressure, reduce batch size
                print(
                    "Memory pressure detected, reducing batch size to 1", file=sys.stderr
                )
                batch_size = 1
                batch = jobs[i : i + 1]

            # Process batch in parallel
            batch_tasks = [processor_func(job, **processor_kwargs) for job in batch]

            self.active_jobs = len(batch_tasks)

            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                results.extend(batch_results)

                self.job_stats["parallel_processed"] += len(batch)

                # Cleanup after batch
                self.memory_pool.cleanup_expired_items()
                gc.collect()

            except Exception as e:
                print(f"Error in parallel batch processing: {e}", file=sys.stderr)
                self.job_stats["errors"] += 1
                raise
            finally:
                self.active_jobs = 0

        self.job_stats["total_processed"] += len(jobs)

        return results

    async def start_monitoring(self) -> None:
        """Start background resource monitoring."""
        if self._monitoring_task is not None:
            print("Monitoring already running", file=sys.stderr)
            return

        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        print("Performance monitoring started", file=sys.stderr)

    async def stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        if self._monitoring_task is None:
            return

        self._stop_monitoring = True
        await self._monitoring_task
        self._monitoring_task = None
        print("Performance monitoring stopped", file=sys.stderr)

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                # Adjust parallelism based on memory
                self.adjust_parallelism()

                # Cleanup memory pool
                if (
                    time.time() - self.memory_pool._last_cleanup
                    > self.config.pool_cleanup_interval_seconds
                ):
                    self.memory_pool.cleanup_expired_items()

                # Track resource usage
                if self.config.enable_resource_tracking:
                    memory_percent, available_gb = self.get_current_memory_usage()
                    self.resource_usage_history.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "memory_percent": memory_percent,
                            "available_gb": available_gb,
                            "active_jobs": self.active_jobs,
                            "current_parallelism": self.current_parallelism,
                        }
                    )

                    # Keep only last 1000 entries
                    if len(self.resource_usage_history) > 1000:
                        self.resource_usage_history = self.resource_usage_history[-1000:]

                await asyncio.sleep(self.config.monitor_interval_seconds)

            except Exception as e:
                print(f"Error in monitoring loop: {e}", file=sys.stderr)
                await asyncio.sleep(self.config.monitor_interval_seconds)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        memory_percent, available_gb = self.get_current_memory_usage()
        pool_stats = self.memory_pool.get_stats()

        return {
            "job_stats": self.job_stats.copy(),
            "memory_pool": {
                "pool_sizes": pool_stats.pool_sizes,
                "hits": pool_stats.hits,
                "misses": pool_stats.misses,
                "evictions": pool_stats.evictions,
                "allocations": pool_stats.allocations,
                "hit_rate": pool_stats.hit_rate,
            },
            "current_state": {
                "parallelism": self.current_parallelism,
                "active_jobs": self.active_jobs,
                "memory_percent": memory_percent,
                "available_gb": available_gb,
            },
            "parallelism_history": self.parallelism_history[-10:],  # Last 10 adjustments
            "resource_history": self.resource_usage_history[-100:],  # Last 100 samples
        }

    def cleanup(self) -> None:
        """Cleanup resources and stop monitoring."""
        self._stop_monitoring = True
        self.memory_pool.clear()
        print("Performance optimizer cleaned up", file=sys.stderr)


def create_performance_optimizer(
    max_parallel_jobs: int = 3,
    memory_pool_size_mb: int = 500,
    enable_parallel: bool = True,
) -> PerformanceOptimizer:
    """
    Create a performance optimizer with common settings.

    Args:
        max_parallel_jobs: Maximum concurrent jobs
        memory_pool_size_mb: Memory pool size in MB
        enable_parallel: Enable parallel processing

    Returns:
        Configured PerformanceOptimizer
    """
    config = OptimizationConfig(
        max_parallel_jobs=max_parallel_jobs,
        memory_pool_size_mb=memory_pool_size_mb,
        enable_parallel_processing=enable_parallel,
    )

    return PerformanceOptimizer(config)


__all__ = [
    "OptimizationConfig",
    "MemoryPoolStats",
    "MemoryPool",
    "PerformanceOptimizer",
    "create_performance_optimizer",
]
