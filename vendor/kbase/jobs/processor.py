"""
Base Job Processor

Provides an abstract base class for job processing with common functionality:
- Async processing loop with graceful shutdown
- Memory management and monitoring
- Timeout handling with watchdog
- Pluggable job handlers

MCPs should subclass BaseJobProcessor and implement their own job handlers.
"""

import asyncio
import gc
import os
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Protocol

import structlog

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from kbase.jobs.queue import JobQueue, get_job_queue
from kbase.jobs.retry import (
    RetryManager,
    NonRetryableError,
    categorize_error,
)

logger = structlog.get_logger(__name__)


# Default timeout configurations
DEFAULT_JOB_TIMEOUTS = {
    "process_document": 1800,  # 30 minutes
    "add_to_qdrant": 2400,     # 40 minutes
    "update_metadata": 900,    # 15 minutes
    "sync_payload": 600,       # 10 minutes
    "default": 1200,           # 20 minutes
}

# Default memory limits
DEFAULT_MEMORY_LIMITS = {
    "max_memory_percent": 85,
    "warning_memory_percent": 78,
    "critical_memory_percent": 92,
    "memory_check_interval": 15,
    "gc_threshold_mb": 300,
    "max_job_memory_mb": 1500,
}


class JobHandler(Protocol):
    """Protocol for job handlers."""

    async def __call__(
        self,
        job: Dict[str, Any],
        processor: "BaseJobProcessor"
    ) -> Any:
        """
        Handle a job.

        Args:
            job: The job dict
            processor: The processor instance (for accessing shared resources)

        Returns:
            Result of job processing
        """
        ...


@dataclass
class ProcessorConfig:
    """Configuration for job processor."""
    poll_interval: float = 5.0  # Seconds between polling for jobs
    shutdown_timeout: float = 30.0  # Seconds to wait for graceful shutdown
    enable_memory_management: bool = True
    enable_watchdog: bool = True
    job_timeouts: Dict[str, int] = None
    memory_limits: Dict[str, int] = None

    def __post_init__(self):
        if self.job_timeouts is None:
            self.job_timeouts = DEFAULT_JOB_TIMEOUTS.copy()
        if self.memory_limits is None:
            self.memory_limits = DEFAULT_MEMORY_LIMITS.copy()


class BaseJobProcessor(ABC):
    """
    Abstract base class for job processors.

    Subclasses should:
    1. Implement handle_job() to process jobs by type
    2. Optionally override on_start() and on_stop() for setup/teardown
    3. Register job handlers using register_handler()
    """

    def __init__(
        self,
        queue: Optional[JobQueue] = None,
        config: Optional[ProcessorConfig] = None,
    ):
        """
        Initialize processor.

        Args:
            queue: JobQueue instance (creates one if None)
            config: Processor configuration
        """
        self.queue = queue or get_job_queue()
        self.config = config or ProcessorConfig()
        self.retry_manager = RetryManager()

        # State tracking
        self._running = False
        self._stop_requested = False
        self._current_job_id: Optional[str] = None
        self._current_job_start: Optional[datetime] = None
        self._stop_current_job = False

        # Watchdog thread
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_active = False

        # Job handlers
        self._handlers: Dict[str, JobHandler] = {}

        # Memory tracking
        self._memory_critical = False
        if PSUTIL_AVAILABLE:
            self._process = psutil.Process()
        else:
            self._process = None

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        """
        Register a handler for a specific job type.

        Args:
            job_type: Type of job (e.g., "process_document")
            handler: Async callable that handles the job
        """
        self._handlers[job_type] = handler
        logger.info("Handler registered", job_type=job_type)

    @abstractmethod
    async def handle_job(self, job: Dict[str, Any]) -> Any:
        """
        Handle a job. Must be implemented by subclasses.

        This method should dispatch to appropriate handlers based on job type,
        or handle jobs directly.

        Args:
            job: The job dict

        Returns:
            Result of job processing

        Raises:
            ValueError: If job type is not supported
        """
        pass

    async def on_start(self) -> None:
        """Called when processor starts. Override for setup."""
        pass

    async def on_stop(self) -> None:
        """Called when processor stops. Override for teardown."""
        pass

    async def run(self) -> None:
        """
        Run the processor loop.

        This starts the main processing loop and handles graceful shutdown.
        """
        self._running = True
        self._stop_requested = False

        # Setup signal handlers
        self._setup_signal_handlers()

        logger.info("Job processor starting")

        try:
            await self.on_start()

            if self.config.enable_watchdog:
                self._start_watchdog()

            await self._process_loop()

        except Exception as e:
            logger.error("Processor crashed", error=str(e))
            raise
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        """Request graceful shutdown."""
        logger.info("Stop requested")
        self._stop_requested = True
        self._running = False

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running and not self._stop_requested:
            try:
                # Check memory pressure
                if self.config.enable_memory_management:
                    if self._check_memory_pressure():
                        await asyncio.sleep(self.config.poll_interval)
                        continue

                # Claim next job
                job = self.queue.claim_job()

                if not job:
                    # No jobs available, wait and retry
                    for _ in range(int(self.config.poll_interval)):
                        if self._stop_requested:
                            break
                        await asyncio.sleep(1)
                    continue

                # Process job
                await self._process_job(job)

                # Small delay between jobs
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error("Error in processing loop", error=str(e))
                await asyncio.sleep(self.config.poll_interval)

    async def _process_job(self, job: Dict[str, Any]) -> None:
        """Process a single job with error handling."""
        job_id = job["id"]
        job_type = job["type"]

        self._current_job_id = job_id
        self._current_job_start = datetime.now()
        self._stop_current_job = False

        logger.info(
            "Processing job",
            job_id=job_id,
            job_type=job_type,
            item_key=job.get("item_key"),
        )

        try:
            # Check for registered handler first
            if job_type in self._handlers:
                handler = self._handlers[job_type]
                result = await self.retry_manager.retry_async(
                    handler,
                    job,
                    self,
                    error_context=f"{job_type}_{job.get('item_key', 'unknown')}",
                )
            else:
                # Fall back to abstract method
                result = await self.retry_manager.retry_async(
                    self.handle_job,
                    job,
                    error_context=f"{job_type}_{job.get('item_key', 'unknown')}",
                )

            # Determine success
            success = False
            if isinstance(result, dict):
                success = result.get("success", False)
            elif result is True:
                success = True
            elif result is not None:
                success = True

            if success:
                self.queue.mark_completed(job, result)
                logger.info("Job completed", job_id=job_id)
            else:
                error_msg = "Job returned failure result"
                if isinstance(result, dict):
                    error_msg = result.get("message", error_msg)
                self.queue.mark_failed(job, error_msg)
                logger.warning("Job failed", job_id=job_id, error=error_msg)

        except NonRetryableError as e:
            error_msg = str(e)
            logger.warning(
                "Job failed (non-retriable)",
                job_id=job_id,
                error=error_msg,
            )
            self.queue.mark_failed(
                job,
                error_msg,
                is_retriable=False,
                error_category=categorize_error(error_msg),
            )

        except Exception as e:
            error_msg = str(e)

            if self._stop_current_job:
                logger.warning(
                    "Job terminated by timeout",
                    job_id=job_id,
                )
                # Already marked as failed by watchdog
            else:
                error_category = categorize_error(error_msg)
                is_retriable = error_category is not None

                logger.error(
                    "Job failed",
                    job_id=job_id,
                    error=error_msg,
                    is_retriable=is_retriable,
                )

                self.queue.mark_failed(
                    job,
                    error_msg,
                    is_retriable=is_retriable,
                    error_category=error_category,
                )

        finally:
            self._current_job_id = None
            self._current_job_start = None

            # Always remove from pending queue
            self.queue.remove_from_pending(job_id)

            # Force garbage collection after each job
            if self.config.enable_memory_management:
                collected = gc.collect()
                if collected > 100:
                    logger.debug("GC collected objects", count=collected)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("Received signal", signal=signum)
            self._stop_requested = True
            self._running = False

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception:
            # May fail in some contexts (e.g., non-main thread)
            pass

    def _start_watchdog(self) -> None:
        """Start watchdog thread for timeout detection."""
        self._watchdog_active = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
        )
        self._watchdog_thread.start()
        logger.debug("Watchdog started")

    def _stop_watchdog(self) -> None:
        """Stop watchdog thread."""
        self._watchdog_active = False
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=5)
        logger.debug("Watchdog stopped")

    def _watchdog_loop(self) -> None:
        """Watchdog loop for timeout detection."""
        while self._watchdog_active:
            try:
                if self._current_job_id and self._current_job_start:
                    elapsed = (datetime.now() - self._current_job_start).total_seconds()

                    # Get timeout for job type
                    # Note: We'd need to store job_type to look up timeout
                    timeout = self.config.job_timeouts.get(
                        "default",
                        DEFAULT_JOB_TIMEOUTS["default"]
                    )

                    if elapsed > timeout:
                        logger.warning(
                            "Job timeout detected",
                            job_id=self._current_job_id,
                            elapsed=elapsed,
                            timeout=timeout,
                        )
                        self._stop_current_job = True

                        # Mark job as failed
                        self.queue.mark_failed(
                            {"id": self._current_job_id, "type": "unknown"},
                            f"Job timeout after {elapsed:.0f}s (limit: {timeout}s)",
                            is_retriable=False,
                        )

            except Exception as e:
                logger.warning("Watchdog error", error=str(e))

            time.sleep(10)  # Check every 10 seconds

    def _check_memory_pressure(self) -> bool:
        """
        Check and handle memory pressure.

        Returns:
            True if processing should be paused
        """
        if not PSUTIL_AVAILABLE or not self._process:
            return False

        try:
            memory = psutil.virtual_memory()
            limits = self.config.memory_limits

            if memory.percent >= limits["critical_memory_percent"]:
                if not self._memory_critical:
                    logger.warning(
                        "Critical memory usage",
                        memory_percent=memory.percent,
                    )
                    self._memory_critical = True

                # Force garbage collection
                gc.collect()
                return True

            elif memory.percent >= limits["warning_memory_percent"]:
                logger.info(
                    "High memory usage",
                    memory_percent=memory.percent,
                )
                gc.collect()

            else:
                self._memory_critical = False

            return False

        except Exception as e:
            logger.warning("Memory check failed", error=str(e))
            return False

    async def _shutdown(self) -> None:
        """Perform shutdown cleanup."""
        logger.info("Processor shutting down")

        self._running = False

        if self.config.enable_watchdog:
            self._stop_watchdog()

        try:
            await self.on_stop()
        except Exception as e:
            logger.error("Error in on_stop", error=str(e))

        logger.info("Processor stopped")

    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not available"}

        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            return {
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent
                },
                "cpu_percent": cpu_percent,
                "current_pid": os.getpid(),
            }
        except Exception as e:
            return {"error": str(e)}


class SimpleJobProcessor(BaseJobProcessor):
    """
    Simple job processor that uses registered handlers.

    This is a concrete implementation that dispatches to registered handlers.
    """

    async def handle_job(self, job: Dict[str, Any]) -> Any:
        """Handle job by dispatching to registered handler."""
        job_type = job["type"]

        if job_type not in self._handlers:
            raise ValueError(f"No handler registered for job type: {job_type}")

        handler = self._handlers[job_type]
        return await handler(job, self)
