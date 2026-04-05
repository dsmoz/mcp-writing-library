"""
Memory Monitoring and Management

Pre-emptive memory monitoring to prevent failures during job processing.
Provides memory status tracking, garbage collection, and memory-based
decision making for adaptive processing.

Based on patterns from mcp-scholar but generalized for reuse.
"""

import gc
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory monitoring."""

    # Memory threshold for pre-emptive action (% of available memory)
    warning_threshold_percent: float = 75.0

    # Memory threshold for critical action (% of available memory)
    critical_threshold_percent: float = 85.0

    # Enable aggressive garbage collection
    enable_aggressive_gc: bool = True

    # Enable pre-emptive memory monitoring
    enable_preemptive_monitoring: bool = True

    # Minimum available memory (MB) to proceed with processing
    min_available_mb: float = 500.0


class MemoryMonitor:
    """
    Pre-emptive memory monitoring to prevent failures.

    Tracks memory usage, provides warnings before critical levels,
    and triggers garbage collection when needed.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory monitor.

        Args:
            config: Memory monitoring configuration
        """
        self.config = config or MemoryConfig()
        self.baseline_memory: Optional[float] = None
        self.peak_memory: float = 0.0
        self._available = PSUTIL_AVAILABLE

        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, memory monitoring disabled")

    def is_available(self) -> bool:
        """Check if memory monitoring is available."""
        return self._available

    def get_memory_status(self) -> Dict:
        """
        Get current memory status.

        Returns:
            Dict with memory usage information
        """
        if not self._available:
            return {
                "available": False,
                "error": "psutil not installed"
            }

        try:
            memory = psutil.virtual_memory()

            return {
                "available": True,
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent,
                "warning": memory.percent >= self.config.warning_threshold_percent,
                "critical": memory.percent >= self.config.critical_threshold_percent
            }
        except Exception as e:
            logger.error("Failed to get memory status", error=str(e))
            return {
                "available": False,
                "error": str(e)
            }

    def check_memory_before_processing(
        self,
        estimated_usage_mb: float
    ) -> Tuple[bool, str]:
        """
        Check if there's enough memory before processing.

        Args:
            estimated_usage_mb: Estimated memory usage for the operation

        Returns:
            Tuple of (can_proceed, message)
        """
        if not self._available:
            # If we can't check memory, allow processing but warn
            return True, "Memory monitoring unavailable, proceeding anyway"

        status = self.get_memory_status()

        if not status.get("available"):
            return True, f"Memory check failed: {status.get('error', 'unknown')}"

        available_mb = status["available_gb"] * 1024

        if status["critical"]:
            return False, f"Critical memory usage: {status['percent']:.1f}% - Cannot proceed"

        if available_mb < self.config.min_available_mb:
            return False, f"Insufficient available memory: {available_mb:.0f}MB (min: {self.config.min_available_mb:.0f}MB)"

        if estimated_usage_mb > available_mb:
            return False, f"Insufficient memory: need {estimated_usage_mb:.0f}MB, have {available_mb:.0f}MB"

        if status["warning"]:
            return True, f"Warning: Memory usage at {status['percent']:.1f}%"

        return True, "Memory OK"

    def trigger_garbage_collection(self, generation: int = 2) -> int:
        """
        Trigger garbage collection.

        Args:
            generation: GC generation to collect (0, 1, or 2)

        Returns:
            Number of objects collected
        """
        if self.config.enable_aggressive_gc:
            logger.debug("Triggering garbage collection", generation=generation)
            collected = gc.collect(generation=generation)
            logger.debug("Garbage collection complete", objects_collected=collected)
            return collected
        return 0

    def set_baseline(self) -> Optional[float]:
        """
        Record baseline memory usage before processing.

        Returns:
            Baseline memory usage in GB, or None if unavailable
        """
        status = self.get_memory_status()

        if status.get("available"):
            self.baseline_memory = status["used_gb"]
            logger.debug("Memory baseline set", baseline_gb=self.baseline_memory)
            return self.baseline_memory

        return None

    def track_peak(self) -> Optional[float]:
        """
        Track peak memory usage during processing.

        Returns:
            Current peak memory in GB, or None if unavailable
        """
        status = self.get_memory_status()

        if status.get("available"):
            current_usage = status["used_gb"]
            if current_usage > self.peak_memory:
                self.peak_memory = current_usage
                logger.debug("New peak memory", peak_gb=self.peak_memory)
            return self.peak_memory

        return None

    def get_memory_delta(self) -> Optional[float]:
        """
        Get memory increase since baseline (in GB).

        Returns:
            Memory delta in GB, or None if baseline not set
        """
        if self.baseline_memory is None:
            return None

        status = self.get_memory_status()

        if status.get("available"):
            return status["used_gb"] - self.baseline_memory

        return None

    def reset_tracking(self) -> None:
        """Reset baseline and peak tracking."""
        self.baseline_memory = None
        self.peak_memory = 0.0
        logger.debug("Memory tracking reset")

    def get_tracking_summary(self) -> Dict:
        """
        Get summary of memory tracking.

        Returns:
            Dict with tracking information
        """
        status = self.get_memory_status()

        return {
            "baseline_gb": self.baseline_memory,
            "peak_gb": self.peak_memory,
            "current_gb": status.get("used_gb"),
            "delta_gb": self.get_memory_delta(),
            "percent": status.get("percent"),
            "warning": status.get("warning", False),
            "critical": status.get("critical", False),
        }

    def should_pause_processing(self) -> Tuple[bool, str]:
        """
        Check if processing should be paused due to memory pressure.

        Returns:
            Tuple of (should_pause, reason)
        """
        if not self._available or not self.config.enable_preemptive_monitoring:
            return False, "Monitoring not enabled"

        status = self.get_memory_status()

        if not status.get("available"):
            return False, "Status unavailable"

        if status["critical"]:
            return True, f"Critical memory usage: {status['percent']:.1f}%"

        return False, "Memory OK"


# Global memory monitor instance
_global_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor(config: Optional[MemoryConfig] = None) -> MemoryMonitor:
    """
    Get the global memory monitor instance.

    Args:
        config: Optional configuration (only used if creating new instance)

    Returns:
        MemoryMonitor instance
    """
    global _global_memory_monitor

    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor(config)

    return _global_memory_monitor


def check_memory_available(estimated_usage_mb: float = 0) -> Tuple[bool, str]:
    """
    Convenience function to check if memory is available.

    Args:
        estimated_usage_mb: Estimated memory usage

    Returns:
        Tuple of (can_proceed, message)
    """
    monitor = get_memory_monitor()
    return monitor.check_memory_before_processing(estimated_usage_mb)


def trigger_gc() -> int:
    """
    Convenience function to trigger garbage collection.

    Returns:
        Number of objects collected
    """
    monitor = get_memory_monitor()
    return monitor.trigger_garbage_collection()
