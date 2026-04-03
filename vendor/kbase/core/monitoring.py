"""
System Monitoring and Resource Optimization

Provides real-time monitoring, alerting, and automatic resource adjustments
for document processing pipelines.

Features:
- CPU, memory, disk monitoring with configurable thresholds
- Alert callbacks for warning/critical conditions
- Background monitoring thread
- Metrics history and summary
- Resource optimization recommendations

Usage:
    from kbase.core.monitoring import (
        SystemMonitor, ResourceOptimizer,
        start_monitoring, get_current_metrics, get_recommendations
    )

    # Start monitoring
    monitor = SystemMonitor()
    monitor.start_monitoring(interval=30)

    # Get current metrics
    metrics = monitor.get_system_metrics()

    # Get optimization recommendations
    optimizer = ResourceOptimizer(monitor)
    recommendations = optimizer.get_recommended_settings()
"""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable

import psutil
import structlog

logger = structlog.get_logger(__name__)


class SystemMonitor:
    """
    Enhanced system monitoring with adaptive thresholds and alerts.

    Monitors:
    - Memory (RAM + swap)
    - CPU usage and load averages
    - Disk usage
    - Current process resource usage
    - Related processes

    Provides continuous monitoring with configurable alert thresholds.
    """

    def __init__(
        self,
        alert_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the system monitor.

        Args:
            alert_callback: Function to call when alerts trigger
                           Signature: callback(alert_type, message, metrics)
            thresholds: Custom threshold values (overrides defaults)
        """
        self.alert_callback = alert_callback or self._default_alert
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: List[Dict[str, Any]] = []
        self.alerts_sent: set = set()

        self.thresholds = {
            "memory_warning": 80.0,
            "memory_critical": 90.0,
            "cpu_warning": 85.0,
            "cpu_critical": 95.0,
            "disk_warning": 85.0,
            "disk_critical": 95.0,
        }

        if thresholds:
            self.thresholds.update(thresholds)

    def _default_alert(self, alert_type: str, message: str, metrics: Dict[str, Any]):
        """Default alert handler - logs warning."""
        logger.warning("System alert", alert_type=alert_type, message=message)

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system metrics.

        Returns:
            Dictionary containing:
            - timestamp: ISO format timestamp
            - memory: RAM and swap usage
            - cpu: CPU percent and load averages
            - disk: Disk usage
            - process: Current process info
            - related_processes: Related process summary
        """
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            disk = psutil.disk_usage('/')

            current_process = psutil.Process()
            process_info = {
                "pid": current_process.pid,
                "memory_rss_mb": current_process.memory_info().rss / (1024 * 1024),
                "memory_vms_mb": current_process.memory_info().vms / (1024 * 1024),
                "memory_percent": current_process.memory_percent(),
                "cpu_percent": current_process.cpu_percent(),
                "num_threads": current_process.num_threads(),
                "num_fds": current_process.num_fds() if hasattr(current_process, 'num_fds') else 0,
            }

            related_processes = self._get_related_processes()

            return {
                "timestamp": datetime.now().isoformat(),
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent,
                    "swap_total_gb": swap.total / (1024**3),
                    "swap_used_gb": swap.used / (1024**3),
                    "swap_percent": swap.percent,
                },
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg_1m": load_avg[0],
                    "load_avg_5m": load_avg[1],
                    "load_avg_15m": load_avg[2],
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": (disk.used / disk.total) * 100,
                },
                "process": process_info,
                "related_processes": related_processes,
            }
        except Exception as e:
            logger.error("Error getting system metrics", error=str(e))
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _get_related_processes(self) -> Dict[str, Any]:
        """Get information about related processes."""
        related = {
            "python_processes": 0,
            "pdf_processes": 0,
            "high_memory_processes": [],
            "high_cpu_processes": []
        }

        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name'].lower()
                    cmdline = ' '.join(proc_info.get('cmdline', []) or []).lower()

                    if 'python' in proc_name:
                        related["python_processes"] += 1
                        if any(keyword in cmdline for keyword in ['pdf', 'docling']):
                            related["pdf_processes"] += 1

                    memory_percent = proc_info.get('memory_percent', 0) or 0
                    cpu_percent = proc_info.get('cpu_percent', 0) or 0

                    if memory_percent > 10:
                        related["high_memory_processes"].append({
                            "pid": proc_info['pid'],
                            "name": proc_name,
                            "memory_percent": memory_percent,
                        })

                    if cpu_percent > 50:
                        related["high_cpu_processes"].append({
                            "pid": proc_info['pid'],
                            "name": proc_name,
                            "cpu_percent": cpu_percent,
                        })

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            related["error"] = str(e)

        return related

    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for alert conditions and return list of alerts.

        Args:
            metrics: Current system metrics

        Returns:
            List of alert dictionaries with type, message, severity, value, threshold
        """
        alerts = []

        try:
            memory_percent = metrics.get("memory", {}).get("percent", 0)

            if memory_percent > self.thresholds["memory_critical"]:
                alert_key = f"memory_critical_{int(memory_percent)}"
                if alert_key not in self.alerts_sent:
                    alerts.append({
                        "type": "memory_critical",
                        "message": f"Critical memory usage: {memory_percent:.1f}%",
                        "severity": "critical",
                        "value": memory_percent,
                        "threshold": self.thresholds["memory_critical"]
                    })
                    self.alerts_sent.add(alert_key)

            elif memory_percent > self.thresholds["memory_warning"]:
                alert_key = f"memory_warning_{int(memory_percent)}"
                if alert_key not in self.alerts_sent:
                    alerts.append({
                        "type": "memory_warning",
                        "message": f"High memory usage: {memory_percent:.1f}%",
                        "severity": "warning",
                        "value": memory_percent,
                        "threshold": self.thresholds["memory_warning"]
                    })
                    self.alerts_sent.add(alert_key)

            cpu_percent = metrics.get("cpu", {}).get("percent", 0)

            if cpu_percent > self.thresholds["cpu_critical"]:
                alert_key = f"cpu_critical_{int(cpu_percent)}"
                if alert_key not in self.alerts_sent:
                    alerts.append({
                        "type": "cpu_critical",
                        "message": f"Critical CPU usage: {cpu_percent:.1f}%",
                        "severity": "critical",
                        "value": cpu_percent,
                        "threshold": self.thresholds["cpu_critical"]
                    })
                    self.alerts_sent.add(alert_key)

            process_memory = metrics.get("process", {}).get("memory_percent", 0)

            if process_memory > 20:
                alert_key = f"process_memory_{int(process_memory)}"
                if alert_key not in self.alerts_sent:
                    alerts.append({
                        "type": "process_memory_high",
                        "message": f"High process memory usage: {process_memory:.1f}%",
                        "severity": "warning",
                        "value": process_memory,
                        "threshold": 20
                    })
                    self.alerts_sent.add(alert_key)

            current_time = time.time()
            self.alerts_sent = {
                alert for alert in self.alerts_sent
                if current_time - hash(alert) % 3600 < 3600
            }

        except Exception as e:
            logger.error("Error checking alerts", error=str(e))

        return alerts

    def start_monitoring(self, interval: int = 30):
        """
        Start continuous monitoring.

        Args:
            interval: Seconds between metric collection
        """
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("System monitoring started", interval=interval)

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")

    def _monitor_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self.get_system_metrics()

                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)

                alerts = self.check_alerts(metrics)
                for alert in alerts:
                    self.alert_callback(alert["type"], alert["message"], metrics)

                time.sleep(interval)

            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                time.sleep(interval)

    def get_metrics_summary(self, last_minutes: int = 30) -> Dict[str, Any]:
        """
        Get summary of metrics for the last N minutes.

        Args:
            last_minutes: Number of minutes to summarize

        Returns:
            Summary dictionary with averages and peaks
        """
        cutoff_time = datetime.now() - timedelta(minutes=last_minutes)

        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]

        if not recent_metrics:
            return {"error": "No recent metrics available"}

        memory_values = [m.get("memory", {}).get("percent", 0) for m in recent_metrics]
        cpu_values = [m.get("cpu", {}).get("percent", 0) for m in recent_metrics]
        process_memory_values = [m.get("process", {}).get("memory_percent", 0) for m in recent_metrics]

        return {
            "period_minutes": last_minutes,
            "sample_count": len(recent_metrics),
            "memory": {
                "avg_percent": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max_percent": max(memory_values) if memory_values else 0,
                "min_percent": min(memory_values) if memory_values else 0,
            },
            "cpu": {
                "avg_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max_percent": max(cpu_values) if cpu_values else 0,
                "min_percent": min(cpu_values) if cpu_values else 0,
            },
            "process": {
                "avg_memory_percent": sum(process_memory_values) / len(process_memory_values) if process_memory_values else 0,
                "max_memory_percent": max(process_memory_values) if process_memory_values else 0,
            },
            "first_sample": recent_metrics[0]["timestamp"],
            "last_sample": recent_metrics[-1]["timestamp"],
        }


class ResourceOptimizer:
    """
    Automatically optimize resource usage based on system conditions.

    Provides recommendations for:
    - Memory limits
    - Processing parameters
    - Timeout values

    Based on current CPU/memory usage levels.
    """

    def __init__(self, system_monitor: SystemMonitor):
        """
        Initialize the resource optimizer.

        Args:
            system_monitor: SystemMonitor instance for getting metrics
        """
        self.system_monitor = system_monitor
        self.optimization_history: List[Dict[str, Any]] = []

    def get_recommended_settings(
        self,
        current_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get recommended settings based on current system state.

        Args:
            current_metrics: Optional pre-fetched metrics

        Returns:
            Dictionary with current_state, settings, and reasoning
        """
        if current_metrics is None:
            current_metrics = self.system_monitor.get_system_metrics()

        memory_percent = current_metrics.get("memory", {}).get("percent", 0)
        cpu_percent = current_metrics.get("cpu", {}).get("percent", 0)
        available_memory_gb = current_metrics.get("memory", {}).get("available_gb", 4)

        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "current_state": {
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
                "available_memory_gb": available_memory_gb,
            },
            "settings": {},
            "reasoning": []
        }

        if memory_percent > 85:
            recommendations["settings"]["memory_limits"] = {
                "max_memory_percent": 80,
                "critical_memory_percent": 88,
                "gc_threshold_mb": 300,
                "max_job_memory_mb": 1000,
            }
            recommendations["settings"]["processing"] = {
                "max_pdf_size_mb": 15,
                "max_document_pages": 30,
                "large_doc_queue_delay": 300,
                "concurrent_jobs": 1,
            }
            recommendations["reasoning"].append("High memory usage detected - using conservative limits")

        elif memory_percent > 70:
            recommendations["settings"]["memory_limits"] = {
                "max_memory_percent": 82,
                "critical_memory_percent": 90,
                "gc_threshold_mb": 350,
                "max_job_memory_mb": 1200,
            }
            recommendations["settings"]["processing"] = {
                "max_pdf_size_mb": 20,
                "max_document_pages": 40,
                "large_doc_queue_delay": 240,
                "concurrent_jobs": 1,
            }

        else:
            recommendations["settings"]["memory_limits"] = {
                "max_memory_percent": 85,
                "critical_memory_percent": 92,
                "gc_threshold_mb": 400,
                "max_job_memory_mb": 1500,
            }
            recommendations["settings"]["processing"] = {
                "max_pdf_size_mb": 25,
                "max_document_pages": 50,
                "large_doc_queue_delay": 180,
                "concurrent_jobs": 1,
            }

        base_timeouts = {
            "add_to_qdrant": 2400,
            "update_metadata": 900,
            "add_to_qdrant_chunk": 1200,
        }

        if cpu_percent > 80 or memory_percent > 80:
            timeout_multiplier = 1.5
            recommendations["reasoning"].append("High system load - increasing timeouts")
        elif cpu_percent < 30 and memory_percent < 50:
            timeout_multiplier = 0.8
        else:
            timeout_multiplier = 1.0

        recommendations["settings"]["timeouts"] = {
            job_type: int(timeout * timeout_multiplier)
            for job_type, timeout in base_timeouts.items()
        }

        if available_memory_gb < 2:
            recommendations["reasoning"].append("Low available memory - reducing concurrent processing")

        return recommendations

    def apply_recommendations(
        self,
        recommendations: Dict[str, Any],
        job_queue: Any
    ) -> Dict[str, Any]:
        """
        Apply recommendations to a job queue configuration.

        Args:
            recommendations: Recommendations from get_recommended_settings
            job_queue: Job queue object with configurable attributes

        Returns:
            Dictionary with applied_settings and errors
        """
        applied = {
            "timestamp": datetime.now().isoformat(),
            "applied_settings": [],
            "errors": []
        }

        try:
            settings = recommendations.get("settings", {})

            if "memory_limits" in settings:
                for key, value in settings["memory_limits"].items():
                    attr_name = key.upper()
                    if hasattr(job_queue, attr_name):
                        setattr(job_queue, attr_name, value)
                        applied["applied_settings"].append(f"Set {key} = {value}")

            if "processing" in settings:
                for key, value in settings["processing"].items():
                    attr_name = key.upper()
                    if hasattr(job_queue, attr_name):
                        setattr(job_queue, attr_name, value)
                        applied["applied_settings"].append(f"Set {key} = {value}")

            if "timeouts" in settings:
                if hasattr(job_queue, 'JOB_TIMEOUTS'):
                    for job_type, timeout in settings["timeouts"].items():
                        job_queue.JOB_TIMEOUTS[job_type] = timeout
                        applied["applied_settings"].append(f"Set timeout for {job_type} = {timeout}s")

            self.optimization_history.append({
                "timestamp": applied["timestamp"],
                "recommendations": recommendations,
                "applied": applied["applied_settings"]
            })

            if len(self.optimization_history) > 50:
                self.optimization_history.pop(0)

            logger.info("Applied optimization settings", count=len(applied["applied_settings"]))

        except Exception as e:
            error_msg = f"Error applying recommendations: {e}"
            applied["errors"].append(error_msg)
            logger.error(error_msg)

        return applied


# =============================================================================
# Global Instances and Convenience Functions
# =============================================================================

_system_monitor: Optional[SystemMonitor] = None
_resource_optimizer: Optional[ResourceOptimizer] = None


def get_system_monitor() -> SystemMonitor:
    """Get or create the global system monitor instance."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor


def get_resource_optimizer() -> ResourceOptimizer:
    """Get or create the global resource optimizer instance."""
    global _resource_optimizer
    if _resource_optimizer is None:
        _resource_optimizer = ResourceOptimizer(get_system_monitor())
    return _resource_optimizer


def start_monitoring(interval: int = 30):
    """Start system monitoring with the global instance."""
    get_system_monitor().start_monitoring(interval)


def stop_monitoring():
    """Stop system monitoring."""
    get_system_monitor().stop_monitoring()


def get_current_metrics() -> Dict[str, Any]:
    """Get current system metrics."""
    return get_system_monitor().get_system_metrics()


def get_recommendations() -> Dict[str, Any]:
    """Get optimization recommendations."""
    return get_resource_optimizer().get_recommended_settings()


def apply_optimizations(job_queue: Any) -> Dict[str, Any]:
    """Apply automatic optimizations to job queue."""
    recommendations = get_recommendations()
    return get_resource_optimizer().apply_recommendations(recommendations, job_queue)


__all__ = [
    # Classes
    "SystemMonitor",
    "ResourceOptimizer",
    # Instance getters
    "get_system_monitor",
    "get_resource_optimizer",
    # Convenience functions
    "start_monitoring",
    "stop_monitoring",
    "get_current_metrics",
    "get_recommendations",
    "apply_optimizations",
]
