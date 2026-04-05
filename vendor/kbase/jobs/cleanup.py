"""
Job Queue and Process Monitor Cleanup System

Automatically manages JSONL files to prevent unbounded growth:
- Creates compressed backups before cleanup
- Rotates old backups
- Maintains configurable retention periods
- Provides emergency recovery mechanisms

Based on mcp-scholar's process_monitor_cleanup.py.
"""

import gzip
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CleanupConfig:
    """Configuration for cleanup operations."""

    # Automatic cleanup thresholds
    max_file_size_mb: float = 10.0  # Trigger cleanup if file > 10MB
    max_record_count: int = 5000  # Trigger cleanup if > 5000 records
    max_age_days: int = 30  # Keep records newer than 30 days

    # Retention policy
    keep_recent_records: int = 1000  # Always keep most recent 1000 records
    keep_recent_days: int = 7  # Always keep last 7 days

    # Backup configuration
    backup_dir: str = "backups"
    max_backups: int = 10  # Keep max 10 backup files
    compress_backups: bool = True  # Use gzip compression

    # Safety settings
    dry_run_mode: bool = False  # Set True for testing
    enable_auto_cleanup: bool = True  # Enable automatic cleanup


# Default configuration
DEFAULT_CLEANUP_CONFIG = CleanupConfig()


class JSONLCleanup:
    """
    Manages backup and cleanup of JSONL files (process monitor, job queues, etc.).

    This is a generalized cleanup class that can work with any JSONL file
    that has a 'timestamp' field in each record.
    """

    def __init__(
        self,
        file_path: Path,
        backup_subdir: str = "general",
        config: Optional[CleanupConfig] = None
    ):
        """
        Initialize cleanup manager.

        Args:
            file_path: Path to the JSONL file to manage
            backup_subdir: Subdirectory under backup_dir for this file's backups
            config: Cleanup configuration
        """
        self.file_path = Path(file_path)
        self.config = config or DEFAULT_CLEANUP_CONFIG

        # Setup backup directory
        self.backup_dir = Path(self.config.backup_dir) / backup_subdir
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def should_cleanup(self) -> Tuple[bool, str]:
        """
        Check if cleanup should be triggered.

        Returns:
            (should_cleanup: bool, reason: str)
        """
        if not self.file_path.exists():
            return False, "File does not exist"

        # Check file size
        file_size_mb = self.file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            return True, f"File size ({file_size_mb:.2f}MB) exceeds threshold"

        # Check record count
        record_count = self._count_records()
        if record_count > self.config.max_record_count:
            return True, f"Record count ({record_count}) exceeds threshold"

        # Check oldest record age
        oldest_age = self._get_oldest_record_age()
        if oldest_age and oldest_age.days > self.config.max_age_days:
            return True, f"Oldest record age ({oldest_age.days} days) exceeds threshold"

        return False, "All thresholds within limits"

    def _count_records(self) -> int:
        """Count total records in file."""
        try:
            with open(self.file_path, 'r') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error("Error counting records", error=str(e))
            return 0

    def _get_oldest_record_age(self) -> Optional[timedelta]:
        """Get age of oldest record."""
        try:
            with open(self.file_path, 'r') as f:
                first_line = f.readline()
                if first_line:
                    record = json.loads(first_line)
                    # Try common timestamp field names
                    timestamp_str = (
                        record.get('timestamp') or
                        record.get('created_at') or
                        record.get('time')
                    )
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        return datetime.now() - timestamp
        except Exception as e:
            logger.error("Error getting oldest record age", error=str(e))
        return None

    def create_backup(self) -> Optional[Path]:
        """
        Create compressed backup of current file.

        Returns:
            Path to backup file or None if failed
        """
        if not self.file_path.exists():
            logger.warning("File does not exist, nothing to backup")
            return None

        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_stem = self.file_path.stem
        backup_name = f"{file_stem}_{timestamp}"

        if self.config.compress_backups:
            backup_path = self.backup_dir / f"{backup_name}.jsonl.gz"

            try:
                # Compress backup
                with open(self.file_path, 'rb') as f_in:
                    with gzip.open(backup_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                logger.info("Created compressed backup", path=str(backup_path))
                return backup_path

            except Exception as e:
                logger.error("Error creating compressed backup", error=str(e))
                return None

        else:
            # Uncompressed backup
            backup_path = self.backup_dir / f"{backup_name}.jsonl"

            try:
                shutil.copy2(self.file_path, backup_path)
                logger.info("Created backup", path=str(backup_path))
                return backup_path

            except Exception as e:
                logger.error("Error creating backup", error=str(e))
                return None

    def cleanup_old_records(self) -> Dict[str, Any]:
        """
        Remove old records from file, keeping recent ones.

        Returns:
            Statistics about cleanup operation
        """
        if not self.file_path.exists():
            return {
                "success": False,
                "message": "File does not exist",
                "records_before": 0,
                "records_after": 0,
                "records_removed": 0,
            }

        # Read all records
        records = []
        try:
            with open(self.file_path, 'r') as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON line")
        except Exception as e:
            return {
                "success": False,
                "message": f"Error reading file: {e}",
                "records_before": 0,
                "records_after": 0,
                "records_removed": 0,
            }

        records_before = len(records)

        # Determine cutoff date
        cutoff_date = datetime.now() - timedelta(days=self.config.keep_recent_days)

        # Filter records
        kept_records = []
        for record in records:
            try:
                # Try common timestamp field names
                timestamp_str = (
                    record.get('timestamp') or
                    record.get('created_at') or
                    record.get('time')
                )
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str)

                    # Keep if recent enough
                    if timestamp >= cutoff_date:
                        kept_records.append(record)
                else:
                    # Keep records without timestamp
                    kept_records.append(record)

            except Exception as e:
                logger.warning("Error parsing record timestamp", error=str(e))
                # Keep problematic records to avoid data loss
                kept_records.append(record)

        # Always keep minimum recent records (sorted by timestamp, most recent first)
        if len(kept_records) > self.config.keep_recent_records:
            # Sort by timestamp descending
            kept_records.sort(
                key=lambda x: x.get('timestamp') or x.get('created_at') or x.get('time') or '',
                reverse=True
            )
            kept_records = kept_records[:self.config.keep_recent_records]

        records_after = len(kept_records)
        records_removed = records_before - records_after

        # Write cleaned records back
        if not self.config.dry_run_mode:
            try:
                # Write to temp file first
                temp_file = self.file_path.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    for record in kept_records:
                        f.write(json.dumps(record) + '\n')

                # Atomic replace
                temp_file.replace(self.file_path)

                logger.info(
                    "Cleanup complete",
                    records_before=records_before,
                    records_after=records_after,
                    records_removed=records_removed
                )

            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error writing cleaned records: {e}",
                    "records_before": records_before,
                    "records_after": 0,
                    "records_removed": 0,
                }

        else:
            logger.info(
                "[DRY RUN] Would cleanup",
                records_before=records_before,
                records_after=records_after,
                records_removed=records_removed
            )

        return {
            "success": True,
            "message": "Cleanup successful",
            "records_before": records_before,
            "records_after": records_after,
            "records_removed": records_removed,
            "cutoff_date": cutoff_date.isoformat(),
            "dry_run": self.config.dry_run_mode,
        }

    def rotate_backups(self) -> int:
        """
        Remove old backups beyond retention limit.

        Returns:
            Number of backups removed
        """
        file_stem = self.file_path.stem

        # Get all backup files for this file
        backup_files = sorted(
            self.backup_dir.glob(f"{file_stem}_*.jsonl*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Newest first
        )

        # Keep only max_backups
        removed_count = 0
        for backup_file in backup_files[self.config.max_backups:]:
            try:
                if not self.config.dry_run_mode:
                    backup_file.unlink()
                    logger.info("Removed old backup", name=backup_file.name)
                else:
                    logger.info("[DRY RUN] Would remove old backup", name=backup_file.name)
                removed_count += 1
            except Exception as e:
                logger.error("Error removing backup", file=str(backup_file), error=str(e))

        return removed_count

    def perform_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform complete cleanup workflow: backup -> cleanup -> rotate.

        Args:
            force: Force cleanup even if thresholds not met

        Returns:
            Statistics about the operation
        """
        # Check if cleanup needed
        should_cleanup, reason = self.should_cleanup()

        if not should_cleanup and not force:
            return {
                "success": True,
                "message": "Cleanup not needed",
                "reason": reason,
                "cleanup_performed": False,
            }

        logger.info("Starting cleanup", reason=reason)

        # Step 1: Create backup
        backup_path = self.create_backup()
        if not backup_path:
            return {
                "success": False,
                "message": "Failed to create backup",
                "cleanup_performed": False,
            }

        # Step 2: Cleanup old records
        cleanup_stats = self.cleanup_old_records()
        if not cleanup_stats["success"]:
            return {
                "success": False,
                "message": "Cleanup failed",
                "backup_path": str(backup_path),
                "cleanup_performed": False,
            }

        # Step 3: Rotate old backups
        removed_backups = self.rotate_backups()

        return {
            "success": True,
            "message": "Cleanup completed successfully",
            "cleanup_performed": True,
            "reason": reason,
            "backup_path": str(backup_path),
            "records_before": cleanup_stats["records_before"],
            "records_after": cleanup_stats["records_after"],
            "records_removed": cleanup_stats["records_removed"],
            "backups_removed": removed_backups,
            "dry_run": self.config.dry_run_mode,
        }

    def restore_from_backup(self, backup_path: Optional[Path] = None) -> bool:
        """
        Restore file from backup.

        Args:
            backup_path: Specific backup to restore, or None for most recent

        Returns:
            True if successful
        """
        file_stem = self.file_path.stem

        if backup_path is None:
            # Find most recent backup
            backup_files = sorted(
                self.backup_dir.glob(f"{file_stem}_*.jsonl*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if not backup_files:
                logger.error("No backups found")
                return False

            backup_path = backup_files[0]

        else:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error("Backup file not found", path=str(backup_path))
                return False

        try:
            # Decompress if needed
            if backup_path.suffix == '.gz':
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(self.file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(backup_path, self.file_path)

            logger.info("Restored from backup", path=str(backup_path))
            return True

        except Exception as e:
            logger.error("Error restoring from backup", error=str(e))
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics about file and backups."""
        file_stem = self.file_path.stem

        stats = {
            "file": {
                "path": str(self.file_path),
                "exists": self.file_path.exists(),
                "size_mb": 0,
                "record_count": 0,
                "oldest_record_age_days": None,
            },
            "backups": {
                "count": 0,
                "total_size_mb": 0,
                "oldest_backup_age_days": None,
            },
            "cleanup_needed": False,
            "cleanup_reason": "",
        }

        if self.file_path.exists():
            stats["file"]["size_mb"] = round(
                self.file_path.stat().st_size / (1024 * 1024), 2
            )
            stats["file"]["record_count"] = self._count_records()

            oldest_age = self._get_oldest_record_age()
            if oldest_age:
                stats["file"]["oldest_record_age_days"] = oldest_age.days

        # Backup statistics
        backup_files = list(self.backup_dir.glob(f"{file_stem}_*.jsonl*"))
        stats["backups"]["count"] = len(backup_files)

        if backup_files:
            stats["backups"]["total_size_mb"] = round(
                sum(f.stat().st_size for f in backup_files) / (1024 * 1024), 2
            )

            oldest_backup = min(backup_files, key=lambda p: p.stat().st_mtime)
            oldest_backup_age = datetime.now() - datetime.fromtimestamp(
                oldest_backup.stat().st_mtime
            )
            stats["backups"]["oldest_backup_age_days"] = oldest_backup_age.days

        # Check if cleanup needed
        should_cleanup, reason = self.should_cleanup()
        stats["cleanup_needed"] = should_cleanup
        stats["cleanup_reason"] = reason

        return stats


class ProcessMonitorCleanup(JSONLCleanup):
    """Specialized cleanup for process_monitor.jsonl files."""

    def __init__(self, queue_dir: str = ".jobs", config: Optional[CleanupConfig] = None):
        """
        Initialize process monitor cleanup.

        Args:
            queue_dir: Job queue directory
            config: Cleanup configuration
        """
        queue_path = Path(queue_dir)
        super().__init__(
            file_path=queue_path / "process_monitor.jsonl",
            backup_subdir="process_monitor",
            config=config
        )


class JobQueueCleanup(JSONLCleanup):
    """Specialized cleanup for completed.jsonl and failed.jsonl files."""

    def __init__(
        self,
        queue_dir: str = ".jobs",
        file_type: str = "completed",
        config: Optional[CleanupConfig] = None
    ):
        """
        Initialize job queue cleanup.

        Args:
            queue_dir: Job queue directory
            file_type: Type of file ("completed" or "failed")
            config: Cleanup configuration
        """
        queue_path = Path(queue_dir)
        super().__init__(
            file_path=queue_path / f"{file_type}.jsonl",
            backup_subdir=f"job_queue_{file_type}",
            config=config
        )


# Convenience functions
def cleanup_all(
    queue_dir: str = ".jobs",
    force: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Cleanup all JSONL files in the queue directory.

    Args:
        queue_dir: Queue directory path
        force: Force cleanup even if not needed
        dry_run: Preview changes without executing

    Returns:
        Combined cleanup statistics
    """
    config = CleanupConfig(dry_run_mode=dry_run)

    results = {
        "process_monitor": None,
        "completed": None,
        "failed": None,
        "total_records_removed": 0,
        "total_backups_removed": 0,
    }

    # Cleanup process monitor
    pm_cleanup = ProcessMonitorCleanup(queue_dir, config)
    if pm_cleanup.file_path.exists():
        results["process_monitor"] = pm_cleanup.perform_cleanup(force=force)
        if results["process_monitor"]["cleanup_performed"]:
            results["total_records_removed"] += results["process_monitor"].get("records_removed", 0)
            results["total_backups_removed"] += results["process_monitor"].get("backups_removed", 0)

    # Cleanup completed jobs
    completed_cleanup = JobQueueCleanup(queue_dir, "completed", config)
    if completed_cleanup.file_path.exists():
        results["completed"] = completed_cleanup.perform_cleanup(force=force)
        if results["completed"]["cleanup_performed"]:
            results["total_records_removed"] += results["completed"].get("records_removed", 0)
            results["total_backups_removed"] += results["completed"].get("backups_removed", 0)

    # Cleanup failed jobs
    failed_cleanup = JobQueueCleanup(queue_dir, "failed", config)
    if failed_cleanup.file_path.exists():
        results["failed"] = failed_cleanup.perform_cleanup(force=force)
        if results["failed"]["cleanup_performed"]:
            results["total_records_removed"] += results["failed"].get("records_removed", 0)
            results["total_backups_removed"] += results["failed"].get("backups_removed", 0)

    return results


def get_all_statistics(queue_dir: str = ".jobs") -> Dict[str, Any]:
    """
    Get statistics for all JSONL files.

    Args:
        queue_dir: Queue directory path

    Returns:
        Combined statistics
    """
    return {
        "process_monitor": ProcessMonitorCleanup(queue_dir).get_statistics(),
        "completed": JobQueueCleanup(queue_dir, "completed").get_statistics(),
        "failed": JobQueueCleanup(queue_dir, "failed").get_statistics(),
    }


# Compatibility functions matching mcp-scholar's process_monitor_cleanup.py API
def perform_process_monitor_cleanup(
    queue_dir: str = ".jobs",
    force: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Perform cleanup on process_monitor.jsonl.

    Compatibility function for mcp-scholar migration.

    Args:
        queue_dir: Queue directory path
        force: Force cleanup even if not needed
        dry_run: Preview changes without executing

    Returns:
        Cleanup statistics
    """
    config = CleanupConfig(dry_run_mode=dry_run)
    cleanup = ProcessMonitorCleanup(queue_dir, config)
    return cleanup.perform_cleanup(force=force)


def get_process_monitor_statistics(queue_dir: str = ".jobs") -> Dict[str, Any]:
    """
    Get statistics for process_monitor.jsonl.

    Compatibility function for mcp-scholar migration.
    Returns structure compatible with mcp-scholar's process_monitor_cleanup.py script.

    Args:
        queue_dir: Queue directory path

    Returns:
        Statistics dictionary with 'monitor_file' and 'backups' keys
    """
    stats = ProcessMonitorCleanup(queue_dir).get_statistics()

    # Transform to match expected format from mcp-scholar
    return {
        "monitor_file": {
            "exists": stats["file"]["exists"],
            "size_mb": stats["file"]["size_mb"],
            "record_count": stats["file"]["record_count"],
            "oldest_record_age_days": stats["file"]["oldest_record_age_days"],
        },
        "backups": stats["backups"],
        "cleanup_needed": stats["cleanup_needed"],
        "cleanup_reason": stats["cleanup_reason"],
    }


def restore_process_monitor_backup(
    queue_dir: str = ".jobs",
    backup_path: Optional[str] = None
) -> bool:
    """
    Restore process_monitor.jsonl from backup.

    Compatibility function for mcp-scholar migration.

    Args:
        queue_dir: Queue directory path
        backup_path: Specific backup to restore, or None for most recent

    Returns:
        True if successful
    """
    cleanup = ProcessMonitorCleanup(queue_dir)
    return cleanup.restore_from_backup(Path(backup_path) if backup_path else None)
