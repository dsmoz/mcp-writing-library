"""
File-based Job Queue

Provides a simple, file-based job queue that works without external dependencies.
Supports both JSONL format (single file per type) and individual JSON files.

Based on patterns from mcp-scholar but generalized for reuse.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Represents a job in the queue."""
    id: str
    type: str
    item_key: str
    collection: str
    status: JobStatus
    created_at: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # Processing metadata
    claimed_at: Optional[str] = None
    claimed_by_pid: Optional[int] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    error: Optional[str] = None
    error_category: Optional[str] = None
    is_retriable: Optional[bool] = None
    retry_count: int = 0
    result: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("status"), str):
            data["status"] = JobStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class JobQueue:
    """
    File-based job queue with thread-safe operations.

    Supports both JSONL format (for mcp-scholar compatibility) and
    individual JSON files (for mcp-cerebellum compatibility).
    """

    def __init__(
        self,
        queue_dir: str = ".jobs",
        use_jsonl: bool = True,
        stale_threshold_minutes: int = 45,
    ):
        """
        Initialize job queue.

        Args:
            queue_dir: Directory for job files
            use_jsonl: If True, use JSONL files; if False, use individual JSON files
            stale_threshold_minutes: Minutes after which a processing job is considered stale
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True)
        self.use_jsonl = use_jsonl
        self.stale_threshold = timedelta(minutes=stale_threshold_minutes)
        self._lock = threading.RLock()

        if use_jsonl:
            # JSONL format (single file per status)
            self.pending_file = self.queue_dir / "pending_jobs.jsonl"
            self.completed_file = self.queue_dir / "completed_jobs.jsonl"
            self.failed_file = self.queue_dir / "failed_jobs.jsonl"
        else:
            # Individual JSON files format
            self.pending_dir = self.queue_dir / "pending"
            self.completed_dir = self.queue_dir / "completed"
            self.failed_dir = self.queue_dir / "failed"
            self.pending_dir.mkdir(exist_ok=True)
            self.completed_dir.mkdir(exist_ok=True)
            self.failed_dir.mkdir(exist_ok=True)

        self.status_file = self.queue_dir / "status.json"

    def add_job(
        self,
        job_type: str,
        item_key: str,
        collection: str = "default",
        **kwargs
    ) -> str:
        """
        Add a new job to the queue.

        Args:
            job_type: Type of job (e.g., "process_document", "add_to_qdrant")
            item_key: Unique identifier for the item being processed
            collection: Collection name
            **kwargs: Additional job-specific data

        Returns:
            Job ID

        Raises:
            ValueError: If a duplicate job exists
        """
        # Check for existing pending jobs with same type and item_key
        existing_jobs = self.get_pending_jobs()
        for existing in existing_jobs:
            if (existing.get("type") == job_type and
                existing.get("item_key") == item_key and
                existing.get("collection") == collection):
                logger.info(
                    "Job already exists",
                    job_type=job_type,
                    item_key=item_key,
                    existing_id=existing["id"],
                )
                return existing["id"]

        # Create new job
        job_id = f"{job_type}_{item_key}_{int(time.time())}"
        job = Job(
            id=job_id,
            type=job_type,
            item_key=item_key,
            collection=collection,
            status=JobStatus.PENDING,
            created_at=datetime.now().isoformat(),
            kwargs=kwargs,
        )

        with self._lock:
            if self.use_jsonl:
                with open(self.pending_file, "a") as f:
                    f.write(json.dumps(job.to_dict()) + "\n")
            else:
                job_file = self.pending_dir / f"{job_id}.json"
                job_file.write_text(json.dumps(job.to_dict(), indent=2))

        self._update_status()
        logger.info(
            "Job added",
            job_id=job_id,
            job_type=job_type,
            item_key=item_key,
        )
        return job_id

    def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get all pending jobs (excludes processing jobs)."""
        jobs = []

        if self.use_jsonl:
            if self.pending_file.exists():
                with open(self.pending_file, "r") as f:
                    for line in f:
                        if line.strip():
                            job = json.loads(line.strip())
                            if job.get("status") != "processing":
                                jobs.append(job)
        else:
            for job_file in self.pending_dir.glob("*.json"):
                try:
                    job = json.loads(job_file.read_text())
                    if job.get("status") != "processing":
                        jobs.append(job)
                except Exception as e:
                    logger.warning(
                        "Failed to read job file",
                        file=str(job_file),
                        error=str(e),
                    )

        return jobs

    def claim_job(self) -> Optional[Dict[str, Any]]:
        """
        Atomically claim the next available job for processing.

        This prevents race conditions when multiple processors try to claim the same job.

        Returns:
            The claimed job dict, or None if no jobs available
        """
        with self._lock:
            if self.use_jsonl:
                return self._claim_job_jsonl()
            else:
                return self._claim_job_files()

    def _claim_job_jsonl(self) -> Optional[Dict[str, Any]]:
        """Claim job from JSONL format."""
        if not self.pending_file.exists():
            return None

        all_jobs = []
        with open(self.pending_file, "r") as f:
            for line in f:
                if line.strip():
                    all_jobs.append(json.loads(line.strip()))

        if not all_jobs:
            return None

        # Find first claimable job
        job_to_claim = None
        job_index = -1
        current_time = datetime.now()

        for i, job in enumerate(all_jobs):
            status = job.get("status", "pending")

            if status == "pending":
                job_to_claim = job
                job_index = i
                break
            elif status == "processing":
                # Check if stale
                claimed_at = job.get("claimed_at")
                if claimed_at:
                    try:
                        claimed_time = datetime.fromisoformat(claimed_at)
                        if current_time - claimed_time > self.stale_threshold:
                            logger.info(
                                "Reclaiming stale job",
                                job_id=job["id"],
                                claimed_at=claimed_at,
                                claimed_by_pid=job.get("claimed_by_pid"),
                            )
                            job_to_claim = job
                            job_index = i
                            break
                    except (ValueError, TypeError):
                        pass

        if job_to_claim is None:
            return None

        # Mark as processing
        current_pid = os.getpid()
        job_to_claim["status"] = "processing"
        job_to_claim["claimed_at"] = current_time.isoformat()
        job_to_claim["claimed_by_pid"] = current_pid

        all_jobs[job_index] = job_to_claim

        # Rewrite file
        with open(self.pending_file, "w") as f:
            for job in all_jobs:
                f.write(json.dumps(job) + "\n")

        logger.info(
            "Job claimed",
            job_id=job_to_claim["id"],
            pid=current_pid,
        )
        return job_to_claim

    def _claim_job_files(self) -> Optional[Dict[str, Any]]:
        """Claim job from individual JSON files format."""
        job_files = sorted(
            self.pending_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime
        )

        current_time = datetime.now()
        current_pid = os.getpid()

        for job_file in job_files:
            try:
                job = json.loads(job_file.read_text())
                status = job.get("status", "pending")

                claimable = False
                if status == "pending":
                    claimable = True
                elif status == "processing":
                    # Check if stale
                    claimed_at = job.get("claimed_at")
                    if claimed_at:
                        try:
                            claimed_time = datetime.fromisoformat(claimed_at)
                            if current_time - claimed_time > self.stale_threshold:
                                logger.info(
                                    "Reclaiming stale job",
                                    job_id=job["id"],
                                    claimed_at=claimed_at,
                                )
                                claimable = True
                        except (ValueError, TypeError):
                            pass

                if claimable:
                    # Mark as processing
                    job["status"] = "processing"
                    job["claimed_at"] = current_time.isoformat()
                    job["claimed_by_pid"] = current_pid
                    job_file.write_text(json.dumps(job, indent=2))

                    logger.info(
                        "Job claimed",
                        job_id=job["id"],
                        pid=current_pid,
                    )
                    return job

            except Exception as e:
                logger.warning(
                    "Failed to process job file",
                    file=str(job_file),
                    error=str(e),
                )

        return None

    def mark_completed(
        self,
        job: Dict[str, Any],
        result: Any = None
    ) -> None:
        """
        Mark a job as completed.

        Args:
            job: The job dict
            result: Optional result data
        """
        with self._lock:
            job_id = job["id"]

            # Check if already completed
            if self._is_already_completed(job_id):
                logger.info("Job already completed", job_id=job_id)
                return

            job["status"] = "completed"
            job["completed_at"] = datetime.now().isoformat()
            job["result"] = result

            if self.use_jsonl:
                with open(self.completed_file, "a") as f:
                    f.write(json.dumps(job) + "\n")
                self._remove_from_pending_jsonl(job_id)
            else:
                # Move file to completed directory
                pending_file = self.pending_dir / f"{job_id}.json"
                completed_file = self.completed_dir / f"{job_id}.json"
                completed_file.write_text(json.dumps(job, indent=2))
                if pending_file.exists():
                    pending_file.unlink()

        self._update_status()
        logger.info("Job completed", job_id=job_id)

    def mark_failed(
        self,
        job: Dict[str, Any],
        error: str,
        is_retriable: bool = False,
        error_category: Optional[str] = None,
    ) -> None:
        """
        Mark a job as failed.

        Args:
            job: The job dict
            error: Error message
            is_retriable: Whether the error is retriable
            error_category: Category of error for analytics
        """
        with self._lock:
            job_id = job["id"]

            # Check if already failed
            if self._is_already_failed(job_id):
                logger.info("Job already marked as failed", job_id=job_id)
                return

            job["status"] = "failed"
            job["failed_at"] = datetime.now().isoformat()
            job["error"] = error
            job["is_retriable"] = is_retriable
            job["error_category"] = error_category

            if self.use_jsonl:
                with open(self.failed_file, "a") as f:
                    f.write(json.dumps(job) + "\n")
                self._remove_from_pending_jsonl(job_id)
            else:
                # Move file to failed directory
                pending_file = self.pending_dir / f"{job_id}.json"
                failed_file = self.failed_dir / f"{job_id}.json"
                failed_file.write_text(json.dumps(job, indent=2))
                if pending_file.exists():
                    pending_file.unlink()

        self._update_status()
        logger.info(
            "Job failed",
            job_id=job_id,
            error=error,
            is_retriable=is_retriable,
        )

    def remove_from_pending(self, job_id: str) -> None:
        """Remove a job from the pending queue."""
        with self._lock:
            if self.use_jsonl:
                self._remove_from_pending_jsonl(job_id)
            else:
                pending_file = self.pending_dir / f"{job_id}.json"
                if pending_file.exists():
                    pending_file.unlink()

        self._update_status()

    def _remove_from_pending_jsonl(self, job_id: str) -> None:
        """Remove job from JSONL pending file."""
        if not self.pending_file.exists():
            return

        jobs = []
        with open(self.pending_file, "r") as f:
            for line in f:
                if line.strip():
                    job = json.loads(line.strip())
                    if job["id"] != job_id:
                        jobs.append(job)

        with open(self.pending_file, "w") as f:
            for job in jobs:
                f.write(json.dumps(job) + "\n")

    def _is_already_completed(self, job_id: str) -> bool:
        """Check if job is already completed."""
        if self.use_jsonl:
            if not self.completed_file.exists():
                return False
            with open(self.completed_file, "r") as f:
                for line in f:
                    if line.strip():
                        job = json.loads(line.strip())
                        if job["id"] == job_id:
                            return True
            return False
        else:
            return (self.completed_dir / f"{job_id}.json").exists()

    def _is_already_failed(self, job_id: str) -> bool:
        """Check if job is already failed."""
        if self.use_jsonl:
            if not self.failed_file.exists():
                return False
            with open(self.failed_file, "r") as f:
                for line in f:
                    if line.strip():
                        job = json.loads(line.strip())
                        if job["id"] == job_id:
                            return True
            return False
        else:
            return (self.failed_dir / f"{job_id}.json").exists()

    def get_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        if self.status_file.exists():
            try:
                return json.loads(self.status_file.read_text())
            except Exception:
                pass
        return {"pending": 0, "processing": 0, "completed": 0, "failed": 0}

    def _update_status(self) -> None:
        """Update the status file with current queue statistics."""
        pending = 0
        processing = 0
        completed = 0
        failed = 0

        if self.use_jsonl:
            if self.pending_file.exists():
                with open(self.pending_file, "r") as f:
                    for line in f:
                        if line.strip():
                            job = json.loads(line.strip())
                            status = job.get("status", "pending")
                            if status == "pending":
                                pending += 1
                            elif status == "processing":
                                processing += 1

            if self.completed_file.exists():
                with open(self.completed_file, "r") as f:
                    completed = sum(1 for line in f if line.strip())

            if self.failed_file.exists():
                with open(self.failed_file, "r") as f:
                    failed = sum(1 for line in f if line.strip())
        else:
            pending = len(list(self.pending_dir.glob("*.json")))
            completed = len(list(self.completed_dir.glob("*.json")))
            failed = len(list(self.failed_dir.glob("*.json")))

        status = {
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "last_updated": datetime.now().isoformat()
        }

        self.status_file.write_text(json.dumps(status, indent=2))

    def retry_failed_jobs(
        self,
        max_retries: int = 3,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retry failed jobs that are marked as retriable.

        Args:
            max_retries: Maximum number of times a job can be retried
            categories: Specific error categories to retry (None for all retriable)

        Returns:
            Dictionary with retry statistics
        """
        results = {
            "total_failed": 0,
            "retriable_found": 0,
            "retried": 0,
            "categories": {}
        }

        failed_jobs = []
        if self.use_jsonl:
            if self.failed_file.exists():
                with open(self.failed_file, "r") as f:
                    for line in f:
                        if line.strip():
                            failed_jobs.append(json.loads(line.strip()))
        else:
            for job_file in self.failed_dir.glob("*.json"):
                try:
                    failed_jobs.append(json.loads(job_file.read_text()))
                except Exception:
                    pass

        results["total_failed"] = len(failed_jobs)

        for job in failed_jobs:
            is_retriable = job.get("is_retriable", False)
            error_category = job.get("error_category")
            retry_count = job.get("retry_count", 0)

            if not is_retriable:
                continue

            results["retriable_found"] += 1

            if categories and error_category not in categories:
                continue

            if retry_count >= max_retries:
                logger.info(
                    "Job reached max retries",
                    job_id=job["id"],
                    retry_count=retry_count,
                )
                continue

            # Add job back to pending
            new_job = job.copy()
            new_job["status"] = "pending"
            new_job["retry_count"] = retry_count + 1
            new_job["retried_at"] = datetime.now().isoformat()

            # Remove failure fields
            for field in ["failed_at", "error", "error_category", "is_retriable"]:
                new_job.pop(field, None)

            with self._lock:
                if self.use_jsonl:
                    with open(self.pending_file, "a") as f:
                        f.write(json.dumps(new_job) + "\n")
                else:
                    job_file = self.pending_dir / f"{new_job['id']}.json"
                    job_file.write_text(json.dumps(new_job, indent=2))

            results["retried"] += 1
            if error_category:
                if error_category not in results["categories"]:
                    results["categories"][error_category] = 0
                results["categories"][error_category] += 1

            logger.info(
                "Job retried",
                job_id=job["id"],
                attempt=retry_count + 1,
                max_retries=max_retries,
            )

        if results["retried"] > 0:
            self._update_status()

        return results

    def cleanup_duplicates(self) -> int:
        """
        Remove duplicate entries from job files.

        Returns:
            Number of duplicates removed
        """
        removed = 0

        if self.use_jsonl:
            for file_path in [self.completed_file, self.failed_file]:
                if file_path.exists():
                    job_entries = {}
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.strip():
                                job = json.loads(line.strip())
                                job_entries[job["id"]] = job

                    original_count = sum(1 for _ in open(file_path) if _.strip())
                    new_count = len(job_entries)

                    with self._lock:
                        with open(file_path, "w") as f:
                            for job in job_entries.values():
                                f.write(json.dumps(job) + "\n")

                    removed += original_count - new_count

        self._update_status()
        return removed


# Singleton instance
_job_queue: Optional[JobQueue] = None


def get_job_queue(
    queue_dir: str = ".jobs",
    use_jsonl: bool = True,
    force_new: bool = False,
) -> JobQueue:
    """
    Get the singleton job queue instance.

    Args:
        queue_dir: Directory for job files
        use_jsonl: If True, use JSONL format
        force_new: If True, create a new instance

    Returns:
        JobQueue instance
    """
    global _job_queue
    if _job_queue is None or force_new:
        _job_queue = JobQueue(queue_dir=queue_dir, use_jsonl=use_jsonl)
    return _job_queue
