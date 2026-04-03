"""
Processor Pause Management

This module provides functionality to pause and resume the job processor
for maintenance, debugging, or when you need to temporarily stop processing jobs.

Features:
- Pause processor for specified duration (minutes or hours)
- Resume processor manually or automatically after duration
- Check pause status
- Cancel active pause
- Persistent pause state across restarts

Based on mcp-zotero-qdrant's processor_pause.py.

Usage:
    from kbase.jobs.pause import ProcessorPauseManager, get_pause_manager

    # Pause for 30 minutes
    manager = ProcessorPauseManager()
    manager.pause_processor(minutes=30)

    # Pause for 2 hours
    manager.pause_processor(hours=2)

    # Check if paused
    if manager.is_paused():
        print(f"Paused until: {manager.get_pause_until()}")

    # Resume early
    manager.resume_processor()
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict

import structlog

logger = structlog.get_logger(__name__)


class ProcessorPauseManager:
    """Manages pausing and resuming the job processor."""

    def __init__(self, queue_dir: str = ".jobs"):
        """
        Initialize pause manager.

        Args:
            queue_dir: Directory for job queue files
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True)
        self.pause_file = self.queue_dir / "processor_pause.json"

    def pause_processor(
        self,
        minutes: int = 0,
        hours: int = 0,
        reason: str = ""
    ) -> Dict:
        """
        Pause the job processor for a specified duration.

        Args:
            minutes: Number of minutes to pause (default: 0)
            hours: Number of hours to pause (default: 0)
            reason: Optional reason for pausing (for logging/tracking)

        Returns:
            Dict with pause status and details

        Raises:
            ValueError: If both minutes and hours are 0 or negative
        """
        if minutes <= 0 and hours <= 0:
            raise ValueError("Must specify positive duration in minutes or hours")

        # Calculate pause duration and end time
        duration = timedelta(minutes=minutes, hours=hours)
        pause_until = datetime.now() + duration

        # Create pause state
        pause_state = {
            "paused": True,
            "paused_at": datetime.now().isoformat(),
            "pause_until": pause_until.isoformat(),
            "duration_minutes": int(duration.total_seconds() / 60),
            "reason": reason or "Manual pause",
            "can_resume_early": True
        }

        # Save pause state
        with open(self.pause_file, 'w') as f:
            json.dump(pause_state, f, indent=2)

        logger.info(
            "Processor paused",
            hours=hours,
            minutes=minutes,
            until=pause_until.isoformat(),
            reason=reason or "Manual pause"
        )

        return {
            "success": True,
            "paused": True,
            "pause_until": pause_until.isoformat(),
            "duration_minutes": int(duration.total_seconds() / 60),
            "reason": pause_state["reason"],
            "message": f"Processor paused until {pause_until.strftime('%Y-%m-%d %H:%M:%S')}"
        }

    def resume_processor(self, force: bool = False) -> Dict:
        """
        Resume the job processor.

        Args:
            force: If True, resume even if pause duration hasn't expired

        Returns:
            Dict with resume status and details
        """
        if not self.pause_file.exists():
            return {
                "success": False,
                "paused": False,
                "message": "Processor is not paused"
            }

        # Read current pause state
        try:
            with open(self.pause_file, 'r') as f:
                pause_state = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Error reading pause state", error=str(e))
            self.pause_file.unlink(missing_ok=True)
            return {
                "success": False,
                "paused": False,
                "message": f"Error reading pause state: {e}"
            }

        if not pause_state.get("paused"):
            return {
                "success": False,
                "paused": False,
                "message": "Processor is not paused"
            }

        # Check if we can resume early
        pause_until = datetime.fromisoformat(pause_state["pause_until"])
        now = datetime.now()

        if now < pause_until and not force:
            remaining_minutes = int((pause_until - now).total_seconds() / 60)
            return {
                "success": False,
                "paused": True,
                "pause_until": pause_state["pause_until"],
                "remaining_minutes": remaining_minutes,
                "message": f"Pause is still active. {remaining_minutes} minutes remaining. Use force=True to resume early."
            }

        # Resume processor by removing pause file
        self.pause_file.unlink()

        resume_type = "early" if now < pause_until else "scheduled"
        logger.info("Processor resumed", resume_type=resume_type)

        return {
            "success": True,
            "paused": False,
            "resumed_type": resume_type,
            "paused_at": pause_state["paused_at"],
            "pause_until": pause_state["pause_until"],
            "message": f"Processor resumed ({resume_type})"
        }

    def is_paused(self) -> bool:
        """
        Check if the processor is currently paused.

        Returns:
            bool: True if paused, False otherwise
        """
        if not self.pause_file.exists():
            return False

        try:
            with open(self.pause_file, 'r') as f:
                pause_state = json.load(f)

            if not pause_state.get("paused"):
                return False

            # Check if pause has expired
            pause_until = datetime.fromisoformat(pause_state["pause_until"])
            now = datetime.now()

            if now >= pause_until:
                # Pause has expired, auto-resume
                self.pause_file.unlink()
                logger.info(
                    "Pause expired, auto-resuming",
                    expired_at=pause_until.isoformat()
                )
                return False

            return True

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Error reading pause state, removing file", error=str(e))
            # If pause file is corrupt, remove it
            self.pause_file.unlink(missing_ok=True)
            return False

    def get_pause_status(self) -> Dict:
        """
        Get detailed pause status.

        Returns:
            Dict with pause status details
        """
        if not self.pause_file.exists():
            return {
                "paused": False,
                "message": "Processor is running normally"
            }

        try:
            with open(self.pause_file, 'r') as f:
                pause_state = json.load(f)

            if not pause_state.get("paused"):
                return {
                    "paused": False,
                    "message": "Processor is running normally"
                }

            pause_until = datetime.fromisoformat(pause_state["pause_until"])
            paused_at = datetime.fromisoformat(pause_state["paused_at"])
            now = datetime.now()

            # Check if expired
            if now >= pause_until:
                self.pause_file.unlink()
                return {
                    "paused": False,
                    "was_paused": True,
                    "expired_at": pause_until.isoformat(),
                    "message": "Pause expired, processor auto-resumed"
                }

            # Calculate remaining time
            remaining = pause_until - now
            remaining_minutes = int(remaining.total_seconds() / 60)
            elapsed = now - paused_at
            elapsed_minutes = int(elapsed.total_seconds() / 60)

            return {
                "paused": True,
                "paused_at": pause_state["paused_at"],
                "pause_until": pause_state["pause_until"],
                "duration_minutes": pause_state["duration_minutes"],
                "remaining_minutes": remaining_minutes,
                "elapsed_minutes": elapsed_minutes,
                "reason": pause_state.get("reason", ""),
                "can_resume_early": pause_state.get("can_resume_early", True),
                "message": f"Processor paused for {remaining_minutes} more minutes (until {pause_until.strftime('%H:%M:%S')})"
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Error reading pause status, removing file", error=str(e))
            # If pause file is corrupt, remove it
            self.pause_file.unlink(missing_ok=True)
            return {
                "paused": False,
                "error": str(e),
                "message": "Pause state corrupted, auto-resumed"
            }

    def get_pause_until(self) -> Optional[str]:
        """
        Get the time when pause will expire.

        Returns:
            ISO format datetime string or None if not paused
        """
        status = self.get_pause_status()
        return status.get("pause_until")

    def cancel_pause(self) -> Dict:
        """
        Cancel an active pause (alias for resume_processor with force=True).

        Returns:
            Dict with cancellation status
        """
        return self.resume_processor(force=True)


# Singleton instance for easy access
_pause_manager: Optional[ProcessorPauseManager] = None


def get_pause_manager(queue_dir: str = ".jobs") -> ProcessorPauseManager:
    """
    Get the global pause manager instance.

    Args:
        queue_dir: Queue directory (only used if creating new instance)

    Returns:
        ProcessorPauseManager instance
    """
    global _pause_manager

    if _pause_manager is None:
        _pause_manager = ProcessorPauseManager(queue_dir)

    return _pause_manager
