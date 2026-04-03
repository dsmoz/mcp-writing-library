"""
System Notifications for Job Processing

This module provides native macOS notifications to alert users when:
- Jobs complete successfully
- Jobs fail with errors
- Queue processing completes

Features:
- Native macOS notification center integration
- Configurable via environment variables
- Graceful fallback if notifications unavailable
- Support for custom notification sounds
- Grouped notifications by job type

Usage:
    from kbase.core import MacNotifier, get_notifier

    # Get global notifier
    notifier = get_notifier()
    notifier.notify_job_completed(job)

    # Or use convenience functions
    from kbase.core import notify_job_completed, notify_job_failed
    notify_job_completed(job)
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class NotificationConfig:
    """Configuration for the notification system."""

    enabled: bool = True
    app_name: str = "Knowledge Base"
    sound_success: str = "Glass"
    sound_failure: str = "Basso"
    sound_warning: str = "Funk"
    get_item_title_fn: Optional[Callable[[str], str]] = None


class MacNotifier:
    """
    macOS notification system using osascript (AppleScript).

    This uses the built-in macOS osascript command, which doesn't require
    any external dependencies and works on all macOS versions.

    Attributes:
        enabled: Whether notifications are enabled
        app_name: Application name to display in notifications
        platform_supported: Whether the platform supports notifications
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        app_name: str = "Knowledge Base",
        get_item_title_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the macOS notifier.

        Args:
            enabled: Whether notifications are enabled. If None, reads from env var
                    NOTIFICATIONS_ENABLED (default: true)
            app_name: Application name to display in notifications
            get_item_title_fn: Optional callback to get human-readable titles for items
        """
        # Check if notifications are enabled via environment variable
        if enabled is None:
            env_enabled = os.getenv("NOTIFICATIONS_ENABLED", "true").lower()
            enabled = env_enabled in ("true", "1", "yes", "on")

        self.enabled = enabled
        self.app_name = app_name
        self.platform_supported = sys.platform == "darwin"
        self.get_item_title_fn = get_item_title_fn

        if self.enabled and not self.platform_supported:
            print(
                f"Warning: Notifications are enabled but not supported on {sys.platform}",
                file=sys.stderr,
            )
            self.enabled = False

    def _send_notification(
        self,
        title: str,
        message: str,
        sound: Optional[str] = None,
        icon: Optional[str] = None,
    ) -> bool:
        """
        Send a notification using osascript (AppleScript).

        Args:
            title: Notification title
            message: Notification message
            sound: Optional sound name (default: None)
            icon: Optional emoji icon (e.g., "check", "x", "books")

        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.enabled or not self.platform_supported:
            return False

        try:
            # Escape quotes in title and message for AppleScript
            title_escaped = title.replace('"', '\\"').replace("'", "\\'")
            message_escaped = message.replace('"', '\\"').replace("'", "\\'")

            # Add icon emoji to title if provided
            if icon:
                title_with_icon = f"{icon} {title_escaped}"
            else:
                title_with_icon = title_escaped

            # Build AppleScript command
            script = (
                f'display notification "{message_escaped}" '
                f'with title "{self.app_name}" '
                f'subtitle "{title_with_icon}"'
            )

            # Add sound if specified
            if sound:
                script += f' sound name "{sound}"'

            # Execute AppleScript
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                print(f"Notification error: {result.stderr}", file=sys.stderr)
                return False

            return True

        except subprocess.TimeoutExpired:
            print("Notification timeout - osascript took too long", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error sending notification: {e}", file=sys.stderr)
            return False

    def _get_item_title(self, item_key: str) -> str:
        """
        Get the title of an item by its key.

        Uses the callback if provided, otherwise returns the key.

        Args:
            item_key: Item key

        Returns:
            Item title or item key if title not found
        """
        if self.get_item_title_fn is None:
            return item_key

        try:
            title = self.get_item_title_fn(item_key)
            if title:
                # Truncate very long titles
                return title[:80] + "..." if len(title) > 80 else title
            return item_key
        except Exception as e:
            print(f"Error fetching item title for {item_key}: {e}", file=sys.stderr)
            return item_key

    def notify_job_completed(self, job: Dict[str, Any]) -> bool:
        """
        Send notification when a job completes successfully.

        Args:
            job: Job dictionary containing job details

        Returns:
            True if notification was sent
        """
        job_type = job.get("type", "Unknown")
        item_key = job.get("item_key", "Unknown")

        # Get human-readable job type
        job_type_name = self._get_job_type_name(job_type)

        # Get item title instead of just the key
        item_title = self._get_item_title(item_key)

        title = f"Job Completed: {job_type_name}"
        message = f"{item_title}"

        # Add additional context if available
        if "result" in job and job["result"]:
            result = job["result"]
            if isinstance(result, dict):
                if "chunks_processed" in result:
                    message += f" - {result['chunks_processed']} chunks"
                elif "documents_added" in result:
                    message += f" - {result['documents_added']} documents"

        return self._send_notification(title, message, sound="Glass", icon="[OK]")

    def notify_job_failed(self, job: Dict[str, Any]) -> bool:
        """
        Send notification when a job fails.

        Args:
            job: Job dictionary containing job details and error

        Returns:
            True if notification was sent
        """
        job_type = job.get("type", "Unknown")
        item_key = job.get("item_key", "Unknown")
        error = job.get("error", "Unknown error")

        # Get human-readable job type
        job_type_name = self._get_job_type_name(job_type)

        # Get item title instead of just the key
        item_title = self._get_item_title(item_key)

        title = f"Job Failed: {job_type_name}"

        # Truncate error message if too long
        error_short = error[:80] + "..." if len(error) > 80 else error
        message = f"{item_title} - {error_short}"

        return self._send_notification(title, message, sound="Basso", icon="[FAIL]")

    def notify_queue_completed(
        self,
        completed_count: int,
        failed_count: int,
        pending_count: int = 0,
    ) -> bool:
        """
        Send notification when the entire queue finishes processing.

        Args:
            completed_count: Number of successfully completed jobs
            failed_count: Number of failed jobs
            pending_count: Number of jobs still pending (optional)

        Returns:
            True if notification was sent
        """
        title = "Queue Processing Complete"

        if failed_count == 0:
            message = f"All {completed_count} jobs completed successfully"
            sound = "Glass"
            icon = "[OK]"
        else:
            message = f"{completed_count} completed, {failed_count} failed"
            sound = "Funk"
            icon = "[!]"

        # Add pending count if there are still jobs in the queue
        if pending_count > 0:
            message += f" - {pending_count} pending"

        return self._send_notification(title, message, sound=sound, icon=icon)

    def notify_batch_completed(
        self,
        job_type: str,
        count: int,
        pending_count: int = 0,
    ) -> bool:
        """
        Send notification when a batch of jobs of the same type completes.

        Args:
            job_type: Type of jobs completed
            count: Number of jobs in the batch
            pending_count: Number of jobs still pending (optional)

        Returns:
            True if notification was sent
        """
        job_type_name = self._get_job_type_name(job_type)
        title = f"Batch Complete: {job_type_name}"
        message = f"{count} items processed"

        # Add pending count if provided
        if pending_count > 0:
            message += f" - {pending_count} remaining"

        return self._send_notification(title, message, sound="Glass", icon="[OK]")

    def notify_queue_status(
        self,
        pending_count: int,
        completed_count: int,
        failed_count: int,
    ) -> bool:
        """
        Send notification about current queue status.

        Args:
            pending_count: Number of jobs pending
            completed_count: Number of jobs completed
            failed_count: Number of jobs failed

        Returns:
            True if notification was sent
        """
        title = "Queue Status Update"

        # Build status message
        parts = []
        if pending_count > 0:
            parts.append(f"{pending_count} pending")
        if completed_count > 0:
            parts.append(f"{completed_count} completed")
        if failed_count > 0:
            parts.append(f"{failed_count} failed")

        message = " - ".join(parts) if parts else "Queue empty"

        # Choose sound and icon based on status
        if failed_count > 0:
            sound = "Funk"
            icon = "[!]"
        elif pending_count > 0:
            sound = None
            icon = "[...]"
        else:
            sound = "Glass"
            icon = "[OK]"

        return self._send_notification(title, message, sound=sound, icon=icon)

    def notify_custom(
        self,
        title: str,
        message: str,
        sound: Optional[str] = None,
        icon: Optional[str] = None,
    ) -> bool:
        """
        Send a custom notification.

        Args:
            title: Notification title
            message: Notification message
            sound: Optional sound name
            icon: Optional icon prefix

        Returns:
            True if notification was sent
        """
        return self._send_notification(title, message, sound=sound, icon=icon)

    def _get_job_type_name(self, job_type: str) -> str:
        """
        Convert job type to human-readable name.

        Args:
            job_type: Internal job type identifier

        Returns:
            Human-readable job type name
        """
        type_names = {
            "add_to_qdrant": "Index to Qdrant",
            "add_to_qdrant_chunk": "Index PDF Chunk",
            "update_metadata": "Update Metadata",
            "add_new_item": "Add New Item",
            "add_document": "Add Document",
            "delete_document": "Delete Document",
            "reindex": "Reindex",
        }
        return type_names.get(job_type, job_type.replace("_", " ").title())


# Global singleton instance
_notifier_instance: Optional[MacNotifier] = None


def get_notifier(
    app_name: Optional[str] = None,
    get_item_title_fn: Optional[Callable[[str], str]] = None,
) -> MacNotifier:
    """
    Get the global notifier instance (singleton pattern).

    Args:
        app_name: Optional app name to use (only used on first call)
        get_item_title_fn: Optional callback to get item titles (only used on first call)

    Returns:
        The global MacNotifier instance
    """
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = MacNotifier(
            app_name=app_name or "Knowledge Base",
            get_item_title_fn=get_item_title_fn,
        )
    return _notifier_instance


def reset_notifier() -> None:
    """Reset the global notifier instance."""
    global _notifier_instance
    _notifier_instance = None


def notify_job_completed(job: Dict[str, Any]) -> bool:
    """
    Convenience function to notify about completed job.

    Args:
        job: Job dictionary

    Returns:
        True if notification was sent
    """
    return get_notifier().notify_job_completed(job)


def notify_job_failed(job: Dict[str, Any]) -> bool:
    """
    Convenience function to notify about failed job.

    Args:
        job: Job dictionary

    Returns:
        True if notification was sent
    """
    return get_notifier().notify_job_failed(job)


def notify_queue_completed(
    completed_count: int,
    failed_count: int,
    pending_count: int = 0,
) -> bool:
    """
    Convenience function to notify about queue completion.

    Args:
        completed_count: Number of completed jobs
        failed_count: Number of failed jobs
        pending_count: Number of jobs still pending

    Returns:
        True if notification was sent
    """
    return get_notifier().notify_queue_completed(completed_count, failed_count, pending_count)


def notify_batch_completed(
    job_type: str,
    count: int,
    pending_count: int = 0,
) -> bool:
    """
    Convenience function to notify about batch completion.

    Args:
        job_type: Type of jobs completed
        count: Number of jobs
        pending_count: Number of jobs still pending

    Returns:
        True if notification was sent
    """
    return get_notifier().notify_batch_completed(job_type, count, pending_count)


def notify_queue_status(
    pending_count: int,
    completed_count: int,
    failed_count: int,
) -> bool:
    """
    Convenience function to notify about queue status.

    Args:
        pending_count: Number of pending jobs
        completed_count: Number of completed jobs
        failed_count: Number of failed jobs

    Returns:
        True if notification was sent
    """
    return get_notifier().notify_queue_status(pending_count, completed_count, failed_count)


__all__ = [
    "NotificationConfig",
    "MacNotifier",
    "get_notifier",
    "reset_notifier",
    "notify_job_completed",
    "notify_job_failed",
    "notify_queue_completed",
    "notify_batch_completed",
    "notify_queue_status",
]
