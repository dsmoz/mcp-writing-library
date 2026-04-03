"""
Connector sync models for kbase-core
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import Field

from kbase.models.base import BaseModel, TimestampMixin


class ConnectorType(str, Enum):
    """Supported connector types."""

    ZOTERO = "zotero"
    MS365 = "ms365"
    GOOGLE = "google"
    FILE_SYSTEM = "file_system"
    WEB_SCRAPER = "web_scraper"


class SyncStatus(str, Enum):
    """Sync operation status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ConnectorSync(BaseModel, TimestampMixin):
    """
    Connector sync tracking model.

    Tracks synchronization operations for connectors.
    """

    id: str = Field(..., description="Sync operation ID")
    user_id: str = Field(..., description="User ID")
    connector_type: ConnectorType = Field(..., description="Connector type")
    collection_id: Optional[str] = Field(None, description="Target collection ID")

    # Status
    status: SyncStatus = Field(default=SyncStatus.PENDING, description="Sync status")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Sync details
    items_total: int = Field(default=0, description="Total items to sync")
    items_processed: int = Field(default=0, description="Items processed")
    items_succeeded: int = Field(default=0, description="Items successfully synced")
    items_failed: int = Field(default=0, description="Items failed to sync")

    # Timestamps
    started_at: Optional[datetime] = Field(None, description="Sync start time")
    completed_at: Optional[datetime] = Field(None, description="Sync completion time")

    # Metadata
    last_sync_token: Optional[str] = Field(
        None,
        description="Token for incremental sync",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Connector configuration",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional sync metadata",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "sync-123",
                "user_id": "user-123",
                "connector_type": "zotero",
                "collection_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "items_total": 100,
                "items_processed": 100,
                "items_succeeded": 98,
                "items_failed": 2,
                "started_at": "2025-01-15T10:00:00Z",
                "completed_at": "2025-01-15T10:15:00Z",
                "last_sync_token": "2025-01-15T10:00:00Z",
                "config": {"library_id": "12345"},
                "metadata": {},
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:15:00Z",
            }
        }
