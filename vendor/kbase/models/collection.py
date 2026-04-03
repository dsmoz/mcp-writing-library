"""
Collection models for kbase-core
"""

from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import Field

from kbase.models.base import BaseModel, IDMixin, MetadataMixin, TimestampMixin


class PermissionLevel(str, Enum):
    """Permission levels for collection access."""

    ADMIN = "admin"  # Full access: read, write, delete, share
    EDITOR = "editor"  # Can read and write
    VIEWER = "viewer"  # Can only read


class CollectionPermission(BaseModel):
    """Permission for a user on a collection."""

    user_id: str = Field(..., description="User ID")
    level: PermissionLevel = Field(..., description="Permission level")


class CollectionBase(BaseModel):
    """Base collection fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Collection name")
    description: Optional[str] = Field(None, description="Collection description")
    tags: List[str] = Field(default_factory=list, description="Collection tags")
    is_public: bool = Field(default=False, description="Whether collection is public")
    allowed_sources: List[str] = Field(
        default_factory=list,
        description="Allowed connector sources (e.g., 'zotero', 'ms365')",
    )


class CollectionCreate(CollectionBase):
    """Schema for creating a new collection."""

    pass


class CollectionUpdate(BaseModel):
    """Schema for updating a collection."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    allowed_sources: Optional[List[str]] = None


class Collection(CollectionBase, IDMixin, TimestampMixin, MetadataMixin):
    """
    Full collection model with all fields.

    Used for database operations and API responses.
    """

    owner_id: str = Field(..., description="User ID of the collection owner")
    permissions: Dict[str, str] = Field(
        default_factory=dict,
        description="User permissions map (user_id -> permission_level)",
    )

    # Statistics (computed)
    document_count: int = Field(default=0, description="Number of documents")
    total_size_bytes: int = Field(default=0, description="Total size of documents")
    last_synced: Optional[str] = Field(None, description="Last sync timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Research Papers",
                "description": "Academic papers and research documents",
                "owner_id": "user-123",
                "tags": ["research", "ml", "ai"],
                "is_public": False,
                "allowed_sources": ["zotero", "file_upload"],
                "permissions": {
                    "user-456": "editor",
                    "user-789": "viewer",
                },
                "document_count": 42,
                "total_size_bytes": 1048576,
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T11:30:00Z",
                "metadata": {},
            }
        }
