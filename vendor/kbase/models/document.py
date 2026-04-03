"""
Document models for kbase-core
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field, HttpUrl

from kbase.models.base import BaseModel, IDMixin, MetadataMixin, TimestampMixin


class DocumentSource(str, Enum):
    """Source types for documents."""

    ZOTERO = "zotero"
    MS365 = "ms365"
    GOOGLE = "google"
    FILE_UPLOAD = "file_upload"
    WEB_SCRAPE = "web_scrape"
    API = "api"


class DocumentMetadata(BaseModel):
    """Extended metadata for documents."""

    # File information
    filename: Optional[str] = None
    file_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None

    # Source-specific
    source_id: Optional[str] = None  # ID in source system
    source_url: Optional[str] = None  # URL in source system
    parent_id: Optional[str] = None  # Parent document if nested

    # Content
    language: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # Timestamps
    source_created_at: Optional[datetime] = None
    source_updated_at: Optional[datetime] = None

    # Processing
    is_indexed: bool = False
    index_version: Optional[str] = None
    chunk_count: Optional[int] = None
    embedding_model: Optional[str] = None

    # Custom fields
    custom: Dict[str, Any] = Field(default_factory=dict)


class DocumentBase(BaseModel):
    """Base document fields."""

    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: Optional[str] = Field(None, description="Full text content")
    source: DocumentSource = Field(..., description="Document source")
    url: Optional[str] = Field(None, description="Document URL")


class DocumentCreate(DocumentBase):
    """Schema for creating a new document."""

    collection_ids: List[UUID] = Field(
        default_factory=list,
        description="Collection IDs to add document to",
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata,
        description="Extended metadata",
    )


class SyncDocumentCreate(BaseModel):
    """
    Simplified document creation schema for MCP server sync operations.

    This model matches the database schema directly and is used by
    SyncDocumentManager for synchronous document creation.
    """

    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: Optional[str] = Field(None, description="Full text content")
    source_type: Optional[str] = Field(None, description="Source type (manual, upload, zotero, etc.)")
    source_id: Optional[str] = Field(None, description="ID in source system")
    collection_id: Optional[UUID] = Field(None, description="Primary collection ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    # File tracking
    file_path: Optional[str] = Field(None, description="Path to source file")
    file_name: Optional[str] = Field(None, description="Original file name")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File content hash")
    file_modified_at: Optional[datetime] = Field(None, description="File modification time")
    mime_type: Optional[str] = Field(None, description="MIME type")
    storage_type: Optional[str] = Field("database", description="Storage type")


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""

    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None


class Document(DocumentBase, IDMixin, TimestampMixin):
    """
    Full document model with all fields.

    Used for database operations and API responses.
    """

    owner_id: str = Field(..., description="User ID of the document owner")
    collection_ids: List[UUID] = Field(
        default_factory=list,
        description="Collections this document belongs to",
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata,
        description="Extended metadata",
    )

    # Vector database reference
    qdrant_ids: List[str] = Field(
        default_factory=list,
        description="Qdrant point IDs for document chunks",
    )

    # Statistics
    view_count: int = Field(default=0, description="Number of views")
    last_accessed: Optional[datetime] = Field(None, description="Last access time")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "660e8400-e29b-41d4-a716-446655440001",
                "title": "Deep Learning Research Paper",
                "content": "Abstract: This paper presents...",
                "source": "zotero",
                "url": "https://example.com/paper.pdf",
                "owner_id": "user-123",
                "collection_ids": ["550e8400-e29b-41d4-a716-446655440000"],
                "metadata": {
                    "filename": "paper.pdf",
                    "file_type": "pdf",
                    "file_size_bytes": 524288,
                    "author": "John Doe",
                    "tags": ["ml", "deep-learning"],
                    "is_indexed": True,
                    "chunk_count": 25,
                },
                "qdrant_ids": ["chunk-1", "chunk-2", "chunk-3"],
                "view_count": 5,
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T11:30:00Z",
            }
        }
