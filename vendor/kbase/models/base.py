"""
Base Pydantic models and mixins for kbase-core
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field


class BaseModel(PydanticBaseModel):
    """
    Base model for all kbase models.

    Provides common configuration and utilities.
    """

    model_config = ConfigDict(
        from_attributes=True,  # Allow ORM mode
        populate_by_name=True,  # Allow population by field name
        str_strip_whitespace=True,  # Strip whitespace from strings
        json_schema_extra={"additionalProperties": False},  # Strict schema
    )


class TimestampMixin:
    """
    Mixin for models with created_at and updated_at timestamps.

    Note: This is a plain class, not a BaseModel subclass, to avoid MRO conflicts.
    """

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the record was last updated",
    )


class IDMixin:
    """
    Mixin for models with UUID primary key.

    Note: This is a plain class, not a BaseModel subclass, to avoid MRO conflicts.
    """

    id: UUID = Field(..., description="Unique identifier")


class MetadataMixin:
    """
    Mixin for models with JSON metadata field.

    Note: This is a plain class, not a BaseModel subclass, to avoid MRO conflicts.
    """

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata as JSON",
    )
