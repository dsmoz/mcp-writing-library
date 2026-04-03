"""
User and authentication models for kbase-core
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import EmailStr, Field

from kbase.models.base import BaseModel, MetadataMixin, TimestampMixin


class UserBase(BaseModel):
    """Base user fields."""

    email: EmailStr = Field(..., description="User email address")
    name: Optional[str] = Field(None, max_length=255, description="User full name")


class UserCreate(UserBase):
    """Schema for creating a new user."""

    password: Optional[str] = Field(None, min_length=8, description="User password")


class UserUpdate(BaseModel):
    """Schema for updating a user."""

    name: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = None


class User(UserBase, TimestampMixin, MetadataMixin):
    """
    Full user model with all fields.

    Used for database operations and API responses.
    """

    id: str = Field(..., description="User ID")
    is_active: bool = Field(default=True, description="Whether user is active")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "user-123",
                "email": "user@example.com",
                "name": "John Doe",
                "is_active": True,
                "last_login": "2025-01-15T10:00:00Z",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-15T10:00:00Z",
                "metadata": {},
            }
        }


class APIKeyBase(BaseModel):
    """Base API key fields."""

    name: Optional[str] = Field(None, max_length=255, description="Key name/description")
    scopes: List[str] = Field(
        default_factory=lambda: ["read", "write"],
        description="API key scopes",
    )


class APIKeyCreate(APIKeyBase):
    """Schema for creating a new API key."""

    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")


class APIKey(APIKeyBase, TimestampMixin, MetadataMixin):
    """
    Full API key model.

    Note: The actual key is never stored, only the hash.
    """

    key_hash: str = Field(..., description="Hashed API key")
    user_id: str = Field(..., description="User ID this key belongs to")
    is_active: bool = Field(default=True, description="Whether key is active")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "key_hash": "sha256:abcdef123456...",
                "user_id": "user-123",
                "name": "Production API Key",
                "scopes": ["read", "write"],
                "is_active": True,
                "last_used": "2025-01-15T10:00:00Z",
                "expires_at": "2026-01-15T00:00:00Z",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-15T10:00:00Z",
                "metadata": {},
            }
        }
