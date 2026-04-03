"""
API Key Manager

Handles API key generation, validation, and management.
"""

import secrets
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from kbase.database.pool import DatabasePool
from kbase.models import APIKey, APIKeyCreate


class APIKeyManager:
    """Manager for API key operations."""

    def __init__(self, pool: DatabasePool):
        """
        Initialize API key manager.

        Args:
            pool: Database connection pool
        """
        self.pool = pool

    def generate_key(self) -> str:
        """
        Generate a secure API key.

        Returns:
            Secure random API key string (32 bytes = 64 hex chars)
        """
        return secrets.token_urlsafe(32)

    async def create(self, data: APIKeyCreate, user_id: str) -> APIKey:
        """
        Create a new API key for a user.

        Args:
            data: API key creation data
            user_id: User ID who owns this key

        Returns:
            Created API key object
        """
        # Generate secure key
        key = self.generate_key()

        query = """
            INSERT INTO api_keys (
                user_id, name, key, expires_at
            )
            VALUES ($1, $2, $3, $4)
            RETURNING id, user_id, name, key, expires_at, created_at, last_used_at
        """

        row = await self.pool.fetchrow(
            query,
            user_id,
            data.name,
            key,
            data.expires_at,
        )

        return APIKey.model_validate(dict(row))

    async def get_by_key(self, key: str) -> Optional[APIKey]:
        """
        Get API key by key string.

        Args:
            key: API key string

        Returns:
            API key object if found, None otherwise
        """
        query = """
            SELECT id, user_id, name, key, expires_at, created_at, last_used_at
            FROM api_keys
            WHERE key = $1
        """

        row = await self.pool.fetchrow(query, key)

        if not row:
            return None

        return APIKey.model_validate(dict(row))

    async def validate_key(self, key: str) -> Optional[str]:
        """
        Validate an API key and return the associated user ID.

        Checks:
        - Key exists
        - Key hasn't expired
        - Updates last_used_at timestamp

        Args:
            key: API key string to validate

        Returns:
            User ID if key is valid, None otherwise
        """
        api_key = await self.get_by_key(key)

        if not api_key:
            return None

        # Check if key has expired
        if api_key.expires_at:
            now = datetime.now(timezone.utc)
            if api_key.expires_at < now:
                return None

        # Update last_used_at
        await self.update_last_used(api_key.id)

        return api_key.user_id

    async def update_last_used(self, key_id: UUID) -> None:
        """
        Update the last_used_at timestamp for an API key.

        Args:
            key_id: API key ID
        """
        query = """
            UPDATE api_keys
            SET last_used_at = $1
            WHERE id = $2
        """

        await self.pool.execute(query, datetime.now(timezone.utc), key_id)

    async def list_for_user(self, user_id: str) -> list[APIKey]:
        """
        List all API keys for a user.

        Args:
            user_id: User ID

        Returns:
            List of API keys
        """
        query = """
            SELECT id, user_id, name, key, expires_at, created_at, last_used_at
            FROM api_keys
            WHERE user_id = $1
            ORDER BY created_at DESC
        """

        rows = await self.pool.fetch(query, user_id)

        return [APIKey.model_validate(dict(row)) for row in rows]

    async def delete(self, key_id: UUID, user_id: str) -> bool:
        """
        Delete an API key.

        Args:
            key_id: API key ID
            user_id: User ID (for authorization check)

        Returns:
            True if deleted, False if not found or unauthorized
        """
        query = """
            DELETE FROM api_keys
            WHERE id = $1 AND user_id = $2
        """

        result = await self.pool.execute(query, key_id, user_id)

        # Check if any rows were deleted
        return result.split()[-1] == "1"

    async def revoke(self, key_id: UUID, user_id: str) -> bool:
        """
        Revoke an API key by setting its expiration to now.

        Args:
            key_id: API key ID
            user_id: User ID (for authorization check)

        Returns:
            True if revoked, False if not found or unauthorized
        """
        query = """
            UPDATE api_keys
            SET expires_at = $1
            WHERE id = $2 AND user_id = $3
        """

        result = await self.pool.execute(
            query,
            datetime.now(timezone.utc),
            key_id,
            user_id,
        )

        # Check if any rows were updated
        return result.split()[-1] == "1"
