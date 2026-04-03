"""
Collection database manager
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

import structlog

from kbase.database.pool import DatabasePool
from kbase.models.collection import (
    Collection,
    CollectionCreate,
    CollectionUpdate,
    PermissionLevel,
)

logger = structlog.get_logger(__name__)


class CollectionManager:
    """
    Manager for collection database operations.

    Handles CRUD operations, permissions, and search for collections.
    """

    def __init__(self, pool: DatabasePool):
        """
        Initialize collection manager.

        Args:
            pool: Database connection pool
        """
        self.pool = pool

    async def create(
        self,
        data: CollectionCreate,
        owner_id: str,
    ) -> Collection:
        """
        Create a new collection.

        Args:
            data: Collection creation data
            owner_id: User ID of the collection owner

        Returns:
            Created collection

        Raises:
            ValueError: If validation fails
        """
        query = """
            INSERT INTO collections (
                name, description, owner_id, tags, is_public,
                allowed_sources, permissions, metadata, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
        """

        now = datetime.utcnow()
        permissions = {owner_id: PermissionLevel.ADMIN.value}

        try:
            row = await self.pool.fetchrow(
                query,
                data.name,
                data.description,
                owner_id,
                data.tags,
                data.is_public,
                data.allowed_sources,
                permissions,
                {},  # metadata
                now,
                now,
            )

            if not row:
                raise RuntimeError("Failed to create collection")

            logger.info(
                "Collection created",
                collection_id=str(row["id"]),
                owner_id=owner_id,
                name=data.name,
            )

            return Collection(**dict(row))

        except Exception as e:
            logger.error("Failed to create collection", error=str(e), owner_id=owner_id)
            raise

    async def get(self, collection_id: UUID) -> Optional[Collection]:
        """
        Get a collection by ID.

        Args:
            collection_id: Collection UUID

        Returns:
            Collection if found, None otherwise
        """
        query = """
            SELECT * FROM collections WHERE id = $1
        """

        try:
            row = await self.pool.fetchrow(query, collection_id)

            if not row:
                return None

            return Collection(**dict(row))

        except Exception as e:
            logger.error(
                "Failed to get collection",
                error=str(e),
                collection_id=str(collection_id),
            )
            raise

    async def list_for_user(
        self,
        user_id: str,
        include_public: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Collection]:
        """
        List collections accessible by a user.

        Returns collections where user is:
        - Owner
        - Has explicit permissions
        - Public collections (if include_public=True)

        Args:
            user_id: User ID
            include_public: Include public collections
            limit: Maximum results
            offset: Result offset for pagination

        Returns:
            List of collections
        """
        query = """
            SELECT * FROM collections
            WHERE 
                owner_id = $1
                OR permissions ? $1
                OR ($2 AND is_public = TRUE)
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
        """

        try:
            rows = await self.pool.fetch(query, user_id, include_public, limit, offset)
            return [Collection(**dict(row)) for row in rows]

        except Exception as e:
            logger.error("Failed to list collections", error=str(e), user_id=user_id)
            raise

    async def update(
        self,
        collection_id: UUID,
        data: CollectionUpdate,
    ) -> Optional[Collection]:
        """
        Update a collection.

        Args:
            collection_id: Collection UUID
            data: Update data

        Returns:
            Updated collection if found, None otherwise
        """
        # Build dynamic update query
        updates = []
        values = []
        param_count = 1

        if data.name is not None:
            updates.append(f"name = ${param_count}")
            values.append(data.name)
            param_count += 1

        if data.description is not None:
            updates.append(f"description = ${param_count}")
            values.append(data.description)
            param_count += 1

        if data.tags is not None:
            updates.append(f"tags = ${param_count}")
            values.append(data.tags)
            param_count += 1

        if data.is_public is not None:
            updates.append(f"is_public = ${param_count}")
            values.append(data.is_public)
            param_count += 1

        if data.allowed_sources is not None:
            updates.append(f"allowed_sources = ${param_count}")
            values.append(data.allowed_sources)
            param_count += 1

        if not updates:
            # No updates provided
            return await self.get(collection_id)

        # Add updated_at
        updates.append(f"updated_at = ${param_count}")
        values.append(datetime.utcnow())
        param_count += 1

        # Add collection_id as last parameter
        values.append(collection_id)

        query = f"""
            UPDATE collections
            SET {', '.join(updates)}
            WHERE id = ${param_count}
            RETURNING *
        """

        try:
            row = await self.pool.fetchrow(query, *values)

            if not row:
                return None

            logger.info("Collection updated", collection_id=str(collection_id))
            return Collection(**dict(row))

        except Exception as e:
            logger.error(
                "Failed to update collection",
                error=str(e),
                collection_id=str(collection_id),
            )
            raise

    async def delete(self, collection_id: UUID) -> bool:
        """
        Delete a collection.

        Args:
            collection_id: Collection UUID

        Returns:
            True if deleted, False if not found
        """
        query = """
            DELETE FROM collections WHERE id = $1
        """

        try:
            result = await self.pool.execute(query, collection_id)
            deleted = result == "DELETE 1"

            if deleted:
                logger.info("Collection deleted", collection_id=str(collection_id))

            return deleted

        except Exception as e:
            logger.error(
                "Failed to delete collection",
                error=str(e),
                collection_id=str(collection_id),
            )
            raise

    async def has_permission(
        self,
        collection_id: UUID,
        user_id: str,
        required_level: PermissionLevel = PermissionLevel.VIEWER,
    ) -> bool:
        """
        Check if a user has permission to access a collection.

        Args:
            collection_id: Collection UUID
            user_id: User ID
            required_level: Minimum required permission level

        Returns:
            True if user has permission
        """
        collection = await self.get(collection_id)

        if not collection:
            return False

        # Owner has all permissions
        if collection.owner_id == user_id:
            return True

        # Check if public and only viewer access required
        if collection.is_public and required_level == PermissionLevel.VIEWER:
            return True

        # Check explicit permissions
        user_level = collection.permissions.get(user_id)
        if not user_level:
            return False

        # Permission hierarchy: admin > editor > viewer
        levels = {
            PermissionLevel.VIEWER.value: 1,
            PermissionLevel.EDITOR.value: 2,
            PermissionLevel.ADMIN.value: 3,
        }

        return levels.get(user_level, 0) >= levels.get(required_level.value, 0)

    async def share(
        self,
        collection_id: UUID,
        user_id: str,
        permission_level: PermissionLevel,
    ) -> Optional[Collection]:
        """
        Share a collection with a user.

        Args:
            collection_id: Collection UUID
            user_id: User to share with
            permission_level: Permission level to grant

        Returns:
            Updated collection if found, None otherwise
        """
        query = """
            UPDATE collections
            SET 
                permissions = permissions || $1::jsonb,
                updated_at = $2
            WHERE id = $3
            RETURNING *
        """

        try:
            import json

            permission_update = json.dumps({user_id: permission_level.value})
            row = await self.pool.fetchrow(
                query,
                permission_update,
                datetime.utcnow(),
                collection_id,
            )

            if not row:
                return None

            logger.info(
                "Collection shared",
                collection_id=str(collection_id),
                user_id=user_id,
                permission_level=permission_level.value,
            )

            return Collection(**dict(row))

        except Exception as e:
            logger.error(
                "Failed to share collection",
                error=str(e),
                collection_id=str(collection_id),
                user_id=user_id,
            )
            raise

    async def revoke_access(
        self,
        collection_id: UUID,
        user_id: str,
    ) -> Optional[Collection]:
        """
        Revoke a user's access to a collection.

        Args:
            collection_id: Collection UUID
            user_id: User to revoke access from

        Returns:
            Updated collection if found, None otherwise
        """
        query = """
            UPDATE collections
            SET 
                permissions = permissions - $1,
                updated_at = $2
            WHERE id = $3
            RETURNING *
        """

        try:
            row = await self.pool.fetchrow(
                query,
                user_id,
                datetime.utcnow(),
                collection_id,
            )

            if not row:
                return None

            logger.info(
                "Collection access revoked",
                collection_id=str(collection_id),
                user_id=user_id,
            )

            return Collection(**dict(row))

        except Exception as e:
            logger.error(
                "Failed to revoke collection access",
                error=str(e),
                collection_id=str(collection_id),
                user_id=user_id,
            )
            raise

    async def get_stats(self, collection_id: UUID) -> Dict:
        """
        Get collection statistics.

        Args:
            collection_id: Collection UUID

        Returns:
            Dictionary with collection statistics
        """
        query = """
            SELECT 
                COUNT(DISTINCT cd.document_id) as document_count,
                COALESCE(SUM((d.metadata->>'file_size_bytes')::bigint), 0) as total_size_bytes
            FROM collections c
            LEFT JOIN collection_documents cd ON c.id = cd.collection_id
            LEFT JOIN documents d ON cd.document_id = d.id
            WHERE c.id = $1
            GROUP BY c.id
        """

        try:
            row = await self.pool.fetchrow(query, collection_id)

            if not row:
                return {
                    "document_count": 0,
                    "total_size_bytes": 0,
                }

            return dict(row)

        except Exception as e:
            logger.error(
                "Failed to get collection stats",
                error=str(e),
                collection_id=str(collection_id),
            )
            raise
