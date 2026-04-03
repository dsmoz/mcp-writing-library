"""
Synchronous database managers for MCP server compatibility.

These provide sync wrappers around database operations for use in
MCP servers that don't require full async.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import structlog

from kbase.database.sync_pool import SyncDatabasePool
from kbase.models.collection import (
    Collection,
    CollectionCreate,
    CollectionUpdate,
    PermissionLevel,
)
from kbase.models.document import Document, DocumentCreate, SyncDocumentCreate

logger = structlog.get_logger(__name__)


class SyncCollectionManager:
    """
    Synchronous manager for collection database operations.
    """

    def __init__(self, pool: SyncDatabasePool):
        """Initialize collection manager with sync pool."""
        self.pool = pool

    def create(
        self,
        data: CollectionCreate,
        owner_id: str,
    ) -> Dict[str, Any]:
        """
        Create a new collection.

        Args:
            data: Collection creation data
            owner_id: User ID of the collection owner

        Returns:
            Created collection as dict
        """
        query = """
            INSERT INTO collections (
                name, description, owner_id, tags, is_public,
                allowed_sources, permissions, metadata, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
        """

        now = datetime.utcnow()
        permissions = {owner_id: PermissionLevel.ADMIN.value}

        import json
        tags_json = json.dumps(data.tags) if data.tags else "[]"
        allowed_sources_json = json.dumps(data.allowed_sources) if data.allowed_sources else "[]"
        permissions_json = json.dumps(permissions)
        metadata_json = json.dumps({})

        try:
            row = self.pool.fetchrow(
                query,
                (
                    data.name,
                    data.description,
                    owner_id,
                    tags_json,
                    data.is_public,
                    allowed_sources_json,
                    permissions_json,
                    metadata_json,
                    now,
                    now,
                ),
            )

            if not row:
                raise RuntimeError("Failed to create collection")

            logger.info(
                "Collection created",
                collection_id=str(row["id"]),
                owner_id=owner_id,
                name=data.name,
            )

            return dict(row)

        except Exception as e:
            logger.error("Failed to create collection", error=str(e), owner_id=owner_id)
            raise

    def get(self, collection_id: Union[UUID, str]) -> Optional[Dict[str, Any]]:
        """
        Get a collection by ID.

        Args:
            collection_id: Collection UUID

        Returns:
            Collection dict if found, None otherwise
        """
        if isinstance(collection_id, str):
            collection_id = UUID(collection_id)

        query = "SELECT * FROM collections WHERE id = %s"

        try:
            row = self.pool.fetchrow(query, (str(collection_id),))
            return dict(row) if row else None

        except Exception as e:
            logger.error(
                "Failed to get collection",
                error=str(e),
                collection_id=str(collection_id),
            )
            raise

    def list(
        self,
        user_id: str,
        include_public: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List collections accessible by a user.

        Args:
            user_id: User ID
            include_public: Include public collections
            limit: Maximum results
            offset: Result offset for pagination

        Returns:
            List of collection dicts
        """
        if include_public:
            query = """
                SELECT * FROM collections
                WHERE
                    owner_id = %s
                    OR is_public = TRUE
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params = (user_id, limit, offset)
        else:
            query = """
                SELECT * FROM collections
                WHERE owner_id = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params = (user_id, limit, offset)

        try:
            rows = self.pool.fetch(query, params)
            return [dict(row) for row in rows]

        except Exception as e:
            logger.error("Failed to list collections", error=str(e), user_id=user_id)
            raise

    def delete(self, collection_id: Union[UUID, str], user_id: str = None) -> bool:
        """
        Delete a collection.

        Args:
            collection_id: Collection UUID
            user_id: Optional user ID for verification (not used currently)

        Returns:
            True if deleted
        """
        if isinstance(collection_id, str):
            collection_id = UUID(collection_id)

        query = "DELETE FROM collections WHERE id = %s"

        try:
            self.pool.execute(query, (str(collection_id),))
            logger.info("Collection deleted", collection_id=str(collection_id))
            return True

        except Exception as e:
            logger.error(
                "Failed to delete collection",
                error=str(e),
                collection_id=str(collection_id),
            )
            raise


class SyncDocumentManager:
    """
    Synchronous manager for document database operations.
    """

    def __init__(self, pool: SyncDatabasePool):
        """Initialize document manager with sync pool."""
        self.pool = pool

    def create(
        self,
        user_id: str,
        document: SyncDocumentCreate,
    ) -> Dict[str, Any]:
        """
        Create a new document.

        Args:
            user_id: User ID creating the document
            document: Document creation data

        Returns:
            Created document as dict
        """
        query = """
            INSERT INTO documents (
                title, content, source_type, source_id,
                owner_id, file_path, file_name, file_size,
                file_hash, file_modified_at, mime_type, storage_type,
                doc_metadata, created_at, modified_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
        """

        now = datetime.utcnow()

        import json
        metadata_json = json.dumps(document.metadata) if document.metadata else "{}"

        try:
            row = self.pool.fetchrow(
                query,
                (
                    document.title,
                    document.content or "",
                    document.source_type,
                    document.source_id,
                    user_id,
                    document.file_path,
                    document.file_name,
                    document.file_size,
                    document.file_hash,
                    document.file_modified_at,
                    document.mime_type,
                    document.storage_type or "database",
                    metadata_json,
                    now,
                    now,
                ),
            )

            if not row:
                raise RuntimeError("Failed to create document")

            logger.info(
                "Document created",
                document_id=str(row["id"]),
                owner_id=user_id,
                title=document.title,
            )

            return dict(row)

        except Exception as e:
            logger.error("Failed to create document", error=str(e), owner_id=user_id)
            raise

    def get(self, document_id: Union[UUID, str]) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.

        Args:
            document_id: Document UUID

        Returns:
            Document dict if found, None otherwise
        """
        if isinstance(document_id, str):
            document_id = UUID(document_id)

        query = "SELECT * FROM documents WHERE id = %s"

        try:
            row = self.pool.fetchrow(query, (str(document_id),))
            return dict(row) if row else None

        except Exception as e:
            logger.error(
                "Failed to get document",
                error=str(e),
                document_id=str(document_id),
            )
            raise

    def list(
        self,
        user_id: str,
        collection_id: Union[UUID, str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List documents for a user.

        Args:
            user_id: User ID
            collection_id: Optional collection filter
            limit: Maximum results
            offset: Result offset

        Returns:
            List of document dicts
        """
        if collection_id:
            if isinstance(collection_id, str):
                collection_id = UUID(collection_id)

            # Query documents in collection via junction table
            query = """
                SELECT d.* FROM documents d
                INNER JOIN collection_documents cd ON d.id = cd.document_id
                WHERE cd.collection_id = %s
                ORDER BY d.created_at DESC
                LIMIT %s OFFSET %s
            """
            params = (str(collection_id), limit, offset)
        else:
            query = """
                SELECT * FROM documents
                WHERE owner_id = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params = (user_id, limit, offset)

        try:
            rows = self.pool.fetch(query, params)
            return [dict(row) for row in rows]

        except Exception as e:
            logger.error("Failed to list documents", error=str(e), user_id=user_id)
            raise

    def delete(self, document_id: Union[UUID, str]) -> bool:
        """
        Delete a document.

        Args:
            document_id: Document UUID

        Returns:
            True if deleted
        """
        if isinstance(document_id, str):
            document_id = UUID(document_id)

        query = "DELETE FROM documents WHERE id = %s"

        try:
            self.pool.execute(query, (str(document_id),))
            logger.info("Document deleted", document_id=str(document_id))
            return True

        except Exception as e:
            logger.error(
                "Failed to delete document",
                error=str(e),
                document_id=str(document_id),
            )
            raise


class SyncAPIKeyManager:
    """
    Synchronous manager for API key operations.
    """

    def __init__(self, pool: SyncDatabasePool):
        """Initialize API key manager with sync pool."""
        self.pool = pool

    def validate_key(self, key: str) -> Optional[str]:
        """
        Validate an API key and return the user ID.

        Args:
            key: API key to validate

        Returns:
            User ID if valid, None otherwise
        """
        query = """
            SELECT user_id FROM api_keys
            WHERE key_hash = %s
            AND is_active = TRUE
            AND (expires_at IS NULL OR expires_at > NOW())
        """

        try:
            row = self.pool.fetchrow(query, (key,))
            if row:
                # Update last_used
                update_query = "UPDATE api_keys SET last_used = NOW() WHERE key_hash = %s"
                self.pool.execute(update_query, (key,))
                return row["user_id"]
            return None

        except Exception as e:
            logger.error("Failed to validate API key", error=str(e))
            return None

    def create(
        self,
        user_id: str,
        name: str,
        key: str,
        expires_at: datetime = None,
    ) -> Dict[str, Any]:
        """
        Create a new API key.

        Args:
            user_id: User ID
            name: Key name
            key: The actual key string
            expires_at: Optional expiration

        Returns:
            Created API key dict
        """
        query = """
            INSERT INTO api_keys (key_hash, user_id, name, expires_at, is_active, created_at)
            VALUES (%s, %s, %s, %s, TRUE, NOW())
            RETURNING *
        """

        try:
            row = self.pool.fetchrow(query, (key, user_id, name, expires_at))
            if not row:
                raise RuntimeError("Failed to create API key")

            logger.info("API key created", user_id=user_id, name=name)
            return dict(row)

        except Exception as e:
            logger.error("Failed to create API key", error=str(e))
            raise


__all__ = [
    "SyncCollectionManager",
    "SyncDocumentManager",
    "SyncAPIKeyManager",
]
