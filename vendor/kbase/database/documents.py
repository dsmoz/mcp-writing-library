"""
Document database manager
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

import structlog

from kbase.database.pool import DatabasePool
from kbase.models.document import (
    Document,
    DocumentCreate,
    DocumentMetadata,
    DocumentUpdate,
)

logger = structlog.get_logger(__name__)


class DocumentManager:
    """
    Manager for document database operations.

    Handles CRUD operations, collection assignments, and document retrieval.
    """

    def __init__(self, pool: DatabasePool):
        """
        Initialize document manager.

        Args:
            pool: Database connection pool
        """
        self.pool = pool

    async def create(
        self,
        data: DocumentCreate,
        owner_id: str,
    ) -> Document:
        """
        Create a new document.

        Args:
            data: Document creation data
            owner_id: User ID of the document owner

        Returns:
            Created document

        Raises:
            ValueError: If validation fails
        """
        # Insert document
        doc_query = """
            INSERT INTO documents (
                title, content, source, url, owner_id,
                metadata, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
        """

        now = datetime.utcnow()

        try:
            async with self.pool.transaction() as conn:
                # Create document
                row = await conn.fetchrow(
                    doc_query,
                    data.title,
                    data.content,
                    data.source.value,
                    data.url,
                    owner_id,
                    data.metadata.model_dump(),
                    now,
                    now,
                )

                if not row:
                    raise RuntimeError("Failed to create document")

                document_id = row["id"]

                # Add to collections
                if data.collection_ids:
                    collection_query = """
                        INSERT INTO collection_documents (collection_id, document_id)
                        VALUES ($1, $2)
                    """
                    for collection_id in data.collection_ids:
                        await conn.execute(collection_query, collection_id, document_id)

                logger.info(
                    "Document created",
                    document_id=str(document_id),
                    owner_id=owner_id,
                    title=data.title,
                    collections=len(data.collection_ids),
                )

                # Fetch full document with collection_ids
                return await self._get_with_collections(conn, document_id)

        except Exception as e:
            logger.error("Failed to create document", error=str(e), owner_id=owner_id)
            raise

    async def get(self, document_id: UUID) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            document_id: Document UUID

        Returns:
            Document if found, None otherwise
        """
        try:
            async with self.pool.acquire() as conn:
                return await self._get_with_collections(conn, document_id)

        except Exception as e:
            logger.error(
                "Failed to get document",
                error=str(e),
                document_id=str(document_id),
            )
            raise

    async def _get_with_collections(self, conn, document_id: UUID) -> Optional[Document]:
        """
        Internal helper to get document with collection IDs.

        Args:
            conn: Database connection
            document_id: Document UUID

        Returns:
            Document with collection_ids populated
        """
        doc_query = """
            SELECT * FROM documents WHERE id = $1
        """

        collections_query = """
            SELECT collection_id FROM collection_documents
            WHERE document_id = $1
        """

        doc_row = await conn.fetchrow(doc_query, document_id)
        if not doc_row:
            return None

        collection_rows = await conn.fetch(collections_query, document_id)
        collection_ids = [row["collection_id"] for row in collection_rows]

        doc_dict = dict(doc_row)
        doc_dict["collection_ids"] = collection_ids
        doc_dict["qdrant_ids"] = []  # TODO: Fetch from tracking table

        return Document(**doc_dict)

    async def list_for_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Document]:
        """
        List documents owned by a user.

        Args:
            user_id: User ID
            limit: Maximum results
            offset: Result offset for pagination

        Returns:
            List of documents
        """
        query = """
            SELECT * FROM documents
            WHERE owner_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
        """

        try:
            rows = await self.pool.fetch(query, user_id, limit, offset)
            
            # For each document, fetch collection IDs
            documents = []
            async with self.pool.acquire() as conn:
                for row in rows:
                    doc = await self._get_with_collections(conn, row["id"])
                    if doc:
                        documents.append(doc)

            return documents

        except Exception as e:
            logger.error("Failed to list documents", error=str(e), user_id=user_id)
            raise

    async def list_for_collection(
        self,
        collection_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Document]:
        """
        List documents in a collection.

        Args:
            collection_id: Collection UUID
            limit: Maximum results
            offset: Result offset for pagination

        Returns:
            List of documents
        """
        query = """
            SELECT d.* FROM documents d
            INNER JOIN collection_documents cd ON d.id = cd.document_id
            WHERE cd.collection_id = $1
            ORDER BY d.created_at DESC
            LIMIT $2 OFFSET $3
        """

        try:
            rows = await self.pool.fetch(query, collection_id, limit, offset)
            
            # For each document, fetch collection IDs
            documents = []
            async with self.pool.acquire() as conn:
                for row in rows:
                    doc = await self._get_with_collections(conn, row["id"])
                    if doc:
                        documents.append(doc)

            return documents

        except Exception as e:
            logger.error(
                "Failed to list documents for collection",
                error=str(e),
                collection_id=str(collection_id),
            )
            raise

    async def update(
        self,
        document_id: UUID,
        data: DocumentUpdate,
    ) -> Optional[Document]:
        """
        Update a document.

        Args:
            document_id: Document UUID
            data: Update data

        Returns:
            Updated document if found, None otherwise
        """
        # Build dynamic update query
        updates = []
        values = []
        param_count = 1

        if data.title is not None:
            updates.append(f"title = ${param_count}")
            values.append(data.title)
            param_count += 1

        if data.content is not None:
            updates.append(f"content = ${param_count}")
            values.append(data.content)
            param_count += 1

        if data.url is not None:
            updates.append(f"url = ${param_count}")
            values.append(data.url)
            param_count += 1

        if data.metadata is not None:
            updates.append(f"metadata = ${param_count}")
            values.append(data.metadata.model_dump())
            param_count += 1

        if not updates:
            # No updates provided
            return await self.get(document_id)

        # Add updated_at
        updates.append(f"updated_at = ${param_count}")
        values.append(datetime.utcnow())
        param_count += 1

        # Add document_id as last parameter
        values.append(document_id)

        query = f"""
            UPDATE documents
            SET {', '.join(updates)}
            WHERE id = ${param_count}
            RETURNING *
        """

        try:
            row = await self.pool.fetchrow(query, *values)

            if not row:
                return None

            logger.info("Document updated", document_id=str(document_id))

            # Fetch with collections
            async with self.pool.acquire() as conn:
                return await self._get_with_collections(conn, document_id)

        except Exception as e:
            logger.error(
                "Failed to update document",
                error=str(e),
                document_id=str(document_id),
            )
            raise

    async def delete(self, document_id: UUID) -> bool:
        """
        Delete a document.

        Args:
            document_id: Document UUID

        Returns:
            True if deleted, False if not found
        """
        query = """
            DELETE FROM documents WHERE id = $1
        """

        try:
            result = await self.pool.execute(query, document_id)
            deleted = result == "DELETE 1"

            if deleted:
                logger.info("Document deleted", document_id=str(document_id))

            return deleted

        except Exception as e:
            logger.error(
                "Failed to delete document",
                error=str(e),
                document_id=str(document_id),
            )
            raise

    async def add_to_collection(
        self,
        document_id: UUID,
        collection_id: UUID,
    ) -> bool:
        """
        Add a document to a collection.

        Args:
            document_id: Document UUID
            collection_id: Collection UUID

        Returns:
            True if added, False if already exists
        """
        query = """
            INSERT INTO collection_documents (collection_id, document_id)
            VALUES ($1, $2)
            ON CONFLICT (collection_id, document_id) DO NOTHING
        """

        try:
            result = await self.pool.execute(query, collection_id, document_id)
            added = result == "INSERT 0 1"

            if added:
                logger.info(
                    "Document added to collection",
                    document_id=str(document_id),
                    collection_id=str(collection_id),
                )

            return added

        except Exception as e:
            logger.error(
                "Failed to add document to collection",
                error=str(e),
                document_id=str(document_id),
                collection_id=str(collection_id),
            )
            raise

    async def remove_from_collection(
        self,
        document_id: UUID,
        collection_id: UUID,
    ) -> bool:
        """
        Remove a document from a collection.

        Args:
            document_id: Document UUID
            collection_id: Collection UUID

        Returns:
            True if removed, False if not found
        """
        query = """
            DELETE FROM collection_documents
            WHERE collection_id = $1 AND document_id = $2
        """

        try:
            result = await self.pool.execute(query, collection_id, document_id)
            removed = result == "DELETE 1"

            if removed:
                logger.info(
                    "Document removed from collection",
                    document_id=str(document_id),
                    collection_id=str(collection_id),
                )

            return removed

        except Exception as e:
            logger.error(
                "Failed to remove document from collection",
                error=str(e),
                document_id=str(document_id),
                collection_id=str(collection_id),
            )
            raise
