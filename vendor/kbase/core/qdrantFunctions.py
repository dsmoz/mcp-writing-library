"""
Qdrant vector database operations for kbase-core.

Provides functions for managing Qdrant collections and
indexing documents with embeddings.
"""

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from kbase.core.logger import get_logger

logger = get_logger(__name__)


def get_qdrant_client():
    """
    Get a synchronous Qdrant client instance.

    Always returns a sync QdrantClient. For async usage, use kbase.vector.get_client().

    Returns:
        QdrantClient instance
    """
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=60,
    )


def collection_exists(collection_name: str) -> bool:
    """
    Check if a collection exists in Qdrant.

    Args:
        collection_name: Name of the collection to check

    Returns:
        True if collection exists, False otherwise
    """
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        return any(col.name == collection_name for col in collections)
    except Exception as e:
        logger.error("Error checking collection existence", error=str(e))
        return False


def add_new_qdrant_collection(
    collection_name: str,
    vector_size: int = 1536,
    distance: str = "cosine",
    hybrid: bool = True,
) -> bool:
    """
    Create a new collection in Qdrant.

    Args:
        collection_name: Name for the new collection
        vector_size: Dimension of vectors (default: 1536 for OpenAI)
        distance: Distance metric ('cosine', 'euclid', 'dot')
        hybrid: If True, create with named vectors (dense + sparse) for hybrid search

    Returns:
        True if collection created successfully, False otherwise
    """
    try:
        from qdrant_client.http.models import (
            Distance,
            VectorParams,
            SparseVectorParams,
            Modifier,
        )

        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }

        client = get_qdrant_client()
        dist = distance_map.get(distance.lower(), Distance.COSINE)

        if hybrid:
            # Hybrid: named dense + sparse vectors
            result = client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=dist),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(modifier=Modifier.IDF),
                },
            )
            logger.info(
                "Created hybrid Qdrant collection",
                collection=collection_name,
                vector_size=vector_size,
                mode="hybrid (dense + sparse)",
            )
        else:
            # Simple: single unnamed dense vector
            result = client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=dist),
            )
            logger.info(
                "Created Qdrant collection",
                collection=collection_name,
                vector_size=vector_size,
                mode="dense only",
            )

        return result

    except Exception as e:
        logger.error(
            "Error creating Qdrant collection",
            collection=collection_name,
            error=str(e),
        )
        return False


def delete_qdrant_collection(collection_name: str) -> bool:
    """
    Delete a collection from Qdrant.

    Args:
        collection_name: Name of collection to delete

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        client = get_qdrant_client()
        result = client.delete_collection(collection_name=collection_name)

        logger.info("Deleted Qdrant collection", collection=collection_name)
        return result

    except Exception as e:
        logger.error(
            "Error deleting Qdrant collection",
            collection=collection_name,
            error=str(e),
        )
        return False


def list_collections() -> Optional[List[Any]]:
    """
    List all collections in Qdrant.

    Returns:
        List of collection objects, or None on error
    """
    try:
        client = get_qdrant_client()
        result = client.get_collections().collections
        return result

    except Exception as e:
        logger.error("Error listing Qdrant collections", error=str(e))
        return None


def vectorize_documents_to_qdrant(
    collection_name: str,
    documents: List[Any],
    ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Add documents to a Qdrant collection with embeddings.

    This function generates embeddings for each document and
    stores them in the specified Qdrant collection.

    Args:
        collection_name: Target collection name
        documents: List of document objects with page_content and metadata
        ids: Optional list of UUIDs for documents

    Returns:
        Dict with success status, vector_point_uids, and message
    """
    try:
        from qdrant_client import models

        from kbase.core.embeddings import EmbeddingGenerator

        client = get_qdrant_client()

        # Verify collection exists
        if not collection_exists(collection_name):
            return {
                "success": False,
                "vector_point_uids": [],
                "message": f"Collection '{collection_name}' does not exist",
            }

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]

        # Initialize embedding generator
        generator = EmbeddingGenerator()

        # Extract text content from documents
        texts = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                texts.append(doc.page_content)
            elif isinstance(doc, dict):
                texts.append(doc.get("page_content", doc.get("content", str(doc))))
            else:
                texts.append(str(doc))

        # Generate embeddings in batch
        logger.info(
            "Generating embeddings for documents",
            count=len(texts),
            collection=collection_name,
        )
        embeddings = generator.generate_batch(texts, show_progress=True)

        # Prepare points for upsert
        points = []
        for i, (doc_id, embedding, doc) in enumerate(zip(ids, embeddings, documents)):
            # Extract metadata
            if hasattr(doc, "metadata"):
                metadata = doc.metadata
            elif isinstance(doc, dict):
                metadata = doc.get("metadata", {})
            else:
                metadata = {}

            # Add text content to metadata for retrieval
            payload = {
                **metadata,
                "text": texts[i],
            }

            points.append(
                models.PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points,
        )

        logger.info(
            "Successfully added documents to Qdrant",
            count=len(points),
            collection=collection_name,
        )

        return {
            "success": True,
            "vector_point_uids": ids,
            "message": f"Successfully added {len(ids)} documents to {collection_name}",
        }

    except Exception as e:
        logger.error(
            "Error adding documents to Qdrant",
            error=str(e),
            collection=collection_name,
        )
        return {
            "success": False,
            "vector_point_uids": [],
            "message": f"Error adding documents: {str(e)}",
        }


async def vectorize_documents_to_qdrant_async(
    collection_name: str,
    documents: List[Any],
    ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Async version of vectorize_documents_to_qdrant.

    Args:
        collection_name: Target collection name
        documents: List of document objects
        ids: Optional list of UUIDs

    Returns:
        Dict with success status and details
    """
    # For now, delegate to sync version
    # Can be optimized with async Qdrant client later
    import asyncio

    return await asyncio.to_thread(
        vectorize_documents_to_qdrant,
        collection_name,
        documents,
        ids,
    )


__all__ = [
    "collection_exists",
    "add_new_qdrant_collection",
    "delete_qdrant_collection",
    "list_collections",
    "vectorize_documents_to_qdrant",
    "vectorize_documents_to_qdrant_async",
    "get_qdrant_client",
]
