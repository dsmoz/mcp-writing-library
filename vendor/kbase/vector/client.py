"""
Qdrant client management for vector database operations.
"""

from typing import Optional

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

logger = structlog.get_logger(__name__)

# Global client instance
_client: Optional[AsyncQdrantClient] = None


async def create_client(
    url: str,
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> AsyncQdrantClient:
    """
    Create and initialize the Qdrant client.

    Args:
        url: Qdrant server URL (e.g., "http://localhost:6333")
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds

    Returns:
        Initialized Qdrant client

    Raises:
        Exception: If client creation fails
    """
    global _client

    try:
        logger.info("Creating Qdrant client", url=url)

        _client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
        )

        # Test connection
        await _client.get_collections()

        logger.info("Qdrant client created successfully", url=url)
        return _client

    except Exception as e:
        logger.error("Failed to create Qdrant client", error=str(e), url=url)
        raise


async def get_client() -> AsyncQdrantClient:
    """
    Get the global Qdrant client instance.

    Returns:
        Qdrant client instance

    Raises:
        RuntimeError: If client hasn't been initialized
    """
    if _client is None:
        raise RuntimeError(
            "Qdrant client not initialized. Call create_client() first."
        )
    return _client


async def close_client() -> None:
    """
    Close the Qdrant client connection.

    Safe to call even if client hasn't been initialized.
    """
    global _client

    if _client is not None:
        try:
            logger.info("Closing Qdrant client")
            await _client.close()
            _client = None
            logger.info("Qdrant client closed")
        except Exception as e:
            logger.error("Error closing Qdrant client", error=str(e))
            raise


async def ensure_collection(
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.COSINE,
    hybrid: bool = True,
) -> bool:
    """
    Ensure a collection exists, create if it doesn't.

    Args:
        collection_name: Name of the collection
        vector_size: Dimension of vectors to store
        distance: Distance metric (COSINE, EUCLID, DOT)
        hybrid: If True, create with named vectors (dense + sparse) for hybrid search

    Returns:
        True if collection was created, False if it already existed

    Raises:
        RuntimeError: If client not initialized
    """
    from qdrant_client.models import SparseVectorParams, Modifier

    client = await get_client()

    try:
        # Check if collection exists
        collections = await client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name in collection_names:
            logger.info("Collection already exists", collection=collection_name)
            return False

        # Create collection with hybrid or simple config
        if hybrid:
            # Hybrid: named dense + sparse vectors
            await client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=distance),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(modifier=Modifier.IDF),
                },
            )
            logger.info(
                "Hybrid collection created",
                collection=collection_name,
                vector_size=vector_size,
                distance=distance.value,
                mode="hybrid (dense + sparse)",
            )
        else:
            # Simple: single unnamed dense vector
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            logger.info(
                "Collection created",
                collection=collection_name,
                vector_size=vector_size,
                distance=distance.value,
                mode="dense only",
            )

        return True

    except Exception as e:
        logger.error(
            "Failed to ensure collection",
            error=str(e),
            collection=collection_name,
        )
        raise


async def get_collection_info(collection_name: str) -> dict:
    """
    Get information about a collection.

    Args:
        collection_name: Name of the collection

    Returns:
        Collection information dictionary

    Raises:
        RuntimeError: If client not initialized
    """
    client = await get_client()

    try:
        info = await client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "config": {
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
            },
        }
    except Exception as e:
        logger.error(
            "Failed to get collection info",
            error=str(e),
            collection=collection_name,
        )
        raise


async def health_check() -> dict:
    """
    Check Qdrant server health.

    Returns:
        Health check information

    Raises:
        RuntimeError: If client not initialized or health check fails
    """
    client = await get_client()

    try:
        collections = await client.get_collections()

        return {
            "status": "healthy",
            "collections_count": len(collections.collections),
            "collections": [c.name for c in collections.collections],
        }
    except Exception as e:
        logger.error("Qdrant health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
        }
