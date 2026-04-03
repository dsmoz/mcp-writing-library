"""
Vector search operations using Qdrant.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct, ScoredPoint

from kbase.vector.client import get_client
from kbase.vector.embeddings import generate_embedding

logger = structlog.get_logger(__name__)


async def index_document(
    collection_name: str,
    document_id: str,
    text_chunks: List[str],
    metadata: Dict[str, Any],
    embedding_model: str = "text-embedding-3-small",
) -> List[str]:
    """
    Index a document by creating embeddings and storing in Qdrant.

    Args:
        collection_name: Qdrant collection name
        document_id: Document UUID (will be converted to string)
        text_chunks: List of text chunks to index
        metadata: Metadata to store with each chunk
        embedding_model: OpenAI embedding model

    Returns:
        List of Qdrant point IDs created

    Raises:
        Exception: If indexing fails
    """
    from kbase.vector.embeddings import generate_embeddings_batch

    client = await get_client()

    if not text_chunks:
        logger.warning("No text chunks to index", document_id=document_id)
        return []

    try:
        # Generate embeddings for all chunks
        embeddings = await generate_embeddings_batch(
            texts=text_chunks,
            model=embedding_model,
        )

        # Create points for Qdrant
        points = []
        point_ids = []

        for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            point_id = f"{document_id}_chunk_{i}"
            point_ids.append(point_id)

            payload = {
                "document_id": str(document_id),
                "chunk_index": i,
                "text": chunk,
                **metadata,
            }

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )

            points.append(point)

        # Upsert points to Qdrant
        await client.upsert(
            collection_name=collection_name,
            points=points,
        )

        logger.info(
            "Document indexed",
            document_id=document_id,
            chunks_count=len(text_chunks),
            collection=collection_name,
        )

        return point_ids

    except Exception as e:
        logger.error(
            "Failed to index document",
            error=str(e),
            document_id=document_id,
            chunks_count=len(text_chunks),
        )
        raise


async def search_similar(
    collection_name: str,
    query_text: str,
    limit: int = 10,
    score_threshold: Optional[float] = None,
    filter_conditions: Optional[Dict[str, Any]] = None,
    embedding_model: str = "text-embedding-3-small",
) -> List[Dict[str, Any]]:
    """
    Search for similar documents using vector similarity.

    Args:
        collection_name: Qdrant collection name
        query_text: Query text to search for
        limit: Maximum number of results
        score_threshold: Minimum similarity score (0-1)
        filter_conditions: Optional filter conditions (e.g., {"document_id": "uuid"})
        embedding_model: OpenAI embedding model

    Returns:
        List of search results with scores and metadata

    Raises:
        Exception: If search fails
    """
    client = await get_client()

    try:
        # Generate query embedding
        query_vector = await generate_embedding(
            text=query_text,
            model=embedding_model,
        )

        # Build filter if conditions provided
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            query_filter = Filter(must=conditions)

        # Search
        results = await client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "document_id": result.payload.get("document_id"),
                "chunk_index": result.payload.get("chunk_index"),
                "text": result.payload.get("text"),
                "metadata": {
                    k: v
                    for k, v in result.payload.items()
                    if k not in ["document_id", "chunk_index", "text"]
                },
            })

        logger.info(
            "Search completed",
            query_length=len(query_text),
            results_count=len(formatted_results),
            collection=collection_name,
        )

        return formatted_results

    except Exception as e:
        logger.error(
            "Search failed",
            error=str(e),
            query_length=len(query_text),
            collection=collection_name,
        )
        raise


async def delete_document_vectors(
    collection_name: str,
    document_id: str,
) -> int:
    """
    Delete all vector points for a document.

    Args:
        collection_name: Qdrant collection name
        document_id: Document UUID

    Returns:
        Number of points deleted

    Raises:
        Exception: If deletion fails
    """
    client = await get_client()

    try:
        # Search for all points with this document_id
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(document_id)),
                )
            ]
        )

        # Scroll to get all points (in batches)
        points_to_delete = []
        offset = None

        while True:
            result, offset = await client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=100,
                offset=offset,
            )

            if not result:
                break

            points_to_delete.extend([point.id for point in result])

            if offset is None:
                break

        # Delete points
        if points_to_delete:
            await client.delete(
                collection_name=collection_name,
                points_selector=points_to_delete,
            )

        logger.info(
            "Document vectors deleted",
            document_id=document_id,
            points_deleted=len(points_to_delete),
            collection=collection_name,
        )

        return len(points_to_delete)

    except Exception as e:
        logger.error(
            "Failed to delete document vectors",
            error=str(e),
            document_id=document_id,
            collection=collection_name,
        )
        raise


async def get_document_chunks(
    collection_name: str,
    document_id: str,
) -> List[Dict[str, Any]]:
    """
    Retrieve all chunks for a document.

    Args:
        collection_name: Qdrant collection name
        document_id: Document UUID

    Returns:
        List of chunks with metadata

    Raises:
        Exception: If retrieval fails
    """
    client = await get_client()

    try:
        # Search for all points with this document_id
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(document_id)),
                )
            ]
        )

        # Scroll to get all points
        chunks = []
        offset = None

        while True:
            result, offset = await client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not result:
                break

            for point in result:
                chunks.append({
                    "id": point.id,
                    "chunk_index": point.payload.get("chunk_index"),
                    "text": point.payload.get("text"),
                    "metadata": {
                        k: v
                        for k, v in point.payload.items()
                        if k not in ["document_id", "chunk_index", "text"]
                    },
                })

            if offset is None:
                break

        # Sort by chunk_index
        chunks.sort(key=lambda x: x.get("chunk_index", 0))

        logger.info(
            "Retrieved document chunks",
            document_id=document_id,
            chunks_count=len(chunks),
            collection=collection_name,
        )

        return chunks

    except Exception as e:
        logger.error(
            "Failed to retrieve document chunks",
            error=str(e),
            document_id=document_id,
            collection=collection_name,
        )
        raise
