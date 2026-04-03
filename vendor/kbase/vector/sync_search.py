"""
Synchronous Vector Search Implementation

This module provides synchronous semantic search using Qdrant.
Uses the sync QdrantClient and sync embedding generation.

Usage:
    from kbase.vector.sync_search import semantic_search
    results = semantic_search("cerebellum_test", "machine learning", limit=10)
"""

import os
from typing import Any, Dict, List, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

import structlog

from kbase.vector.sync_client import get_qdrant_client
from kbase.vector.sync_embeddings import generate_embedding, generate_sparse_vector
from kbase.vector.sync_indexing import is_hybrid_collection

logger = structlog.get_logger(__name__)


def semantic_search(
    collection_name: str,
    query: str,
    limit: int = 10,
    score_threshold: Optional[float] = None,
    filter_conditions: Optional[Dict[str, Any]] = None,
    hybrid_weight: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using vector similarity.

    Automatically detects if collection uses hybrid vectors (dense + sparse)
    and performs appropriate search.

    Args:
        collection_name: Qdrant collection name
        query: Search query text
        limit: Maximum number of results (default: 10)
        score_threshold: Minimum similarity score (0-1)
        filter_conditions: Optional filter conditions (e.g., {"document_id": "uuid"})
        hybrid_weight: Weight for dense vs sparse in hybrid search (0-1, higher = more dense)

    Returns:
        List of search results with scores and metadata:
            - id: Qdrant point ID
            - score: Similarity score (0-1, higher is better)
            - document_id: Document UUID
            - chunk_index: Chunk index within document
            - text: Text content of the chunk
            - metadata: Additional metadata

    Raises:
        Exception: If search fails
    """
    try:
        client = get_qdrant_client()

        # Check if collection uses hybrid vectors
        is_hybrid = is_hybrid_collection(collection_name)

        # Generate query embedding
        logger.info(
            "Generating query embedding",
            query_length=len(query),
            hybrid=is_hybrid,
        )
        query_vector = generate_embedding(query)

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

        # Perform search (hybrid or simple)
        logger.info(
            "Performing semantic search",
            collection=collection_name,
            limit=limit,
            has_filter=filter_conditions is not None,
            hybrid=is_hybrid,
        )

        if is_hybrid:
            # Hybrid search: combine dense and sparse
            from qdrant_client.models import (
                SparseVector,
                Prefetch,
                FusionQuery,
                Fusion,
            )

            # Generate sparse vector
            sparse_indices, sparse_values = generate_sparse_vector(query)

            # Use query_points for hybrid search with RRF fusion
            results = client.query_points(
                collection_name=collection_name,
                prefetch=[
                    # Dense search
                    Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=limit * 2,
                    ),
                    # Sparse search
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                        using="sparse",
                        limit=limit * 2,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )

            # query_points returns QueryResponse with .points attribute
            results = results.points
        else:
            # Simple dense-only search
            results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            ).points

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "document_id": result.payload.get("document_id"),
                "chunk_index": result.payload.get("chunk_index"),
                "text": result.payload.get("text"),
                "title": result.payload.get("title"),
                "metadata": {
                    k: v
                    for k, v in result.payload.items()
                    if k not in ["document_id", "chunk_index", "text", "title"]
                },
            })

        logger.info(
            "Search completed",
            query_length=len(query),
            results_count=len(formatted_results),
            collection=collection_name,
        )

        return formatted_results

    except Exception as e:
        logger.error(
            "Search failed",
            error=str(e),
            query_length=len(query),
            collection=collection_name,
        )
        raise


def search_by_document_id(
    collection_name: str,
    document_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Retrieve all chunks for a specific document.

    Args:
        collection_name: Qdrant collection name
        document_id: Document UUID

    Returns:
        List of chunks with metadata, sorted by chunk_index
    """
    try:
        client = get_qdrant_client()

        # Build filter for document_id
        query_filter = Filter(
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
            result, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
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
                    "title": point.payload.get("title"),
                    "metadata": {
                        k: v
                        for k, v in point.payload.items()
                        if k not in ["document_id", "chunk_index", "text", "title"]
                    },
                })

            if offset is None or len(chunks) >= limit:
                break

        # Sort by chunk_index
        chunks.sort(key=lambda x: x.get("chunk_index", 0))

        logger.info(
            "Retrieved document chunks",
            document_id=document_id,
            chunks_count=len(chunks),
            collection=collection_name,
        )

        return chunks[:limit]

    except Exception as e:
        logger.error(
            "Failed to retrieve document chunks",
            error=str(e),
            document_id=document_id,
            collection=collection_name,
        )
        raise


def get_collection_stats(collection_name: str) -> Dict[str, Any]:
    """
    Get statistics for a Qdrant collection.

    Args:
        collection_name: Qdrant collection name

    Returns:
        Collection statistics including points count and vector info
    """
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection_name=collection_name)

        return {
            "name": collection_name,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    except Exception as e:
        logger.error(
            "Failed to get collection stats",
            error=str(e),
            collection=collection_name,
        )
        raise
