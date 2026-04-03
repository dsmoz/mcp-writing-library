"""
Hybrid Search Implementation for Qdrant

This module implements hybrid search combining dense semantic vectors with sparse BM25 vectors
using Qdrant's Query API with prefetch and fusion.

Hybrid search combines:
- Dense vectors (768D): OpenAI text-embedding-3-small for semantic understanding
- Sparse vectors (BM25): Keyword matching for technical terms and acronyms
- RRF fusion: Reciprocal Rank Fusion for optimal score blending

Performance:
- Semantic search: ~200-300ms
- BM25 search: ~100-200ms
- Fusion: ~10-20ms
- Total: ~400-600ms (still under 1 second)

Usage:
    from kbase.vector import hybrid_search, compare_search_methods

    # Basic hybrid search
    results = hybrid_search(client, "zotero_hybrid", "HIV prevention strategies")

    # With filters
    results = hybrid_search(
        client,
        "zotero_hybrid",
        "climate change",
        filters={"payload.itemType": {"operator": "match", "value": "article"}},
    )

    # Compare search methods
    comparison = compare_search_methods(client, "zotero_hybrid", "query")
"""

import re
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from qdrant_client import QdrantClient, models

from kbase.vector.hybrid_embeddings import get_dense_embedding, get_sparse_embedding

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _is_date_value(v: Any) -> bool:
    return isinstance(v, datetime) or (
        isinstance(v, str) and bool(_ISO_DATE_RE.match(v))
    )


def _parse_date(v: Any) -> datetime:
    return datetime.fromisoformat(v) if isinstance(v, str) else v


def build_filter(
    filters: Dict[str, Any],
    logic: str = "must",
) -> models.Filter:
    """
    Build Qdrant filter from filter dictionary.

    Args:
        filters: Filter dictionary with field names as keys and filter configs as values.
                 Each filter config should have "operator" and "value" keys.
        logic: Filter logic - "must" (AND), "should" (OR), or "must_not" (NOT)

    Returns:
        Qdrant Filter object

    Example:
        filter = build_filter({
            "payload.itemType": {"operator": "match", "value": "article"},
            "payload.year": {"operator": "range", "gte": 2020},
        })

    Supported operators:
        - "match": Exact value match
        - "match_any": Match any value in a list
        - "match_text": Full-text match
        - "range": Numeric range (gte, lte, gt, lt); auto-detects ISO date strings
          and uses DatetimeRange instead of Range
        - "is_null": Match items where the field is null
        - "is_empty": Match items where the field is empty
        - "nested_filter": Nested AND/OR group; requires "logic" and "filters" keys
          instead of "value". Keys starting with "_" are used for grouping only and
          are not passed as field names to Qdrant.
          Example: {"_group": {"operator": "nested_filter", "logic": "should",
                                "filters": {"tags": {"operator": "match_any",
                                                     "value": ["HIV", "SRHR"]}}}}
    """
    conditions: List[Any] = []

    for field, filter_config in filters.items():
        operator = filter_config.get("operator", "match")
        value = filter_config.get("value")

        condition: Optional[Any] = None

        if operator == "match":
            condition = models.FieldCondition(
                key=field,
                match=models.MatchValue(value=value),
            )
        elif operator == "match_any":
            condition = models.FieldCondition(
                key=field,
                match=models.MatchAny(
                    any=value if isinstance(value, list) else [value]
                ),
            )
        elif operator == "match_text":
            condition = models.FieldCondition(
                key=field,
                match=models.MatchText(text=value),
            )
        elif operator == "range":
            range_config: Dict[str, Any] = {
                k: filter_config[k]
                for k in ("gte", "lte", "gt", "lt")
                if k in filter_config
            }
            if any(_is_date_value(v) for v in range_config.values()):
                condition = models.FieldCondition(
                    key=field,
                    range=models.DatetimeRange(
                        **{k: _parse_date(v) for k, v in range_config.items()}
                    ),
                )
            else:
                condition = models.FieldCondition(
                    key=field,
                    range=models.Range(**range_config),
                )
        elif operator == "is_null":
            condition = models.IsNullCondition(
                is_null=models.PayloadField(key=field)
            )
        elif operator == "is_empty":
            condition = models.IsEmptyCondition(
                is_empty=models.PayloadField(key=field)
            )
        elif operator == "nested_filter":
            sub_logic = filter_config.get("logic", "should")
            sub_filters = filter_config.get("filters", {})
            sub_filter = build_filter(sub_filters, sub_logic)
            conditions.append(sub_filter)
            continue

        if condition is not None:
            conditions.append(condition)

    # Build filter based on logic
    if logic == "should":
        return models.Filter(should=conditions)
    elif logic == "must_not":
        return models.Filter(must_not=conditions)
    else:  # default to "must"
        return models.Filter(must=conditions)


def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 10,
    fusion_method: str = "rrf",
    prefetch_limit: int = 20,
    filters: Optional[Dict[str, Any]] = None,
    filter_logic: str = "must",
    dense_vector_name: str = "dense",
    sparse_vector_name: str = "sparse",
    get_dense_fn: Optional[Callable[[str], List[float]]] = None,
    get_sparse_fn: Optional[Callable[[str], models.SparseVector]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining semantic and keyword search.

    This function uses Qdrant's Query API with prefetch and fusion to combine
    results from both dense (semantic) and sparse (BM25) vector searches.

    Args:
        client: Qdrant client instance
        collection_name: Qdrant collection name (e.g., "zotero_hybrid")
        query: Search query text
        top_k: Number of final results to return (default: 10)
        fusion_method: Fusion algorithm to use (default: "rrf")
            - "rrf": Reciprocal Rank Fusion (recommended)
            - "dbsf": Distribution-Based Score Fusion
        prefetch_limit: Number of candidates to retrieve from each method (default: 20)
        filters: Optional metadata filters to apply
        filter_logic: How to combine filters - "must" (AND), "should" (OR), "must_not" (NOT)
        dense_vector_name: Name of the dense vector field in the collection
        sparse_vector_name: Name of the sparse vector field in the collection
        get_dense_fn: Custom function to get dense embeddings (default: get_dense_embedding)
        get_sparse_fn: Custom function to get sparse embeddings (default: get_sparse_embedding)
        verbose: Whether to print progress messages (default: True)

    Returns:
        List of search results with scores and payloads

    Example:
        from qdrant_client import QdrantClient
        from kbase.vector import hybrid_search

        client = QdrantClient(url="http://localhost:6333")
        results = hybrid_search(client, "zotero_hybrid", "HIV prevention strategies")

        for result in results:
            print(f"Score: {result['score']}, ID: {result['id']}")
    """
    try:
        # Use custom embedding functions or defaults
        dense_fn = get_dense_fn or get_dense_embedding
        sparse_fn = get_sparse_fn or get_sparse_embedding

        # Generate embeddings
        if verbose:
            print(
                f"Generating embeddings for query: {query[:50]}...",
                file=sys.stderr,
            )
        dense_embedding = dense_fn(query)
        sparse_embedding = sparse_fn(query)

        # Build filter if provided
        query_filter: Optional[models.Filter] = None
        if filters:
            query_filter = build_filter(filters, filter_logic)

        # Perform hybrid search using Query API
        if verbose:
            print(
                f"Performing hybrid search with {fusion_method} fusion...",
                file=sys.stderr,
            )

        # Select fusion type
        fusion = models.Fusion.RRF if fusion_method == "rrf" else models.Fusion.DBSF

        # Convert SparseVector to dict for pydantic v2 compatibility
        sparse_query = (
            {"indices": sparse_embedding.indices, "values": sparse_embedding.values}
            if hasattr(sparse_embedding, "indices")
            else sparse_embedding
        )

        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                # Branch 1: Dense vector search (semantic)
                models.Prefetch(
                    query=dense_embedding,
                    using=dense_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                # Branch 2: Sparse vector search (BM25)
                models.Prefetch(
                    query=sparse_query,
                    using=sparse_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=fusion),
            limit=top_k,
            with_payload=True,
        )

        # Convert to list of dicts for easier handling
        search_results: List[Dict[str, Any]] = []
        for point in results.points:
            result = {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
            search_results.append(result)

        if verbose:
            print(
                f"Hybrid search completed: {len(search_results)} results",
                file=sys.stderr,
            )

        return search_results

    except Exception as e:
        print(f"Error in hybrid search: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return []


def compare_search_methods(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 10,
    dense_vector_name: str = "dense",
    sparse_vector_name: str = "sparse",
    get_dense_fn: Optional[Callable[[str], List[float]]] = None,
    get_sparse_fn: Optional[Callable[[str], models.SparseVector]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare results from semantic-only, BM25-only, and hybrid search.

    Useful for testing and understanding the contribution of each method.

    Args:
        client: Qdrant client instance
        collection_name: Qdrant collection name
        query: Search query text
        top_k: Number of results to return from each method
        dense_vector_name: Name of the dense vector field
        sparse_vector_name: Name of the sparse vector field
        get_dense_fn: Custom function to get dense embeddings
        get_sparse_fn: Custom function to get sparse embeddings

    Returns:
        Dictionary with keys:
            - "semantic": Semantic search results
            - "bm25": BM25 search results
            - "hybrid_rrf": Hybrid search with RRF fusion
            - "hybrid_dbsf": Hybrid search with DBSF fusion

    Example:
        comparison = compare_search_methods(client, "zotero_hybrid", "query")
        print(f"Semantic results: {len(comparison['semantic'])}")
        print(f"BM25 results: {len(comparison['bm25'])}")
        print(f"Hybrid RRF results: {len(comparison['hybrid_rrf'])}")
    """
    try:
        # Use custom embedding functions or defaults
        dense_fn = get_dense_fn or get_dense_embedding
        sparse_fn = get_sparse_fn or get_sparse_embedding

        # Generate embeddings
        dense_embedding = dense_fn(query)
        sparse_embedding = sparse_fn(query)

        results: Dict[str, List[Dict[str, Any]]] = {}

        # 1. Semantic-only search
        print("Running semantic-only search...", file=sys.stderr)
        semantic_results = client.query_points(
            collection_name=collection_name,
            query=dense_embedding,
            using=dense_vector_name,
            limit=top_k,
            with_payload=True,
        )
        results["semantic"] = [
            {"id": p.id, "score": p.score, "payload": p.payload}
            for p in semantic_results.points
        ]

        # 2. BM25-only search
        print("Running BM25-only search...", file=sys.stderr)
        sparse_query = (
            {"indices": sparse_embedding.indices, "values": sparse_embedding.values}
            if hasattr(sparse_embedding, "indices")
            else sparse_embedding
        )
        bm25_results = client.query_points(
            collection_name=collection_name,
            query=sparse_query,
            using=sparse_vector_name,
            limit=top_k,
            with_payload=True,
        )
        results["bm25"] = [
            {"id": p.id, "score": p.score, "payload": p.payload}
            for p in bm25_results.points
        ]

        # 3. Hybrid with RRF
        print("Running hybrid search (RRF)...", file=sys.stderr)
        results["hybrid_rrf"] = hybrid_search(
            client,
            collection_name,
            query,
            top_k,
            fusion_method="rrf",
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            get_dense_fn=get_dense_fn,
            get_sparse_fn=get_sparse_fn,
            verbose=False,
        )

        # 4. Hybrid with DBSF
        print("Running hybrid search (DBSF)...", file=sys.stderr)
        results["hybrid_dbsf"] = hybrid_search(
            client,
            collection_name,
            query,
            top_k,
            fusion_method="dbsf",
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            get_dense_fn=get_dense_fn,
            get_sparse_fn=get_sparse_fn,
            verbose=False,
        )

        return results

    except Exception as e:
        print(f"Error comparing search methods: {e}", file=sys.stderr)
        return {}


__all__ = [
    "build_filter",
    "hybrid_search",
    "compare_search_methods",
]
