"""
Vector database operations using Qdrant and embeddings.

This module provides:
- Qdrant client management (async and sync)
- OpenAI-compatible embedding generation
- Document indexing and similarity search
- BM25 sparse vector encoding for hybrid search
- Hybrid search with dense + sparse vectors and RRF fusion
"""

from kbase.vector.client import (
    close_client,
    create_client,
    ensure_collection,
    get_client,
    get_collection_info,
    health_check,
)
from kbase.vector.embeddings import (
    generate_embedding,
    generate_embeddings_batch,
    get_embedding_dimensions,
    get_openai_client,
    init_openai_client,
    truncate_text,
)
from kbase.vector.search import (
    delete_document_vectors,
    get_document_chunks,
    index_document,
    search_similar,
)

# BM25 Encoder for sparse vectors
from kbase.vector.bm25_encoder import (
    BM25Encoder,
    BM25Stats,
    encode_text_to_sparse_vector,
    get_bm25_encoder,
    set_bm25_encoder,
)

# Hybrid embeddings (dense + sparse)
from kbase.vector.hybrid_embeddings import (
    get_dense_embedding,
    get_dense_embeddings_batch,
    get_hybrid_embeddings,
    get_hybrid_embeddings_batch,
    get_sparse_embedding,
)

# Hybrid search with RRF fusion
from kbase.vector.hybrid_search import (
    build_filter,
    compare_search_methods,
    hybrid_search,
)

# Sync Qdrant client + payload index management
from kbase.vector.sync_client import (
    check_connection,
    ensure_payload_indexes,
    get_qdrant_client,
    reset_client,
)

# OpenAI-compatible embedding client with TTL cache
from kbase.vector.embedding_client import (
    get_embedding_with_cache,
    get_openai_compatible_client,
    reset_embedding_client,
)

__all__ = [
    # Qdrant client (async)
    "create_client",
    "get_client",
    "close_client",
    "ensure_collection",
    "get_collection_info",
    "health_check",
    # OpenAI embeddings (async)
    "init_openai_client",
    "get_openai_client",
    "generate_embedding",
    "generate_embeddings_batch",
    "get_embedding_dimensions",
    "truncate_text",
    # Search operations (async)
    "index_document",
    "search_similar",
    "delete_document_vectors",
    "get_document_chunks",
    # BM25 Encoder (sync)
    "BM25Encoder",
    "BM25Stats",
    "get_bm25_encoder",
    "set_bm25_encoder",
    "encode_text_to_sparse_vector",
    # Hybrid embeddings (sync)
    "get_dense_embedding",
    "get_dense_embeddings_batch",
    "get_sparse_embedding",
    "get_hybrid_embeddings",
    "get_hybrid_embeddings_batch",
    # Hybrid search (sync)
    "build_filter",
    "hybrid_search",
    "compare_search_methods",
    # Sync Qdrant client
    "get_qdrant_client",
    "reset_client",
    "check_connection",
    "ensure_payload_indexes",
    # Embedding client with TTL cache
    "get_openai_compatible_client",
    "reset_embedding_client",
    "get_embedding_with_cache",
]
