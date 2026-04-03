"""Shared utilities for kbase-core."""

from kbase.utils.file_utils import check_file_changed, compute_sha256
from kbase.utils.cache import TTLCache
from kbase.utils.embedding_cache import (
    EmbeddingCache,
    SparseEmbeddingCache,
    get_dense_embedding_cached,
    get_sparse_embedding_cached,
    clear_cache,
    get_cache_stats,
)

__all__ = [
    # File hashing
    "compute_sha256",
    "check_file_changed",
    # In-process TTL cache
    "TTLCache",
    # Embedding caches (LRU + TTL)
    "EmbeddingCache",
    "SparseEmbeddingCache",
    "get_dense_embedding_cached",
    "get_sparse_embedding_cached",
    "clear_cache",
    "get_cache_stats",
]
