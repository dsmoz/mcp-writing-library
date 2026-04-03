"""Thread-safe LRU+TTL caches for dense and sparse embedding vectors.

Caching embeddings avoids redundant LM Studio / OpenAI API calls when the same
text is embedded multiple times in a session.

    Cache hit:  <1ms (memory lookup)
    Cache miss: 50-150ms (API call to embedding provider)

Both caches use SHA256 hashes as keys to handle long texts efficiently.

Usage:
    from kbase.utils.embedding_cache import (
        EmbeddingCache,
        SparseEmbeddingCache,
        get_dense_embedding_cached,
        get_sparse_embedding_cached,
    )

    # Module-level singletons (max_size=200, ttl=30 min):
    embedding = get_dense_embedding_cached("funding for HIV prevention")
    sparse = get_sparse_embedding_cached("HIV prevention Mozambique")

    # Custom instance:
    cache = EmbeddingCache(max_size=500, ttl_minutes=60)
"""

import hashlib
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import models


class EmbeddingCache:
    """Thread-safe LRU cache with TTL for dense embedding vectors.

    Args:
        max_size:    Maximum number of entries (default: 200)
        ttl_minutes: Time-to-live in minutes (default: 30)
    """

    def __init__(self, max_size: int = 200, ttl_minutes: int = 30):
        self._cache: OrderedDict[str, Tuple[List[float], datetime]] = OrderedDict()
        self._max_size = max_size
        self._ttl = timedelta(minutes=ttl_minutes)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Return cached embedding, or None if missing or expired."""
        key = self._make_key(text)
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            embedding, timestamp = self._cache[key]
            if datetime.now() - timestamp > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return embedding

    def set(self, text: str, embedding: List[float]) -> None:
        """Store embedding, evicting LRU entry if at capacity."""
        key = self._make_key(text)
        with self._lock:
            if key in self._cache:
                self._cache[key] = (embedding, datetime.now())
                self._cache.move_to_end(key)
                return
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (embedding, datetime.now())

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def size(self) -> int:
        """Current number of cached entries."""
        with self._lock:
            return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Cache statistics: size, hits, misses, hit_rate, ttl_minutes."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "ttl_minutes": self._ttl.total_seconds() / 60,
            }


class SparseEmbeddingCache:
    """Thread-safe LRU cache with TTL for sparse (BM25) embedding vectors.

    Identical API to EmbeddingCache but stores `models.SparseVector` values.

    Args:
        max_size:    Maximum number of entries (default: 200)
        ttl_minutes: Time-to-live in minutes (default: 30)
    """

    def __init__(self, max_size: int = 200, ttl_minutes: int = 30):
        self._cache: OrderedDict[str, Tuple[models.SparseVector, datetime]] = OrderedDict()
        self._max_size = max_size
        self._ttl = timedelta(minutes=ttl_minutes)
        self._lock = threading.Lock()

    def _make_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[models.SparseVector]:
        """Return cached sparse vector, or None if missing or expired."""
        key = self._make_key(text)
        with self._lock:
            if key not in self._cache:
                return None
            embedding, timestamp = self._cache[key]
            if datetime.now() - timestamp > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return embedding

    def set(self, text: str, embedding: models.SparseVector) -> None:
        """Store sparse vector, evicting LRU entry if at capacity."""
        key = self._make_key(text)
        with self._lock:
            if key in self._cache:
                self._cache[key] = (embedding, datetime.now())
                self._cache.move_to_end(key)
                return
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (embedding, datetime.now())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# ---------------------------------------------------------------------------
# Module-level singletons and convenience functions
# ---------------------------------------------------------------------------

_dense_cache = EmbeddingCache(max_size=200, ttl_minutes=30)
_sparse_cache = SparseEmbeddingCache(max_size=200, ttl_minutes=30)


def get_dense_embedding_cached(text: str) -> List[float]:
    """Get dense embedding with caching via kbase.vector.hybrid_embeddings.

    Uses the module-level singleton (max_size=200, ttl=30 min).
    Cache hit: <1ms. Cache miss: delegates to get_dense_embedding().
    """
    cached = _dense_cache.get(text)
    if cached is not None:
        return cached
    from kbase.vector.hybrid_embeddings import get_dense_embedding
    embedding = get_dense_embedding(text)
    _dense_cache.set(text, embedding)
    return embedding


def get_sparse_embedding_cached(text: str) -> models.SparseVector:
    """Get sparse (BM25) embedding with caching via kbase.vector.hybrid_embeddings.

    Uses the module-level singleton (max_size=200, ttl=30 min).
    """
    cached = _sparse_cache.get(text)
    if cached is not None:
        return cached
    from kbase.vector.hybrid_embeddings import get_sparse_embedding
    embedding = get_sparse_embedding(text)
    _sparse_cache.set(text, embedding)
    return embedding


def clear_cache() -> None:
    """Clear both dense and sparse embedding caches."""
    _dense_cache.clear()
    _sparse_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Return statistics for both caches.

    Returns:
        Dict with dense_size, sparse_size, max_size, ttl_minutes, and
        full hit/miss stats from the dense cache.
    """
    stats = _dense_cache.stats()
    stats["dense_size"] = stats.pop("size")
    stats["sparse_size"] = _sparse_cache.size()
    return stats
