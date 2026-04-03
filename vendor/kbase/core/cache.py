"""
Caching service for kbase-core.

Provides both in-memory (LRU with TTL) and Redis-based caching
for search results, documents, and embeddings.
"""

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel


class CacheConfig(BaseModel):
    """Configuration for the cache service."""

    # Redis configuration (optional - if not provided, uses in-memory cache)
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

    # Cache behavior
    default_ttl: int = 300  # 5 minutes
    key_prefix: str = "kbase:"

    # In-memory cache settings
    max_memory_items: int = 1000
    use_redis: bool = False


class InMemoryCache:
    """
    LRU cache with TTL for in-memory caching.

    Thread-safe implementation with automatic eviction.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize the in-memory cache.

        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for each entry in seconds
        """
        self._cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _hash_key(self, key: str) -> str:
        """Generate hash key."""
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value for key.

        Args:
            key: The cache key to look up.

        Returns:
            Cached value if found and not expired, None otherwise.
        """
        hashed_key = self._hash_key(key)

        with self._lock:
            if hashed_key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[hashed_key]

            # Check TTL
            if datetime.now() - timestamp > self._ttl:
                del self._cache[hashed_key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(hashed_key)
            self._hits += 1

            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Cache a value.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Optional TTL override in seconds.
        """
        hashed_key = self._hash_key(key)

        with self._lock:
            if hashed_key in self._cache:
                self._cache[hashed_key] = (value, datetime.now())
                self._cache.move_to_end(hashed_key)
                return

            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[hashed_key] = (value, datetime.now())

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        hashed_key = self._hash_key(key)
        with self._lock:
            if hashed_key in self._cache:
                del self._cache[hashed_key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "ttl_seconds": self._ttl.total_seconds(),
            }


class CacheService:
    """
    Unified caching service supporting both in-memory and Redis backends.

    Provides methods for caching search results, documents, and collections
    with automatic key generation and invalidation patterns.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the cache service.

        Args:
            config: Cache configuration. If None, uses defaults (in-memory).
        """
        self.config = config or CacheConfig()
        self._redis_client = None
        self._memory_cache = InMemoryCache(
            max_size=self.config.max_memory_items,
            ttl_seconds=self.config.default_ttl,
        )

        if self.config.use_redis:
            self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis

            self._redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=True,
            )
            # Test connection
            self._redis_client.ping()
        except Exception as e:
            from kbase.core.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(
                "Redis connection failed, falling back to in-memory cache",
                error=str(e),
            )
            self._redis_client = None

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.config.key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        full_key = self._make_key(key)

        if self._redis_client:
            try:
                value = self._redis_client.get(full_key)
                if value:
                    return json.loads(value)
                return None
            except Exception:
                pass

        return self._memory_cache.get(full_key)

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON serializable).
            ttl: Optional TTL in seconds.
        """
        full_key = self._make_key(key)
        ttl = ttl or self.config.default_ttl

        if self._redis_client:
            try:
                self._redis_client.setex(full_key, ttl, json.dumps(value))
                return
            except Exception:
                pass

        self._memory_cache.set(full_key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        full_key = self._make_key(key)

        if self._redis_client:
            try:
                return bool(self._redis_client.delete(full_key))
            except Exception:
                pass

        return self._memory_cache.delete(full_key)

    async def invalidate_document(self, doc_id: str) -> int:
        """
        Invalidate all cache entries related to a document.

        Args:
            doc_id: Document ID.

        Returns:
            Number of keys invalidated.
        """
        pattern = f"doc:{doc_id}:*"
        return await self.invalidate_pattern(pattern)

    async def invalidate_collection(self, collection_id: str) -> int:
        """
        Invalidate all cache entries related to a collection.

        Args:
            collection_id: Collection ID.

        Returns:
            Number of keys invalidated.
        """
        pattern = f"collection:{collection_id}:*"
        return await self.invalidate_pattern(pattern)

    async def invalidate_global_search(self) -> int:
        """
        Invalidate all search cache entries.

        Returns:
            Number of keys invalidated.
        """
        pattern = "search:*"
        return await self.invalidate_pattern(pattern)

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Glob pattern for keys to invalidate.

        Returns:
            Number of keys invalidated.
        """
        full_pattern = self._make_key(pattern)
        count = 0

        if self._redis_client:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis_client.scan(
                        cursor=cursor, match=full_pattern, count=100
                    )
                    if keys:
                        count += self._redis_client.delete(*keys)
                    if cursor == 0:
                        break
                return count
            except Exception:
                pass

        # For in-memory cache, we'd need to iterate (less efficient)
        # Just clear for now
        self._memory_cache.clear()
        return 1

    def generate_search_key(
        self,
        query: str,
        collection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a deterministic cache key for a search query.

        Args:
            query: Search query text.
            collection_id: Optional collection ID filter.
            filters: Optional additional filters.

        Returns:
            Cache key string.
        """
        key_parts = [query]
        if collection_id:
            key_parts.append(f"c:{collection_id}")
        if filters:
            # Sort filters for deterministic ordering
            sorted_filters = json.dumps(filters, sort_keys=True)
            key_parts.append(f"f:{sorted_filters}")

        combined = "|".join(key_parts)
        key_hash = hashlib.md5(combined.encode("utf-8")).hexdigest()[:16]
        return f"search:{key_hash}"

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics.
        """
        stats = {"backend": "redis" if self._redis_client else "memory"}

        if self._redis_client:
            try:
                info = self._redis_client.info("memory")
                stats["redis_memory"] = info.get("used_memory_human", "unknown")
                stats["redis_keys"] = self._redis_client.dbsize()
            except Exception:
                pass
        else:
            stats.update(self._memory_cache.stats())

        return stats

    async def clear_all(self) -> None:
        """Clear all cache entries."""
        if self._redis_client:
            try:
                pattern = self._make_key("*")
                cursor = 0
                while True:
                    cursor, keys = self._redis_client.scan(
                        cursor=cursor, match=pattern, count=100
                    )
                    if keys:
                        self._redis_client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception:
                pass

        self._memory_cache.clear()


# Global cache instance
_cache_service: Optional[CacheService] = None


async def init_cache(config: Optional[CacheConfig] = None) -> CacheService:
    """
    Initialize the global cache service.

    Args:
        config: Cache configuration.

    Returns:
        Initialized CacheService instance.
    """
    global _cache_service
    _cache_service = CacheService(config)
    return _cache_service


def get_cache() -> CacheService:
    """
    Get the global cache service.

    Returns:
        CacheService instance.

    Raises:
        RuntimeError: If cache not initialized.
    """
    if _cache_service is None:
        # Auto-initialize with defaults
        return CacheService()
    return _cache_service


__all__ = [
    "CacheConfig",
    "CacheService",
    "InMemoryCache",
    "init_cache",
    "get_cache",
]
