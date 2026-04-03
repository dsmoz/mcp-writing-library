"""
Embedding generation for kbase-core.

Provides dense semantic embeddings using OpenAI-compatible APIs
with support for batching and caching.
"""

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests

from kbase.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    api_key: str
    model: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    vector_size: int = 1536
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 2.0


class EmbeddingCache:
    """
    Thread-safe LRU cache with TTL for embeddings.

    Reduces API calls by caching recently used embeddings.
    """

    def __init__(self, max_size: int = 1000, ttl_minutes: int = 60):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum cached embeddings
            ttl_minutes: Time-to-live in minutes
        """
        from collections import OrderedDict

        self._cache: OrderedDict[str, Tuple[List[float], datetime]] = OrderedDict()
        self._max_size = max_size
        self._ttl = timedelta(minutes=ttl_minutes)
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        combined = f"{model}:{text}"
        return hashlib.md5(combined.encode("utf-8")).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if exists and not expired."""
        key = self._hash_text(text, model)

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

    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        key = self._hash_text(text, model)

        with self._lock:
            if key in self._cache:
                self._cache[key] = (embedding, datetime.now())
                self._cache.move_to_end(key)
                return

            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (embedding, datetime.now())

    def clear(self) -> None:
        """Clear all cached embeddings."""
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
            }


class EmbeddingGenerator:
    """
    Generate embeddings using OpenAI-compatible APIs.

    Supports single and batch embedding generation with
    automatic retries, rate limiting, and caching.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        vector_size: int = 1536,
        use_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl_minutes: int = 60,
    ):
        """
        Initialize the embedding generator.

        Args:
            api_key: API key (defaults to EMBEDDING_API_KEY or OPENAI_API_KEY env)
            model: Embedding model name
            base_url: API base URL (defaults to EMBEDDING_BASE_URL or OpenAI)
            vector_size: Expected embedding dimensions
            use_cache: Enable caching
            cache_size: Maximum cached embeddings
            cache_ttl_minutes: Cache TTL in minutes
        """
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        self.vector_size = vector_size

        if not self.api_key:
            raise ValueError("API key required. Set EMBEDDING_API_KEY or OPENAI_API_KEY environment variable.")

        self._cache = EmbeddingCache(max_size=cache_size, ttl_minutes=cache_ttl_minutes) if use_cache else None
        self._batch_size = 100
        self._max_retries = 3
        self._retry_delay = 2.0

    def _make_request(
        self,
        texts: List[str],
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> List[List[float]]:
        """
        Make embedding API request with retries.

        Args:
            texts: List of texts to embed
            max_retries: Number of retry attempts
            retry_delay: Base delay between retries

        Returns:
            List of embedding vectors
        """
        url = f"{self.base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "input": texts if len(texts) > 1 else texts[0],
            "model": self.model,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()

                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]

                return embeddings

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "Embedding API request failed",
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=max_retries,
                    )
                    raise

                wait_time = retry_delay * (2**attempt)
                logger.warning(
                    "Embedding API request failed, retrying",
                    error=str(e),
                    attempt=attempt + 1,
                    wait_seconds=wait_time,
                )
                time.sleep(wait_time)

        return []

    def generate(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (list of floats)
        """
        # Check cache
        if self._cache:
            cached = self._cache.get(text, self.model)
            if cached is not None:
                return cached

        embeddings = self._make_request([text])
        embedding = embeddings[0]

        # Cache result
        if self._cache:
            self._cache.set(text, self.model, embedding)

        return embedding

    def generate_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Uses batched API calls for 5-10x speedup over individual calls.

        Args:
            texts: List of texts to embed
            batch_size: Texts per API call (default: 100)
            show_progress: Log progress

        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            return []

        batch_size = batch_size or self._batch_size
        all_embeddings: List[List[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        # Check cache for all texts first
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        cached_embeddings: Dict[int, List[float]] = {}

        if self._cache:
            for i, text in enumerate(texts):
                cached = self._cache.get(text, self.model)
                if cached is not None:
                    cached_embeddings[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

            if show_progress and cached_embeddings:
                logger.info(
                    "Cache hits for batch embedding",
                    cached=len(cached_embeddings),
                    uncached=len(uncached_texts),
                )
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Process uncached texts in batches
        if uncached_texts:
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i : i + batch_size]
                batch_num = i // batch_size + 1

                # Rate limiting between batches
                if i > 0:
                    time.sleep(self._retry_delay)

                embeddings = self._make_request(batch_texts)

                # Cache results
                if self._cache:
                    for text, embedding in zip(batch_texts, embeddings):
                        self._cache.set(text, self.model, embedding)

                # Map back to original indices
                for j, embedding in enumerate(embeddings):
                    original_idx = uncached_indices[i + j]
                    cached_embeddings[original_idx] = embedding

                if show_progress:
                    logger.info(
                        "Batch embedding progress",
                        batch=batch_num,
                        total_batches=total_batches,
                        texts_in_batch=len(batch_texts),
                    )

        # Reconstruct in original order
        all_embeddings = [cached_embeddings[i] for i in range(len(texts))]

        return all_embeddings

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.stats()
        return {"caching": "disabled"}

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self._cache:
            self._cache.clear()


# Global instance for convenience
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get or create the global embedding generator.

    Returns:
        EmbeddingGenerator instance
    """
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using global generator.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    return get_embedding_generator().generate(text)


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts using global generator.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    return get_embedding_generator().generate_batch(texts)


__all__ = [
    "EmbeddingConfig",
    "EmbeddingCache",
    "EmbeddingGenerator",
    "get_embedding_generator",
    "generate_embedding",
    "generate_embeddings_batch",
]
