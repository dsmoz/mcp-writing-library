"""OpenAI-compatible embedding client with TTL cache.

Provides a shared client for LM Studio / OpenAI-compatible embedding APIs.
Includes in-process TTL cache to avoid re-embedding identical texts within
a session, and a 3-attempt retry for transient API failures.

Environment variables:
    EMBEDDING_BASE_URL: API base URL (default: http://localhost:1234/v1)
    EMBEDDING_API_KEY:  API key (default: "local")
    EMBEDDING_MODEL:    Model name (default: nomic-embed-text-v1.5)
"""

import logging
import os
import time
from typing import Optional

from openai import OpenAI

from kbase.utils.cache import TTLCache

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None
_embed_cache = TTLCache(ttl=300)


def get_openai_compatible_client(
    base_url: str = None,
    api_key: str = None,
) -> OpenAI:
    """Get or create OpenAI-compatible embedding client (singleton).

    Args:
        base_url: API base URL. Falls back to EMBEDDING_BASE_URL env var,
                  then http://localhost:1234/v1.
        api_key:  API key. Falls back to EMBEDDING_API_KEY env var, then "local".

    Returns:
        OpenAI client instance
    """
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=base_url or os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1"),
            api_key=api_key or os.getenv("EMBEDDING_API_KEY", "local"),
        )
    return _client


def reset_embedding_client() -> None:
    """Force-create a new client on next call (use after connection errors)."""
    global _client
    _client = None


def get_embedding_with_cache(
    text: str,
    model: str = None,
    ttl: int = 300,
) -> list[float]:
    """
    Generate a dense embedding with in-process TTL caching.

    Text is truncated to 8000 chars before embedding to stay within typical
    token limits. Results are cached for `ttl` seconds — avoids redundant
    API calls when the same query or chunk is embedded multiple times.

    Args:
        text:  Text to embed (truncated to 8000 chars)
        model: Embedding model name. Falls back to EMBEDDING_MODEL env var,
               then "nomic-embed-text-v1.5".
        ttl:   Cache TTL in seconds (default 300). Ignored if the cache was
               already created; use reset_embedding_client() to reset cache.

    Returns:
        List of floats (embedding vector)

    Raises:
        RuntimeError: If embedding fails after 3 attempts
    """
    text = text[:8000]

    cached = _embed_cache.get(text)
    if cached is not None:
        return cached

    model = model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
    last_exc = None

    for attempt in range(3):
        try:
            response = get_openai_compatible_client().embeddings.create(
                model=model,
                input=text,
            )
            vec = response.data[0].embedding
            _embed_cache.set(text, vec)
            return vec
        except Exception as e:
            last_exc = e
            logger.warning(
                f"Embedding attempt {attempt + 1}/3 failed ({type(e).__name__}): {e}"
            )
            if attempt < 2:
                time.sleep(5)

    raise RuntimeError(f"Embedding failed after 3 attempts: {last_exc}") from last_exc
