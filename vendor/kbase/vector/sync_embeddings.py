"""
Synchronous Embedding Generation

This module provides synchronous embedding generation using OpenAI API.
Uses requests library for HTTP calls (sync instead of async).

Usage:
    from kbase.vector.sync_embeddings import generate_embedding
    embedding = generate_embedding("Hello world")
"""

import os
from typing import List, Optional

import requests
import structlog

logger = structlog.get_logger(__name__)

# Global OpenAI settings
_openai_base_url: Optional[str] = None
_openai_api_key: Optional[str] = None
_openai_model: Optional[str] = None


def init_openai(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "text-embedding-3-small",
) -> None:
    """
    Initialize OpenAI settings for embedding generation.

    Args:
        api_key: OpenAI API key
        base_url: API base URL (default: OpenAI)
        model: Embedding model name
    """
    global _openai_base_url, _openai_api_key, _openai_model
    _openai_base_url = base_url
    _openai_api_key = api_key
    _openai_model = model
    logger.info("OpenAI embeddings initialized", base_url=base_url, model=model)


def _get_openai_settings():
    """Get OpenAI settings from globals or environment."""
    base_url = _openai_base_url or os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    api_key = _openai_api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = _openai_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    if not api_key:
        raise RuntimeError(
            "OpenAI API key not set. Either call init_openai() or set "
            "EMBEDDING_API_KEY or OPENAI_API_KEY environment variable."
        )

    return base_url, api_key, model


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as list of floats

    Raises:
        RuntimeError: If API key not configured
        Exception: If embedding generation fails
    """
    base_url, api_key, model = _get_openai_settings()

    url = f"{base_url}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "input": text,
        "model": model,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        embedding = data["data"][0]["embedding"]

        logger.debug(
            "Generated embedding",
            text_length=len(text),
            embedding_dimensions=len(embedding),
            model=model,
        )

        return embedding

    except Exception as e:
        logger.error(
            "Failed to generate embedding",
            error=str(e),
            text_length=len(text),
            model=model,
        )
        raise


def generate_embeddings_batch(
    texts: List[str],
    batch_size: int = 100,
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches.

    This is 5-10x faster than calling generate_embedding() for each text.

    Args:
        texts: List of texts to embed
        batch_size: Max texts per API call (default: 100)

    Returns:
        List of embedding vectors

    Raises:
        RuntimeError: If API key not configured
        Exception: If embedding generation fails
    """
    if not texts:
        return []

    base_url, api_key, model = _get_openai_settings()

    url = f"{base_url}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        payload = {
            "input": batch_texts,
            "model": model,
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()

            data = response.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                "Generated batch embeddings",
                batch_size=len(batch_texts),
                total_processed=len(all_embeddings),
                total_texts=len(texts),
            )

        except Exception as e:
            logger.error(
                "Failed to generate batch embeddings",
                error=str(e),
                batch_start=i,
                batch_size=len(batch_texts),
            )
            raise

    logger.info(
        "Generated embeddings for batch",
        total_texts=len(texts),
        total_embeddings=len(all_embeddings),
        model=model,
    )

    return all_embeddings


def generate_sparse_vector(text: str) -> tuple[List[int], List[float]]:
    """
    Generate a sparse vector for BM25-style keyword search.

    Uses simple tokenization and term frequency counting.
    The IDF modifier is applied by Qdrant at query time.

    Args:
        text: Text to convert to sparse vector

    Returns:
        Tuple of (indices, values) for sparse vector
    """
    import re
    from collections import Counter

    # Simple tokenization: lowercase, split on non-alphanumeric, filter short tokens
    tokens = re.findall(r'\b[a-z0-9]{2,}\b', text.lower())

    if not tokens:
        return [], []

    # Count term frequencies
    term_counts = Counter(tokens)

    # Convert to sparse vector format
    # Use hash of token as index (mod large prime for reasonable index range)
    HASH_MOD = 2**20  # ~1M possible indices

    indices = []
    values = []

    for token, count in term_counts.items():
        # Use hash of token as index
        token_hash = hash(token) % HASH_MOD
        indices.append(token_hash)
        values.append(float(count))

    return indices, values


def generate_sparse_vectors_batch(texts: List[str]) -> List[tuple[List[int], List[float]]]:
    """
    Generate sparse vectors for multiple texts.

    Args:
        texts: List of texts to convert

    Returns:
        List of (indices, values) tuples
    """
    return [generate_sparse_vector(text) for text in texts]


# Re-export for convenience
get_embeddings = generate_embeddings_batch
