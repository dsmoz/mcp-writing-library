"""
Hybrid Embeddings for Dense and Sparse Vectors

This module provides functions to generate both dense (semantic) and sparse (BM25)
embeddings for hybrid search in Qdrant.

Dense embeddings: OpenAI-compatible API (semantic understanding)
Sparse embeddings: BM25 algorithm (keyword matching)

Usage:
    from kbase.vector import (
        get_dense_embedding,
        get_sparse_embedding,
        get_hybrid_embeddings,
        get_hybrid_embeddings_batch,
    )

    # Single embedding
    dense, sparse = get_hybrid_embeddings("search query")

    # Batch embeddings (5-10x faster)
    embeddings = get_hybrid_embeddings_batch(["query1", "query2", "query3"])

Environment Variables:
    EMBEDDING_BASE_URL: API base URL (e.g., "https://api.openai.com/v1")
    EMBEDDING_MODEL: Model name (default: "text-embedding-3-small")
    EMBEDDING_API_KEY: API key for authentication
"""

import os
import sys
import time
from typing import List, Optional, Tuple

import requests
from qdrant_client import models

from kbase.vector.bm25_encoder import get_bm25_encoder


def get_dense_embedding(
    text: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[float]:
    """
    Generate dense semantic embedding using OpenAI-compatible API.

    Args:
        text: Text to embed
        base_url: API base URL (default: from env EMBEDDING_BASE_URL)
        model: Model name (default: from env EMBEDDING_MODEL)
        api_key: API key (default: from env EMBEDDING_API_KEY)

    Returns:
        Dense vector embedding (list of floats)

    Raises:
        requests.HTTPError: If API request fails
        ValueError: If API returns unexpected response

    Example:
        embedding = get_dense_embedding("semantic search query")
    """
    if base_url is None:
        base_url = os.getenv("EMBEDDING_BASE_URL")
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    if api_key is None:
        api_key = os.getenv("EMBEDDING_API_KEY")

    if not base_url:
        raise ValueError("EMBEDDING_BASE_URL not set")
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY not set")

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
        return data["data"][0]["embedding"]

    except requests.RequestException as e:
        print(f"Error getting dense embedding: {e}", file=sys.stderr)
        raise


def get_dense_embeddings_batch(
    texts: List[str],
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 100,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = True,
) -> List[List[float]]:
    """
    Generate dense semantic embeddings for multiple texts in batches.

    This function sends multiple texts to the API in a single request,
    which is 5-10x faster than calling get_dense_embedding() for each text.

    Args:
        texts: List of texts to embed
        base_url: API base URL (default: from env EMBEDDING_BASE_URL)
        model: Model name (default: from env EMBEDDING_MODEL)
        api_key: API key (default: from env EMBEDDING_API_KEY)
        batch_size: Max texts per API call (default: 100, max: 2048)
        max_retries: Number of retry attempts on failure (default: 3)
        retry_delay: Base delay between retries in seconds (default: 2.0)
        verbose: Whether to print progress messages (default: True)

    Returns:
        List of dense vector embeddings (one per input text)

    Raises:
        requests.HTTPError: If API request fails after all retries

    Note:
        - OpenAI API supports up to 2,048 texts per request
        - Batch size of 50-100 is recommended for optimal performance
        - Uses exponential backoff on retries (2s, 4s, 8s)
    """
    if base_url is None:
        base_url = os.getenv("EMBEDDING_BASE_URL")
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    if api_key is None:
        api_key = os.getenv("EMBEDDING_API_KEY")

    if not base_url:
        raise ValueError("EMBEDDING_BASE_URL not set")
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY not set")

    url = f"{base_url}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    # Process in batches to stay under API limits
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                # Rate limiting: add delay between batches (except first)
                if i > 0 and attempt == 0:
                    time.sleep(retry_delay)

                payload = {
                    "input": batch_texts,  # Send array of texts
                    "model": model,
                }

                response = requests.post(
                    url, json=payload, headers=headers, timeout=60
                )
                response.raise_for_status()

                data = response.json()
                # Extract embeddings in order
                batch_embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(batch_embeddings)

                # Log progress
                if verbose:
                    print(
                        f"  Dense embeddings: batch {batch_num}/{total_batches} "
                        f"({len(batch_texts)} texts)",
                        file=sys.stderr,
                    )
                break  # Success - exit retry loop

            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    print(
                        f"Error getting dense embeddings for batch {batch_num} "
                        f"after {max_retries} attempts: {e}",
                        file=sys.stderr,
                    )
                    raise

                # Exponential backoff
                wait_time = retry_delay * (2**attempt)
                print(
                    f"  Batch {batch_num} failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait_time}s: {e}",
                    file=sys.stderr,
                )
                time.sleep(wait_time)

    return all_embeddings


def _hash_sparse_fallback(text: str) -> models.SparseVector:
    """
    Hash-based TF-IDF sparse vector fallback when no BM25 encoder is available.

    Uses deterministic MD5 hashing to map terms to 20-bit integer indices.
    Produces consistent index space across all servers using this function,
    enabling cross-collection sparse search even without a trained encoder.

    This is intentionally kept as a fallback — BM25 with a fitted encoder
    is more accurate. Use train_bm25_encoder.py to build a proper encoder.
    """
    import hashlib
    import math
    import re
    from collections import Counter

    tokens = [t for t in re.findall(r'[a-zA-Z0-9]+', text.lower()) if len(t) > 2]
    if not tokens:
        return models.SparseVector(indices=[0], values=[0.0])

    total = len(tokens)
    term_counts = Counter(tokens)
    indices = []
    values = []

    for term, count in term_counts.items():
        term_hash = int(hashlib.md5(term.encode()).hexdigest(), 16) % (2 ** 20)
        tf = count / total
        weight = tf * math.log(1.0 + 1.0 / tf)
        indices.append(term_hash)
        values.append(float(weight))

    return models.SparseVector(indices=indices, values=values)


def get_sparse_embedding(
    text: str,
    encoder_path: Optional[str] = None,
) -> models.SparseVector:
    """
    Generate sparse BM25 embedding.

    Uses a fitted BM25 encoder when available (.data/bm25_encoder.pkl by default).
    Falls back to deterministic hash-based TF-IDF if no encoder is found — this
    ensures cross-collection sparse search still works, though with lower accuracy
    than a trained BM25 encoder.

    Args:
        text: Text to embed
        encoder_path: Optional path to BM25 encoder file.
            Default: .data/bm25_encoder.pkl (relative to working directory).

    Returns:
        Qdrant SparseVector with BM25 scores (or hash-TF-IDF if no encoder).
    """
    encoder = get_bm25_encoder(encoder_path)

    if encoder is None:
        return _hash_sparse_fallback(text)

    return encoder.encode(text)


def get_hybrid_embeddings(
    text: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    encoder_path: Optional[str] = None,
) -> Tuple[List[float], models.SparseVector]:
    """
    Generate both dense and sparse embeddings for hybrid search.

    Args:
        text: Text to embed
        base_url: API base URL for dense embeddings
        model: Model name for dense embeddings
        api_key: API key for dense embeddings
        encoder_path: Path to BM25 encoder for sparse embeddings

    Returns:
        Tuple of (dense_vector, sparse_vector)

    Example:
        dense, sparse = get_hybrid_embeddings("HIV prevention strategies")
    """
    dense = get_dense_embedding(text, base_url, model, api_key)
    sparse = get_sparse_embedding(text, encoder_path)

    return dense, sparse


def get_hybrid_embeddings_batch(
    texts: List[str],
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    encoder_path: Optional[str] = None,
    batch_size: int = 100,
    verbose: bool = True,
) -> List[Tuple[List[float], models.SparseVector]]:
    """
    Generate hybrid embeddings for multiple texts efficiently.

    Uses TRUE batch embedding for dense vectors (single API call for all texts).
    BM25 sparse vectors are generated locally (already fast).

    This provides 5-10x speedup over calling get_hybrid_embeddings() for each text.

    Args:
        texts: List of texts to embed
        base_url: API base URL for dense embeddings
        model: Model name for dense embeddings
        api_key: API key for dense embeddings
        encoder_path: Path to BM25 encoder for sparse embeddings
        batch_size: Max texts per dense embedding API call
        verbose: Whether to print progress messages

    Returns:
        List of (dense_vector, sparse_vector) tuples

    Note:
        - Dense embeddings: batched API calls (5-10x faster)
        - Sparse embeddings: local BM25 encoding (no API calls)
        - Total speedup: 5-10x for dense + sparse combined
    """
    # Get ALL dense embeddings in ONE or FEW API calls (batched)
    dense_embeddings = get_dense_embeddings_batch(
        texts,
        base_url=base_url,
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Get sparse embeddings (local, fast) — falls back to hash-TF-IDF if no encoder
    encoder = get_bm25_encoder(encoder_path)
    if encoder is None:
        sparse_embeddings = [_hash_sparse_fallback(text) for text in texts]
    else:
        sparse_embeddings = [encoder.encode(text) for text in texts]

    # Zip together
    return list(zip(dense_embeddings, sparse_embeddings))


__all__ = [
    "get_dense_embedding",
    "get_dense_embeddings_batch",
    "get_sparse_embedding",
    "get_hybrid_embeddings",
    "get_hybrid_embeddings_batch",
]
