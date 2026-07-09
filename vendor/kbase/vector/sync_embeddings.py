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


# Sparse-vector index space for the deterministic md5 fallback encoder.
# Token md5 hashes are taken mod this value. A fitted BM25 encoder, when
# present, uses its own (sequential) vocab index space instead; both stay
# within this 2**20 bound so the Postgres ``sparsevec(1048576)`` mirror fits.
SPARSE_HASH_MOD = 2**20  # ~1M possible indices


def dedupe_sparse(
    indices: List[int], values: List[float]
) -> tuple[List[int], List[float]]:
    """
    Collapse duplicate sparse-vector indices, aggregating their values.

    Qdrant rejects a SparseVector whose indices are not unique
    (422: "indices: must be unique"). Distinct tokens can map to the same
    index via hash collision (mod SPARSE_HASH_MOD), so any sparse vector
    must be deduped before upsert/query. Colliding indices have their
    values summed; the result is sorted by index for determinism.

    Args:
        indices: Sparse vector indices (may contain duplicates)
        values: Parallel list of values

    Returns:
        (indices, values) with unique, ascending indices
    """
    from collections import defaultdict

    agg: dict = defaultdict(float)
    for i, v in zip(indices, values):
        agg[i] += v
    items = sorted(agg.items())
    return [i for i, _ in items], [v for _, v in items]


def _require_fitted_encoder() -> bool:
    """Whether to hard-fail rather than silently use the md5 fallback."""
    return os.getenv("BM25_REQUIRE_FITTED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def generate_sparse_vector(text: str) -> tuple[List[int], List[float]]:
    """
    Generate a DETERMINISTIC sparse vector for BM25-style keyword search.

    HISTORY / BUG: this function used to derive indices from the builtin
    ``hash(token) % SPARSE_HASH_MOD``. CPython salts ``hash(str)`` per process
    (PYTHONHASHSEED), so the indices produced when a document was *indexed*
    never matched the indices produced for the same terms at *query* time in a
    different process. The hybrid sparse arm therefore returned effectively
    nothing and hybrid search silently collapsed to dense-only in production.

    FIX: delegate to :func:`kbase.vector.hybrid_embeddings.get_sparse_embedding`,
    which uses a fitted BM25 vocab encoder when an artifact is available
    (``BM25_ENCODER_PATH`` or the default ``.data/bm25_encoder.pkl``) and a
    deterministic md5-hashed TF fallback otherwise. Both the index path
    (:func:`generate_sparse_vectors_batch`) and the query path (``sync_search``)
    call this one function, so both use the identical deterministic encoder and
    their indices reproducibly line up across processes.

    See ``docs/sparse-encoder-fix-runbook.md``.

    Args:
        text: Text to convert to sparse vector

    Returns:
        Tuple of (indices, values) for sparse vector. Indices are unique
        and ascending (deduped against md5 hash collisions). Empty input
        (no usable tokens) yields ``([], [])``.
    """
    # Local imports keep the dependency one-directional (hybrid_embeddings and
    # bm25_encoder never import this module) and avoid import-time cost for
    # callers that never touch sparse vectors.
    from kbase.vector.hybrid_embeddings import get_sparse_embedding
    from kbase.vector.bm25_encoder import get_bm25_encoder

    encoder_path = os.getenv("BM25_ENCODER_PATH") or None

    if _require_fitted_encoder() and get_bm25_encoder(encoder_path) is None:
        raise RuntimeError(
            "BM25_REQUIRE_FITTED is set but no fitted BM25 encoder was found "
            f"(BM25_ENCODER_PATH={encoder_path or '.data/bm25_encoder.pkl'}). "
            "Refusing to fall back to the md5 encoder, which lives in a "
            "different index space and would silently break hybrid search "
            "against fitted-encoder data. Deploy the fitted artifact (see the "
            "runbook) or unset BM25_REQUIRE_FITTED."
        )

    sv = get_sparse_embedding(text, encoder_path=encoder_path)

    # get_sparse_embedding returns an all-zero sentinel ([0], [0.0]) for text
    # with no usable tokens. Drop zero-weight entries so empty input yields an
    # empty vector (historical contract) and no bogus index-0 mass is stored.
    # Real BM25/TF weights are always strictly positive, so this only strips
    # the sentinel, never a legitimate term.
    pairs = [
        (int(i), float(v))
        for i, v in zip(sv.indices, sv.values)
        if float(v) != 0.0
    ]
    if not pairs:
        return [], []

    indices = [i for i, _ in pairs]
    values = [v for _, v in pairs]

    # A fitted vocab yields unique indices; the md5 fallback can collide.
    # Qdrant requires unique indices, so dedupe (and sort) before returning.
    return dedupe_sparse(indices, values)


def generate_sparse_vectors_batch(texts: List[str]) -> List[tuple[List[int], List[float]]]:
    """
    Generate sparse vectors for multiple texts.

    Uses the same deterministic encoder as :func:`generate_sparse_vector`, so
    indexed vectors reproducibly match query vectors across processes.

    Args:
        texts: List of texts to convert

    Returns:
        List of (indices, values) tuples
    """
    return [generate_sparse_vector(text) for text in texts]


# Re-export for convenience
get_embeddings = generate_embeddings_batch
