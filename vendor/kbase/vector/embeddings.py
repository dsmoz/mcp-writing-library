"""
Embedding generation using OpenAI API.
"""

from typing import List

import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger(__name__)

# Global OpenAI client
_openai_client: AsyncOpenAI | None = None


def init_openai_client(api_key: str, base_url: str | None = None) -> None:
    """
    Initialize the OpenAI client (or compatible API like LM Studio).

    Args:
        api_key: OpenAI API key (or local placeholder for LM Studio)
        base_url: Optional base URL for OpenAI-compatible APIs (e.g., LM Studio)
    """
    global _openai_client
    if base_url:
        _openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        logger.info("OpenAI-compatible client initialized", base_url=base_url)
    else:
        _openai_client = AsyncOpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")


def get_openai_client() -> AsyncOpenAI:
    """
    Get the OpenAI client instance.

    Returns:
        OpenAI client

    Raises:
        RuntimeError: If client not initialized
    """
    if _openai_client is None:
        raise RuntimeError(
            "OpenAI client not initialized. Call init_openai_client() first."
        )
    return _openai_client


async def generate_embedding(
    text: str,
    model: str = "text-embedding-3-small",
) -> List[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed
        model: OpenAI embedding model

    Returns:
        Embedding vector as list of floats

    Raises:
        RuntimeError: If client not initialized
        Exception: If embedding generation fails
    """
    client = get_openai_client()

    try:
        response = await client.embeddings.create(
            model=model,
            input=text,
        )

        embedding = response.data[0].embedding

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


async def generate_embeddings_batch(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model
        batch_size: Maximum texts per API request

    Returns:
        List of embedding vectors

    Raises:
        RuntimeError: If client not initialized
        Exception: If embedding generation fails
    """
    client = get_openai_client()

    if not texts:
        return []

    embeddings = []

    try:
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            response = await client.embeddings.create(
                model=model,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

            logger.debug(
                "Generated batch embeddings",
                batch_size=len(batch),
                total_processed=len(embeddings),
                total_texts=len(texts),
                model=model,
            )

        logger.info(
            "Generated embeddings for batch",
            total_texts=len(texts),
            total_embeddings=len(embeddings),
            model=model,
        )

        return embeddings

    except Exception as e:
        logger.error(
            "Failed to generate batch embeddings",
            error=str(e),
            texts_count=len(texts),
            model=model,
        )
        raise


async def get_embedding_dimensions(model: str = "text-embedding-3-small") -> int:
    """
    Get the dimensions of embeddings for a model.

    Uses a test embedding to determine dimensions.

    Args:
        model: OpenAI embedding model

    Returns:
        Number of dimensions

    Raises:
        RuntimeError: If client not initialized
        Exception: If test embedding fails
    """
    try:
        test_embedding = await generate_embedding("test", model=model)
        return len(test_embedding)
    except Exception as e:
        logger.error(
            "Failed to get embedding dimensions",
            error=str(e),
            model=model,
        )
        raise


def truncate_text(text: str, max_tokens: int = 8000) -> str:
    """
    Truncate text to fit within token limit.

    Simple character-based truncation (approximate).
    For more accurate tokenization, use tiktoken.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens (approximate)

    Returns:
        Truncated text
    """
    # Rough estimate: 1 token ≈ 4 characters
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    logger.warning(
        "Text truncated for embedding",
        original_length=len(text),
        truncated_length=len(truncated),
        max_tokens=max_tokens,
    )

    return truncated
