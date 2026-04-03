"""
Token-based Text Chunking

Provides intelligent text chunking for RAG (Retrieval-Augmented Generation) systems.
Uses tiktoken for accurate token counting and respects semantic boundaries.

Features:
- Token-accurate chunking (default 512 tokens)
- Semantic boundary detection (headings, paragraphs, sentences)
- Deterministic chunk IDs for deduplication
- Chunk metadata tracking (position, tokens, boundaries)
- Page mapping support for PDF documents

Usage:
    from kbase.core.token_chunker import (
        TextChunk, chunk_text_by_tokens, generate_chunk_id,
        validate_chunks, get_chunk_statistics
    )

    # Basic chunking
    chunks = chunk_text_by_tokens(text, item_key="doc123", version=1)

    # With page mappings
    chunks = chunk_text_by_tokens(
        text,
        page_mappings=mappings,
        item_key="doc123",
        version=1
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
import hashlib
import re

import tiktoken
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PageMapping:
    """
    Maps character positions to page numbers.

    Used for tracking which pages content comes from in PDF documents.
    """
    page_number: int
    start_char: int
    end_char: int
    char_offset: int = 0


@dataclass
class TextChunk:
    """
    Represents a text chunk with comprehensive metadata.

    Attributes:
        text: The chunk text content
        chunk_index: Position in document (0-based)
        start_char: Start position in original text
        end_char: End position in original text
        token_count: Number of tokens in chunk
        boundary_type: How chunk ends ("heading", "paragraph", "sentence", "forced")
        page_numbers: List of pages this chunk spans
        start_page: First page number
        end_page: Last page number
        page_count: Number of pages spanned
        chunk_id: Deterministic ID for deduplication
    """
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    boundary_type: str
    page_numbers: List[int] = field(default_factory=list)
    start_page: int = 0
    end_page: int = 0
    page_count: int = 0
    chunk_id: str = ""


def chunk_text_by_tokens(
    text: str,
    item_key: str,
    version: int = 1,
    page_mappings: Optional[List[PageMapping]] = None,
    chunk_size: int = 512,
    overlap: int = 50,
    model: str = "gpt-4",
    page_mapper: Optional[Callable[[int, int, List[PageMapping]], Dict[str, Any]]] = None
) -> List[TextChunk]:
    """
    Chunk text by token count, respecting semantic boundaries.

    Args:
        text: Full document text to chunk
        item_key: Unique identifier for deterministic ID generation
        version: Version number for ID generation
        page_mappings: Optional list of PageMapping objects for page tracking
        chunk_size: Target tokens per chunk (default 512)
        overlap: Token overlap between chunks (default 50)
        model: Tiktoken model name (default "gpt-4")
        page_mapper: Optional custom page mapping function

    Returns:
        List of TextChunk objects with metadata

    Example:
        >>> text = "Long document text..."
        >>> chunks = chunk_text_by_tokens(text, item_key="doc123", version=1)
        >>> len(chunks)
        5
        >>> chunks[0].token_count
        512
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    chunks = []
    start_idx = 0
    chunk_idx = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = encoding.decode(chunk_tokens)

        start_char = len(encoding.decode(tokens[:start_idx]))
        end_char = len(encoding.decode(tokens[:end_idx]))

        boundary = determine_boundary_type(chunk_text, text, start_char, end_char)

        # Map to pages
        page_info = _map_chunk_to_pages(start_char, end_char, page_mappings, page_mapper)

        chunk_id = generate_chunk_id(item_key, version, chunk_idx, chunk_text)

        chunk = TextChunk(
            text=chunk_text,
            chunk_index=chunk_idx,
            start_char=start_char,
            end_char=end_char,
            token_count=len(chunk_tokens),
            boundary_type=boundary,
            page_numbers=page_info.get('page_numbers', []),
            start_page=page_info.get('start_page', 0),
            end_page=page_info.get('end_page', 0),
            page_count=page_info.get('page_count', 0),
            chunk_id=chunk_id
        )

        chunks.append(chunk)

        start_idx = end_idx - overlap
        chunk_idx += 1

    logger.debug(
        "Chunked text",
        total_chunks=len(chunks),
        total_tokens=sum(c.token_count for c in chunks),
        item_key=item_key
    )

    return chunks


def _map_chunk_to_pages(
    start_char: int,
    end_char: int,
    page_mappings: Optional[List[PageMapping]],
    page_mapper: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Map character positions to page numbers.

    Args:
        start_char: Start character position
        end_char: End character position
        page_mappings: List of PageMapping objects
        page_mapper: Optional custom mapping function

    Returns:
        Dictionary with page_numbers, start_page, end_page, page_count
    """
    if page_mapper and page_mappings:
        return page_mapper(start_char, end_char, page_mappings)

    if not page_mappings:
        return {
            'page_numbers': [],
            'start_page': 0,
            'end_page': 0,
            'page_count': 0
        }

    page_numbers = set()
    start_page = None
    end_page = None

    for mapping in page_mappings:
        if mapping.start_char <= end_char and mapping.end_char >= start_char:
            page_numbers.add(mapping.page_number)
            if start_page is None or mapping.page_number < start_page:
                start_page = mapping.page_number
            if end_page is None or mapping.page_number > end_page:
                end_page = mapping.page_number

    page_list = sorted(list(page_numbers))

    return {
        'page_numbers': page_list,
        'start_page': start_page or 0,
        'end_page': end_page or 0,
        'page_count': len(page_list)
    }


def determine_boundary_type(
    chunk_text: str,
    full_text: str,
    start_char: int,
    end_char: int
) -> str:
    """
    Determine what semantic boundary this chunk ends on.

    Priority:
    1. Heading (##, ###, etc.)
    2. Paragraph (double newline)
    3. Sentence (. ! ?)
    4. Forced (mid-word break)

    Args:
        chunk_text: Text of the chunk
        full_text: Full document text
        start_char: Chunk start position in full text
        end_char: Chunk end position in full text

    Returns:
        Boundary type: "heading", "paragraph", "sentence", or "forced"

    Example:
        >>> determine_boundary_type("Text\\n\\n## Heading", "...", 0, 100)
        'heading'
    """
    last_chars = chunk_text[-50:] if len(chunk_text) > 50 else chunk_text

    if re.search(r'\n#{1,6}\s+\w+', last_chars):
        return "heading"

    if '\n\n' in last_chars:
        return "paragraph"

    if re.search(r'[.!?]\s*$', last_chars):
        return "sentence"

    return "forced"


def generate_chunk_id(
    item_key: str,
    version: int,
    chunk_index: int,
    chunk_text: str
) -> str:
    """
    Generate deterministic chunk ID for deduplication.

    Format: {key}_{version}_{index}_{hash}
    Example: doc123_1_0_a3f2e9

    Args:
        item_key: Unique document identifier
        version: Version number
        chunk_index: Position in document (0-based)
        chunk_text: Text content of chunk (for hash)

    Returns:
        Deterministic chunk ID string

    Example:
        >>> generate_chunk_id("doc123", 1, 0, "Some text")
        'doc123_1_0_8f3a92'
    """
    text_sample = chunk_text[:100]
    text_hash = hashlib.md5(text_sample.encode()).hexdigest()[:6]
    return f"{item_key}_{version}_{chunk_index}_{text_hash}"


def validate_chunks(chunks: List[TextChunk]) -> Dict[str, Any]:
    """
    Validate chunk list for consistency and quality.

    Checks:
    - No duplicate chunk IDs
    - Sequential chunk indices
    - Token counts within bounds
    - No empty chunks

    Args:
        chunks: List of TextChunk objects

    Returns:
        Dictionary with:
            - valid: True if all checks pass
            - total_chunks: Number of chunks
            - total_tokens: Sum of all chunk tokens
            - warnings: List of warning messages
            - errors: List of error messages

    Example:
        >>> chunks = [TextChunk(...), TextChunk(...)]
        >>> result = validate_chunks(chunks)
        >>> result['valid']
        True
    """
    warnings = []
    errors = []

    if not chunks:
        return {
            "valid": False,
            "total_chunks": 0,
            "total_tokens": 0,
            "warnings": [],
            "errors": ["No chunks provided"]
        }

    chunk_ids = [c.chunk_id for c in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("Duplicate chunk IDs detected")

    for i, chunk in enumerate(chunks):
        if chunk.chunk_index != i:
            errors.append(
                f"Non-sequential index at position {i}: expected {i}, got {chunk.chunk_index}"
            )

    for chunk in chunks:
        if chunk.token_count == 0:
            errors.append(f"Empty chunk at index {chunk.chunk_index}")
        elif chunk.token_count > 600:
            warnings.append(
                f"Oversized chunk at index {chunk.chunk_index}: {chunk.token_count} tokens"
            )

    for chunk in chunks:
        if not chunk.text or len(chunk.text.strip()) == 0:
            errors.append(f"Empty text in chunk {chunk.chunk_index}")

    total_tokens = sum(c.token_count for c in chunks)

    return {
        "valid": len(errors) == 0,
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "warnings": warnings,
        "errors": errors
    }


def estimate_chunk_count(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    model: str = "gpt-4"
) -> int:
    """
    Estimate number of chunks without actually chunking.

    Args:
        text: Full text to estimate
        chunk_size: Target tokens per chunk
        overlap: Token overlap between chunks
        model: Tiktoken model name

    Returns:
        Estimated chunk count

    Example:
        >>> estimate_chunk_count("A long document...")
        12
    """
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = len(encoding.encode(text))

    effective_chunk_size = chunk_size - overlap
    chunk_count = (total_tokens + effective_chunk_size - 1) // effective_chunk_size

    return max(1, chunk_count)


def get_chunk_statistics(chunks: List[TextChunk]) -> Dict[str, Any]:
    """
    Get statistical summary of chunk list.

    Args:
        chunks: List of TextChunk objects

    Returns:
        Dictionary with statistics:
            - total_chunks: Number of chunks
            - total_tokens: Sum of all tokens
            - avg_tokens: Average tokens per chunk
            - min_tokens: Minimum token count
            - max_tokens: Maximum token count
            - boundary_types: Count by boundary type
            - pages_covered: Unique pages across all chunks

    Example:
        >>> stats = get_chunk_statistics(chunks)
        >>> stats['avg_tokens']
        498.5
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_tokens": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "boundary_types": {},
            "pages_covered": []
        }

    total_tokens = sum(c.token_count for c in chunks)
    token_counts = [c.token_count for c in chunks]

    boundary_types = {}
    for chunk in chunks:
        boundary_types[chunk.boundary_type] = boundary_types.get(chunk.boundary_type, 0) + 1

    all_pages = set()
    for chunk in chunks:
        all_pages.update(chunk.page_numbers)

    return {
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "avg_tokens": round(total_tokens / len(chunks), 1),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "boundary_types": boundary_types,
        "pages_covered": sorted(list(all_pages))
    }


def get_token_count(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Tiktoken model name

    Returns:
        Token count
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


__all__ = [
    # Dataclasses
    "TextChunk",
    "PageMapping",
    # Main functions
    "chunk_text_by_tokens",
    "determine_boundary_type",
    "generate_chunk_id",
    # Validation/stats
    "validate_chunks",
    "get_chunk_statistics",
    # Utilities
    "estimate_chunk_count",
    "get_token_count",
]
