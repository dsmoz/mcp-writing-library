"""
Token-Based Text Chunker

This module provides token-based text chunking with semantic boundary detection.
Based on the chunking strategy from mcp-zotero-qdrant.

Features:
- Token-based sizing (default 512 tokens with 50 token overlap)
- Semantic boundary detection (heading > paragraph > sentence > forced)
- Deterministic chunk IDs for deduplication
- Page mapping support

Usage:
    from kbase.vector.chunker import chunk_text, TextChunk
    chunks = chunk_text(text, document_id="doc123")
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Try to import tiktoken, fallback to character-based if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using character-based chunking")


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""

    text: str                           # Chunk content
    chunk_index: int                    # Position in document (0-indexed)
    start_char: int                     # Start character position
    end_char: int                       # End character position
    token_count: int                    # Approximate token count
    boundary_type: str                  # "heading", "paragraph", "sentence", "forced"
    chunk_id: str                       # Deterministic ID for deduplication

    # Optional page mapping
    page_numbers: List[int] = field(default_factory=list)
    start_page: Optional[int] = None
    end_page: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "boundary_type": self.boundary_type,
            "chunk_id": self.chunk_id,
            "page_numbers": self.page_numbers,
            "start_page": self.start_page,
            "end_page": self.end_page,
        }


def _get_encoding(model: str = "gpt-4"):
    """Get tiktoken encoding for the specified model."""
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoding) -> int:
    """Count tokens in text."""
    if encoding is None:
        # Fallback: approximate 1 token = 4 characters
        return len(text) // 4
    return len(encoding.encode(text))


def _generate_chunk_id(
    document_id: str,
    version: int,
    chunk_index: int,
    text: str,
) -> str:
    """
    Generate a deterministic chunk ID for deduplication.

    Format: {document_id}_{version}_{chunk_index}_{content_hash}
    """
    # Create hash of first 100 chars for uniqueness
    content_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
    return f"{document_id}_{version}_{chunk_index}_{content_hash}"


def _determine_boundary_type(text: str) -> str:
    """
    Determine the semantic boundary type at the end of text.

    Priority: heading > paragraph > sentence > forced
    """
    # Check last 100 characters for boundary markers
    tail = text[-100:] if len(text) > 100 else text

    # Heading: ends with markdown heading or similar
    if re.search(r'\n#{1,6}\s+\w+', tail):
        return "heading"

    # Paragraph: ends with double newline
    if re.search(r'\n\n\s*$', tail):
        return "paragraph"

    # Sentence: ends with sentence-ending punctuation
    if re.search(r'[.!?]\s*$', tail):
        return "sentence"

    # Forced: no natural boundary found
    return "forced"


def _find_best_break_point(
    text: str,
    target_pos: int,
    search_range: int = 100,
) -> int:
    """
    Find the best position to break text near target_pos.

    Prefers breaking at semantic boundaries:
    1. Paragraph breaks (double newline)
    2. Sentence endings (. ! ?)
    3. Line breaks
    4. Word boundaries (space)
    """
    if target_pos >= len(text):
        return len(text)

    # Search window around target position
    start = max(0, target_pos - search_range)
    end = min(len(text), target_pos + search_range)
    window = text[start:end]

    # Relative target position in window
    rel_target = target_pos - start

    # Priority 1: Paragraph break (double newline)
    para_matches = list(re.finditer(r'\n\n', window))
    if para_matches:
        # Find closest to target
        best = min(para_matches, key=lambda m: abs(m.end() - rel_target))
        return start + best.end()

    # Priority 2: Sentence ending
    sent_matches = list(re.finditer(r'[.!?]\s+', window))
    if sent_matches:
        best = min(sent_matches, key=lambda m: abs(m.end() - rel_target))
        return start + best.end()

    # Priority 3: Line break
    line_matches = list(re.finditer(r'\n', window))
    if line_matches:
        best = min(line_matches, key=lambda m: abs(m.end() - rel_target))
        return start + best.end()

    # Priority 4: Word boundary
    word_matches = list(re.finditer(r'\s+', window))
    if word_matches:
        best = min(word_matches, key=lambda m: abs(m.end() - rel_target))
        return start + best.end()

    # Fallback: use target position
    return target_pos


def chunk_text(
    text: str,
    document_id: str,
    version: int = 1,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    model: str = "gpt-4",
    page_mappings: Optional[List[Tuple[int, int, int]]] = None,
) -> List[TextChunk]:
    """
    Chunk text into smaller pieces based on token count.

    Args:
        text: Full text to chunk
        document_id: Document identifier for chunk IDs
        version: Document version (default: 1)
        chunk_size: Target tokens per chunk (default: 512)
        chunk_overlap: Overlap tokens between chunks (default: 50)
        model: Model for tokenization (default: "gpt-4")
        page_mappings: Optional list of (page_num, start_char, end_char) tuples

    Returns:
        List of TextChunk objects

    Example:
        >>> chunks = chunk_text("Long document text...", document_id="doc123")
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.chunk_index}: {len(chunk.text)} chars")
    """
    if not text or not text.strip():
        return []

    encoding = _get_encoding(model)

    # Calculate character-per-token ratio for this text
    total_tokens = _count_tokens(text, encoding)
    chars_per_token = len(text) / max(total_tokens, 1)

    # Convert token targets to character targets
    target_chunk_chars = int(chunk_size * chars_per_token)
    target_overlap_chars = int(chunk_overlap * chars_per_token)

    chunks = []
    start_pos = 0
    chunk_index = 0

    while start_pos < len(text):
        # Calculate end position
        end_pos = start_pos + target_chunk_chars

        if end_pos >= len(text):
            # Last chunk - take everything remaining
            end_pos = len(text)
        else:
            # Find best break point
            end_pos = _find_best_break_point(text, end_pos)

        # Extract chunk text
        chunk_text_content = text[start_pos:end_pos].strip()

        if not chunk_text_content:
            start_pos = end_pos
            continue

        # Calculate token count for this chunk
        chunk_tokens = _count_tokens(chunk_text_content, encoding)

        # Determine boundary type
        boundary_type = _determine_boundary_type(chunk_text_content)

        # Generate deterministic chunk ID
        chunk_id = _generate_chunk_id(document_id, version, chunk_index, chunk_text_content)

        # Map to pages if page mappings provided
        page_nums = []
        start_page = None
        end_page = None

        if page_mappings:
            for page_num, page_start, page_end in page_mappings:
                # Check if chunk overlaps with this page
                if start_pos < page_end and end_pos > page_start:
                    page_nums.append(page_num)

            if page_nums:
                start_page = min(page_nums)
                end_page = max(page_nums)

        # Create chunk object
        chunk = TextChunk(
            text=chunk_text_content,
            chunk_index=chunk_index,
            start_char=start_pos,
            end_char=end_pos,
            token_count=chunk_tokens,
            boundary_type=boundary_type,
            chunk_id=chunk_id,
            page_numbers=page_nums,
            start_page=start_page,
            end_page=end_page,
        )

        chunks.append(chunk)
        chunk_index += 1

        # Move to next chunk with overlap
        # Ensure we make significant progress to avoid infinite loops
        min_progress = max(target_chunk_chars // 2, 100)  # At least half chunk or 100 chars
        next_start = end_pos - target_overlap_chars
        start_pos = max(next_start, start_pos + min_progress)

    logger.info(
        "Text chunked",
        document_id=document_id,
        total_chars=len(text),
        total_tokens=total_tokens,
        num_chunks=len(chunks),
        avg_tokens_per_chunk=total_tokens // max(len(chunks), 1),
    )

    return chunks


def chunk_text_simple(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Simple character-based chunking for quick use.

    Args:
        text: Text to chunk
        chunk_size: Characters per chunk
        chunk_overlap: Overlap characters

    Returns:
        List of text strings
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Find break point
        end = _find_best_break_point(text, end, search_range=50)
        chunks.append(text[start:end].strip())

        start = end - chunk_overlap
        start = max(start, start + 1)  # Ensure progress

    return [c for c in chunks if c]  # Filter empty chunks
