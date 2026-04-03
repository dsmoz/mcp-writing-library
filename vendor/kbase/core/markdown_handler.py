"""
Markdown Handling Utilities for Contextual Chunking

This module provides utilities for:
- Checking if markdown files exist
- Reading existing markdown content
- Extracting page number mappings from <!-- Page N --> markers
- Mapping text chunks to specific pages
- Validating markdown structure

Usage:
    from kbase.core import (
        PageMapping,
        extract_page_mappings,
        map_chunk_to_pages,
        validate_markdown_structure,
    )

    # Extract page mappings from markdown
    mappings = extract_page_mappings(markdown_content)

    # Map a chunk to its pages
    page_info = map_chunk_to_pages(chunk_start, chunk_end, mappings)
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class PageMapping:
    """
    Maps page number to character positions in markdown.

    Attributes:
        page_number: The page number (1-indexed)
        start_char: Character position where this page starts
        end_char: Character position where this page ends (-1 if last page)
        marker_position: Position of the <!-- Page N --> marker
    """

    page_number: int
    start_char: int
    end_char: int
    marker_position: int


@dataclass
class ChunkPageInfo:
    """
    Information about which pages a chunk spans.

    Attributes:
        page_numbers: List of page numbers the chunk spans
        start_page: First page number
        end_page: Last page number
        page_count: Number of pages spanned
        total_pages: Total pages in the document
    """

    page_numbers: List[int]
    start_page: Optional[int]
    end_page: Optional[int]
    page_count: int
    total_pages: int


@dataclass
class MarkdownValidation:
    """
    Result of markdown structure validation.

    Attributes:
        valid: True if structure is valid for processing
        has_page_markers: True if page markers were found
        page_count: Number of pages detected
        character_count: Total character count
        warnings: List of warning messages
    """

    valid: bool
    has_page_markers: bool
    page_count: int
    character_count: int
    warnings: List[str]


def check_markdown_exists(
    item_key: str,
    base_path: str = ".data",
    subfolder: str = "",
) -> bool:
    """
    Check if markdown file exists for an item.

    Args:
        item_key: Item key (e.g., 'ABC12345')
        base_path: Base directory for markdown files
        subfolder: Optional subfolder within base_path

    Returns:
        True if markdown file exists, False otherwise

    Example:
        >>> check_markdown_exists("ABC12345", ".data", "zotero")
        True
    """
    if subfolder:
        markdown_path = Path(base_path) / subfolder / f"{item_key}.md"
    else:
        markdown_path = Path(base_path) / f"{item_key}.md"
    return markdown_path.exists()


def read_markdown_file(
    item_key: str,
    base_path: str = ".data",
    subfolder: str = "",
) -> Optional[str]:
    """
    Read existing markdown file.

    Args:
        item_key: Item key (e.g., 'ABC12345')
        base_path: Base directory for markdown files
        subfolder: Optional subfolder within base_path

    Returns:
        Markdown content as string, or None if file doesn't exist

    Example:
        >>> content = read_markdown_file("ABC12345", ".data", "zotero")
        >>> len(content)
        5000
    """
    if subfolder:
        markdown_path = Path(base_path) / subfolder / f"{item_key}.md"
    else:
        markdown_path = Path(base_path) / f"{item_key}.md"

    if markdown_path.exists():
        return markdown_path.read_text(encoding="utf-8")
    return None


def extract_page_mappings(markdown: str) -> List[PageMapping]:
    """
    Extract page number mappings from markdown.

    Parses <!-- Page N --> markers and creates character position mappings.

    Args:
        markdown: Markdown content with page markers

    Returns:
        List of PageMapping objects, sorted by page number

    Example:
        >>> markdown = "Text\\n<!-- Page 1 -->\\nMore text\\n<!-- Page 2 -->\\nEnd"
        >>> mappings = extract_page_mappings(markdown)
        >>> len(mappings)
        2
        >>> mappings[0].page_number
        1
    """
    page_pattern = r"<!-- Page (\d+) -->"
    mappings: List[PageMapping] = []

    for match in re.finditer(page_pattern, markdown):
        page_num = int(match.group(1))
        marker_pos = match.start()

        mappings.append(
            PageMapping(
                page_number=page_num,
                start_char=marker_pos,
                end_char=-1,  # Will be set to next page's start
                marker_position=marker_pos,
            )
        )

    # Set end positions
    for i in range(len(mappings)):
        if i < len(mappings) - 1:
            mappings[i].end_char = mappings[i + 1].start_char
        else:
            mappings[i].end_char = len(markdown)

    return mappings


def map_chunk_to_pages(
    chunk_start: int,
    chunk_end: int,
    page_mappings: List[PageMapping],
) -> ChunkPageInfo:
    """
    Determine which page(s) a text chunk spans.

    Args:
        chunk_start: Character position where chunk starts
        chunk_end: Character position where chunk ends
        page_mappings: List of PageMapping objects from extract_page_mappings()

    Returns:
        ChunkPageInfo with page span information

    Example:
        >>> mappings = [PageMapping(1, 0, 100, 0), PageMapping(2, 100, 200, 100)]
        >>> info = map_chunk_to_pages(50, 150, mappings)
        >>> info.page_numbers
        [1, 2]
        >>> info.page_count
        2
    """
    pages_spanned: List[int] = []

    for mapping in page_mappings:
        # Check if chunk overlaps with this page
        if not (chunk_end < mapping.start_char or chunk_start > mapping.end_char):
            pages_spanned.append(mapping.page_number)

    if not pages_spanned:
        return ChunkPageInfo(
            page_numbers=[],
            start_page=None,
            end_page=None,
            page_count=0,
            total_pages=len(page_mappings),
        )

    return ChunkPageInfo(
        page_numbers=pages_spanned,
        start_page=pages_spanned[0],
        end_page=pages_spanned[-1],
        page_count=len(pages_spanned),
        total_pages=len(page_mappings),
    )


def map_chunk_to_pages_dict(
    chunk_start: int,
    chunk_end: int,
    page_mappings: List[PageMapping],
) -> Dict:
    """
    Determine which page(s) a text chunk spans (dict version).

    This is a compatibility function that returns a dict instead of dataclass.

    Args:
        chunk_start: Character position where chunk starts
        chunk_end: Character position where chunk ends
        page_mappings: List of PageMapping objects

    Returns:
        Dictionary with page span information

    Example:
        >>> mappings = [PageMapping(1, 0, 100, 0), PageMapping(2, 100, 200, 100)]
        >>> map_chunk_to_pages_dict(50, 150, mappings)
        {'page_numbers': [1, 2], 'start_page': 1, 'end_page': 2, 'page_count': 2, 'total_pages': 2}
    """
    info = map_chunk_to_pages(chunk_start, chunk_end, page_mappings)
    return {
        "page_numbers": info.page_numbers,
        "start_page": info.start_page,
        "end_page": info.end_page,
        "page_count": info.page_count,
        "total_pages": info.total_pages,
    }


def get_all_markdown_files(
    base_path: str = ".data",
    subfolder: str = "",
) -> List[str]:
    """
    Get list of all item keys that have markdown files.

    Scans the specified folder for *.md files and extracts item keys.

    Args:
        base_path: Base directory for markdown files
        subfolder: Optional subfolder within base_path

    Returns:
        List of item keys (e.g., ['ABC12345', 'DEF67890'])

    Example:
        >>> keys = get_all_markdown_files(".data", "zotero")
        >>> len(keys)
        150
    """
    if subfolder:
        folder = Path(base_path) / subfolder
    else:
        folder = Path(base_path)

    if not folder.exists():
        return []

    # Get all .md files, sorted alphabetically
    markdown_files = sorted(folder.glob("*.md"))

    # Extract item keys from filenames (e.g., ABC12345.md -> ABC12345)
    return [md_file.stem for md_file in markdown_files]


def validate_markdown_structure(markdown: str) -> MarkdownValidation:
    """
    Validate markdown file structure for contextual chunking.

    Checks for:
    - Page markers present
    - Non-empty content
    - Minimum content length

    Args:
        markdown: Markdown content to validate

    Returns:
        MarkdownValidation with validation results

    Example:
        >>> markdown = "Text\\n<!-- Page 1 -->\\nMore text"
        >>> result = validate_markdown_structure(markdown)
        >>> result.valid
        True
        >>> result.page_count
        1
    """
    warnings: List[str] = []

    # Check for empty content
    if not markdown or len(markdown.strip()) == 0:
        return MarkdownValidation(
            valid=False,
            has_page_markers=False,
            page_count=0,
            character_count=0,
            warnings=["Markdown is empty"],
        )

    # Extract page mappings
    page_mappings = extract_page_mappings(markdown)
    has_page_markers = len(page_mappings) > 0

    if not has_page_markers:
        warnings.append("No page markers found - page tracking unavailable")

    # Check minimum content length
    if len(markdown) < 100:
        warnings.append(
            f"Very short content ({len(markdown)} chars) - may not chunk well"
        )

    return MarkdownValidation(
        valid=True,
        has_page_markers=has_page_markers,
        page_count=len(page_mappings),
        character_count=len(markdown),
        warnings=warnings,
    )


def validate_markdown_structure_dict(markdown: str) -> Dict:
    """
    Validate markdown file structure (dict version).

    This is a compatibility function that returns a dict instead of dataclass.

    Args:
        markdown: Markdown content to validate

    Returns:
        Dictionary with validation results
    """
    result = validate_markdown_structure(markdown)
    return {
        "valid": result.valid,
        "has_page_markers": result.has_page_markers,
        "page_count": result.page_count,
        "character_count": result.character_count,
        "warnings": result.warnings,
    }


def create_page_marker(page_number: int) -> str:
    """
    Create a page marker comment.

    Args:
        page_number: Page number to create marker for

    Returns:
        Page marker string

    Example:
        >>> create_page_marker(5)
        '<!-- Page 5 -->'
    """
    return f"<!-- Page {page_number} -->"


def insert_page_markers(
    content: str,
    page_breaks: List[int],
) -> str:
    """
    Insert page markers at specified character positions.

    Args:
        content: Original content
        page_breaks: List of character positions where pages start

    Returns:
        Content with page markers inserted

    Example:
        >>> content = "First page content. Second page content."
        >>> insert_page_markers(content, [0, 20])
        '<!-- Page 1 -->First page content. <!-- Page 2 -->Second page content.'
    """
    if not page_breaks:
        return content

    # Sort breaks in descending order to insert from end to start
    # (so positions don't shift)
    sorted_breaks = sorted(enumerate(page_breaks, 1), key=lambda x: x[1], reverse=True)

    result = content
    for page_num, position in sorted_breaks:
        marker = create_page_marker(page_num)
        result = result[:position] + marker + result[position:]

    return result


__all__ = [
    "PageMapping",
    "ChunkPageInfo",
    "MarkdownValidation",
    "check_markdown_exists",
    "read_markdown_file",
    "extract_page_mappings",
    "map_chunk_to_pages",
    "map_chunk_to_pages_dict",
    "get_all_markdown_files",
    "validate_markdown_structure",
    "validate_markdown_structure_dict",
    "create_page_marker",
    "insert_page_markers",
]
