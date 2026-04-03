"""
Unified document processing for kbase-core.

Provides text extraction, chunking, and processing for various
document formats (PDF, DOCX, TXT, Markdown, etc.).
"""

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from kbase.core.logger import get_logger

logger = get_logger(__name__)


class DocumentSource(Enum):
    """Source type for documents."""

    LOCAL_FILE = "local_file"
    URL = "url"
    S3 = "s3"
    BLOB = "blob"
    TEXT = "text"


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""

    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    boundary_type: str  # "heading", "paragraph", "sentence", "forced"
    page_numbers: List[int] = field(default_factory=list)
    start_page: int = 0
    end_page: int = 0
    page_count: int = 0
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate deterministic chunk ID."""
        content_hash = hashlib.md5(self.text.encode("utf-8")).hexdigest()[:8]
        return f"chunk_{self.chunk_index}_{content_hash}"


@dataclass
class ProcessedDocument:
    """Result of document processing."""

    full_text: str
    chunks: List[TextChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0
    source_path: Optional[str] = None
    source_type: DocumentSource = DocumentSource.TEXT
    processing_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.full_text.split())
        if not self.char_count:
            self.char_count = len(self.full_text)


def generate_chunk_id(
    document_key: str, version: int, chunk_index: int, chunk_text: str
) -> str:
    """
    Generate deterministic chunk ID for deduplication.

    Args:
        document_key: Unique document identifier
        version: Document version number
        chunk_index: Index of chunk within document
        chunk_text: Text content of chunk

    Returns:
        Deterministic chunk ID string
    """
    content_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()[:8]
    return f"{document_key}_{version}_{chunk_index}_{content_hash}"


def determine_boundary_type(
    chunk_text: str,
    full_text: str,
    start_char: int,
    end_char: int,
) -> str:
    """
    Determine semantic boundary type for a chunk.

    Priority:
    1. Heading (##, ###, etc.)
    2. Paragraph (double newline)
    3. Sentence (. ! ?)
    4. Forced (mid-word break)

    Args:
        chunk_text: Text of the chunk
        full_text: Full document text
        start_char: Chunk start position
        end_char: Chunk end position

    Returns:
        Boundary type string
    """
    last_chars = chunk_text[-50:] if len(chunk_text) > 50 else chunk_text

    # Check for heading
    if re.search(r"\n#{1,6}\s+\w+", last_chars):
        return "heading"

    # Check for paragraph break
    if re.search(r"\n\n", last_chars):
        return "paragraph"

    # Check for sentence end
    if re.search(r"[.!?]\s*$", last_chars.strip()):
        return "sentence"

    return "forced"


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    document_key: str = "doc",
    version: int = 1,
) -> List[TextChunk]:
    """
    Chunk text by token count, respecting semantic boundaries.

    Args:
        text: Full document text to chunk
        chunk_size: Target tokens per chunk (default 512)
        overlap: Token overlap between chunks (default 50)
        document_key: Document identifier for chunk IDs
        version: Document version for chunk IDs

    Returns:
        List of TextChunk objects with metadata
    """
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        # Fallback to simple character-based chunking
        logger.warning("tiktoken not available, using character-based chunking")
        return _chunk_text_by_chars(text, chunk_size * 4, overlap * 4, document_key, version)

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
        chunk_id = generate_chunk_id(document_key, version, chunk_idx, chunk_text)

        chunk = TextChunk(
            text=chunk_text,
            chunk_index=chunk_idx,
            start_char=start_char,
            end_char=end_char,
            token_count=len(chunk_tokens),
            boundary_type=boundary,
            chunk_id=chunk_id,
        )

        chunks.append(chunk)

        # Safety check: if we've reached the end, stop
        if end_idx >= len(tokens):
            break

        # Move to next chunk position, ensuring forward progress
        next_start = end_idx - overlap
        if next_start <= start_idx:
            # Overlap is larger than chunk, just move to end
            next_start = end_idx
        start_idx = next_start
        chunk_idx += 1

    return chunks


def _chunk_text_by_chars(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200,
    document_key: str = "doc",
    version: int = 1,
) -> List[TextChunk]:
    """Fallback character-based chunking."""
    chunks = []
    start_idx = 0
    chunk_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))

        # Try to end at sentence boundary
        if end_idx < len(text):
            for sep in [". ", "! ", "? ", "\n\n", "\n"]:
                last_sep = text[start_idx:end_idx].rfind(sep)
                if last_sep > chunk_size // 2:
                    end_idx = start_idx + last_sep + len(sep)
                    break

        chunk_text = text[start_idx:end_idx]
        boundary = determine_boundary_type(chunk_text, text, start_idx, end_idx)
        chunk_id = generate_chunk_id(document_key, version, chunk_idx, chunk_text)

        chunk = TextChunk(
            text=chunk_text,
            chunk_index=chunk_idx,
            start_char=start_idx,
            end_char=end_idx,
            token_count=len(chunk_text) // 4,  # Approximate
            boundary_type=boundary,
            chunk_id=chunk_id,
        )

        chunks.append(chunk)

        # Safety check: if we've reached the end, stop
        if end_idx >= len(text):
            break

        # Move to next chunk position, ensuring forward progress
        next_start = end_idx - overlap
        if next_start <= start_idx:
            # Overlap is larger than chunk, just move to end
            next_start = end_idx
        start_idx = next_start
        chunk_idx += 1

    return chunks


class UnifiedDocumentProcessor:
    """
    Unified processor for various document formats.

    Supports PDF, DOCX, TXT, Markdown, and other text formats.
    Uses docling for advanced document parsing when available.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_docling: bool = True,
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target tokens per chunk
            chunk_overlap: Token overlap between chunks
            use_docling: Use docling for PDF/DOCX parsing
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_docling = use_docling
        self._docling_available = None

    def _check_docling(self) -> bool:
        """Check if docling is available."""
        if self._docling_available is None:
            try:
                import docling

                self._docling_available = True
            except ImportError:
                self._docling_available = False
                logger.info("docling not available, using fallback processors")
        return self._docling_available

    def process_document(
        self,
        source: DocumentSource,
        source_path: str,
        apply_contextual_retrieval: bool = False,
        document_key: Optional[str] = None,
        version: int = 1,
    ) -> ProcessedDocument:
        """
        Process a document and extract text with chunking.

        Args:
            source: Source type (LOCAL_FILE, URL, TEXT, etc.)
            source_path: Path to file or URL or raw text
            apply_contextual_retrieval: Apply contextual retrieval optimization
            document_key: Optional document identifier
            version: Document version for chunk IDs

        Returns:
            ProcessedDocument with full text and chunks
        """
        if document_key is None:
            document_key = hashlib.md5(source_path.encode()).hexdigest()[:12]

        try:
            # Extract text based on source type
            if source == DocumentSource.TEXT:
                full_text = source_path
                metadata = {"source": "text"}
                page_count = 1
            elif source == DocumentSource.LOCAL_FILE:
                full_text, metadata, page_count = self._extract_from_file(source_path)
            elif source == DocumentSource.URL:
                full_text, metadata, page_count = self._extract_from_url(source_path)
            else:
                full_text = source_path
                metadata = {"source": str(source)}
                page_count = 1

            # Chunk the text
            chunks = chunk_text_by_tokens(
                full_text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
                document_key=document_key,
                version=version,
            )

            return ProcessedDocument(
                full_text=full_text,
                chunks=chunks,
                metadata=metadata,
                page_count=page_count,
                source_path=source_path,
                source_type=source,
            )

        except Exception as e:
            logger.error(
                "Error processing document",
                source=str(source),
                path=source_path,
                error=str(e),
            )
            return ProcessedDocument(
                full_text="",
                source_path=source_path,
                source_type=source,
                processing_errors=[str(e)],
            )

    def _extract_from_file(
        self, file_path: str
    ) -> tuple[str, Dict[str, Any], int]:
        """Extract text from a local file."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix == ".docx":
            return self._extract_docx(file_path)
        elif suffix in [".txt", ".md", ".markdown"]:
            return self._extract_text(file_path)
        else:
            # Try as plain text
            return self._extract_text(file_path)

    def _extract_pdf(self, file_path: str) -> tuple[str, Dict[str, Any], int]:
        """Extract text from PDF."""
        # Try docling first
        if self.use_docling and self._check_docling():
            try:
                return self._extract_with_docling(file_path)
            except Exception as e:
                logger.warning(f"docling failed, falling back to PyMuPDF: {e}")

        # Fallback to PyMuPDF
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            full_text = "\n\n".join(text_parts)

            metadata = {
                "page_count": len(doc),
                "format": "pdf",
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
            }
            page_count = len(doc)
            doc.close()

            return full_text, metadata, page_count

        except ImportError:
            raise ImportError("PyMuPDF (fitz) required for PDF processing")

    def _extract_docx(self, file_path: str) -> tuple[str, Dict[str, Any], int]:
        """Extract text from DOCX."""
        # Try docling first
        if self.use_docling and self._check_docling():
            try:
                return self._extract_with_docling(file_path)
            except Exception as e:
                logger.warning(f"docling failed, falling back to python-docx: {e}")

        # Fallback to python-docx
        try:
            import docx

            doc = docx.Document(file_path)
            text_parts = [para.text for para in doc.paragraphs]
            full_text = "\n\n".join(text_parts)

            metadata = {
                "format": "docx",
                "paragraph_count": len(doc.paragraphs),
            }

            return full_text, metadata, 1

        except ImportError:
            raise ImportError("python-docx required for DOCX processing")

    def _extract_text(self, file_path: str) -> tuple[str, Dict[str, Any], int]:
        """Extract text from plain text file."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            full_text = f.read()

        metadata = {
            "format": Path(file_path).suffix.lstrip("."),
            "size_bytes": Path(file_path).stat().st_size,
        }

        return full_text, metadata, 1

    def _extract_with_docling(
        self, file_path: str
    ) -> tuple[str, Dict[str, Any], int]:
        """Extract using docling for advanced parsing."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)

        full_text = result.document.export_to_markdown()

        metadata = {
            "format": Path(file_path).suffix.lstrip("."),
            "docling_version": "2.0",
        }

        # Try to get page count
        page_count = 1
        if hasattr(result.document, "pages"):
            page_count = len(result.document.pages)

        return full_text, metadata, page_count

    def _extract_from_url(self, url: str) -> tuple[str, Dict[str, Any], int]:
        """Extract text from URL."""
        import requests

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "pdf" in content_type:
            # Download and process as PDF
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(response.content)
                temp_path = f.name

            try:
                return self._extract_pdf(temp_path)
            finally:
                Path(temp_path).unlink()

        # Assume HTML/text
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []

            def handle_data(self, data):
                self.text_parts.append(data.strip())

        parser = TextExtractor()
        parser.feed(response.text)
        full_text = " ".join(filter(None, parser.text_parts))

        metadata = {
            "url": url,
            "content_type": content_type,
        }

        return full_text, metadata, 1


__all__ = [
    "DocumentSource",
    "TextChunk",
    "ProcessedDocument",
    "UnifiedDocumentProcessor",
    "chunk_text_by_tokens",
    "generate_chunk_id",
    "determine_boundary_type",
]
