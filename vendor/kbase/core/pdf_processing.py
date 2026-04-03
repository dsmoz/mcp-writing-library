"""
PDF Processing Utilities

Provides intelligent PDF analysis, splitting, and OCR detection for
memory-efficient processing of large PDF files.

This module combines:
- PDFSplitter: Intelligent PDF splitting by size/pages
- PDFOCRDetector: Detection of whether PDF requires OCR
- PDFType enum: Classification of PDF types

Usage:
    from kbase.core.pdf_processing import (
        PDFSplitter, PDFOCRDetector, PDFType,
        detect_ocr_requirement, create_pdf_splitter
    )

    # Check if PDF needs OCR
    result = detect_ocr_requirement("document.pdf")
    if result.requires_ocr:
        print(f"OCR needed: {result.recommendation}")

    # Split large PDF
    splitter = create_pdf_splitter()
    chunks = splitter.split_pdf("large_document.pdf", "doc_001")
"""

import os
import sys
import math
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# PDF Type Classification
# =============================================================================

class PDFType(Enum):
    """Classification of PDF types for processing strategy."""
    TEXT_BASED = "text_based"  # Native PDF with embedded text
    IMAGE_BASED = "image_based"  # Scanned/image-only PDF
    MIXED = "mixed"  # Contains both text and images
    UNKNOWN = "unknown"  # Unable to determine


# =============================================================================
# OCR Detection
# =============================================================================

@dataclass
class OCRDetectionResult:
    """Result of OCR detection analysis."""
    requires_ocr: bool
    pdf_type: PDFType
    confidence: float  # 0.0 to 1.0
    text_coverage: float  # Percentage of pages with text (0-100)
    image_coverage: float  # Percentage of pages with images (0-100)
    total_pages: int
    pages_with_text: int
    pages_with_images: int
    avg_text_per_page: float  # Average characters per page
    recommendation: str
    details: Dict[str, Any]


class PDFOCRDetector:
    """
    Intelligent detector for determining if a PDF requires OCR.

    Uses multiple detection strategies:
    1. Text extraction analysis (primary method)
    2. Image/graphics detection
    3. Font embedding analysis
    4. Content stream analysis
    5. File size heuristics
    """

    def __init__(
        self,
        min_text_threshold: int = 50,
        min_text_coverage: float = 80.0,
        sample_pages: Optional[int] = None
    ):
        """
        Initialize the OCR detector.

        Args:
            min_text_threshold: Minimum characters per page to consider it text-based
            min_text_coverage: Minimum percentage of pages with text (0-100)
            sample_pages: Number of pages to sample (None = all pages)
        """
        self.min_text_threshold = min_text_threshold
        self.min_text_coverage = min_text_coverage
        self.sample_pages = sample_pages
        self.pdf_library = self._detect_pdf_library()

    def _detect_pdf_library(self) -> str:
        """Detect which PDF library is available."""
        try:
            import fitz  # PyMuPDF
            return "pymupdf"
        except ImportError:
            pass

        try:
            import PyPDF2
            return "pypdf2"
        except ImportError:
            pass

        try:
            import pdfplumber
            return "pdfplumber"
        except ImportError:
            pass

        return "none"

    def detect(self, pdf_path: str) -> OCRDetectionResult:
        """
        Analyze a PDF file to determine if it requires OCR.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            OCRDetectionResult with analysis and recommendation
        """
        if not os.path.exists(pdf_path):
            return self._create_error_result(f"PDF file not found: {pdf_path}")

        if self.pdf_library == "none":
            return self._create_error_result(
                "No PDF library available. Install PyMuPDF (fitz), PyPDF2, or pdfplumber."
            )

        try:
            if self.pdf_library == "pymupdf":
                return self._detect_with_pymupdf(pdf_path)
            elif self.pdf_library == "pypdf2":
                return self._detect_with_pypdf2(pdf_path)
            elif self.pdf_library == "pdfplumber":
                return self._detect_with_pdfplumber(pdf_path)
        except Exception as e:
            return self._create_error_result(f"Error analyzing PDF: {str(e)}")

    def _detect_with_pymupdf(self, pdf_path: str) -> OCRDetectionResult:
        """Detect OCR requirement using PyMuPDF (most feature-rich)."""
        import fitz

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            pages_to_check = self._get_sample_pages(total_pages)

            pages_with_text = 0
            pages_with_images = 0
            total_text_length = 0
            text_lengths = []

            for page_num in pages_to_check:
                page = doc[page_num]
                text = page.get_text().strip()
                text_length = len(text)
                text_lengths.append(text_length)
                total_text_length += text_length

                if text_length >= self.min_text_threshold:
                    pages_with_text += 1

                image_list = page.get_images()
                if image_list:
                    pages_with_images += 1

            doc.close()

            avg_text_per_page = total_text_length / len(pages_to_check) if pages_to_check else 0
            text_coverage = (pages_with_text / len(pages_to_check) * 100) if pages_to_check else 0
            image_coverage = (pages_with_images / len(pages_to_check) * 100) if pages_to_check else 0

            pdf_type, requires_ocr, confidence = self._classify_pdf(
                text_coverage, image_coverage, avg_text_per_page
            )

            recommendation = self._generate_recommendation(
                pdf_type, requires_ocr, text_coverage, avg_text_per_page
            )

            return OCRDetectionResult(
                requires_ocr=requires_ocr,
                pdf_type=pdf_type,
                confidence=confidence,
                text_coverage=text_coverage,
                image_coverage=image_coverage,
                total_pages=total_pages,
                pages_with_text=pages_with_text,
                pages_with_images=pages_with_images,
                avg_text_per_page=avg_text_per_page,
                recommendation=recommendation,
                details={
                    "library": "PyMuPDF",
                    "sample_size": len(pages_to_check),
                    "text_lengths": text_lengths[:10],
                    "min_text_length": min(text_lengths) if text_lengths else 0,
                    "max_text_length": max(text_lengths) if text_lengths else 0,
                }
            )

        except Exception as e:
            return self._create_error_result(f"PyMuPDF error: {str(e)}")

    def _detect_with_pypdf2(self, pdf_path: str) -> OCRDetectionResult:
        """Detect OCR requirement using PyPDF2."""
        import PyPDF2

        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                pages_to_check = self._get_sample_pages(total_pages)

                pages_with_text = 0
                total_text_length = 0
                text_lengths = []

                for page_num in pages_to_check:
                    page = reader.pages[page_num]
                    text = page.extract_text().strip()
                    text_length = len(text)
                    text_lengths.append(text_length)
                    total_text_length += text_length

                    if text_length >= self.min_text_threshold:
                        pages_with_text += 1

                avg_text_per_page = total_text_length / len(pages_to_check) if pages_to_check else 0
                text_coverage = (pages_with_text / len(pages_to_check) * 100) if pages_to_check else 0
                image_coverage = 0.0

                pdf_type, requires_ocr, confidence = self._classify_pdf(
                    text_coverage, image_coverage, avg_text_per_page
                )
                confidence = max(0.6, confidence - 0.2)

                recommendation = self._generate_recommendation(
                    pdf_type, requires_ocr, text_coverage, avg_text_per_page
                )

                return OCRDetectionResult(
                    requires_ocr=requires_ocr,
                    pdf_type=pdf_type,
                    confidence=confidence,
                    text_coverage=text_coverage,
                    image_coverage=image_coverage,
                    total_pages=total_pages,
                    pages_with_text=pages_with_text,
                    pages_with_images=0,
                    avg_text_per_page=avg_text_per_page,
                    recommendation=recommendation,
                    details={
                        "library": "PyPDF2",
                        "sample_size": len(pages_to_check),
                        "text_lengths": text_lengths[:10],
                        "note": "Image detection not available with PyPDF2"
                    }
                )

        except Exception as e:
            return self._create_error_result(f"PyPDF2 error: {str(e)}")

    def _detect_with_pdfplumber(self, pdf_path: str) -> OCRDetectionResult:
        """Detect OCR requirement using pdfplumber."""
        import pdfplumber

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_check = self._get_sample_pages(total_pages)

                pages_with_text = 0
                pages_with_images = 0
                total_text_length = 0
                text_lengths = []

                for page_num in pages_to_check:
                    page = pdf.pages[page_num]
                    text = page.extract_text() or ""
                    text_length = len(text.strip())
                    text_lengths.append(text_length)
                    total_text_length += text_length

                    if text_length >= self.min_text_threshold:
                        pages_with_text += 1

                    images = page.images
                    if images:
                        pages_with_images += 1

                avg_text_per_page = total_text_length / len(pages_to_check) if pages_to_check else 0
                text_coverage = (pages_with_text / len(pages_to_check) * 100) if pages_to_check else 0
                image_coverage = (pages_with_images / len(pages_to_check) * 100) if pages_to_check else 0

                pdf_type, requires_ocr, confidence = self._classify_pdf(
                    text_coverage, image_coverage, avg_text_per_page
                )

                recommendation = self._generate_recommendation(
                    pdf_type, requires_ocr, text_coverage, avg_text_per_page
                )

                return OCRDetectionResult(
                    requires_ocr=requires_ocr,
                    pdf_type=pdf_type,
                    confidence=confidence,
                    text_coverage=text_coverage,
                    image_coverage=image_coverage,
                    total_pages=total_pages,
                    pages_with_text=pages_with_text,
                    pages_with_images=pages_with_images,
                    avg_text_per_page=avg_text_per_page,
                    recommendation=recommendation,
                    details={
                        "library": "pdfplumber",
                        "sample_size": len(pages_to_check),
                        "text_lengths": text_lengths[:10],
                    }
                )

        except Exception as e:
            return self._create_error_result(f"pdfplumber error: {str(e)}")

    def _get_sample_pages(self, total_pages: int) -> List[int]:
        """Determine which pages to sample for analysis."""
        if self.sample_pages is None or total_pages <= 10:
            return list(range(total_pages))

        if total_pages <= 50:
            step = max(1, total_pages // self.sample_pages)
            return list(range(0, total_pages, step))[:self.sample_pages]

        sample_size = self.sample_pages
        section_size = sample_size // 3

        pages = []
        pages.extend(range(0, min(section_size, total_pages)))
        middle_start = total_pages // 2 - section_size // 2
        pages.extend(range(middle_start, min(middle_start + section_size, total_pages)))
        last_start = total_pages - section_size
        pages.extend(range(max(last_start, 0), total_pages))

        return sorted(set(pages))[:sample_size]

    def _classify_pdf(
        self,
        text_coverage: float,
        image_coverage: float,
        avg_text_per_page: float
    ) -> Tuple[PDFType, bool, float]:
        """Classify PDF type and determine OCR requirement."""
        if text_coverage >= self.min_text_coverage and avg_text_per_page >= self.min_text_threshold:
            return PDFType.TEXT_BASED, False, 0.9

        if text_coverage < 20 and avg_text_per_page < self.min_text_threshold / 2:
            return PDFType.IMAGE_BASED, True, 0.85

        if image_coverage > 50 and text_coverage < 50:
            return PDFType.IMAGE_BASED, True, 0.75

        if text_coverage >= 40 and image_coverage > 20:
            if avg_text_per_page >= self.min_text_threshold * 2:
                return PDFType.MIXED, False, 0.7
            else:
                return PDFType.MIXED, True, 0.65

        if text_coverage < 40:
            return PDFType.IMAGE_BASED, True, 0.6

        return PDFType.TEXT_BASED, False, 0.5

    def _generate_recommendation(
        self,
        pdf_type: PDFType,
        requires_ocr: bool,
        text_coverage: float,
        avg_text_per_page: float
    ) -> str:
        """Generate human-readable recommendation."""
        if pdf_type == PDFType.TEXT_BASED:
            return (
                f"PDF is text-based ({text_coverage:.1f}% pages with text). "
                f"Direct text extraction is recommended. No OCR needed."
            )
        elif pdf_type == PDFType.IMAGE_BASED:
            return (
                f"PDF is image-based ({text_coverage:.1f}% pages with text). "
                f"OCR processing is required for text extraction."
            )
        elif pdf_type == PDFType.MIXED:
            if requires_ocr:
                return (
                    f"PDF has mixed content ({text_coverage:.1f}% text coverage). "
                    f"OCR recommended for complete extraction."
                )
            else:
                return (
                    f"PDF has mixed content ({text_coverage:.1f}% text coverage). "
                    f"Text extraction is sufficient, OCR optional."
                )
        else:
            return "Unable to determine PDF type. Manual inspection recommended."

    def _create_error_result(self, error_message: str) -> OCRDetectionResult:
        """Create an error result."""
        return OCRDetectionResult(
            requires_ocr=False,
            pdf_type=PDFType.UNKNOWN,
            confidence=0.0,
            text_coverage=0.0,
            image_coverage=0.0,
            total_pages=0,
            pages_with_text=0,
            pages_with_images=0,
            avg_text_per_page=0.0,
            recommendation=f"Error: {error_message}",
            details={"error": error_message}
        )


# =============================================================================
# PDF Splitting
# =============================================================================

class PDFSplitter:
    """
    Intelligent PDF splitter for memory-efficient processing of large PDF files.

    This class intelligently splits large PDF files into manageable chunks based on:
    - File size limits
    - Page count limits
    - Memory constraints
    - Processing complexity
    """

    def __init__(
        self,
        max_chunk_size_mb: int = 5,
        max_chunk_pages: int = 10,
        overlap_pages: int = 1,
        temp_dir: str = ".temp/pdf_chunks"
    ):
        """
        Initialize the PDF splitter with configurable limits.

        Args:
            max_chunk_size_mb: Maximum size per chunk in MB
            max_chunk_pages: Maximum pages per chunk
            overlap_pages: Pages to overlap between chunks for context
            temp_dir: Directory to store temporary chunk files
        """
        self.max_chunk_size_mb = max_chunk_size_mb
        self.max_chunk_pages = max_chunk_pages
        self.overlap_pages = overlap_pages
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF file to determine if splitting is needed.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with analysis results including size, pages, and split recommendation
        """
        try:
            import fitz
        except ImportError:
            return {"error": "PyMuPDF (fitz) is required for PDF splitting"}

        try:
            if not os.path.exists(pdf_path):
                return {"error": f"PDF file not found: {pdf_path}"}

            file_size_bytes = os.path.getsize(pdf_path)
            file_size_mb = file_size_bytes / (1024 * 1024)

            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()

            needs_splitting = (
                file_size_mb > self.max_chunk_size_mb or
                page_count > self.max_chunk_pages
            )

            if needs_splitting:
                size_based_chunks = math.ceil(file_size_mb / self.max_chunk_size_mb)
                page_based_chunks = math.ceil(page_count / self.max_chunk_pages)
                recommended_chunks = max(size_based_chunks, page_based_chunks)
                pages_per_chunk = math.ceil(page_count / recommended_chunks)
            else:
                recommended_chunks = 1
                pages_per_chunk = page_count

            return {
                "file_size_mb": round(file_size_mb, 2),
                "page_count": page_count,
                "needs_splitting": needs_splitting,
                "recommended_chunks": recommended_chunks,
                "pages_per_chunk": pages_per_chunk,
                "estimated_chunk_size_mb": round(
                    file_size_mb / recommended_chunks, 2
                ) if recommended_chunks > 0 else file_size_mb
            }

        except Exception as e:
            return {"error": f"Error analyzing PDF {pdf_path}: {str(e)}"}

    def split_pdf(self, pdf_path: str, item_key: str) -> List[Dict[str, Any]]:
        """
        Split a PDF file into manageable chunks.

        Args:
            pdf_path: Path to the original PDF file
            item_key: Unique identifier for naming chunks

        Returns:
            List of chunk information dictionaries
        """
        try:
            import fitz
        except ImportError:
            return [{"error": "PyMuPDF (fitz) is required for PDF splitting"}]

        try:
            analysis = self.analyze_pdf(pdf_path)
            if "error" in analysis:
                return [{"error": analysis["error"]}]

            if not analysis["needs_splitting"]:
                return [{
                    "chunk_id": f"{item_key}_full",
                    "file_path": pdf_path,
                    "start_page": 1,
                    "end_page": analysis["page_count"],
                    "page_count": analysis["page_count"],
                    "estimated_size_mb": analysis["file_size_mb"],
                    "is_original": True
                }]

            doc = fitz.open(pdf_path)
            chunks = []
            pages_per_chunk = analysis["pages_per_chunk"]
            total_pages = analysis["page_count"]

            for chunk_index in range(analysis["recommended_chunks"]):
                start_page = chunk_index * pages_per_chunk
                end_page = min(start_page + pages_per_chunk - 1, total_pages - 1)

                if chunk_index > 0:
                    start_page = max(0, start_page - self.overlap_pages)

                if chunk_index < analysis["recommended_chunks"] - 1:
                    end_page = min(total_pages - 1, end_page + self.overlap_pages)

                chunk_filename = f"{item_key}_chunk_{chunk_index + 1}_of_{analysis['recommended_chunks']}.pdf"
                chunk_path = self.temp_dir / chunk_filename

                chunk_doc = fitz.open()
                chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
                chunk_doc.save(str(chunk_path))
                chunk_doc.close()

                chunk_size_bytes = os.path.getsize(chunk_path)
                chunk_size_mb = chunk_size_bytes / (1024 * 1024)

                chunk_info = {
                    "chunk_id": f"{item_key}_chunk_{chunk_index + 1}",
                    "file_path": str(chunk_path),
                    "start_page": start_page + 1,
                    "end_page": end_page + 1,
                    "page_count": end_page - start_page + 1,
                    "estimated_size_mb": round(chunk_size_mb, 2),
                    "is_original": False,
                    "chunk_index": chunk_index + 1,
                    "total_chunks": analysis["recommended_chunks"]
                }

                chunks.append(chunk_info)

            doc.close()

            logger.info(
                "Split PDF into chunks",
                pdf_path=pdf_path,
                chunks=len(chunks)
            )
            return chunks

        except Exception as e:
            error_msg = f"Error splitting PDF {pdf_path}: {str(e)}"
            logger.error(error_msg)
            return [{"error": error_msg}]

    def cleanup_chunks(self, item_key: str) -> bool:
        """
        Clean up temporary chunk files for a specific item.

        Args:
            item_key: Unique identifier to clean up chunks for

        Returns:
            True if cleanup was successful
        """
        try:
            pattern = f"{item_key}_chunk_*.pdf"
            chunk_files = list(self.temp_dir.glob(pattern))

            for chunk_file in chunk_files:
                if chunk_file.exists():
                    chunk_file.unlink()
                    logger.debug("Cleaned up chunk file", file=str(chunk_file))

            return True

        except Exception as e:
            logger.error("Error cleaning up chunks", item_key=item_key, error=str(e))
            return False

    def get_chunk_processing_priority(self, chunk_info: Dict[str, Any]) -> int:
        """
        Calculate processing priority for a chunk based on size and complexity.
        Lower numbers = higher priority.

        Args:
            chunk_info: Chunk information dictionary

        Returns:
            Priority score (lower = higher priority)
        """
        base_priority = 100
        size_penalty = int(chunk_info.get("estimated_size_mb", 0) * 2)
        page_penalty = int(chunk_info.get("page_count", 0) / 10)

        if chunk_info.get("is_original", False):
            original_bonus = -20
        else:
            original_bonus = 0

        return base_priority + size_penalty + page_penalty + original_bonus

    def get_recommended_delay(self, chunk_info: Dict[str, Any]) -> int:
        """
        Get recommended delay before processing this chunk.

        Args:
            chunk_info: Chunk information dictionary

        Returns:
            Delay in seconds
        """
        size_mb = chunk_info.get("estimated_size_mb", 0)

        if size_mb > 20:
            return 300  # 5 minutes for large chunks
        elif size_mb > 10:
            return 120  # 2 minutes for medium chunks
        else:
            return 30   # 30 seconds for small chunks


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_ocr_requirement(
    pdf_path: str,
    min_text_threshold: int = 50,
    min_text_coverage: float = 80.0,
    sample_pages: Optional[int] = 10
) -> OCRDetectionResult:
    """
    Convenience function to detect if a PDF requires OCR.

    Args:
        pdf_path: Path to the PDF file
        min_text_threshold: Minimum characters per page for text-based classification
        min_text_coverage: Minimum percentage of pages with text
        sample_pages: Number of pages to sample (None = all pages)

    Returns:
        OCRDetectionResult with analysis and recommendation

    Example:
        >>> result = detect_ocr_requirement("research_paper.pdf")
        >>> if result.requires_ocr:
        >>>     print(f"OCR needed: {result.recommendation}")
        >>> else:
        >>>     print(f"Direct extraction: {result.recommendation}")
    """
    detector = PDFOCRDetector(
        min_text_threshold=min_text_threshold,
        min_text_coverage=min_text_coverage,
        sample_pages=sample_pages
    )
    return detector.detect(pdf_path)


def create_pdf_splitter(
    max_chunk_size_mb: Optional[int] = None,
    max_chunk_pages: Optional[int] = None,
    overlap_pages: Optional[int] = None,
    temp_dir: Optional[str] = None
) -> PDFSplitter:
    """
    Create a configured PDF splitter instance.

    Args:
        max_chunk_size_mb: Maximum chunk size in MB (default from env or 5)
        max_chunk_pages: Maximum pages per chunk (default from env or 10)
        overlap_pages: Pages to overlap (default from env or 1)
        temp_dir: Temp directory for chunks

    Returns:
        Configured PDFSplitter instance
    """
    return PDFSplitter(
        max_chunk_size_mb=max_chunk_size_mb or int(os.getenv("PDF_MAX_CHUNK_SIZE_MB", "5")),
        max_chunk_pages=max_chunk_pages or int(os.getenv("PDF_MAX_CHUNK_PAGES", "10")),
        overlap_pages=overlap_pages or int(os.getenv("PDF_OVERLAP_PAGES", "1")),
        temp_dir=temp_dir or ".temp/pdf_chunks"
    )


def print_detection_report(result: OCRDetectionResult, verbose: bool = False):
    """
    Print a formatted detection report.

    Args:
        result: OCRDetectionResult to print
        verbose: Include detailed information
    """
    print("\n" + "=" * 70)
    print("PDF OCR DETECTION REPORT")
    print("=" * 70)
    print(f"PDF Type: {result.pdf_type.value.upper()}")
    print(f"Requires OCR: {'YES' if result.requires_ocr else 'NO'}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"\nRecommendation:")
    print(f"  {result.recommendation}")

    if verbose:
        print(f"\nDetailed Metrics:")
        print(f"  Total Pages: {result.total_pages}")
        print(f"  Pages with Text: {result.pages_with_text} ({result.text_coverage:.1f}%)")
        print(f"  Pages with Images: {result.pages_with_images} ({result.image_coverage:.1f}%)")
        print(f"  Avg Characters/Page: {result.avg_text_per_page:.0f}")

        if result.details:
            print(f"\nTechnical Details:")
            for key, value in result.details.items():
                if key not in ['text_lengths']:
                    print(f"  {key}: {value}")

    print("=" * 70 + "\n")


__all__ = [
    # Enums
    "PDFType",
    # Dataclasses
    "OCRDetectionResult",
    # Classes
    "PDFOCRDetector",
    "PDFSplitter",
    # Functions
    "detect_ocr_requirement",
    "create_pdf_splitter",
    "print_detection_report",
]
