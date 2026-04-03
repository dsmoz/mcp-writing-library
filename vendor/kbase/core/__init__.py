"""
Core business logic modules for kbase.

This module provides:
- Cache: Redis/in-memory caching with TTL
- Logger: Structured logging with structlog
- Document Processor: Text extraction and chunking
- Embeddings: OpenAI-compatible embedding generation
- Qdrant Functions: Vector database operations
- PDF Processing: PDF splitting and OCR detection
- Token Chunker: Semantic text chunking with token counting
- Monitoring: System resource monitoring and optimization
"""

try:
    from kbase.core.cache import (
        CacheConfig,
        CacheService,
        InMemoryCache,
        get_cache,
        init_cache,
    )
except ImportError as e:
    raise ImportError(f"kbase.core.cache failed to import: {e}. Check that all dependencies are installed.") from e

try:
    from kbase.core.document_processor import (
        DocumentSource,
        ProcessedDocument,
        TextChunk,
        UnifiedDocumentProcessor,
        chunk_text_by_tokens,
        determine_boundary_type,
        generate_chunk_id,
    )
except ImportError as e:
    raise ImportError(f"kbase.core.document_processor failed to import: {e}. Check tiktoken/openai dependencies.") from e

try:
    from kbase.core.embeddings import (
        EmbeddingCache,
        EmbeddingConfig,
        EmbeddingGenerator,
        generate_embedding,
        generate_embeddings_batch,
        get_embedding_generator,
    )
except ImportError as e:
    raise ImportError(f"kbase.core.embeddings failed to import: {e}.") from e

from kbase.core.logger import configure_structlog, get_logger

try:
    from kbase.core.qdrantFunctions import (
        add_new_qdrant_collection,
        collection_exists,
        delete_qdrant_collection,
        get_qdrant_client,
        list_collections,
        vectorize_documents_to_qdrant,
        vectorize_documents_to_qdrant_async,
    )
except ImportError as e:
    raise ImportError(f"kbase.core.qdrantFunctions failed to import: {e}. Check qdrant-client dependency.") from e

# PDF Processing
try:
    from kbase.core.pdf_processing import (
        PDFType,
        OCRDetectionResult,
        PDFOCRDetector,
        PDFSplitter,
        detect_ocr_requirement,
        create_pdf_splitter,
        print_detection_report,
    )
except ImportError as e:
    raise ImportError(f"kbase.core.pdf_processing failed to import: {e}. Check PyPDF2 dependency.") from e

# Token Chunker (standalone, more flexible than document_processor's version)
try:
    from kbase.core.token_chunker import (
        TextChunk as TokenTextChunk,  # Alias to avoid conflict
        PageMapping,
        chunk_text_by_tokens as token_chunk_text,  # Alias
        validate_chunks,
        get_chunk_statistics,
        estimate_chunk_count,
        get_token_count,
    )
except ImportError as e:
    raise ImportError(f"kbase.core.token_chunker failed to import: {e}. Check tiktoken dependency.") from e

# System Monitoring
try:
    from kbase.core.monitoring import (
        SystemMonitor,
        ResourceOptimizer,
        get_system_monitor,
        get_resource_optimizer,
        start_monitoring,
        stop_monitoring,
        get_current_metrics,
        get_recommendations,
        apply_optimizations,
    )
except ImportError as e:
    raise ImportError(f"kbase.core.monitoring failed to import: {e}. Check psutil dependency.") from e

# Notifications (Phase 3)
try:
    from kbase.core.notifications import (
        NotificationConfig,
        MacNotifier,
        get_notifier,
        reset_notifier,
        notify_job_completed,
        notify_job_failed,
        notify_queue_completed,
        notify_batch_completed,
        notify_queue_status,
    )
except ImportError as e:
    raise ImportError(f"kbase.core.notifications failed to import: {e}.") from e

# Markdown Handler (Phase 3)
try:
    from kbase.core.markdown_handler import (
        PageMapping as MarkdownPageMapping,  # Alias to avoid conflict with token_chunker
        ChunkPageInfo,
        MarkdownValidation,
        check_markdown_exists,
        read_markdown_file,
        extract_page_mappings,
        map_chunk_to_pages,
        map_chunk_to_pages_dict,
        get_all_markdown_files,
        validate_markdown_structure,
        validate_markdown_structure_dict,
        create_page_marker,
        insert_page_markers,
    )
except ImportError as e:
    raise ImportError(f"kbase.core.markdown_handler failed to import: {e}.") from e

# Context Generator (v1.2.0)
try:
    from kbase.core.context_generator import (
        build_metadata_context,
        validate_metadata_quality,
        ensure_document_metadata,
        generate_metadata_from_content,
        generate_document_context,
        init_llm_client,
        get_llm_client,
        is_contextual_retrieval_available,
        ContextGenerator,
        generate_context_for_indexing,
    )
except ImportError as e:
    raise ImportError(f"kbase.core.context_generator failed to import: {e}.") from e

__all__ = [
    # Cache
    "CacheConfig",
    "CacheService",
    "InMemoryCache",
    "init_cache",
    "get_cache",
    # Logger
    "get_logger",
    "configure_structlog",
    # Document Processor
    "DocumentSource",
    "TextChunk",
    "ProcessedDocument",
    "UnifiedDocumentProcessor",
    "chunk_text_by_tokens",
    "generate_chunk_id",
    "determine_boundary_type",
    # Embeddings
    "EmbeddingConfig",
    "EmbeddingCache",
    "EmbeddingGenerator",
    "get_embedding_generator",
    "generate_embedding",
    "generate_embeddings_batch",
    # Qdrant Functions
    "collection_exists",
    "add_new_qdrant_collection",
    "delete_qdrant_collection",
    "list_collections",
    "vectorize_documents_to_qdrant",
    "vectorize_documents_to_qdrant_async",
    "get_qdrant_client",
    # PDF Processing
    "PDFType",
    "OCRDetectionResult",
    "PDFOCRDetector",
    "PDFSplitter",
    "detect_ocr_requirement",
    "create_pdf_splitter",
    "print_detection_report",
    # Token Chunker
    "TokenTextChunk",
    "PageMapping",
    "token_chunk_text",
    "validate_chunks",
    "get_chunk_statistics",
    "estimate_chunk_count",
    "get_token_count",
    # System Monitoring
    "SystemMonitor",
    "ResourceOptimizer",
    "get_system_monitor",
    "get_resource_optimizer",
    "start_monitoring",
    "stop_monitoring",
    "get_current_metrics",
    "get_recommendations",
    "apply_optimizations",
    # Notifications (Phase 3)
    "NotificationConfig",
    "MacNotifier",
    "get_notifier",
    "reset_notifier",
    "notify_job_completed",
    "notify_job_failed",
    "notify_queue_completed",
    "notify_batch_completed",
    "notify_queue_status",
    # Markdown Handler (Phase 3)
    "MarkdownPageMapping",
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
    # Context Generator (v1.2.0)
    "build_metadata_context",
    "validate_metadata_quality",
    "ensure_document_metadata",
    "generate_metadata_from_content",
    "generate_document_context",
    "init_llm_client",
    "get_llm_client",
    "is_contextual_retrieval_available",
    "ContextGenerator",
    "generate_context_for_indexing",
]
