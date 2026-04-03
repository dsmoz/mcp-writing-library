"""
Synchronous Document Indexing

This module provides synchronous document indexing for Qdrant.
Handles chunking, embedding generation, and upserting to Qdrant.

Based on the indexing strategy from mcp-zotero-qdrant.

Usage:
    from kbase.vector.sync_indexing import index_document, index_documents_batch

    # Index a single document
    point_ids = index_document(
        collection_name="cerebellum",
        document_id="doc-123",
        title="My Document",
        content="Document content here...",
        metadata={"source": "manual"}
    )

    # Index multiple documents in batch
    results = index_documents_batch(
        collection_name="cerebellum",
        documents=[
            {"document_id": "doc-1", "title": "Doc 1", "content": "..."},
            {"document_id": "doc-2", "title": "Doc 2", "content": "..."},
        ]
    )
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from qdrant_client.models import PointStruct

import structlog

from kbase.vector.sync_client import get_qdrant_client
from kbase.vector.sync_embeddings import (
    generate_embedding,
    generate_embeddings_batch,
    generate_sparse_vector,
    generate_sparse_vectors_batch,
)
from kbase.vector.chunker import chunk_text, TextChunk
from kbase.core.context_generator import (
    # Fast approach (default)
    build_metadata_context,
    validate_metadata_quality,
    # LLM-based (for poor metadata)
    generate_metadata_from_content,
    generate_document_context,
    # Legacy per-chunk LLM approach
    ContextGenerator,
    is_contextual_retrieval_available,
    CONTEXT_DELIMITER,
)

logger = structlog.get_logger(__name__)

# Configuration constants
DEFAULT_CHUNK_SIZE = 512        # Tokens per chunk
DEFAULT_CHUNK_OVERLAP = 50      # Token overlap
EMBEDDING_BATCH_SIZE = 100      # Texts per API call
QDRANT_UPSERT_BATCH = 500       # Points per upsert


def is_hybrid_collection(collection_name: str) -> bool:
    """
    Check if a collection uses hybrid vectors (named dense + sparse).

    Args:
        collection_name: Qdrant collection name

    Returns:
        True if collection has named vectors, False if simple dense
    """
    client = get_qdrant_client()

    try:
        info = client.get_collection(collection_name=collection_name)
        vectors_config = info.config.params.vectors

        # If vectors_config is a dict, it has named vectors (hybrid)
        # If it's VectorParams directly, it's simple dense
        return isinstance(vectors_config, dict)

    except Exception as e:
        logger.warning(
            "Could not determine collection type, assuming simple",
            collection=collection_name,
            error=str(e),
        )
        return False


def index_document(
    collection_name: str,
    document_id: str,
    title: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    enable_contextual_retrieval: bool = True,
    max_context_doc_chars: int = 12000,
    context_mode: str = "auto",
) -> List[str]:
    """
    Index a single document to Qdrant with hybrid contextual retrieval.

    Pipeline:
    1. Chunk text into smaller pieces
    2. Generate context (metadata-based or LLM-based depending on context_mode)
    3. Generate embeddings for each chunk (with context prepended)
    4. Upsert points to Qdrant

    Context Modes:
    - "auto" (default): Use metadata if good quality, else use LLM for document-level context
    - "metadata": Always use metadata-based context (fast, no LLM)
    - "llm_document": Use LLM to generate ONE document-level context
    - "llm_chunk": Use LLM to generate context per chunk (slow, legacy)
    - "none": No context prepending

    Args:
        collection_name: Qdrant collection name
        document_id: Unique document identifier
        title: Document title (stored in payload)
        content: Full document text
        metadata: Additional metadata to store (title, tags, creators, date, etc.)
        chunk_size: Target tokens per chunk
        chunk_overlap: Overlap between chunks
        enable_contextual_retrieval: Enable context prepending (default True)
        max_context_doc_chars: Max document chars for LLM context generation
        context_mode: Context generation mode (auto|metadata|llm_document|llm_chunk|none)

    Returns:
        List of Qdrant point IDs created

    Raises:
        Exception: If indexing fails
    """
    if not content or not content.strip():
        logger.warning("Empty content, skipping indexing", document_id=document_id)
        return []

    metadata = metadata or {}

    try:
        # Step 1: Chunk the text
        logger.info(
            "Chunking document",
            document_id=document_id,
            content_length=len(content),
        )

        chunks = chunk_text(
            text=content,
            document_id=document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            logger.warning("No chunks generated", document_id=document_id)
            return []

        # Step 2: Generate context (hybrid approach)
        chunk_texts = [chunk.text for chunk in chunks]
        contexts = [None] * len(chunks)  # Store contexts for payload
        document_context = None  # Single context for all chunks

        if enable_contextual_retrieval and context_mode != "none":
            # Determine actual context mode
            actual_mode = context_mode

            if context_mode == "auto":
                # Check metadata quality to decide approach
                validation = validate_metadata_quality(
                    title=title,
                    metadata=metadata,
                )

                if validation["is_valid"]:
                    actual_mode = "metadata"
                    logger.info(
                        "Using metadata-based context (fast)",
                        document_id=document_id,
                        quality_score=validation["quality_score"],
                    )
                elif is_contextual_retrieval_available():
                    actual_mode = "llm_document"
                    logger.info(
                        "Metadata poor, using LLM document-level context",
                        document_id=document_id,
                        missing_fields=validation["missing_fields"],
                    )
                else:
                    actual_mode = "metadata"
                    logger.info(
                        "Metadata poor but LLM not available, using metadata anyway",
                        document_id=document_id,
                    )

            # Generate context based on mode
            if actual_mode == "metadata":
                # FAST: Metadata-based context (instant, no LLM)
                document_context = build_metadata_context(
                    title=title,
                    metadata=metadata,
                    chunk_info={"chunk_index": 0, "total_chunks": len(chunks)},
                )
                contexts = [document_context] * len(chunks)

                logger.info(
                    "Generated metadata-based context",
                    document_id=document_id,
                    context_preview=document_context[:100] if document_context else None,
                )

            elif actual_mode == "llm_document":
                # MEDIUM: ONE LLM call for document-level context
                if is_contextual_retrieval_available():
                    try:
                        document_context = generate_document_context(
                            content=content,
                            max_content_chars=max_context_doc_chars,
                        )
                        contexts = [document_context] * len(chunks)

                        logger.info(
                            "Generated LLM document-level context",
                            document_id=document_id,
                            context_preview=document_context[:100] if document_context else None,
                        )
                    except Exception as e:
                        logger.warning(
                            "LLM document context failed, falling back to metadata",
                            document_id=document_id,
                            error=str(e),
                        )
                        document_context = build_metadata_context(
                            title=title,
                            metadata=metadata,
                        )
                        contexts = [document_context] * len(chunks)
                else:
                    logger.warning(
                        "llm_document mode requested but LLM not available",
                        document_id=document_id,
                    )

            elif actual_mode == "llm_chunk":
                # SLOW: Per-chunk LLM context (legacy approach)
                if is_contextual_retrieval_available():
                    logger.info(
                        "Generating per-chunk LLM contexts (slow)",
                        document_id=document_id,
                        num_chunks=len(chunks),
                    )

                    context_generator = ContextGenerator(
                        multilingual=True,
                        max_document_chars=max_context_doc_chars,
                    )
                    contextualized_chunks = context_generator.generate_contexts(
                        full_document=content,
                        chunks=chunk_texts,
                        title=title,
                        metadata=metadata,
                    )

                    # Use contextualized text for embedding
                    chunk_texts = [c.contextualized_text for c in contextualized_chunks]
                    contexts = [c.context for c in contextualized_chunks]

                    logger.info(
                        "Generated per-chunk LLM contexts",
                        document_id=document_id,
                        sample_context=contexts[0][:100] if contexts else None,
                    )
                else:
                    logger.warning(
                        "llm_chunk mode requested but LLM not available",
                        document_id=document_id,
                    )

            # Prepend document context to chunk texts (for non-llm_chunk modes)
            if document_context and actual_mode != "llm_chunk":
                chunk_texts = [
                    f"{document_context}{CONTEXT_DELIMITER}{chunk_text}"
                    for chunk_text in chunk_texts
                ]

        elif enable_contextual_retrieval:
            logger.debug(
                "Contextual retrieval disabled via context_mode=none",
                document_id=document_id,
            )

        # Step 3: Generate embeddings for all chunks (with context prepended)
        has_context = any(c is not None for c in contexts)

        # Check if collection uses hybrid vectors
        is_hybrid = is_hybrid_collection(collection_name)

        logger.info(
            "Generating embeddings",
            document_id=document_id,
            num_chunks=len(chunks),
            has_context=has_context,
            context_mode=context_mode if enable_contextual_retrieval else "disabled",
            hybrid=is_hybrid,
        )

        # Generate dense embeddings
        embeddings = generate_embeddings_batch(chunk_texts)

        # Generate sparse vectors for hybrid collections
        sparse_vectors = None
        if is_hybrid:
            sparse_vectors = generate_sparse_vectors_batch(chunk_texts)
            logger.debug(
                "Generated sparse vectors",
                document_id=document_id,
                num_vectors=len(sparse_vectors),
            )

        # Step 4: Create Qdrant points
        from qdrant_client.models import SparseVector

        points = []
        point_ids = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Use UUID for point ID (Qdrant requires UUID or int)
            point_id = str(uuid4())
            point_ids.append(point_id)

            # Build payload with document info + chunk info
            payload = {
                "document_id": str(document_id),
                "title": title,
                "text": chunk.text,  # Original chunk text (for display)
                "chunk_index": chunk.chunk_index,
                "chunk_id": chunk.chunk_id,  # Deterministic ID for deduplication
                "token_count": chunk.token_count,
                "boundary_type": chunk.boundary_type,
                **metadata,
            }

            # Add page info if available
            if chunk.page_numbers:
                payload["page_numbers"] = chunk.page_numbers
                payload["start_page"] = chunk.start_page
                payload["end_page"] = chunk.end_page

            # Add context if contextual retrieval was used
            if contexts[i]:
                payload["context"] = contexts[i]
                payload["contextualized"] = True
            else:
                payload["contextualized"] = False

            # Create point with appropriate vector format
            if is_hybrid and sparse_vectors:
                # Hybrid: named vectors (dense + sparse)
                indices, values = sparse_vectors[i]
                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense": embedding,
                        "sparse": SparseVector(indices=indices, values=values),
                    },
                    payload=payload,
                )
            else:
                # Simple: single unnamed dense vector
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )

            points.append(point)

        # Step 5: Upsert to Qdrant
        logger.info(
            "Upserting to Qdrant",
            document_id=document_id,
            num_points=len(points),
            collection=collection_name,
        )

        client = get_qdrant_client()

        # Upsert in batches to avoid timeout
        for i in range(0, len(points), QDRANT_UPSERT_BATCH):
            batch_points = points[i:i + QDRANT_UPSERT_BATCH]
            client.upsert(
                collection_name=collection_name,
                points=batch_points,
            )

        logger.info(
            "Document indexed successfully",
            document_id=document_id,
            num_chunks=len(chunks),
            collection=collection_name,
        )

        return point_ids

    except Exception as e:
        logger.error(
            "Failed to index document",
            error=str(e),
            document_id=document_id,
        )
        raise


def index_documents_batch(
    collection_name: str,
    documents: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """
    Index multiple documents to Qdrant efficiently.

    Processes documents one at a time but batches embedding generation
    within each document for efficiency.

    Args:
        collection_name: Qdrant collection name
        documents: List of document dicts with keys:
            - document_id: Unique identifier
            - title: Document title
            - content: Full text
            - metadata: Optional additional metadata
        chunk_size: Target tokens per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Dict with:
            - success: True if all documents indexed
            - total_documents: Number of documents processed
            - total_chunks: Total chunks created
            - indexed: List of successfully indexed document IDs
            - failed: List of failed document IDs with errors

    Example:
        >>> docs = [
        ...     {"document_id": "1", "title": "Doc 1", "content": "..."},
        ...     {"document_id": "2", "title": "Doc 2", "content": "..."},
        ... ]
        >>> result = index_documents_batch("cerebellum", docs)
        >>> print(f"Indexed {len(result['indexed'])} documents")
    """
    results = {
        "success": True,
        "total_documents": len(documents),
        "total_chunks": 0,
        "indexed": [],
        "failed": [],
    }

    for doc in documents:
        document_id = doc.get("document_id")
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})

        if not document_id:
            results["failed"].append({
                "document_id": None,
                "error": "Missing document_id",
            })
            continue

        try:
            point_ids = index_document(
                collection_name=collection_name,
                document_id=document_id,
                title=title,
                content=content,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            results["indexed"].append(document_id)
            results["total_chunks"] += len(point_ids)

        except Exception as e:
            results["failed"].append({
                "document_id": document_id,
                "error": str(e),
            })
            results["success"] = False

    logger.info(
        "Batch indexing complete",
        total_documents=results["total_documents"],
        indexed=len(results["indexed"]),
        failed=len(results["failed"]),
        total_chunks=results["total_chunks"],
    )

    return results


def delete_document_vectors(
    collection_name: str,
    document_id: str,
) -> int:
    """
    Delete all vector points for a document.

    Args:
        collection_name: Qdrant collection name
        document_id: Document UUID

    Returns:
        Number of points deleted
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_qdrant_client()

    try:
        # Build filter for document_id
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(document_id)),
                )
            ]
        )

        # Scroll to find all points
        points_to_delete = []
        offset = None

        while True:
            result, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=100,
                offset=offset,
            )

            if not result:
                break

            points_to_delete.extend([point.id for point in result])

            if offset is None:
                break

        # Delete points
        if points_to_delete:
            client.delete(
                collection_name=collection_name,
                points_selector=points_to_delete,
            )

        logger.info(
            "Document vectors deleted",
            document_id=document_id,
            points_deleted=len(points_to_delete),
            collection=collection_name,
        )

        return len(points_to_delete)

    except Exception as e:
        logger.error(
            "Failed to delete document vectors",
            error=str(e),
            document_id=document_id,
        )
        raise


def ensure_collection(
    collection_name: str,
    vector_size: int = 768,
    recreate: bool = False,
    hybrid: bool = True,
) -> bool:
    """
    Ensure a Qdrant collection exists with proper configuration.

    Args:
        collection_name: Name of the collection
        vector_size: Dimension of vectors (default: 768 for text-embedding-3-small)
        recreate: If True, delete and recreate collection
        hybrid: If True, create collection with named vectors (dense + sparse) for hybrid search

    Returns:
        True if collection was created, False if it already existed
    """
    from qdrant_client.models import (
        Distance,
        VectorParams,
        SparseVectorParams,
        Modifier,
    )

    client = get_qdrant_client()

    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name in collection_names:
            if recreate:
                logger.info("Deleting existing collection", collection=collection_name)
                client.delete_collection(collection_name=collection_name)
            else:
                logger.info("Collection already exists", collection=collection_name)
                return False

        # Create collection with hybrid or simple config
        if hybrid:
            # Hybrid search: named dense + sparse vectors
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        modifier=Modifier.IDF,
                    ),
                },
            )

            logger.info(
                "Hybrid collection created",
                collection=collection_name,
                vector_size=vector_size,
                mode="hybrid (dense + sparse)",
            )
        else:
            # Simple: single unnamed dense vector
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

            logger.info(
                "Collection created",
                collection=collection_name,
                vector_size=vector_size,
                mode="dense only",
            )

        # Create payload indexes for filterable fields
        from qdrant_client.models import PayloadSchemaType
        for field in ("language", "domain", "doc_type"):
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        logger.info("Payload indexes created", collection=collection_name)

        return True

    except Exception as e:
        logger.error(
            "Failed to ensure collection",
            error=str(e),
            collection=collection_name,
        )
        raise


def check_document_indexed(
    collection_name: str,
    document_id: str,
) -> Dict[str, Any]:
    """
    Check if a document is indexed in Qdrant.

    Args:
        collection_name: Qdrant collection name
        document_id: Document UUID

    Returns:
        Dict with:
            - indexed: True if document has vectors
            - chunk_count: Number of chunks found
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_qdrant_client()

    try:
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(document_id)),
                )
            ]
        )

        # Count points with this document_id
        result = client.count(
            collection_name=collection_name,
            count_filter=filter_condition,
            exact=True,
        )

        return {
            "indexed": result.count > 0,
            "chunk_count": result.count,
        }

    except Exception as e:
        logger.error(
            "Failed to check document indexed",
            error=str(e),
            document_id=document_id,
        )
        return {
            "indexed": False,
            "chunk_count": 0,
            "error": str(e),
        }
