"""
Writing passages tool: store and search exemplary writing passages.
"""
from typing import Optional, List
from uuid import uuid4
import structlog

from src.sentry import capture_tool_error
from src.tools.collections import get_collection_names
from src.tools.qdrant_errors import handle_qdrant_error
from src.tools.registry import VALID_DOC_TYPES, VALID_DOMAINS, VALID_LANGUAGES
from src.tools.styles import VALID_STYLES

logger = structlog.get_logger(__name__)

# Imported here so tests can patch src.tools.passages.*
try:
    from kbase.vector.sync_indexing import index_document, delete_document_vectors, check_document_indexed
    from kbase.vector.sync_search import semantic_search
    from kbase.vector.sync_client import get_qdrant_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue
except ImportError:
    index_document = None  # type: ignore
    delete_document_vectors = None  # type: ignore
    check_document_indexed = None  # type: ignore
    semantic_search = None  # type: ignore
    get_qdrant_client = None  # type: ignore
    Filter = None  # type: ignore
    FieldCondition = None  # type: ignore
    MatchValue = None  # type: ignore


def add_passage(
    text: str,
    doc_type: str = "general",
    language: str = "en",
    domain: str = "general",
    quality_notes: str = "",
    tags: Optional[List[str]] = None,
    source: str = "manual",
    style: Optional[List[str]] = None,
    rubric_section: Optional[str] = None,
    client_id: str = "default",
) -> dict:
    """Store an exemplary writing passage in the user's writing_passages collection."""
    if doc_type not in VALID_DOC_TYPES:
        return {
            "success": False,
            "error": f"Invalid doc_type '{doc_type}'. Must be one of: {sorted(VALID_DOC_TYPES)}",
        }
    if language not in VALID_LANGUAGES:
        return {
            "success": False,
            "error": f"Invalid language '{language}'. Must be one of: {sorted(VALID_LANGUAGES)}",
        }
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    style = style or []
    unknown_styles = [s for s in style if s not in VALID_STYLES]
    warnings = []
    if unknown_styles:
        warnings.append(
            f"Unknown style(s) ignored: {unknown_styles}. Valid styles: {sorted(VALID_STYLES)}"
        )
        style = [s for s in style if s in VALID_STYLES]

    document_id = str(uuid4())
    collection = get_collection_names(client_id)["passages"]
    title = f"[{doc_type.upper()} | {language.upper()}] {text[:60]}..."

    metadata = {
        "client_id": client_id,
        "doc_type": doc_type,
        "language": language,
        "domain": domain,
        "quality_notes": quality_notes,
        "tags": tags or [],
        "source": source,
        "entry_type": "passage",
        "style": style,
    }
    if rubric_section is not None:
        metadata["rubric_section"] = rubric_section

    try:
        point_ids = index_document(
            collection_name=collection,
            document_id=document_id,
            title=title,
            content=text,
            metadata=metadata,
            context_mode="metadata",
        )
        return {
            "success": True,
            "document_id": document_id,
            "chunks_created": len(point_ids),
            "collection": collection,
            "warnings": warnings,
        }
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="add_passage", collection=collection, client_id=client_id)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("Failed to add passage", error=str(e))
        capture_tool_error(e, tool_name="add_passage", client_id=client_id)
        return {"success": False, "error": str(e)}


def search_passages(
    query: str,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    style: Optional[str] = None,
    rubric_section: Optional[str] = None,
    top_k: int = 5,
    client_id: str = "default",
) -> dict:
    """Search for exemplary writing passages by semantic similarity."""
    collection = get_collection_names(client_id)["passages"]

    filter_conditions = {}
    if doc_type:
        filter_conditions["doc_type"] = doc_type
    if language:
        filter_conditions["language"] = language
    if domain:
        filter_conditions["domain"] = domain
    if rubric_section:
        filter_conditions["rubric_section"] = rubric_section

    # Over-fetch when style post-filtering is needed (reduces result count)
    fetch_k = top_k * 3 if style else top_k

    try:
        raw_results = semantic_search(
            collection_name=collection,
            query=query,
            limit=fetch_k,
            filter_conditions=filter_conditions if filter_conditions else None,
        )

        results = []
        for r in raw_results:
            meta = r.get("metadata", {})
            entry = {
                "score": round(r["score"], 4),
                "text": r.get("text", ""),
                "title": r.get("title", ""),
                "doc_type": meta.get("doc_type"),
                "language": meta.get("language"),
                "domain": meta.get("domain"),
                "quality_notes": meta.get("quality_notes"),
                "tags": meta.get("tags", []),
                "style": meta.get("style", []),
                "source": meta.get("source"),
                "document_id": r.get("document_id"),
            }
            if meta.get("rubric_section"):
                entry["rubric_section"] = meta["rubric_section"]
            results.append(entry)

        # Post-filter by style — kbase-core uses MatchValue which cannot match list fields
        if style:
            results = [r for r in results if style in r.get("style", [])]

        results = results[:top_k]

        return {"success": True, "results": results, "total": len(results)}
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="search_passages", collection=collection, client_id=client_id)
        if qdrant_result is not None:
            qdrant_result["results"] = []
            return qdrant_result
        logger.error("Passage search failed", error=str(e))
        capture_tool_error(e, tool_name="search_passages", client_id=client_id)
        return {"success": False, "error": str(e), "results": []}


def delete_passage(document_id: str, client_id: str = "default") -> dict:
    """Delete a passage from the user's writing_passages collection by document_id."""
    collection = get_collection_names(client_id)["passages"]
    try:
        check_result = check_document_indexed(
            collection_name=collection,
            document_id=document_id,
        )
        if not check_result.get("indexed"):
            return {
                "success": False,
                "error": f"No passage found with document_id '{document_id}'",
            }

        delete_document_vectors(collection_name=collection, document_id=document_id)
        return {"success": True, "document_id": document_id, "deleted": True}
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="delete_passage", collection=collection, document_id=document_id)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("Failed to delete passage", error=str(e), document_id=document_id)
        capture_tool_error(e, tool_name="delete_passage", document_id=document_id)
        return {"success": False, "error": str(e)}


def batch_add_passages(items: list, client_id: str = "default") -> dict:
    """Add multiple writing passages in a single call. Never raises; collects per-item errors."""
    results = []
    succeeded = 0
    failed = 0

    for i, item in enumerate(items):
        if not isinstance(item, dict) or not item.get("text"):
            failed += 1
            results.append({
                "index": i,
                "success": False,
                "error": "item must be a dict with a non-empty 'text' field",
            })
            continue

        result = add_passage(
            text=item["text"],
            doc_type=item.get("doc_type", "general"),
            language=item.get("language", "en"),
            domain=item.get("domain", "general"),
            quality_notes=item.get("quality_notes", ""),
            tags=item.get("tags"),
            source=item.get("source", "manual"),
            style=item.get("style"),
            rubric_section=item.get("rubric_section"),
            client_id=client_id,
        )
        result["index"] = i
        results.append(result)
        if result.get("success"):
            succeeded += 1
        else:
            failed += 1

    total = len(items)
    return {
        "success": True,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }


def record_correction(
    original: str,
    corrected: str,
    issue_type: str,
    doc_type: str = "general",
    language: str = "en",
    domain: str = "general",
    source: str = "manual",
    client_id: str = "default",
) -> dict:
    """Store a before/after correction pair in the passages collection.

    The original is tagged 'ai-corrected' (negative example).
    The corrected version is tagged 'human-corrected' (positive example).
    Both are stored with the same correction_id so they can be retrieved as a pair.
    """
    if doc_type not in VALID_DOC_TYPES:
        return {
            "success": False,
            "error": f"Invalid doc_type '{doc_type}'. Must be one of: {sorted(VALID_DOC_TYPES)}",
        }
    if language not in VALID_LANGUAGES:
        return {
            "success": False,
            "error": f"Invalid language '{language}'. Must be one of: {sorted(VALID_LANGUAGES)}",
        }
    if not original or not original.strip():
        return {"success": False, "error": "original cannot be empty"}
    if not corrected or not corrected.strip():
        return {"success": False, "error": "corrected cannot be empty"}

    from uuid import uuid4
    correction_id = str(uuid4())
    collection = get_collection_names(client_id)["passages"]

    results = {}
    for role, text, style_tag in [
        ("original", original, "ai-corrected"),
        ("corrected", corrected, "human-corrected"),
    ]:
        document_id = str(uuid4())
        title = f"[CORRECTION:{role.upper()} | {doc_type.upper()} | {language.upper()}] {text[:60]}..."
        metadata = {
            "client_id": client_id,
            "doc_type": doc_type,
            "language": language,
            "domain": domain,
            "quality_notes": f"Correction {role}: {issue_type}",
            "tags": ["correction", role, issue_type],
            "source": source,
            "entry_type": "correction",
            "style": [style_tag],
            "correction_id": correction_id,
            "correction_role": role,
            "issue_type": issue_type,
        }
        try:
            point_ids = index_document(
                collection_name=collection,
                document_id=document_id,
                title=title,
                content=text,
                metadata=metadata,
                context_mode="metadata",
            )
            results[role] = {
                "success": True,
                "document_id": document_id,
                "chunks_created": len(point_ids),
            }
        except Exception as e:
            qdrant_result = handle_qdrant_error(e, tool_name="record_correction", collection=collection, role=role, client_id=client_id)
            if qdrant_result is not None:
                results[role] = qdrant_result
            else:
                logger.error("Failed to record correction", role=role, error=str(e))
                capture_tool_error(e, tool_name="record_correction", role=role, client_id=client_id)
                results[role] = {"success": False, "error": str(e)}

    overall_success = all(v.get("success") for v in results.values())
    return {
        "success": overall_success,
        "correction_id": correction_id,
        "collection": collection,
        "original": results.get("original", {}),
        "corrected": results.get("corrected", {}),
    }


def update_passage(
    document_id: str,
    text: Optional[str] = None,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    quality_notes: Optional[str] = None,
    tags: Optional[List[str]] = None,
    source: Optional[str] = None,
    style: Optional[List[str]] = None,
    client_id: str = "default",
) -> dict:
    """Update a passage by deleting and re-indexing with merged metadata."""
    updated_fields = [
        f for f, v in {
            "text": text, "doc_type": doc_type, "language": language,
            "domain": domain, "quality_notes": quality_notes, "tags": tags,
            "source": source, "style": style,
        }.items() if v is not None
    ]
    if not updated_fields:
        return {"success": False, "error": "At least one field must be provided to update"}

    # Validate provided fields before touching Qdrant
    if doc_type is not None and doc_type not in VALID_DOC_TYPES:
        return {
            "success": False,
            "error": f"Invalid doc_type '{doc_type}'. Must be one of: {sorted(VALID_DOC_TYPES)}",
        }
    if language is not None and language not in VALID_LANGUAGES:
        return {
            "success": False,
            "error": f"Invalid language '{language}'. Must be one of: {sorted(VALID_LANGUAGES)}",
        }
    if domain is not None and domain not in VALID_DOMAINS:
        return {
            "success": False,
            "error": f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}",
        }

    collection = get_collection_names(client_id)["passages"]

    try:
        # Fetch current document to merge metadata
        client = get_qdrant_client()
        filter_condition = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=str(document_id)))]
        )
        results, _ = client.scroll(
            collection_name=collection,
            scroll_filter=filter_condition,
            limit=1,
            with_payload=True,
        )
        if not results:
            return {
                "success": False,
                "error": f"No passage found with document_id '{document_id}'",
            }

        existing_payload = results[0].payload or {}

        # Merge: new values override existing
        merged_text = text if text is not None else existing_payload.get("text", "")
        merged_doc_type = doc_type if doc_type is not None else existing_payload.get("doc_type", "general")
        merged_language = language if language is not None else existing_payload.get("language", "en")
        merged_domain = domain if domain is not None else existing_payload.get("domain", "general")
        merged_quality_notes = quality_notes if quality_notes is not None else existing_payload.get("quality_notes", "")
        merged_tags = tags if tags is not None else existing_payload.get("tags", [])
        merged_source = source if source is not None else existing_payload.get("source", "manual")
        merged_style = style if style is not None else existing_payload.get("style", [])

        if not merged_text or not merged_text.strip():
            return {"success": False, "error": "text cannot be empty"}

        # NOTE: This operation is non-atomic. If re-indexing fails after deletion,
        # the document will be lost. The original payload is preserved below and
        # included in the error response so the caller can recover.
        delete_document_vectors(collection_name=collection, document_id=document_id)

        # Re-index with same document_id using add_passage logic
        style_list = merged_style or []
        unknown_styles = [s for s in style_list if s not in VALID_STYLES]
        warnings = []
        if unknown_styles:
            warnings.append(
                f"Unknown style(s) ignored: {unknown_styles}. Valid styles: {sorted(VALID_STYLES)}"
            )
            style_list = [s for s in style_list if s in VALID_STYLES]

        title = f"[{merged_doc_type.upper()} | {merged_language.upper()}] {merged_text[:60]}..."
        metadata = {
            "client_id": client_id,
            "doc_type": merged_doc_type,
            "language": merged_language,
            "domain": merged_domain,
            "quality_notes": merged_quality_notes,
            "tags": merged_tags,
            "source": merged_source,
            "entry_type": "passage",
            "style": style_list,
        }

        try:
            point_ids = index_document(
                collection_name=collection,
                document_id=document_id,
                title=title,
                content=merged_text,
                metadata=metadata,
                context_mode="metadata",
            )
        except Exception as index_error:
            logger.error(
                "Re-index failed after deletion",
                error=str(index_error),
                document_id=document_id,
            )
            capture_tool_error(index_error, tool_name="update_passage", phase="re-index", document_id=document_id)
            return {
                "success": False,
                "error": f"Re-index failed after deletion: {index_error}. Original payload preserved for recovery.",
                "document_id": document_id,
                "original_payload": existing_payload,
            }

        return {
            "success": True,
            "document_id": document_id,
            "updated_fields": updated_fields,
            "chunks_created": len(point_ids),
            "warnings": warnings,
        }

    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="update_passage", collection=collection, document_id=document_id)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("Failed to update passage", error=str(e), document_id=document_id)
        capture_tool_error(e, tool_name="update_passage", document_id=document_id)
        return {"success": False, "error": str(e)}
