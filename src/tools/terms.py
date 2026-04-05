"""
Terminology dictionary tool: store and search consultant vocabulary entries.
"""
from typing import Optional
from uuid import uuid4
import structlog

from src.sentry import capture_tool_error
from src.tools.collections import get_collection_names
from src.tools.registry import VALID_DOMAINS, VALID_LANGUAGES_TERMS as VALID_LANGUAGES

logger = structlog.get_logger(__name__)

# Module-level imports so tests can patch src.tools.terms.*
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


def add_term(
    preferred: str,
    avoid: str = "",
    domain: str = "general",
    language: str = "en",
    why: str = "",
    example_bad: str = "",
    example_good: str = "",
    client_id: str = "default",
) -> dict:
    """Store a terminology entry in the user's writing_terms collection."""
    if not preferred or not preferred.strip():
        return {"success": False, "error": "preferred term cannot be empty"}
    if domain not in VALID_DOMAINS:
        return {
            "success": False,
            "error": f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}",
        }

    document_id = str(uuid4())
    collection = get_collection_names(client_id)["terms"]

    content_parts = [
        f"Preferred term: {preferred}",
        f"Avoid: {avoid}" if avoid else "",
        f"Why: {why}" if why else "",
        f"Bad example: {example_bad}" if example_bad else "",
        f"Good example: {example_good}" if example_good else "",
        f"Domain: {domain}",
        f"Language: {language}",
    ]
    content = "\n".join(p for p in content_parts if p)

    metadata = {
        "client_id": client_id,
        "preferred": preferred,
        "avoid": avoid,
        "domain": domain,
        "language": language,
        "why": why,
        "example_bad": example_bad,
        "example_good": example_good,
        "entry_type": "term",
    }

    try:
        point_ids = index_document(
            collection_name=collection,
            document_id=document_id,
            title=preferred,
            content=content,
            metadata=metadata,
            context_mode="metadata",
        )
        return {
            "success": True,
            "document_id": document_id,
            "chunks_created": len(point_ids),
            "collection": collection,
        }
    except Exception as e:
        logger.error("Failed to add term", error=str(e))
        capture_tool_error(e, tool_name="add_term", client_id=client_id)
        return {"success": False, "error": str(e)}


def search_terms(
    query: str,
    domain: Optional[str] = None,
    language: Optional[str] = None,
    top_k: int = 8,
    client_id: str = "default",
    include_shared: bool = True,
) -> dict:
    """Search the user's terminology dictionary and (optionally) the shared published pool.

    Personal terms take precedence — if the same preferred term appears in both, the personal
    entry is returned and the shared duplicate is dropped.
    """
    from src.tools.collections import get_core_collection_names

    filter_conditions: dict = {}
    if domain:
        filter_conditions["domain"] = domain
    if language:
        filter_conditions["language"] = language

    def _fetch(collection: str, source: str) -> list:
        try:
            raw = semantic_search(
                collection_name=collection,
                query=query,
                limit=top_k,
                filter_conditions=filter_conditions if filter_conditions else None,
            )
            out = []
            for r in raw:
                meta = r.get("metadata", {})
                out.append({
                    "score": round(r["score"], 4),
                    "preferred": meta.get("preferred", r.get("title", "")),
                    "avoid": meta.get("avoid", ""),
                    "domain": meta.get("domain"),
                    "language": meta.get("language"),
                    "why": meta.get("why", ""),
                    "example_bad": meta.get("example_bad", ""),
                    "example_good": meta.get("example_good", ""),
                    "document_id": r.get("document_id"),
                    "source": source,
                })
            return out
        except Exception:
            return []

    personal = _fetch(get_collection_names(client_id)["terms"], "personal")

    shared: list = []
    if include_shared:
        shared_col = get_core_collection_names().get("terms_shared")
        if shared_col:
            shared = _fetch(shared_col, "shared")

    # Merge: personal wins; deduplicate by preferred term (case-insensitive)
    seen_preferred: set = {r["preferred"].lower() for r in personal}
    merged = list(personal)
    for r in shared:
        if r["preferred"].lower() not in seen_preferred:
            merged.append(r)
            seen_preferred.add(r["preferred"].lower())

    # Re-sort by score and cap at top_k
    merged.sort(key=lambda x: x["score"], reverse=True)
    merged = merged[:top_k]

    return {"success": True, "results": merged, "total": len(merged)}


def delete_term(document_id: str, client_id: str = "default") -> dict:
    """Delete a term from the user's writing_terms collection by document_id."""
    collection = get_collection_names(client_id)["terms"]
    try:
        check_result = check_document_indexed(
            collection_name=collection,
            document_id=document_id,
        )
        if not check_result.get("indexed"):
            return {
                "success": False,
                "error": f"No term found with document_id '{document_id}'",
            }

        delete_document_vectors(collection_name=collection, document_id=document_id)
        return {"success": True, "document_id": document_id, "deleted": True}
    except Exception as e:
        logger.error("Failed to delete term", error=str(e), document_id=document_id)
        capture_tool_error(e, tool_name="delete_term", document_id=document_id)
        return {"success": False, "error": str(e)}


def batch_add_terms(items: list, client_id: str = "default") -> dict:
    """Add multiple terminology entries in a single call. Never raises; collects per-item errors."""
    results = []
    succeeded = 0
    failed = 0

    for i, item in enumerate(items):
        if not isinstance(item, dict) or not item.get("preferred"):
            failed += 1
            results.append({
                "index": i,
                "success": False,
                "error": "item must be a dict with a non-empty 'preferred' field",
            })
            continue

        result = add_term(
            preferred=item["preferred"],
            avoid=item.get("avoid", ""),
            domain=item.get("domain", "general"),
            language=item.get("language", "en"),
            why=item.get("why", ""),
            example_bad=item.get("example_bad", ""),
            example_good=item.get("example_good", ""),
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


def update_term(
    document_id: str,
    preferred: Optional[str] = None,
    avoid: Optional[str] = None,
    domain: Optional[str] = None,
    language: Optional[str] = None,
    why: Optional[str] = None,
    example_bad: Optional[str] = None,
    example_good: Optional[str] = None,
    client_id: str = "default",
) -> dict:
    """Update a term by deleting and re-indexing with merged metadata."""
    updated_fields = [
        f for f, v in {
            "preferred": preferred, "avoid": avoid, "domain": domain,
            "language": language, "why": why,
            "example_bad": example_bad, "example_good": example_good,
        }.items() if v is not None
    ]
    if not updated_fields:
        return {"success": False, "error": "At least one field must be provided to update"}

    # Validate provided fields before touching Qdrant
    if domain is not None and domain not in VALID_DOMAINS:
        return {
            "success": False,
            "error": f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}",
        }
    if language is not None and language not in VALID_LANGUAGES:
        return {
            "success": False,
            "error": f"Invalid language '{language}'. Must be one of: {sorted(VALID_LANGUAGES)}",
        }

    collection = get_collection_names(client_id)["terms"]

    try:
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
                "error": f"No term found with document_id '{document_id}'",
            }

        existing_payload = results[0].payload or {}

        # Merge: new values override existing
        merged_preferred = preferred if preferred is not None else existing_payload.get("preferred", "")
        merged_avoid = avoid if avoid is not None else existing_payload.get("avoid", "")
        merged_domain = domain if domain is not None else existing_payload.get("domain", "general")
        merged_language = language if language is not None else existing_payload.get("language", "en")
        merged_why = why if why is not None else existing_payload.get("why", "")
        merged_example_bad = example_bad if example_bad is not None else existing_payload.get("example_bad", "")
        merged_example_good = example_good if example_good is not None else existing_payload.get("example_good", "")

        if not merged_preferred or not merged_preferred.strip():
            return {"success": False, "error": "preferred term cannot be empty"}

        # NOTE: This operation is non-atomic. If re-indexing fails after deletion,
        # the document will be lost. The original payload is preserved below and
        # included in the error response so the caller can recover.
        delete_document_vectors(collection_name=collection, document_id=document_id)

        # Re-index with same document_id using add_term logic
        content_parts = [
            f"Preferred term: {merged_preferred}",
            f"Avoid: {merged_avoid}" if merged_avoid else "",
            f"Why: {merged_why}" if merged_why else "",
            f"Bad example: {merged_example_bad}" if merged_example_bad else "",
            f"Good example: {merged_example_good}" if merged_example_good else "",
            f"Domain: {merged_domain}",
            f"Language: {merged_language}",
        ]
        content = "\n".join(p for p in content_parts if p)

        metadata = {
            "client_id": client_id,
            "preferred": merged_preferred,
            "avoid": merged_avoid,
            "domain": merged_domain,
            "language": merged_language,
            "why": merged_why,
            "example_bad": merged_example_bad,
            "example_good": merged_example_good,
            "entry_type": "term",
        }

        try:
            point_ids = index_document(
                collection_name=collection,
                document_id=document_id,
                title=merged_preferred,
                content=content,
                metadata=metadata,
                context_mode="metadata",
            )
        except Exception as index_error:
            logger.error(
                "Re-index failed after deletion",
                error=str(index_error),
                document_id=document_id,
            )
            capture_tool_error(index_error, tool_name="update_term", phase="re-index", document_id=document_id)
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
        }

    except Exception as e:
        logger.error("Failed to update term", error=str(e), document_id=document_id)
        capture_tool_error(e, tool_name="update_term", document_id=document_id)
        return {"success": False, "error": str(e)}
