"""
Writing passages tool: store and search exemplary writing passages.
"""
from typing import Optional, List
from uuid import uuid4
import structlog

from src.tools.collections import get_collection_names
from src.tools.styles import VALID_STYLES

logger = structlog.get_logger(__name__)

VALID_DOC_TYPES = {
    "executive-summary", "concept-note", "policy-brief",
    "report", "email", "general"
}
VALID_LANGUAGES = {"en", "pt"}
VALID_DOMAINS = {"srhr", "governance", "climate", "general", "m-and-e"}

# Imported here so tests can patch src.tools.passages.index_document
try:
    from kbase.vector.sync_indexing import index_document
    from kbase.vector.sync_search import semantic_search
except ImportError:
    index_document = None  # type: ignore
    semantic_search = None  # type: ignore


def add_passage(
    text: str,
    doc_type: str = "general",
    language: str = "en",
    domain: str = "general",
    quality_notes: str = "",
    tags: Optional[List[str]] = None,
    source: str = "manual",
    style: Optional[List[str]] = None,
) -> dict:
    """Store an exemplary writing passage in the writing_passages collection."""
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
    collection = get_collection_names()["passages"]
    title = f"[{doc_type.upper()} | {language.upper()}] {text[:60]}..."

    metadata = {
        "doc_type": doc_type,
        "language": language,
        "domain": domain,
        "quality_notes": quality_notes,
        "tags": tags or [],
        "source": source,
        "entry_type": "passage",
        "style": style,
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
        return {
            "success": True,
            "document_id": document_id,
            "chunks_created": len(point_ids),
            "collection": collection,
            "warnings": warnings,
        }
    except Exception as e:
        logger.error("Failed to add passage", error=str(e))
        return {"success": False, "error": str(e)}


def search_passages(
    query: str,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    style: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """Search for exemplary writing passages by semantic similarity."""
    collection = get_collection_names()["passages"]

    filter_conditions = {}
    if doc_type:
        filter_conditions["doc_type"] = doc_type
    if language:
        filter_conditions["language"] = language
    if domain:
        filter_conditions["domain"] = domain

    # Over-fetch when style filtering is needed (post-filter reduces result count)
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
            results.append({
                "score": round(r["score"], 4),
                "text": r.get("text", ""),
                "title": r.get("title", ""),
                "doc_type": r.get("metadata", {}).get("doc_type"),
                "language": r.get("metadata", {}).get("language"),
                "domain": r.get("metadata", {}).get("domain"),
                "quality_notes": r.get("metadata", {}).get("quality_notes"),
                "tags": r.get("metadata", {}).get("tags", []),
                "style": r.get("metadata", {}).get("style", []),
                "source": r.get("metadata", {}).get("source"),
                "document_id": r.get("document_id"),
            })

        # Post-filter by style — kbase-core uses MatchValue which cannot match list fields
        if style:
            results = [r for r in results if style in r.get("style", [])]

        results = results[:top_k]

        return {"success": True, "results": results, "total": len(results)}
    except Exception as e:
        logger.error("Passage search failed", error=str(e))
        return {"success": False, "error": str(e), "results": []}
