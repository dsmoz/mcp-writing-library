"""
Terminology dictionary tool: store and search consultant vocabulary entries.
"""
from typing import Optional
from uuid import uuid4
import structlog

from src.tools.collections import get_collection_names

logger = structlog.get_logger(__name__)

VALID_DOMAINS = {"srhr", "governance", "climate", "general", "m-and-e"}
VALID_LANGUAGES = {"en", "pt", "both"}

# Module-level imports so tests can patch src.tools.terms.index_document
try:
    from kbase.vector.sync_indexing import index_document
    from kbase.vector.sync_search import semantic_search
except ImportError:
    index_document = None  # type: ignore
    semantic_search = None  # type: ignore


def add_term(
    preferred: str,
    avoid: str = "",
    domain: str = "general",
    language: str = "en",
    why: str = "",
    example_bad: str = "",
    example_good: str = "",
) -> dict:
    """Store a terminology entry in the writing_terms collection."""
    if not preferred or not preferred.strip():
        return {"success": False, "error": "preferred term cannot be empty"}
    if domain not in VALID_DOMAINS:
        return {
            "success": False,
            "error": f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}",
        }

    document_id = str(uuid4())
    collection = get_collection_names()["terms"]

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
        return {"success": False, "error": str(e)}


def search_terms(
    query: str,
    domain: Optional[str] = None,
    language: Optional[str] = None,
    top_k: int = 8,
) -> dict:
    """Search the terminology dictionary for relevant entries."""
    collection = get_collection_names()["terms"]

    filter_conditions = {}
    if domain:
        filter_conditions["domain"] = domain
    if language:
        filter_conditions["language"] = language

    try:
        raw_results = semantic_search(
            collection_name=collection,
            query=query,
            limit=top_k,
            filter_conditions=filter_conditions if filter_conditions else None,
        )

        results = []
        for r in raw_results:
            meta = r.get("metadata", {})
            results.append({
                "score": round(r["score"], 4),
                "preferred": meta.get("preferred", r.get("title", "")),
                "avoid": meta.get("avoid", ""),
                "domain": meta.get("domain"),
                "language": meta.get("language"),
                "why": meta.get("why", ""),
                "example_bad": meta.get("example_bad", ""),
                "example_good": meta.get("example_good", ""),
                "document_id": r.get("document_id"),
            })

        return {"success": True, "results": results, "total": len(results)}
    except Exception as e:
        logger.error("Term search failed", error=str(e))
        return {"success": False, "error": str(e), "results": []}
