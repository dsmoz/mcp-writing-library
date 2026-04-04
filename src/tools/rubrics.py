"""
Rubric alignment checker: store and score document sections against evaluation criteria.
"""
from typing import Optional, List
from uuid import uuid4
import structlog

from src.sentry import capture_tool_error
from src.tools.collections import get_collection_names

logger = structlog.get_logger(__name__)

# framework is a free-form label — no closed enum. Use lowercase slugs (e.g. "lambda",
# "usaid", "undp", "oca-2025", "ds-moz-editorial"). list_rubric_frameworks() discovers all stored values.

try:
    from kbase.vector.sync_indexing import index_document
    from kbase.vector.sync_search import semantic_search
    from kbase.vector.sync_client import get_qdrant_client
except ImportError:
    index_document = None  # type: ignore
    semantic_search = None  # type: ignore
    get_qdrant_client = None  # type: ignore


def add_rubric_criterion(
    framework: str,
    section: str,
    criterion: str,
    weight: float = 1.0,
    red_flags: Optional[List[str]] = None,
) -> dict:
    """
    Store one evaluation criterion in the writing_rubrics Qdrant collection.

    Args:
        framework: Evaluation framework slug — any lowercase slug (e.g. "usaid", "undp", "lambda", "oca-2025")
        section: Section name (e.g. 'technical-approach', 'financial-management', 'methodology')
        criterion: The criterion description (what evaluators look for)
        weight: Relative importance 0.1–2.0 (default 1.0)
        red_flags: Phrases/patterns evaluators penalise (optional)

    Returns:
        {success, document_id, chunks_created, collection} on success
    """
    framework = framework.lower().strip()
    if not framework:
        return {"success": False, "error": "framework cannot be empty"}

    if not criterion or not criterion.strip():
        return {"success": False, "error": "criterion cannot be empty"}

    if not section or not section.strip():
        return {"success": False, "error": "section cannot be empty"}

    # Clamp weight to valid range
    weight = max(0.1, min(2.0, weight))

    if index_document is None:
        return {"success": False, "error": "kbase library is not available"}

    document_id = str(uuid4())
    collection = get_collection_names()["rubrics"]
    title = f"[{framework.upper()} | {section}] {criterion[:60]}"
    metadata = {
        "framework": framework,
        "section": section,
        "weight": weight,
        "red_flags": red_flags or [],
        "entry_type": "rubric_criterion",
    }

    try:
        point_ids = index_document(
            collection_name=collection,
            document_id=document_id,
            title=title,
            content=criterion,
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
        logger.error("Failed to add rubric criterion", error=str(e))
        capture_tool_error(e, tool_name="add_rubric_criterion", framework=framework)
        return {"success": False, "error": str(e)}


def score_against_rubric(
    text: str,
    framework: str,
    section: Optional[str] = None,
    top_k: int = 5,
    doc_context: str = None,
) -> dict:
    """
    Score a document section against all stored criteria for a given framework.

    Args:
        text: Document section to score
        framework: Donor name to filter criteria
        section: Optional section filter (e.g. "technical-approach")
        top_k: Number of criteria to match (default 5)
        doc_context: Optional free-text context about the document type (e.g. "annual report").
            Not stored — informational only.

    Returns:
        {success, framework, section, text_length, criteria_matched, overall_score,
         verdict, criteria, doc_context} on success
    """
    if doc_context:
        logger.debug("score_against_rubric context", doc_context=doc_context)
    framework = framework.lower().strip()
    if not framework:
        return {"success": False, "error": "framework cannot be empty"}

    if semantic_search is None:
        return {"success": False, "error": "kbase library is not available"}

    collection = get_collection_names()["rubrics"]

    filter_conditions: dict = {"framework": framework}
    if section:
        filter_conditions["section"] = section

    try:
        raw_results = semantic_search(
            collection_name=collection,
            query=text,
            limit=top_k,
            filter_conditions=filter_conditions,
        )
    except Exception as e:
        logger.error("score_against_rubric search failed", error=str(e))
        capture_tool_error(e, tool_name="score_against_rubric", framework=framework)
        return {"success": False, "error": str(e)}

    if not raw_results:
        section_info = f" and section '{section}'" if section else ""
        return {
            "success": False,
            "error": f"No rubric criteria found for framework '{framework}'{section_info}. Add criteria first with add_rubric_criterion().",
        }

    criteria = []
    raw_scores = []
    weights = []

    for result in raw_results:
        meta = result.get("metadata", {})
        weight = meta.get("weight", 1.0)
        raw_score = result.get("score", 0.0)
        weighted_score = raw_score * weight

        criteria.append({
            "criterion": result.get("text") or result.get("title", ""),
            "section": meta.get("section", ""),
            "score": round(raw_score, 4),
            "weighted_score": round(weighted_score, 4),
            "weight": weight,
            "red_flags": meta.get("red_flags", []),
            "document_id": result.get("document_id", ""),
        })
        raw_scores.append(raw_score)
        weights.append(weight)

    # Weighted average: sum(raw_score * weight) / sum(weights)
    overall_score = sum(s * w for s, w in zip(raw_scores, weights)) / sum(weights) if weights else 0.0
    overall_score = round(overall_score, 4)

    if overall_score >= 0.7:
        verdict = "strong"
    elif overall_score >= 0.5:
        verdict = "adequate"
    else:
        verdict = "weak"

    return {
        "success": True,
        "framework": framework,
        "section": section,
        "text_length": len(text),
        "criteria_matched": len(criteria),
        "overall_score": overall_score,
        "verdict": verdict,
        "criteria": criteria,
        "doc_context": doc_context,
    }


def list_rubric_frameworks() -> dict:
    """
    Return all frameworks that have at least one criterion stored.

    Returns:
        {success, frameworks: [{framework, criterion_count}], total_frameworks, total_criteria}
    """
    if get_qdrant_client is None:
        return {"success": False, "error": "kbase library is not available"}

    collection = get_collection_names()["rubrics"]

    try:
        client = get_qdrant_client()
        framework_counts: dict[str, int] = {}
        offset = None

        while True:
            results, next_offset = client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                payload = point.payload or {}
                framework = payload.get("framework")
                if framework:
                    framework_counts[framework] = framework_counts.get(framework, 0) + 1

            if next_offset is None:
                break
            offset = next_offset

        frameworks = sorted(
            [{"framework": d, "criterion_count": c} for d, c in framework_counts.items()],
            key=lambda x: x["framework"],
        )

        return {
            "success": True,
            "frameworks": frameworks,
            "total_frameworks": len(frameworks),
            "total_criteria": sum(framework_counts.values()),
        }

    except Exception as e:
        logger.error("list_rubric_frameworks failed", error=str(e))
        capture_tool_error(e, tool_name="list_rubric_frameworks")
        return {"success": False, "error": str(e)}
