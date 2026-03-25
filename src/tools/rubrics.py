"""
Donor rubric alignment checker: store and score proposal text against donor evaluation criteria.
"""
from typing import Optional, List
from uuid import uuid4
import structlog

from src.tools.collections import get_collection_names

logger = structlog.get_logger(__name__)

VALID_DONORS = {"usaid", "undp", "global-fund", "eu", "general"}

try:
    from kbase.vector.sync_indexing import index_document
    from kbase.vector.sync_search import semantic_search
    from kbase.vector.sync_client import get_qdrant_client
except ImportError:
    index_document = None  # type: ignore
    semantic_search = None  # type: ignore
    get_qdrant_client = None  # type: ignore


def add_rubric_criterion(
    donor: str,
    section: str,
    criterion: str,
    weight: float = 1.0,
    red_flags: Optional[List[str]] = None,
) -> dict:
    """
    Store one evaluation criterion in the writing_rubrics Qdrant collection.

    Args:
        donor: Donor name — must be one of: usaid, undp, global-fund, eu, general
        section: Proposal section name (e.g. "technical-approach", "sustainability")
        criterion: The criterion description (what evaluators look for)
        weight: Relative importance 0.1–2.0 (default 1.0)
        red_flags: Phrases/patterns evaluators penalise (optional)

    Returns:
        {success, document_id, chunks_created, collection} on success
    """
    donor = donor.lower()
    if donor not in VALID_DONORS:
        return {
            "success": False,
            "error": f"Invalid donor '{donor}'. Must be one of: {sorted(VALID_DONORS)}",
        }

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
    title = f"[{donor.upper()} | {section}] {criterion[:60]}"
    metadata = {
        "donor": donor,
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
        return {"success": False, "error": str(e)}


def score_against_rubric(
    text: str,
    donor: str,
    section: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Score a proposal text against all stored criteria for a given donor.

    Args:
        text: Proposal text to score
        donor: Donor name to filter criteria
        section: Optional section filter (e.g. "technical-approach")
        top_k: Number of criteria to match (default 5)

    Returns:
        {success, donor, section, text_length, criteria_matched, overall_score,
         verdict, criteria} on success
    """
    donor = donor.lower()
    if donor not in VALID_DONORS:
        return {
            "success": False,
            "error": f"Invalid donor '{donor}'. Must be one of: {sorted(VALID_DONORS)}",
        }

    if semantic_search is None:
        return {"success": False, "error": "kbase library is not available"}

    collection = get_collection_names()["rubrics"]

    filter_conditions: dict = {"donor": donor}
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
        return {"success": False, "error": str(e)}

    if not raw_results:
        section_info = f" and section '{section}'" if section else ""
        return {
            "success": False,
            "error": f"No rubric criteria found for donor '{donor}'{section_info}. Add criteria first with add_rubric_criterion().",
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
        "donor": donor,
        "section": section,
        "text_length": len(text),
        "criteria_matched": len(criteria),
        "overall_score": overall_score,
        "verdict": verdict,
        "criteria": criteria,
    }


def list_rubric_donors() -> dict:
    """
    Return all donors that have at least one criterion stored.

    Returns:
        {success, donors: [{donor, criterion_count}], total_donors, total_criteria}
    """
    if get_qdrant_client is None:
        return {"success": False, "error": "kbase library is not available"}

    collection = get_collection_names()["rubrics"]

    try:
        client = get_qdrant_client()
        donor_counts: dict[str, int] = {}
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
                donor = payload.get("donor")
                if donor:
                    donor_counts[donor] = donor_counts.get(donor, 0) + 1

            if next_offset is None:
                break
            offset = next_offset

        donors = sorted(
            [{"donor": d, "criterion_count": c} for d, c in donor_counts.items()],
            key=lambda x: x["donor"],
        )

        return {
            "success": True,
            "donors": donors,
            "total_donors": len(donors),
            "total_criteria": sum(donor_counts.values()),
        }

    except Exception as e:
        logger.error("list_rubric_donors failed", error=str(e))
        return {"success": False, "error": str(e)}
