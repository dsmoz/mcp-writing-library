"""
Contribution and moderation tools.

Flow:
    User calls contribute_term() / contribute_thesaurus_entry() / contribute_rubric() / contribute_template()
    → entry stored in writing_contributions with status="pending"
    → Telegram notification sent to admin

    Admin calls list_contributions(status="pending")
    → reviews entries

    Admin calls review_contribution(contribution_id, action="publish"|"reject")
    → publish: copies entry into target shared collection (writing_terms_shared, writing_thesaurus, etc.)
    → reject: updates status + stores reason

    search_terms() queries user's personal terms first, then writing_terms_shared.

Shared collections that accept contributions:
    terms      → writing_terms_shared
    thesaurus  → writing_thesaurus
    rubrics    → writing_rubrics
    templates  → writing_templates
"""
import json
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import structlog

from src.sentry import capture_tool_error
from src.tools.qdrant_errors import handle_qdrant_error

logger = structlog.get_logger(__name__)

VALID_TARGETS = ("terms", "thesaurus", "rubrics", "templates")

try:
    from kbase.vector.sync_indexing import index_document, ensure_collection
    from kbase.vector.sync_search import semantic_search
    from kbase.vector.sync_client import get_qdrant_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    _kbase_available = True
except ImportError:
    index_document = None        # type: ignore
    ensure_collection = None     # type: ignore
    semantic_search = None       # type: ignore
    get_qdrant_client = None     # type: ignore
    Filter = None                # type: ignore
    FieldCondition = None        # type: ignore
    MatchValue = None            # type: ignore
    _kbase_available = False


def _contributions_collection() -> str:
    from src.tools.collections import get_core_collection_names
    return get_core_collection_names()["contributions"]


def _target_collection(target: str) -> str:
    from src.tools.collections import get_core_collection_names
    mapping = {
        "terms": get_core_collection_names()["terms_shared"],
        "thesaurus": get_core_collection_names()["thesaurus"],
        "rubrics": get_core_collection_names()["rubrics"],
        "templates": get_core_collection_names()["templates"],
    }
    return mapping[target]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embed_text_for_contribution(target: str, payload: dict) -> str:
    """Build embed text from contribution payload based on target type."""
    if target == "terms":
        parts = [
            f"Preferred term: {payload.get('preferred', '')}",
            f"Avoid: {payload.get('avoid', '')}" if payload.get("avoid") else "",
            f"Why: {payload.get('why', '')}" if payload.get("why") else "",
            f"Domain: {payload.get('domain', '')}",
        ]
    elif target == "thesaurus":
        parts = [
            payload.get("headword", ""),
            payload.get("definition", ""),
            payload.get("why_avoid", ""),
        ]
    elif target == "rubrics":
        parts = [
            payload.get("criterion", ""),
            payload.get("framework", ""),
            payload.get("section", ""),
        ]
    elif target == "templates":
        sections = payload.get("sections", [])
        parts = [
            payload.get("framework", ""),
            payload.get("doc_type", ""),
        ] + [s.get("name", "") + ": " + s.get("description", "") for s in sections]
    else:
        parts = [str(v) for v in payload.values() if isinstance(v, str)]

    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Contribute
# ---------------------------------------------------------------------------

def contribute(
    target: str,
    payload: dict,
    contributed_by: str,
    note: str = "",
) -> dict:
    """
    Submit an entry to the moderation queue for a shared collection.

    Args:
        target: Target shared collection — terms|thesaurus|rubrics|templates
        payload: The entry fields (same structure as the target collection)
        contributed_by: client_id of the submitting user
        note: Optional note to the moderator

    Returns:
        {success, contribution_id, status: "pending"}
    """
    if not _kbase_available or index_document is None:
        return {"success": False, "error": "kbase not available"}

    if target not in VALID_TARGETS:
        return {
            "success": False,
            "error": f"Invalid target '{target}'. Must be one of: {VALID_TARGETS}",
        }
    if not payload:
        return {"success": False, "error": "payload cannot be empty"}
    if not contributed_by:
        return {"success": False, "error": "contributed_by cannot be empty"}

    contribution_id = str(uuid4())
    now = _now_iso()

    metadata = {
        "contribution_id": contribution_id,
        "contributed_by": contributed_by,
        "target_collection": target,
        "status": "pending",
        "submitted_at": now,
        "reviewed_at": None,
        "reviewed_by": None,
        "rejection_reason": None,
        "note": note,
        "entry_type": "contribution",
        # Store payload as JSON string (Qdrant payload values must be scalar/list)
        "payload_json": json.dumps(payload),
    }

    embed_text = _embed_text_for_contribution(target, payload)
    title = f"[CONTRIBUTION:{target.upper()}] {contributed_by} — {now[:10]}"

    try:
        index_document(
            collection_name=_contributions_collection(),
            document_id=contribution_id,
            title=title,
            content=embed_text or title,
            metadata=metadata,
            context_mode="metadata",
        )
        logger.info("Contribution submitted", contribution_id=contribution_id, target=target, by=contributed_by)
        return {
            "success": True,
            "contribution_id": contribution_id,
            "target": target,
            "status": "pending",
            "submitted_at": now,
        }
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="contribute", collection=_contributions_collection(), target=target, contributed_by=contributed_by)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("Failed to submit contribution", error=str(e))
        capture_tool_error(e, tool_name="contribute", target=target, contributed_by=contributed_by)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# List contributions
# ---------------------------------------------------------------------------

def list_contributions(
    status: Optional[str] = "pending",
    target: Optional[str] = None,
    contributed_by: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """
    List contributions from the moderation queue.

    Args:
        status: Filter by status — pending|published|rejected|all (default: pending)
        target: Filter by target collection — terms|thesaurus|rubrics|templates
        contributed_by: Filter to a specific contributor (used for mine=True pattern)
        limit: Max results (default 50)

    Returns:
        {success, contributions: [...], total}
    """
    if get_qdrant_client is None:
        return {"success": False, "error": "kbase not available"}

    try:
        client = get_qdrant_client()
        must_conditions = []

        if status and status != "all":
            must_conditions.append(
                FieldCondition(key="status", match=MatchValue(value=status))
            )
        if target:
            must_conditions.append(
                FieldCondition(key="target_collection", match=MatchValue(value=target))
            )
        if contributed_by:
            must_conditions.append(
                FieldCondition(key="contributed_by", match=MatchValue(value=contributed_by))
            )

        scroll_filter = Filter(must=must_conditions) if must_conditions else None

        results, _ = client.scroll(
            collection_name=_contributions_collection(),
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        contributions = []
        for point in results:
            p = point.payload or {}
            entry = {
                "contribution_id": p.get("contribution_id"),
                "contributed_by": p.get("contributed_by"),
                "target": p.get("target_collection"),
                "status": p.get("status"),
                "submitted_at": p.get("submitted_at"),
                "reviewed_at": p.get("reviewed_at"),
                "reviewed_by": p.get("reviewed_by"),
                "rejection_reason": p.get("rejection_reason"),
                "note": p.get("note", ""),
                "payload": json.loads(p.get("payload_json", "{}")),
            }
            contributions.append(entry)

        # Sort by submitted_at descending
        contributions.sort(key=lambda x: x.get("submitted_at") or "", reverse=True)

        return {"success": True, "contributions": contributions, "total": len(contributions)}

    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="list_contributions", collection=_contributions_collection())
        if qdrant_result is not None:
            qdrant_result["contributions"] = []
            return qdrant_result
        logger.error("list_contributions failed", error=str(e))
        capture_tool_error(e, tool_name="list_contributions")
        return {"success": False, "error": str(e), "contributions": []}


# ---------------------------------------------------------------------------
# Review (admin only)
# ---------------------------------------------------------------------------

def review_contribution(
    contribution_id: str,
    action: str,
    reviewed_by: str,
    rejection_reason: str = "",
) -> dict:
    """
    Publish or reject a pending contribution.

    On publish: copies the entry into the target shared collection.
    On reject: marks status=rejected with reason; entry kept for audit.

    Args:
        contribution_id: UUID of the contribution to review
        action: "publish" or "reject"
        reviewed_by: admin client_id performing the review
        rejection_reason: Required when action="reject"

    Returns:
        {success, contribution_id, action, target_collection}
    """
    if get_qdrant_client is None or index_document is None:
        return {"success": False, "error": "kbase not available"}

    if action not in ("publish", "reject"):
        return {"success": False, "error": "action must be 'publish' or 'reject'"}
    if action == "reject" and not rejection_reason:
        return {"success": False, "error": "rejection_reason is required when action='reject'"}

    try:
        client = get_qdrant_client()
        results, _ = client.scroll(
            collection_name=_contributions_collection(),
            scroll_filter=Filter(
                must=[FieldCondition(
                    key="contribution_id", match=MatchValue(value=contribution_id)
                )]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            return {"success": False, "error": f"Contribution '{contribution_id}' not found"}

        point = results[0]
        p = point.payload or {}

        if p.get("status") != "pending":
            return {
                "success": False,
                "error": f"Contribution is already '{p.get('status')}' — cannot review again",
            }

        target = p.get("target_collection")
        payload = json.loads(p.get("payload_json", "{}"))
        now = _now_iso()

        # Update the contribution record status
        client.set_payload(
            collection_name=_contributions_collection(),
            payload={
                "status": action + "ed",   # "published" | "rejected"
                "reviewed_at": now,
                "reviewed_by": reviewed_by,
                "rejection_reason": rejection_reason if action == "reject" else None,
            },
            points=[point.id],
        )

        if action == "publish":
            _publish_to_shared(target, payload, contribution_id)

        logger.info(
            "Contribution reviewed",
            contribution_id=contribution_id,
            action=action,
            by=reviewed_by,
        )
        return {
            "success": True,
            "contribution_id": contribution_id,
            "action": action,
            "target_collection": target,
            "reviewed_at": now,
        }

    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="review_contribution", collection=_contributions_collection(), contribution_id=contribution_id)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("review_contribution failed", error=str(e))
        capture_tool_error(e, tool_name="review_contribution", contribution_id=contribution_id)
        return {"success": False, "error": str(e)}


def _publish_to_shared(target: str, payload: dict, source_contribution_id: str) -> None:
    """Write the approved entry into the target shared collection."""
    from kbase.vector.sync_indexing import ensure_collection as _ensure
    from src.tools.collections import get_core_collection_names

    VECTOR_SIZE = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    collection_name = _target_collection(target)

    # Ensure collection exists
    try:
        _ensure(collection_name=collection_name, vector_size=VECTOR_SIZE, hybrid=True)
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="_publish_to_shared", collection=collection_name)
        if qdrant_result is None:
            logger.warning("ensure_collection failed before publish", collection=collection_name, error=str(e))

    document_id = str(uuid4())
    payload = dict(payload)
    payload["source_contribution_id"] = source_contribution_id
    payload["entry_type"] = payload.get("entry_type", target.rstrip("s"))  # "term", "thesaurus", etc.

    embed_text = _embed_text_for_contribution(target, payload)
    title = f"[{target.upper()}] {embed_text[:60]}"

    try:
        index_document(
            collection_name=collection_name,
            document_id=document_id,
            title=title,
            content=embed_text or title,
            metadata=payload,
            context_mode="metadata",
        )
        logger.info("Contribution published to shared collection", target=target, collection=collection_name)
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="_publish_to_shared", collection=collection_name, target=target, contribution_id=source_contribution_id)
        if qdrant_result is not None:
            logger.error("Qdrant error publishing contribution", error=qdrant_result["error"])
        else:
            logger.error("Failed to publish contribution to shared collection", error=str(e))
        capture_tool_error(e, tool_name="_publish_to_shared", target=target, contribution_id=source_contribution_id)
        raise


# ---------------------------------------------------------------------------
# Convenience wrappers with validated payloads
# ---------------------------------------------------------------------------

def contribute_term(
    preferred: str,
    contributed_by: str,
    avoid: str = "",
    domain: str = "general",
    language: str = "en",
    why: str = "",
    example_bad: str = "",
    example_good: str = "",
    note: str = "",
) -> dict:
    from src.tools.registry import VALID_DOMAINS, VALID_LANGUAGES_TERMS
    if not preferred.strip():
        return {"success": False, "error": "preferred cannot be empty"}
    if domain not in VALID_DOMAINS:
        return {"success": False, "error": f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}"}
    if language not in VALID_LANGUAGES_TERMS:
        return {"success": False, "error": f"Invalid language '{language}'. Must be one of: {sorted(VALID_LANGUAGES_TERMS)}"}

    payload = {
        "preferred": preferred, "avoid": avoid, "domain": domain,
        "language": language, "why": why,
        "example_bad": example_bad, "example_good": example_good,
        "entry_type": "term",
    }
    return contribute(target="terms", payload=payload, contributed_by=contributed_by, note=note)


def contribute_thesaurus_entry(
    headword: str,
    contributed_by: str,
    language: str = "en",
    domain: str = "general",
    definition: str = "",
    part_of_speech: str = "verb",
    register: str = "neutral",
    alternatives: Optional[list] = None,
    collocations: Optional[list] = None,
    why_avoid: str = "",
    example_bad: str = "",
    example_good: str = "",
    note: str = "",
) -> dict:
    if not headword.strip():
        return {"success": False, "error": "headword cannot be empty"}

    payload = {
        "headword": headword, "language": language, "domain": domain,
        "definition": definition, "part_of_speech": part_of_speech,
        "register": register,
        "alternatives": json.dumps(alternatives or []),
        "collocations": json.dumps(collocations or []),
        "why_avoid": why_avoid, "example_bad": example_bad, "example_good": example_good,
        "source": "contributed",
        "entry_type": "thesaurus",
    }
    return contribute(target="thesaurus", payload=payload, contributed_by=contributed_by, note=note)


def contribute_rubric(
    framework: str,
    section: str,
    criterion: str,
    contributed_by: str,
    weight: float = 1.0,
    red_flags: Optional[list] = None,
    note: str = "",
) -> dict:
    if not framework.strip():
        return {"success": False, "error": "framework cannot be empty"}
    if not criterion.strip():
        return {"success": False, "error": "criterion cannot be empty"}

    payload = {
        "framework": framework.lower().strip(),
        "section": section, "criterion": criterion,
        "weight": weight, "red_flags": red_flags or [],
        "entry_type": "rubric_criterion",
    }
    return contribute(target="rubrics", payload=payload, contributed_by=contributed_by, note=note)


def contribute_template(
    framework: str,
    doc_type: str,
    sections: list,
    contributed_by: str,
    note: str = "",
) -> dict:
    from src.tools.registry import VALID_DOC_TYPES
    if not framework.strip():
        return {"success": False, "error": "framework cannot be empty"}
    if doc_type not in VALID_DOC_TYPES:
        return {"success": False, "error": f"Invalid doc_type '{doc_type}'. Must be one of: {sorted(VALID_DOC_TYPES)}"}
    if not sections:
        return {"success": False, "error": "sections cannot be empty"}

    payload = {
        "framework": framework.lower().strip(),
        "doc_type": doc_type,
        "sections": sections,
        "section_count": len(sections),
        "entry_type": "template",
    }
    return contribute(target="templates", payload=payload, contributed_by=contributed_by, note=note)
