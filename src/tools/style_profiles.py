"""
Style profile tools: save and retrieve writing style profiles extracted from samples.

Style extraction is LLM-assisted: Claude analyses 2–5 writing samples in the conversation
and produces a structured profile. These tools handle persistence and retrieval in Qdrant.
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4
import structlog

from src.sentry import capture_tool_error
from src.tools.collections import ensure_user_collections_once
from src.tools.qdrant_errors import handle_qdrant_error
from src.tools.styles import VALID_STYLES
from src.tools.registry import VALID_CHANNELS

logger = structlog.get_logger(__name__)


def _style_profiles_collection(client_id: str = "default") -> str:
    from src.tools.collections import get_user_collection_names
    return get_user_collection_names(client_id)["style_profiles"]

try:
    from kbase.vector.sync_indexing import index_document, delete_document_vectors, check_document_indexed
    from kbase.vector.sync_search import semantic_search
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    import os
    _qdrant_available = True
except ImportError:
    index_document = None  # type: ignore
    delete_document_vectors = None  # type: ignore
    check_document_indexed = None  # type: ignore
    semantic_search = None  # type: ignore
    _qdrant_available = False


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    import os
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def save_style_profile(
    name: str,
    style_scores: dict,
    rules: list,
    anti_patterns: list,
    sample_excerpts: list,
    description: str = "",
    source_documents: Optional[list] = None,
    channel: Optional[str] = None,
    client_id: str = "default",
) -> dict:
    """
    Save a writing style profile extracted from writing samples.

    After analysing 2–5 writing samples against the 14 style dimensions,
    call this to persist the profile in Qdrant for future retrieval and search.

    Args:
        name: Unique profile name (e.g. "danilo-voice-pt", "lambda-proposals")
        style_scores: Dict mapping style labels to scores 0.0–1.0
                      (e.g. {"narrative": 0.8, "conversational": 0.7})
        rules: Writing rules inferred from the samples
               (e.g. ["Uses em-dashes for asides", "Avoids passive voice"])
        anti_patterns: Patterns absent or contrary to this style
                       (e.g. ["'leverage'", "Excessive nominalisations"])
        sample_excerpts: Short representative quotes from the writing samples
        description: Human-readable summary of the style
        source_documents: Names or titles of the writing samples analysed

    Returns:
        {success, name, document_id, warnings}
    """
    if not name or not name.strip():
        return {"success": False, "error": "name cannot be empty"}
    if not style_scores:
        return {"success": False, "error": "style_scores cannot be empty"}
    if not rules and not sample_excerpts:
        return {"success": False, "error": "provide at least one rule or sample_excerpt"}

    ensure_user_collections_once(client_id)

    # Validate channel
    warnings = []
    if channel is not None and channel not in VALID_CHANNELS:
        warnings.append(
            f"Unknown channel '{channel}'. Valid channels: {sorted(VALID_CHANNELS)}. "
            "Saving anyway — use list_style_profiles() to filter by channel."
        )

    # Validate style score keys
    unknown_styles = [k for k in style_scores if k not in VALID_STYLES]
    if unknown_styles:
        warnings.append(
            f"Unknown style key(s) in style_scores: {unknown_styles}. "
            f"Valid styles: {sorted(VALID_STYLES)}"
        )

    # Clamp score values to [0.0, 1.0]
    clamped = {k: max(0.0, min(1.0, float(v))) for k, v in style_scores.items()}
    if clamped != style_scores:
        warnings.append("Some style_scores values were clamped to [0.0, 1.0].")

    document_id = str(uuid4())
    created_at = datetime.utcnow().isoformat()

    payload = {
        "client_id": client_id,
        "name": name.strip(),
        "description": description,
        "style_scores": clamped,
        "rules": rules,
        "anti_patterns": anti_patterns,
        "sample_excerpts": sample_excerpts,
        "source_documents": source_documents or [],
        "channel": channel,
        "created_at": created_at,
        "entry_type": "style_profile",
    }

    # Build embed text from all descriptive fields
    embed_parts = []
    if description:
        embed_parts.append(description)
    if rules:
        embed_parts.extend(rules)
    if anti_patterns:
        embed_parts.extend(anti_patterns)
    if sample_excerpts:
        embed_parts.extend(sample_excerpts)
    embed_text = "\n".join(embed_parts)

    title = f"[STYLE PROFILE] {name}"

    try:
        point_ids = index_document(
            collection_name=_style_profiles_collection(client_id),
            document_id=document_id,
            title=title,
            content=embed_text,
            metadata=payload,
            context_mode="metadata",
        )
        return {
            "success": True,
            "name": name,
            "document_id": document_id,
            "chunks_created": len(point_ids),
            "warnings": warnings,
        }
    except Exception as e:
        col = _style_profiles_collection(client_id)
        qdrant_result = handle_qdrant_error(e, tool_name="save_style_profile", collection=col, name=name, client_id=client_id)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("Failed to save style profile", name=name, error=str(e))
        capture_tool_error(e, tool_name="save_style_profile", name=name, client_id=client_id)
        return {"success": False, "error": str(e)}


def load_style_profile(name: str, client_id: str = "default") -> dict:
    """
    Load a saved style profile by exact name.

    Args:
        name: Profile name as used in save_style_profile

    Returns:
        {success, profile} or {success: False, error}
    """
    if not name or not name.strip():
        return {"success": False, "error": "name cannot be empty"}

    ensure_user_collections_once(client_id)

    try:
        client = _get_qdrant_client()
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        results, _ = client.scroll(
            collection_name=_style_profiles_collection(client_id),
            scroll_filter=Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=name.strip()))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            return {"success": False, "error": f"No style profile found with name '{name}'"}

        payload = results[0].payload or {}
        return {"success": True, "profile": payload}
    except Exception as e:
        col = _style_profiles_collection(client_id)
        qdrant_result = handle_qdrant_error(e, tool_name="load_style_profile", collection=col, name=name)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("Failed to load style profile", name=name, error=str(e))
        capture_tool_error(e, tool_name="load_style_profile", name=name)
        return {"success": False, "error": str(e)}


def delete_style_profile(document_id: str, client_id: str = "default") -> dict:
    """Delete all chunks for a style profile by document_id from the user's collection.

    Cross-user deletion is prevented by collection-prefix isolation: the collection
    resolves to `{client_id}_writing_style_profiles`, so callers can only delete
    documents that live under their own tenant.
    """
    collection = _style_profiles_collection(client_id)
    try:
        if check_document_indexed is None or delete_document_vectors is None:
            return {"success": False, "error": "kbase indexing unavailable"}

        check_result = check_document_indexed(
            collection_name=collection,
            document_id=document_id,
        )
        chunks = check_result.get("chunk_count", 0)
        if chunks == 0:
            return {"success": True, "document_id": document_id, "chunks_deleted": 0}

        delete_document_vectors(collection_name=collection, document_id=document_id)
        return {"success": True, "document_id": document_id, "chunks_deleted": chunks}
    except Exception as e:
        qdrant_result = handle_qdrant_error(
            e, tool_name="delete_style_profile", collection=collection, document_id=document_id
        )
        if qdrant_result is not None:
            return qdrant_result
        logger.error("Failed to delete style profile", error=str(e), document_id=document_id)
        capture_tool_error(e, tool_name="delete_style_profile", document_id=document_id)
        return {"success": False, "error": str(e)}


def update_style_profile(
    name: str,
    new_style_scores: Optional[dict] = None,
    new_rules: Optional[list] = None,
    new_anti_patterns: Optional[list] = None,
    new_sample_excerpts: Optional[list] = None,
    new_source_documents: Optional[list] = None,
    description: Optional[str] = None,
    channel: Optional[str] = None,
    score_weight: float = 0.3,
    client_id: str = "default",
) -> dict:
    """
    Merge new evidence into an existing style profile without overwriting it.

    Scores are blended: new = (1 - score_weight) * existing + score_weight * new_scores.
    Rules and anti_patterns are unioned; exact duplicates are removed.
    Sample excerpts are appended (capped at 20 to keep the embed text manageable).

    Args:
        name: Profile name to update (must already exist)
        new_style_scores: New dimension scores to blend in (0.0–1.0 per dimension)
        new_rules: Additional rules to union into the existing list
        new_anti_patterns: Additional anti-patterns to union into the existing list
        new_sample_excerpts: New representative quotes to append
        new_source_documents: Additional source document names to record
        description: Replace the description if provided
        score_weight: Weight given to new scores when blending (default 0.3 = 30% new, 70% existing)

    Returns:
        {success, name, document_id, updated_fields, chunks_created, warnings}
    """
    if not name or not name.strip():
        return {"success": False, "error": "name cannot be empty"}
    if not (0.0 < score_weight <= 1.0):
        return {"success": False, "error": "score_weight must be between 0.0 (exclusive) and 1.0"}

    ensure_user_collections_once(client_id)

    # Load existing profile
    load_result = load_style_profile(name, client_id=client_id)
    if not load_result["success"]:
        return load_result

    existing = load_result["profile"]
    warnings = []
    updated_fields = []

    # Blend style_scores
    merged_scores = dict(existing.get("style_scores", {}))
    if new_style_scores:
        unknown = [k for k in new_style_scores if k not in VALID_STYLES]
        if unknown:
            warnings.append(f"Unknown style key(s) ignored: {unknown}")
        for k, v in new_style_scores.items():
            if k in VALID_STYLES:
                clamped_v = max(0.0, min(1.0, float(v)))
                existing_v = merged_scores.get(k, clamped_v)
                merged_scores[k] = round(
                    (1 - score_weight) * existing_v + score_weight * clamped_v, 4
                )
        updated_fields.append("style_scores")

    # Union rules (preserve order, deduplicate)
    merged_rules = list(existing.get("rules", []))
    if new_rules:
        for r in new_rules:
            if r not in merged_rules:
                merged_rules.append(r)
        updated_fields.append("rules")

    # Union anti_patterns
    merged_anti = list(existing.get("anti_patterns", []))
    if new_anti_patterns:
        for a in new_anti_patterns:
            if a not in merged_anti:
                merged_anti.append(a)
        updated_fields.append("anti_patterns")

    # Append sample_excerpts (cap at 20)
    merged_excerpts = list(existing.get("sample_excerpts", []))
    if new_sample_excerpts:
        for e in new_sample_excerpts:
            if e not in merged_excerpts:
                merged_excerpts.append(e)
        merged_excerpts = merged_excerpts[-20:]
        updated_fields.append("sample_excerpts")

    # Append source_documents
    merged_sources = list(existing.get("source_documents", []))
    if new_source_documents:
        for s in new_source_documents:
            if s not in merged_sources:
                merged_sources.append(s)
        updated_fields.append("source_documents")

    merged_description = description if description is not None else existing.get("description", "")
    if description is not None:
        updated_fields.append("description")

    # Merge channel
    merged_channel = channel if channel is not None else existing.get("channel")
    if channel is not None:
        updated_fields.append("channel")

    if not updated_fields:
        return {"success": False, "error": "At least one field must be provided to update"}

    # Delete old profile and re-index with merged data
    try:
        client = _get_qdrant_client()
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        old_results, _ = client.scroll(
            collection_name=_style_profiles_collection(client_id),
            scroll_filter=Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=name.strip()))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if old_results and delete_document_vectors is not None:
            delete_document_vectors(
                collection_name=_style_profiles_collection(client_id),
                document_id=old_results[0].payload.get("document_id", str(old_results[0].id)),
            )
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="update_style_profile", collection=_style_profiles_collection(client_id), name=name)
        if qdrant_result is None:
            logger.warning("Could not delete old profile vectors", name=name, error=str(e))

    # Re-save with merged data
    result = save_style_profile(
        name=name,
        style_scores=merged_scores,
        rules=merged_rules,
        anti_patterns=merged_anti,
        sample_excerpts=merged_excerpts,
        description=merged_description,
        source_documents=merged_sources,
        channel=merged_channel,
        client_id=client_id,
    )

    if result.get("success"):
        result["updated_fields"] = updated_fields
        result["score_weight_used"] = score_weight
        result["warnings"] = warnings + result.get("warnings", [])

    return result


def harvest_corrections_to_profile(
    profile_name: str,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    min_corrections: int = 3,
    top_k: int = 20,
    client_id: str = "default",
) -> dict:
    """
    Scan human-corrected passages in the library and propose additions to a style profile.

    Retrieves passages tagged 'human-corrected' (stored via record_correction()),
    extracts distinct issue_type categories found, and surfaces them as candidate
    rules and anti-patterns for the agent to propose to the user.

    The agent presents the candidates; the user approves or skips each one;
    approved ones are merged via update_style_profile().

    Args:
        profile_name: Profile to enrich (must already exist)
        language: Filter corrections by language (en|pt). If omitted, returns all.
        domain: Filter corrections by domain. If omitted, returns all.
        min_corrections: Minimum number of human-corrected passages required
                         before returning candidates (default 3)
        top_k: Max corrections to retrieve for analysis (default 20)

    Returns:
        {
            success, profile_name, corrections_found,
            candidates: [
                {
                    type: "rule" | "anti_pattern",
                    text: str,
                    source_issue_type: str,
                    example_corrected: str   (first 120 chars of a supporting correction)
                }
            ],
            insufficient_data: bool,
            note: str (present when insufficient_data=True)
        }
    """
    if not profile_name or not profile_name.strip():
        return {"success": False, "error": "profile_name cannot be empty"}

    # Verify profile exists
    load_result = load_style_profile(profile_name, client_id=client_id)
    if not load_result["success"]:
        return load_result

    if semantic_search is None:
        return {"success": False, "error": "kbase not available — semantic search required"}

    try:
        from src.tools.collections import get_collection_names
        collection = get_collection_names(client_id)["passages"]

        filter_conditions: dict = {"entry_type": "correction"}
        if language:
            filter_conditions["language"] = language
        if domain:
            filter_conditions["domain"] = domain

        raw = semantic_search(
            collection_name=collection,
            query="human correction improved writing quality",
            limit=top_k * 4,  # over-fetch for post-filter
            filter_conditions=filter_conditions,
        )

        # Post-filter to human-corrected only
        human_corrections = [
            r for r in raw
            if "human-corrected" in r.get("metadata", {}).get("style", [])
        ][:top_k]

        if len(human_corrections) < min_corrections:
            return {
                "success": True,
                "profile_name": profile_name,
                "corrections_found": len(human_corrections),
                "candidates": [],
                "insufficient_data": True,
                "note": (
                    f"Only {len(human_corrections)} human-corrected passage(s) found "
                    f"(minimum {min_corrections} required). "
                    "Call record_correction() more to build the corpus."
                ),
            }

        # Group by issue_type and build candidates
        seen_issue_types: dict = {}
        for r in human_corrections:
            meta = r.get("metadata", {})
            issue_type = meta.get("issue_type", "unspecified")
            if issue_type not in seen_issue_types:
                seen_issue_types[issue_type] = r.get("text", "")

        existing_profile = load_result["profile"]
        existing_rules = set(existing_profile.get("rules", []))
        existing_anti = set(existing_profile.get("anti_patterns", []))

        # Map issue_types to candidate rules / anti-patterns
        _ISSUE_TO_RULE = {
            "hollow-intensifier": {
                "type": "anti_pattern",
                "text": "Hollow intensifiers: avoid 'it is important to note that', 'it is crucial that'",
            },
            "passive-voice": {
                "type": "rule",
                "text": "Prefer active voice — restructure passive constructions",
            },
            "ai-patterns": {
                "type": "anti_pattern",
                "text": "AI-sounding openers and connectors: avoid Furthermore, Moreover, Additionally as paragraph starters",
            },
            "missing-connector": {
                "type": "rule",
                "text": "Every paragraph requires a connector to the preceding idea",
            },
            "deficit-framing": {
                "type": "anti_pattern",
                "text": "Deficit framing: avoid 'victims', 'suffering from', 'the poor'",
            },
            "grandiose-opener": {
                "type": "anti_pattern",
                "text": "Grandiose openers: avoid 'Against this backdrop', 'The evidence is unequivocal'",
            },
            "sentence-monotony": {
                "type": "rule",
                "text": "Vary sentence length deliberately — mix short, medium, and long sentences",
            },
            "generic-closing": {
                "type": "anti_pattern",
                "text": "Generic closings: avoid 'In conclusion, this report has shown...'",
            },
            "mechanical-listing": {
                "type": "anti_pattern",
                "text": "Mechanical listing: avoid Firstly / Secondly / Thirdly / Finally as paragraph openers",
            },
        }

        candidates = []
        for issue_type, example_text in seen_issue_types.items():
            mapping = _ISSUE_TO_RULE.get(issue_type)
            if mapping is None:
                # Unknown issue type — surface as a generic rule candidate
                mapping = {
                    "type": "rule",
                    "text": f"Address '{issue_type}' issues consistently",
                }
            # Skip if already in the profile
            if mapping["text"] in existing_rules or mapping["text"] in existing_anti:
                continue
            candidates.append({
                "type": mapping["type"],
                "text": mapping["text"],
                "source_issue_type": issue_type,
                "example_corrected": example_text[:120],
            })

        return {
            "success": True,
            "profile_name": profile_name,
            "corrections_found": len(human_corrections),
            "candidates": candidates,
            "insufficient_data": False,
        }

    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="harvest_corrections_to_profile", collection=_style_profiles_collection(client_id), profile_name=profile_name)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("harvest_corrections_to_profile failed", error=str(e))
        capture_tool_error(e, tool_name="harvest_corrections_to_profile", profile_name=profile_name)
        return {"success": False, "error": str(e)}


def search_style_profiles(
    text: str,
    top_k: int = 3,
    client_id: str = "default",
    channel: Optional[str] = None,
) -> dict:
    """
    Find the style profiles most semantically similar to a text sample.

    Use this to identify which saved style a piece of writing most closely matches.

    Args:
        text: A writing sample to compare against saved profiles
        top_k: Number of results to return (default 3)
        channel: Optional channel filter (linkedin|facebook|instagram|email|report|...)

    Returns:
        {success, results: [{score, profile}], total}
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    ensure_user_collections_once(client_id)

    try:
        filter_conditions = {"channel": channel} if channel else None
        raw_results = semantic_search(
            collection_name=_style_profiles_collection(client_id),
            query=text,
            limit=top_k,
            filter_conditions=filter_conditions,
        )

        results = []
        for r in raw_results:
            results.append({
                "score": round(r["score"], 4),
                "profile": r.get("metadata", {}),
            })

        return {"success": True, "results": results, "total": len(results)}
    except Exception as e:
        col = _style_profiles_collection(client_id)
        qdrant_result = handle_qdrant_error(e, tool_name="search_style_profiles", collection=col, client_id=client_id)
        if qdrant_result is not None:
            qdrant_result["results"] = []
            return qdrant_result
        logger.error("Style profile search failed", error=str(e))
        capture_tool_error(e, tool_name="search_style_profiles", client_id=client_id)
        return {"success": False, "error": str(e), "results": []}


def list_style_profiles(
    channel: Optional[str] = None,
    client_id: str = "default",
    limit: int = 50,
) -> dict:
    """
    List all saved style profiles, optionally filtered by channel.

    Args:
        channel: Filter to profiles tagged with a specific channel
                 (linkedin|facebook|instagram|email|report|proposal|...)
        limit: Max profiles to return (default 50)

    Returns:
        {success, profiles: [{name, description, channel, style_scores, created_at, document_id}], total}
    """
    ensure_user_collections_once(client_id)

    try:
        client = _get_qdrant_client()
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        scroll_filter = None
        if channel:
            scroll_filter = Filter(
                must=[FieldCondition(key="channel", match=MatchValue(value=channel))]
            )

        results, _ = client.scroll(
            collection_name=_style_profiles_collection(client_id),
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        profiles = []
        for point in results:
            p = point.payload or {}
            profiles.append({
                "name": p.get("name"),
                "description": p.get("description", ""),
                "channel": p.get("channel"),
                "style_scores": p.get("style_scores", {}),
                "created_at": p.get("created_at"),
                "document_id": p.get("document_id"),
            })

        # Sort by name for stable ordering
        profiles.sort(key=lambda x: (x.get("name") or "").lower())

        return {"success": True, "profiles": profiles, "total": len(profiles)}
    except Exception as e:
        col = _style_profiles_collection(client_id)
        qdrant_result = handle_qdrant_error(e, tool_name="list_style_profiles", collection=col, client_id=client_id)
        if qdrant_result is not None:
            qdrant_result["profiles"] = []
            return qdrant_result
        logger.error("list_style_profiles failed", error=str(e))
        capture_tool_error(e, tool_name="list_style_profiles", client_id=client_id)
        return {"success": False, "error": str(e), "profiles": []}


def get_style_injection_context(
    name: str,
    client_id: str = "default",
    max_excerpts: int = 3,
    max_rules: int = 10,
    max_anti_patterns: int = 5,
) -> dict:
    """Format a saved style profile as a few-shot-ready prompt block.

    Research (2026) shows 3–5 concrete excerpts + explicit rules deliver up to
    23.5× higher style fidelity than description-only prompts. This returns an
    injection_block ready to paste into a system prompt.

    Args:
        name: Profile name (must exist for this client_id).
        max_excerpts: Cap on sample excerpts included (default 3).
        max_rules: Cap on rules included (default 10).
        max_anti_patterns: Cap on anti-patterns included (default 5).

    Returns:
        {success, profile_name, channel, injection_block,
         rules_count, anti_patterns_count, excerpts_count}
    """
    loaded = load_style_profile(name=name, client_id=client_id)
    if not loaded.get("success"):
        return loaded

    profile = loaded["profile"]
    rules = list(profile.get("rules", []))[:max_rules]
    anti = list(profile.get("anti_patterns", []))[:max_anti_patterns]
    excerpts = list(profile.get("sample_excerpts", []))[:max_excerpts]
    channel = profile.get("channel") or "general"

    lines = [
        f"## Style Profile: {profile.get('name', name)}",
        f"Channel: {channel}",
    ]
    description = profile.get("description")
    if description:
        lines.append(f"Description: {description}")
    lines.append("")

    if rules:
        lines.append("### Rules")
        for i, r in enumerate(rules, 1):
            lines.append(f"{i}. {r}")
        lines.append("")

    if anti:
        lines.append("### Avoid")
        for i, a in enumerate(anti, 1):
            lines.append(f"{i}. {a}")
        lines.append("")

    if excerpts:
        lines.append("### Example Excerpts")
        for i, ex in enumerate(excerpts, 1):
            lines.append(f"Example {i}:")
            lines.append(str(ex).strip())
            lines.append("")

    injection_block = "\n".join(lines).rstrip() + "\n"

    return {
        "success": True,
        "profile_name": profile.get("name", name),
        "channel": channel,
        "injection_block": injection_block,
        "rules_count": len(rules),
        "anti_patterns_count": len(anti),
        "excerpts_count": len(excerpts),
    }
