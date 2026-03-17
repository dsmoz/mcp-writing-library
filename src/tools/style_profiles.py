"""
Style profile tools: save and retrieve writing style profiles extracted from samples.

Style extraction is LLM-assisted: Claude analyses 2–5 writing samples in the conversation
and produces a structured profile. These tools handle persistence and retrieval in Qdrant.
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4
import structlog

from src.tools.styles import VALID_STYLES

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "writing_style_profiles"

try:
    from kbase.vector.sync_indexing import index_document
    from kbase.vector.sync_search import semantic_search
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    import os
    _qdrant_available = True
except ImportError:
    index_document = None  # type: ignore
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

    # Validate style score keys
    warnings = []
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
        "name": name.strip(),
        "description": description,
        "style_scores": clamped,
        "rules": rules,
        "anti_patterns": anti_patterns,
        "sample_excerpts": sample_excerpts,
        "source_documents": source_documents or [],
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
            collection_name=COLLECTION_NAME,
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
        logger.error("Failed to save style profile", name=name, error=str(e))
        return {"success": False, "error": str(e)}


def load_style_profile(name: str) -> dict:
    """
    Load a saved style profile by exact name.

    Args:
        name: Profile name as used in save_style_profile

    Returns:
        {success, profile} or {success: False, error}
    """
    if not name or not name.strip():
        return {"success": False, "error": "name cannot be empty"}

    try:
        client = _get_qdrant_client()
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
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
        logger.error("Failed to load style profile", name=name, error=str(e))
        return {"success": False, "error": str(e)}


def search_style_profiles(text: str, top_k: int = 3) -> dict:
    """
    Find the style profiles most semantically similar to a text sample.

    Use this to identify which saved style a piece of writing most closely matches.

    Args:
        text: A writing sample to compare against saved profiles
        top_k: Number of results to return (default 3)

    Returns:
        {success, results: [{score, profile}], total}
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    try:
        raw_results = semantic_search(
            collection_name=COLLECTION_NAME,
            query=text,
            limit=top_k,
        )

        results = []
        for r in raw_results:
            results.append({
                "score": round(r["score"], 4),
                "profile": r.get("metadata", {}),
            })

        return {"success": True, "results": results, "total": len(results)}
    except Exception as e:
        logger.error("Style profile search failed", error=str(e))
        return {"success": False, "error": str(e), "results": []}
