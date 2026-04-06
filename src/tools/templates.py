"""
Document structure enforcement: store and check templates for any framework or client.
"""
from typing import List
from uuid import uuid4
import structlog

from src.sentry import capture_tool_error
from src.tools.collections import get_collection_names
from src.tools.qdrant_errors import handle_qdrant_error
from src.tools.registry import VALID_DOC_TYPES

logger = structlog.get_logger(__name__)

# framework is a free-form label — no closed enum. Use lowercase slugs
# (e.g. "undp", "usaid", "lambda", "ds-moz"). list_templates() discovers all stored values.

# Thresholds for embedding-based (cosine) scoring
_EMBED_THRESHOLD_PRESENT = 0.55
_EMBED_THRESHOLD_PARTIAL = 0.35

# Thresholds for keyword-based scoring (lower range ~[0, 0.2])
_KW_THRESHOLD_PRESENT = 0.15
_KW_THRESHOLD_PARTIAL = 0.05

try:
    from kbase.vector.sync_indexing import index_document
    from kbase.vector.sync_search import semantic_search
    from kbase.vector.sync_client import get_qdrant_client
except ImportError:
    index_document = None  # type: ignore
    semantic_search = None  # type: ignore
    get_qdrant_client = None  # type: ignore

try:
    from kbase.vector.sync_embeddings import generate_embedding as _generate_embedding
    _embedding_available = True
except ImportError:
    _generate_embedding = None  # type: ignore
    _embedding_available = False


def _cosine(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x ** 2 for x in a) ** 0.5
    nb = sum(x ** 2 for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _keyword_coverage(section_name: str, section_description: str, paragraph: str) -> float:
    """
    Fallback coverage scoring using keyword matching when embeddings are unavailable.
    Returns a score 0.0–1.0 based on proportion of section name/description words found.
    """
    import re
    para_lower = paragraph.lower()
    # Tokenise section name and description
    words = re.findall(r'\b[a-z]{3,}\b', (section_name + " " + section_description).lower())
    # Remove very common stop words
    stop = {"the", "and", "for", "this", "that", "with", "from", "are", "has", "have",
            "been", "will", "its", "their", "each", "they", "what", "how", "all"}
    words = [w for w in words if w not in stop]
    if not words:
        return 0.0
    matches = sum(1 for w in words if w in para_lower)
    return matches / len(words)


def add_template(framework: str, doc_type: str, sections: list) -> dict:
    """
    Store a proposal template (list of required sections) for a framework+doc_type combination.

    Args:
        framework: Evaluation framework slug — any lowercase slug (e.g. "undp", "lambda", "ds-moz")
        doc_type: Document type — must be one of the valid doc_types (see registry)
        sections: List of section dicts. Each must have:
                  - name (str): Section name
                  - description (str): What this section should contain
                  - required (bool, optional): Whether mandatory (default True)
                  - order (int, optional): Expected position 1-based (default = list index + 1)

    Returns:
        {success, document_id, chunks_created, framework, doc_type, section_count} on success,
        or {success: False, error} on invalid input
    """
    framework = framework.lower().strip()
    doc_type = doc_type.lower()

    if not framework:
        return {"success": False, "error": "framework cannot be empty"}

    if doc_type not in VALID_DOC_TYPES:
        return {
            "success": False,
            "error": f"Invalid doc_type '{doc_type}'. Must be one of: {sorted(VALID_DOC_TYPES)}",
        }

    if not sections or not isinstance(sections, list):
        return {"success": False, "error": "sections must be a non-empty list"}

    # Validate and normalise each section
    normalised = []
    for i, sec in enumerate(sections):
        if "name" not in sec:
            return {"success": False, "error": f"Section at index {i} is missing required key 'name'"}
        if "description" not in sec:
            return {"success": False, "error": f"Section at index {i} is missing required key 'description'"}
        normalised.append({
            "name": sec["name"],
            "description": sec["description"],
            "required": sec.get("required", True),
            "order": sec.get("order", i + 1),
        })

    if index_document is None:
        return {"success": False, "error": "kbase library is not available"}

    document_id = str(uuid4())
    collection = get_collection_names()["templates"]
    title = f"[{framework.upper()} | {doc_type}] Template"

    # Concatenate section names + descriptions for embedding
    content_parts = []
    for sec in normalised:
        content_parts.append(f"{sec['name']}: {sec['description']}")
    content = "\n".join(content_parts)

    metadata = {
        "framework": framework,
        "doc_type": doc_type,
        "sections": normalised,
        "section_count": len(normalised),
        "entry_type": "template",
    }

    try:
        point_ids = index_document(
            collection_name=collection,
            document_id=document_id,
            title=title,
            content=content,
            metadata=metadata,
            context_mode="metadata",
        )
        return {
            "success": True,
            "document_id": document_id,
            "chunks_created": len(point_ids),
            "framework": framework,
            "doc_type": doc_type,
            "section_count": len(normalised),
        }
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="add_template", collection=collection, framework=framework, doc_type=doc_type)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("Failed to add template", error=str(e))
        capture_tool_error(e, tool_name="add_template", framework=framework, doc_type=doc_type)
        return {"success": False, "error": str(e)}


def check_structure(text: str, framework: str, doc_type: str) -> dict:
    """
    Check whether a document draft covers all required sections from the stored template.

    Args:
        text: The document draft text to check
        framework: Evaluation framework slug — any lowercase slug (e.g. "undp", "lambda", "ds-moz")
        doc_type: Document type — must be one of the valid doc_types (see registry)

    Returns:
        {success, framework, doc_type, template_document_id, total_sections, required_sections,
         present_count, partial_count, missing_count, verdict, sections, missing_required}
        verdict is "complete" (0 missing required) or "incomplete" (>0 missing required)
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    framework = framework.lower().strip()
    doc_type = doc_type.lower()

    if not framework:
        return {"success": False, "error": "framework cannot be empty"}

    if doc_type not in VALID_DOC_TYPES:
        return {
            "success": False,
            "error": f"Invalid doc_type '{doc_type}'. Must be one of: {sorted(VALID_DOC_TYPES)}",
        }

    if semantic_search is None:
        return {"success": False, "error": "kbase library is not available"}

    collection = get_collection_names()["templates"]

    # Retrieve the template
    try:
        raw_results = semantic_search(
            collection_name=collection,
            query=f"{framework} {doc_type} template",
            limit=1,
            filter_conditions={"framework": framework, "doc_type": doc_type},
        )
    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="check_structure", collection=collection, framework=framework, doc_type=doc_type)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("check_structure search failed", error=str(e))
        capture_tool_error(e, tool_name="check_structure", framework=framework, doc_type=doc_type)
        return {"success": False, "error": str(e)}

    if not raw_results:
        return {
            "success": False,
            "error": f"No template found for framework '{framework}' doc_type '{doc_type}'",
        }

    template_result = raw_results[0]
    template_doc_id = template_result.get("document_id", "")
    metadata = template_result.get("metadata", {})
    sections = metadata.get("sections", [])

    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # Whether embeddings are globally available (per-section fallback is still possible)
    embeddings_globally_available = _embedding_available and _generate_embedding is not None

    section_results = []
    missing_required: List[str] = []
    present_count = 0
    partial_count = 0
    missing_required_count = 0
    sections_used_embedding = 0
    sections_used_keyword = 0

    for sec in sections:
        sec_name = sec.get("name", "")
        sec_desc = sec.get("description", "")
        sec_required = sec.get("required", True)
        query_text = f"{sec_name}: {sec_desc}"

        # Compute coverage score across paragraphs — try embedding first, fall back per section
        coverage_score = 0.0
        section_scoring_method = "keyword"

        if embeddings_globally_available:
            try:
                sec_embedding = _generate_embedding(query_text)
                para_scores = []
                for para in paragraphs:
                    try:
                        para_embedding = _generate_embedding(para)
                        sim = _cosine(sec_embedding, para_embedding)
                        para_scores.append(sim)
                    except Exception:
                        # Embedding failed for this paragraph — skip
                        pass
                if para_scores:
                    coverage_score = max(para_scores)
                    section_scoring_method = "embedding"
                else:
                    # All paragraph embeddings failed — fall back to keyword for this section
                    logger.warning(
                        "Embedding failed for section, using keyword fallback",
                        section=sec_name,
                        error="all paragraph embeddings failed",
                    )
            except Exception as e:
                # Section embedding failed — fall back to keyword for this section
                logger.warning(
                    "Embedding failed for section, using keyword fallback",
                    section=sec_name,
                    error=str(e),
                )

        if section_scoring_method == "keyword":
            # Keyword fallback: max keyword coverage across paragraphs
            para_scores = [_keyword_coverage(sec_name, sec_desc, para) for para in paragraphs]
            coverage_score = max(para_scores) if para_scores else 0.0

        # Tally scoring method usage
        if section_scoring_method == "embedding":
            sections_used_embedding += 1
        else:
            sections_used_keyword += 1

        # Classify status using appropriate thresholds
        if section_scoring_method == "embedding":
            threshold_present = _EMBED_THRESHOLD_PRESENT
            threshold_partial = _EMBED_THRESHOLD_PARTIAL
        else:
            threshold_present = _KW_THRESHOLD_PRESENT
            threshold_partial = _KW_THRESHOLD_PARTIAL

        if coverage_score >= threshold_present:
            status = "present"
            present_count += 1
        elif coverage_score >= threshold_partial:
            status = "partial"
            partial_count += 1
        else:
            status = "missing"
            if sec_required:
                missing_required.append(sec_name)
                missing_required_count += 1

        section_results.append({
            "name": sec_name,
            "required": sec_required,
            "status": status,
            "coverage_score": round(coverage_score, 4),
            "scoring_method": section_scoring_method,
        })

    required_sections = sum(1 for s in sections if s.get("required", True))
    verdict = "complete" if missing_required_count == 0 else "incomplete"

    # Determine top-level scoring_method
    if sections_used_embedding > 0 and sections_used_keyword == 0:
        top_scoring_method = "embedding"
    elif sections_used_keyword > 0 and sections_used_embedding == 0:
        top_scoring_method = "keyword"
    else:
        top_scoring_method = "mixed"

    return {
        "success": True,
        "framework": framework,
        "doc_type": doc_type,
        "template_document_id": template_doc_id,
        "total_sections": len(sections),
        "required_sections": required_sections,
        "present_count": present_count,
        "partial_count": partial_count,
        "missing_count": missing_required_count,
        "verdict": verdict,
        "scoring_method": top_scoring_method,
        "sections": section_results,
        "missing_required": missing_required,
    }


def list_templates() -> dict:
    """
    Return all stored templates.

    Returns:
        {success, templates: [{framework, doc_type, section_count, document_id}], total}
        Sorted by framework then doc_type.
    """
    if get_qdrant_client is None:
        return {"success": False, "error": "kbase library is not available"}

    collection = get_collection_names()["templates"]

    try:
        client = get_qdrant_client()
        templates = []
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
                if payload.get("entry_type") == "template":
                    templates.append({
                        "framework": payload.get("framework", ""),
                        "doc_type": payload.get("doc_type", ""),
                        "section_count": payload.get("section_count", 0),
                        "document_id": payload.get("document_id", str(point.id)),
                    })

            if next_offset is None:
                break
            offset = next_offset

        templates.sort(key=lambda x: (x["framework"], x["doc_type"]))

        return {
            "success": True,
            "templates": templates,
            "total": len(templates),
        }

    except Exception as e:
        qdrant_result = handle_qdrant_error(e, tool_name="list_templates", collection=collection)
        if qdrant_result is not None:
            return qdrant_result
        logger.error("list_templates failed", error=str(e))
        capture_tool_error(e, tool_name="list_templates")
        return {"success": False, "error": str(e)}
