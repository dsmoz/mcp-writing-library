"""
Taxonomy introspection: discover which payload values are actually populated
in the caller's per-user collections.

Used for:
  - `list_taxonomy()` tool — let callers validate filter values before searching.
  - Zero-result hints — search_passages / search_terms surface populated values
    when a filter combination matches nothing.
"""
from typing import Iterable, Optional
import structlog

from src.tools.collections import get_collection_names

logger = structlog.get_logger(__name__)

try:
    from kbase.vector.sync_client import get_qdrant_client
except ImportError:
    get_qdrant_client = None  # type: ignore


# Fields each collection exposes as filterable facets.
PASSAGE_FACETS = ("doc_type", "language", "domain", "style", "rubric_section", "tags")
TERM_FACETS = ("domain", "language")
STYLE_PROFILE_FACETS = ("channel",)

# Scroll cap: large enough to surface every distinct tag in a normal user
# corpus, small enough to keep latency bounded.
_SCROLL_LIMIT = 2000


def get_distinct_values(
    collection: str,
    fields: Iterable[str],
    limit: int = _SCROLL_LIMIT,
) -> dict[str, list[str]]:
    """Scroll a collection and return distinct payload values per requested field.

    List-valued payload fields (e.g. ``tags``, ``style``) are flattened.
    Returns empty dict on any client error so callers can no-op silently.
    """
    if get_qdrant_client is None:
        return {}

    out: dict[str, set[str]] = {f: set() for f in fields}

    try:
        client = get_qdrant_client()
        offset = None
        scanned = 0
        while scanned < limit:
            batch_size = min(256, limit - scanned)
            points, offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for p in points:
                payload = p.payload or {}
                for f in fields:
                    v = payload.get(f)
                    if v is None or v == "":
                        continue
                    if isinstance(v, list):
                        for item in v:
                            if item:
                                out[f].add(str(item))
                    else:
                        out[f].add(str(v))
            scanned += len(points)
            if offset is None:
                break
    except Exception as e:
        logger.debug("Distinct-values scroll failed", collection=collection, error=str(e))
        return {}

    return {f: sorted(vals) for f, vals in out.items()}


def build_zero_result_hint(
    collection: str,
    requested_filters: dict[str, str],
    facets: Iterable[str],
) -> Optional[dict]:
    """Produce a hint payload for callers when a filtered search returns zero hits.

    The hint enumerates which requested filter values are absent from the
    corpus and which alternatives exist. Returns ``None`` if introspection
    fails or the corpus is empty.
    """
    if not requested_filters:
        return None

    distinct = get_distinct_values(collection, facets)
    if not distinct:
        return None

    if all(len(v) == 0 for v in distinct.values()):
        return {
            "message": "Collection is empty. No entries to filter against.",
            "available_values": {},
        }

    mismatches = {}
    for field, value in requested_filters.items():
        available = distinct.get(field, [])
        if value not in available:
            mismatches[field] = {
                "requested": value,
                "available": available,
            }

    if not mismatches:
        return {
            "message": (
                "Filter values exist in the corpus, but no entries match the "
                "combination together with the query."
            ),
            "available_values": distinct,
        }

    parts = [
        f"No entries matched {field}={value!r}. Available: {info['available']}"
        for field, info in mismatches.items()
        for value in [info["requested"]]
    ]
    return {
        "message": " ".join(parts),
        "filter_mismatches": mismatches,
        "available_values": distinct,
    }


def list_taxonomy(client_id: str = "default") -> dict:
    """Return populated facet values for the caller's per-user collections.

    Lets calling skills validate filter values before issuing search queries
    instead of trial-and-error empty results.
    """
    names = get_collection_names(client_id)

    return {
        "success": True,
        "client_id": client_id,
        "passages": {
            "collection": names["passages"],
            "values": get_distinct_values(names["passages"], PASSAGE_FACETS),
        },
        "terms": {
            "collection": names["terms"],
            "values": get_distinct_values(names["terms"], TERM_FACETS),
        },
        "style_profiles": {
            "collection": names["style_profiles"],
            "values": get_distinct_values(names["style_profiles"], STYLE_PROFILE_FACETS),
        },
    }
