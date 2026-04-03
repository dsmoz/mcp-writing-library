"""
Export tool: scroll all points from a Qdrant collection and return JSON or CSV.
"""
import csv
import io
import structlog

from src.tools.collections import get_collection_names

logger = structlog.get_logger(__name__)

try:
    from kbase.vector.sync_client import get_qdrant_client
except ImportError:
    get_qdrant_client = None  # type: ignore


def _resolve_collection(collection: str, user_id: str = "default") -> str | None:
    """Return the Qdrant collection name for a logical alias or literal name."""
    names = get_collection_names(user_id)
    if collection in names:
        return names[collection]
    # Accept the literal Qdrant collection name too
    if collection in names.values():
        return collection
    return None


def export_library(collection: str, output_format: str = "json", user_id: str = "default") -> dict:
    """
    Export all points from a writing library collection.

    Scrolls the entire collection in batches and returns payloads as JSON or CSV.

    Args:
        collection: Logical alias ("passages", "terms", "style_profiles", "rubrics") or
                    the literal Qdrant collection name.
        output_format: Output format — "json" (default) or "csv".

    Returns:
        {success, collection, count, format, data} on success,
        or {success: False, error} on failure.
    """
    if get_qdrant_client is None:
        return {"success": False, "error": "kbase library is not available"}

    collection_name = _resolve_collection(collection, user_id)
    if collection_name is None:
        names = get_collection_names(user_id)
        return {
            "success": False,
            "error": (
                f"Unknown collection '{collection}'. "
                f"Valid aliases: {sorted(names.keys())}. "
                f"Valid Qdrant names: {sorted(names.values())}."
            ),
        }

    if output_format not in ("json", "csv"):
        return {
            "success": False,
            "error": f"Invalid format '{output_format}'. Must be 'json' or 'csv'.",
        }

    try:
        client = get_qdrant_client()
        payloads = []
        offset = None

        while True:
            results, next_offset = client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                payloads.append(point.payload or {})

            if next_offset is None:
                break
            offset = next_offset

        count = len(payloads)

        if output_format == "json":
            return {
                "success": True,
                "collection": collection_name,
                "count": count,
                "format": "json",
                "data": payloads,
            }

        # CSV: derive headers from the union of all payload keys
        if not payloads:
            return {
                "success": True,
                "collection": collection_name,
                "count": 0,
                "format": "csv",
                "data": "",
            }

        all_keys: list[str] = []
        seen: set[str] = set()
        for payload in payloads:
            for k in payload.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=all_keys,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for payload in payloads:
            # Stringify list/dict values for CSV compatibility
            row = {
                k: (str(v) if isinstance(v, (list, dict)) else v)
                for k, v in payload.items()
            }
            writer.writerow(row)

        return {
            "success": True,
            "collection": collection_name,
            "count": count,
            "format": "csv",
            "data": output.getvalue(),
        }

    except Exception as e:
        logger.error("export_library failed", collection=collection_name, error=str(e))
        return {"success": False, "error": str(e)}
