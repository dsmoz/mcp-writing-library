"""
Shared helpers for the sparse-encoder maintenance scripts.

These back three operational scripts that together recover from the builtin-
``hash()`` sparse-encoder bug (see ``docs/sparse-encoder-fix-runbook.md``):

    train_bm25_encoder.py   — fit ONE shared BM25 vocab over every writing
                              collection and save the artifact.
    reindex_sparse.py       — regenerate only the "sparse" named vector of
                              every stored point with the deterministic encoder.
    which_encoder.py        — diagnose whether stored sparse vectors are
                              reproducible by the current encoder.

Keeping the Qdrant plumbing in one place means the fit corpus, the reindex
corpus and the diagnostic all agree on which collections and which text field
matter.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


def bootstrap() -> Path:
    """Mirror ``main.py``: make the project root + vendored ``kbase`` importable
    and load ``.env``. Returns the project root."""
    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "vendor"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except Exception:
        # dotenv is optional; env vars may already be set (e.g. in Railway).
        pass
    return root


# Default artifact location, matched to hybrid_embeddings.get_sparse_embedding
# and BM25_ENCODER_PATH.
DEFAULT_ENCODER_PATH = os.getenv("BM25_ENCODER_PATH") or os.path.join(
    ".data", "bm25_encoder.pkl"
)

# Per-user collection suffixes (see src/tools/collections.py) + shared/core ones.
_WRITING_SUFFIXES = (
    "_writing_passages",
    "_writing_terms",
    "_writing_style_profiles",
    "_writing_rubrics",
    "_writing_templates",
    "_writing_thesaurus",
)
_CORE_COLLECTIONS = ("writing_terms_shared", "writing_contributions")


def is_writing_collection(name: str) -> bool:
    """True if ``name`` is one of the writing library's collections."""
    return name in _CORE_COLLECTIONS or any(
        name.endswith(suffix) for suffix in _WRITING_SUFFIXES
    )


def collection_is_hybrid(client, name: str) -> bool:
    """True if the collection has a named ``sparse`` sparse-vector config."""
    info = client.get_collection(collection_name=name)
    sparse_conf = getattr(info.config.params, "sparse_vectors", None)
    return bool(sparse_conf) and "sparse" in sparse_conf


def list_hybrid_writing_collections(
    client, only: Optional[Iterable[str]] = None
) -> List[str]:
    """Return the sorted names of hybrid writing collections.

    Args:
        client: Qdrant client.
        only: If given, restrict to this explicit set of names (still filtered
            to those that are actually hybrid). If None, auto-discover every
            writing collection.
    """
    only_set = set(only) if only is not None else None
    all_names = [c.name for c in client.get_collections().collections]
    result: List[str] = []
    for name in sorted(all_names):
        if only_set is not None:
            if name not in only_set:
                continue
        elif not is_writing_collection(name):
            continue
        try:
            if collection_is_hybrid(client, name):
                result.append(name)
        except Exception:
            # A collection that vanished or can't be described is simply skipped.
            continue
    return result


def scroll_points(
    client,
    collection: str,
    with_vectors: bool = False,
    batch: int = 256,
) -> Iterator:
    """Yield every point in ``collection`` (payload always, vectors optional)."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        for point in points:
            yield point
        if offset is None:
            break


def point_text(point) -> str:
    """The searchable text a point was indexed from (payload['text'])."""
    payload = getattr(point, "payload", None) or {}
    return (payload.get("text") or "").strip()
