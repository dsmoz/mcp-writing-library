#!/usr/bin/env python3
"""
Regenerate stored sparse vectors with the deterministic encoder.

Every sparse vector currently stored in Qdrant was produced by the old builtin-
``hash()`` encoder and is unreproducible garbage (see
docs/sparse-encoder-fix-runbook.md). This script recomputes each point's
"sparse" named vector from its own ``payload['text']`` using the fixed,
deterministic ``generate_sparse_vector`` and writes it back.

Only the sparse vector is touched — the dense vector and the payload are left
exactly as they are (``update_vectors`` updates named vectors in place). A full,
expensive dense re-embed is NOT required; dense vectors were always fine.

Prerequisite: the fitted BM25 artifact must already be deployed here (same file
that every instance will serve queries with). Run this AFTER
scripts/train_bm25_encoder.py and BEFORE serving fitted-encoder queries.

Usage
-----
    uv run python scripts/reindex_sparse.py --dry-run       # count only, no writes
    uv run python scripts/reindex_sparse.py                 # reindex all collections
    uv run python scripts/reindex_sparse.py --collections usr_abc_writing_passages
"""
import argparse
import sys

from _sparse_ops import (
    bootstrap,
    list_hybrid_writing_collections,
    point_text,
    scroll_points,
)


def main() -> int:
    bootstrap()

    parser = argparse.ArgumentParser(
        description="Regenerate stored sparse vectors deterministically."
    )
    parser.add_argument("--collections", nargs="*", default=None,
                        help="Explicit collections (default: all hybrid writing collections).")
    parser.add_argument("--batch", type=int, default=256, help="Upsert batch size.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report, but write nothing.")
    args = parser.parse_args()

    from qdrant_client import models

    from kbase.vector.sync_client import get_qdrant_client
    from kbase.vector.sync_embeddings import generate_sparse_vector
    from kbase.vector.bm25_encoder import get_bm25_encoder

    client = get_qdrant_client()

    # Announce which encoder is active so an accidental md5-fallback reindex is
    # visible rather than silent.
    if get_bm25_encoder() is not None:
        print("Encoder: FITTED BM25 vocab", file=sys.stderr)
    else:
        print(
            "Encoder: md5 deterministic FALLBACK (no fitted artifact found). "
            "This is deterministic and correct, but lower quality than a fitted "
            "encoder and lives in a different index space — make sure queries "
            "will use the SAME encoder.",
            file=sys.stderr,
        )

    collections = list_hybrid_writing_collections(client, only=args.collections)
    if not collections:
        print("No hybrid writing collections found; nothing to reindex.", file=sys.stderr)
        return 1

    grand_updated = grand_skipped = 0
    for name in collections:
        updated = skipped = 0
        buffer: list = []

        def flush():
            nonlocal updated
            if not buffer:
                return
            if not args.dry_run:
                client.update_vectors(collection_name=name, points=list(buffer))
            updated += len(buffer)
            buffer.clear()

        for point in scroll_points(client, name, with_vectors=False, batch=args.batch):
            indices, values = generate_sparse_vector(point_text(point))
            if not indices:
                # No usable tokens — an empty sparse vector contributes nothing;
                # leave whatever is there rather than pushing an empty vector.
                skipped += 1
                continue
            buffer.append(
                models.PointVectors(
                    id=point.id,
                    vector={"sparse": models.SparseVector(indices=indices, values=values)},
                )
            )
            if len(buffer) >= args.batch:
                flush()
        flush()

        verb = "would update" if args.dry_run else "updated"
        print(f"  {name}: {verb} {updated} points ({skipped} skipped, empty text)",
              file=sys.stderr)
        grand_updated += updated
        grand_skipped += skipped

    verb = "Would update" if args.dry_run else "Updated"
    print(f"\n{verb} {grand_updated} sparse vectors across {len(collections)} "
          f"collection(s); {grand_skipped} skipped.", file=sys.stderr)
    if args.dry_run:
        print("Dry run — nothing was written.", file=sys.stderr)
    else:
        print("Verify with: uv run python scripts/which_encoder.py", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
