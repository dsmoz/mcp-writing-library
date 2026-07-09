#!/usr/bin/env python3
"""
Diagnose whether stored sparse vectors are reproducible by the current encoder.

Two checks, per collection, over a bounded sample:

  1. STORED-VECTOR SELF-CONSISTENCY
     Pull each point WITH its stored sparse vector, recompute the sparse vector
     from the point's OWN ``payload['text']`` with the current deterministic
     ``generate_sparse_vector``, and measure how well the recomputed nonzero
     index set overlaps the stored one (recall = |stored ∩ recomputed| / |stored|).

       * With the OLD builtin-hash() data this is ~0% — proof the stored sparse
         vectors are garbage and a reindex is required.
       * After scripts/reindex_sparse.py with the SAME encoder, this is ~100%.

  2. TEXT-QUERY SELF-RANK
     For each sampled point, issue a sparse-only query built from its own text
     and confirm (a) the sparse arm is NON-EMPTY and (b) the source point ranks
     highly (ideally #1). This exercises the exact query path used by
     sync_search's hybrid prefetch.

Usage
-----
    uv run python scripts/which_encoder.py
    uv run python scripts/which_encoder.py --collections usr_abc_writing_passages --sample 300
"""
import argparse
import statistics
import sys

from _sparse_ops import (
    bootstrap,
    list_hybrid_writing_collections,
    point_text,
    scroll_points,
)


def _stored_sparse_indices(point) -> set:
    """Extract the stored 'sparse' vector's index set from a scrolled point,
    tolerating both object and dict representations."""
    vec = getattr(point, "vector", None)
    if not isinstance(vec, dict):
        return set()
    sparse = vec.get("sparse")
    if sparse is None:
        return set()
    indices = getattr(sparse, "indices", None)
    if indices is None and isinstance(sparse, dict):
        indices = sparse.get("indices")
    return set(int(i) for i in (indices or []))


def main() -> int:
    bootstrap()

    parser = argparse.ArgumentParser(
        description="Diagnose sparse-vector reproducibility."
    )
    parser.add_argument("--collections", nargs="*", default=None,
                        help="Explicit collections (default: all hybrid writing collections).")
    parser.add_argument("--sample", type=int, default=200,
                        help="Max points to sample per collection (default: 200).")
    parser.add_argument("--topk", type=int, default=10,
                        help="Top-k for the text-query self-rank test (default: 10).")
    parser.add_argument("--no-query-test", action="store_true",
                        help="Skip the text-query self-rank test (self-consistency only).")
    args = parser.parse_args()

    from qdrant_client import models

    from kbase.vector.sync_client import get_qdrant_client
    from kbase.vector.sync_embeddings import generate_sparse_vector
    from kbase.vector.bm25_encoder import get_bm25_encoder

    client = get_qdrant_client()

    print("Active encoder:",
          "FITTED BM25 vocab" if get_bm25_encoder() is not None
          else "md5 deterministic fallback",
          file=sys.stderr)

    collections = list_hybrid_writing_collections(client, only=args.collections)
    if not collections:
        print("No hybrid writing collections found.", file=sys.stderr)
        return 1

    overall_ok = True
    for name in collections:
        recalls = []
        empty_recompute = 0
        sampled = 0
        sample_points = []

        for point in scroll_points(client, name, with_vectors=True):
            stored = _stored_sparse_indices(point)
            text = point_text(point)
            recomputed_idx, _ = generate_sparse_vector(text)
            recomputed = set(recomputed_idx)
            if not recomputed:
                empty_recompute += 1
            if stored:
                inter = len(stored & recomputed)
                recalls.append(inter / len(stored))
            sample_points.append((point.id, text))
            sampled += 1
            if sampled >= args.sample:
                break

        if not sampled:
            print(f"  {name}: empty collection, skipped", file=sys.stderr)
            continue

        mean_recall = statistics.mean(recalls) if recalls else float("nan")
        print(f"\n[{name}] sampled {sampled} points", file=sys.stderr)
        print(f"  self-consistency recall (stored ∩ recomputed / stored): "
              f"{mean_recall:.1%}" if recalls else
              "  self-consistency recall: n/a (no stored sparse vectors)",
              file=sys.stderr)
        print(f"  points whose text recomputes to an EMPTY sparse vector: "
              f"{empty_recompute}/{sampled}", file=sys.stderr)
        if recalls and mean_recall < 0.99:
            overall_ok = False
            print("  ⚠️  stored sparse vectors are NOT reproducible by the "
                  "current encoder — run scripts/reindex_sparse.py.", file=sys.stderr)

        if args.no_query_test:
            continue

        # Text-query self-rank: does a sparse-only query from a point's own text
        # surface that point?
        hits_at_1 = hits_at_k = nonempty = considered = 0
        ranks = []
        for pid, text in sample_points[: min(50, len(sample_points))]:
            indices, values = generate_sparse_vector(text)
            if not indices:
                continue
            nonempty += 1
            considered += 1
            res = client.query_points(
                collection_name=name,
                query=models.SparseVector(indices=indices, values=values),
                using="sparse",
                limit=args.topk,
                with_payload=False,
            ).points
            ids = [p.id for p in res]
            if ids and ids[0] == pid:
                hits_at_1 += 1
            if pid in ids:
                hits_at_k += 1
                ranks.append(ids.index(pid) + 1)
        if considered:
            print(f"  text-query self-rank over {considered} points: "
                  f"hit@1={hits_at_1/considered:.1%}, "
                  f"hit@{args.topk}={hits_at_k/considered:.1%}, "
                  f"median rank={statistics.median(ranks) if ranks else 'n/a'}, "
                  f"non-empty sparse arm={nonempty}/{considered}", file=sys.stderr)
            if hits_at_k / considered < 0.9:
                overall_ok = False
                print("  ⚠️  source passages do not rank in top-k on their own "
                      "text — sparse arm is not working as expected.", file=sys.stderr)
        else:
            print("  text-query self-rank: no non-empty sparse queries to test.",
                  file=sys.stderr)

    print("\nRESULT:", "OK — sparse vectors are reproducible and self-queries rank."
          if overall_ok else
          "PROBLEM — see warnings above (reindex needed or encoder mismatch).",
          file=sys.stderr)
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
