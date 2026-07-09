#!/usr/bin/env python3
"""
Fit the shared BM25 sparse encoder.

Why this exists
---------------
The sparse (keyword) arm of hybrid search used to derive its indices from
Python's builtin ``hash()``, which is salted per process (PYTHONHASHSEED). Index-
time and query-time indices therefore never agreed and hybrid search silently
collapsed to dense-only. The fix makes ``generate_sparse_vector`` deterministic,
preferring a *fitted* BM25 vocab encoder when this artifact is present.

This script fits ONE encoder over the union of every writing collection's text.
A single shared vocabulary (identical index space on every server) is what keeps
cross-collection / cross-tenant sparse search coherent — do not fit a separate
encoder per collection.

Usage
-----
    uv run python scripts/train_bm25_encoder.py
    uv run python scripts/train_bm25_encoder.py --output .data/bm25_encoder.pkl
    uv run python scripts/train_bm25_encoder.py --collections usr_abc_writing_passages writing_terms_shared

After fitting, deploy the artifact to EVERY writing-library instance (same file),
then run scripts/reindex_sparse.py before serving queries. See
docs/sparse-encoder-fix-runbook.md.
"""
import argparse
import sys

from _sparse_ops import (
    DEFAULT_ENCODER_PATH,
    bootstrap,
    list_hybrid_writing_collections,
    point_text,
    scroll_points,
)

# pgvector mirror stores sparse_embedding as sparsevec(1048576) and the adapter
# shifts indices +1 (1-based), so the largest 0-based vocab index must stay
# strictly below this. A fitted vocab is sequential [0, vocab_size), so the
# guard is simply vocab_size <= SPARSE_DIM.
SPARSE_DIM = 2**20  # 1_048_576


def main() -> int:
    root = bootstrap()

    parser = argparse.ArgumentParser(description="Fit the shared BM25 sparse encoder.")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_ENCODER_PATH,
        help=f"Where to write the encoder (default: {DEFAULT_ENCODER_PATH})",
    )
    parser.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help="Explicit collection names to fit on (default: auto-discover all "
        "hybrid writing collections).",
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=1,
        help="Refuse to fit if fewer than this many documents were collected.",
    )
    args = parser.parse_args()

    from kbase.vector.sync_client import get_qdrant_client
    from kbase.vector.bm25_encoder import BM25Encoder

    client = get_qdrant_client()

    collections = list_hybrid_writing_collections(client, only=args.collections)
    if not collections:
        print("No hybrid writing collections found; nothing to fit.", file=sys.stderr)
        return 1

    print(f"Fitting shared BM25 encoder over {len(collections)} collection(s):", file=sys.stderr)
    documents = []
    for name in collections:
        count = 0
        for point in scroll_points(client, name, with_vectors=False):
            text = point_text(point)
            if text:
                documents.append(text)
                count += 1
        print(f"  {name}: {count} documents", file=sys.stderr)

    if len(documents) < args.min_docs:
        print(
            f"Collected only {len(documents)} documents (< --min-docs "
            f"{args.min_docs}); refusing to fit a degenerate encoder.",
            file=sys.stderr,
        )
        return 1

    encoder = BM25Encoder()
    encoder.fit(documents, verbose=True)

    vocab_size = len(encoder.vocab_to_index)
    if vocab_size > SPARSE_DIM:
        print(
            f"ERROR: fitted vocabulary ({vocab_size}) exceeds the sparse index "
            f"space ({SPARSE_DIM}). The Qdrant sparse arm and the Postgres "
            f"sparsevec({SPARSE_DIM}) mirror cannot represent it. Reduce the "
            "corpus or raise the sparse dimension before proceeding.",
            file=sys.stderr,
        )
        return 1

    encoder.save(args.output)

    stats = encoder.get_stats()
    print("", file=sys.stderr)
    print("Fitted BM25 encoder:", file=sys.stderr)
    print(f"  documents:        {len(documents)}", file=sys.stderr)
    print(f"  vocabulary size:  {vocab_size} (limit {SPARSE_DIM})", file=sys.stderr)
    print(f"  avg doc length:   {stats.average_document_length:.1f} tokens", file=sys.stderr)
    print(f"  saved to:         {args.output}", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "Next: deploy this artifact to every instance (BM25_ENCODER_PATH), set "
        "BM25_REQUIRE_FITTED=1, then run scripts/reindex_sparse.py.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
