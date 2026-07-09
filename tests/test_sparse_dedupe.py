"""
Tests for the sparse-vector encoder.

Two bugs are guarded here:

1. Qdrant 422 "[indices: must be unique]" — distinct tokens can collide onto the
   same sparse index, and Qdrant rejects a SparseVector with duplicate indices.
   dedupe_sparse() collapses colliding indices (summing their values) before the
   vector is returned.

2. Non-deterministic indices (the big one) — generate_sparse_vector() used to
   derive indices from the builtin ``hash(token)``, which CPython salts per
   process (PYTHONHASHSEED). Indices produced when a document was *indexed*
   therefore never matched the indices produced for the same terms at *query*
   time in a different process, so the hybrid sparse arm returned nothing and
   hybrid search silently collapsed to dense-only. The encoder is now
   deterministic (a fitted BM25 vocab when an artifact is present, else a
   deterministic md5-hashed TF fallback).
"""
import hashlib
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from kbase.vector.sync_embeddings import dedupe_sparse, generate_sparse_vector

FAILING_CRITERION = (
    "Presents ranked priorities, each structured as priority -> evidence -> "
    "specific recommendation -> responsible actor (Global Fund, CCM, "
    "government), traceable to dialogue findings."
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _strictly_increasing(xs) -> bool:
    return all(a < b for a, b in zip(xs, xs[1:]))


@pytest.fixture()
def _reset_bm25_cache():
    """Reset the global BM25 encoder cache around a test that loads an artifact,
    so it does not leak into tests that expect the md5 fallback."""
    import kbase.vector.bm25_encoder as be

    yield
    be._bm25_encoder = None
    be._encoder_path = None


class TestDedupeSparse:
    def test_collapses_duplicate_indices_summing_values(self):
        idx, val = dedupe_sparse([5, 1, 5, 1, 5], [1.0, 2.0, 3.0, 4.0, 1.0])
        assert idx == [1, 5]
        assert val == [6.0, 5.0]  # index 1: 2+4, index 5: 1+3+1

    def test_already_unique_returns_sorted(self):
        idx, val = dedupe_sparse([3, 1, 2], [10.0, 20.0, 30.0])
        assert idx == [1, 2, 3]
        assert val == [20.0, 30.0, 10.0]

    def test_empty(self):
        assert dedupe_sparse([], []) == ([], [])

    def test_output_unique_and_ascending(self):
        idx, _ = dedupe_sparse([9, 9, 9, 4, 4, 1], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert len(idx) == len(set(idx))
        assert _strictly_increasing(idx)


class TestGenerateSparseVector:
    def test_failing_criterion_has_unique_indices(self):
        indices, values = generate_sparse_vector(FAILING_CRITERION)
        assert len(indices) == len(set(indices)), "indices must be unique (Qdrant 422 guard)"
        assert len(indices) == len(values)
        assert _strictly_increasing(indices)

    def test_deduped_even_when_encoder_yields_collisions(self, monkeypatch):
        # generate_sparse_vector delegates to hybrid_embeddings.get_sparse_embedding;
        # force it to return duplicate indices and confirm the Qdrant-422 dedupe
        # guard still collapses them (summing values) after the refactor.
        from qdrant_client.models import SparseVector

        def fake_get_sparse_embedding(text, encoder_path=None):
            return SparseVector(indices=[7, 3, 7, 3, 7], values=[1.0, 2.0, 1.0, 2.0, 1.0])

        monkeypatch.setattr(
            "kbase.vector.hybrid_embeddings.get_sparse_embedding",
            fake_get_sparse_embedding,
        )
        indices, values = generate_sparse_vector("anything")
        assert indices == [3, 7]
        assert values == [4.0, 3.0]  # index 3: 2+2, index 7: 1+1+1

    def test_result_is_valid_qdrant_sparse_vector(self):
        from qdrant_client.models import SparseVector

        indices, values = generate_sparse_vector(FAILING_CRITERION)
        sv = SparseVector(indices=indices, values=values)
        assert len(sv.indices) == len(set(sv.indices))
        assert len(sv.indices) == len(sv.values)

    def test_indices_within_sparse_dimension(self):
        # Every index must fit the Postgres sparsevec(1048576) mirror (and the
        # +1 shift the adapter applies), i.e. stay strictly below 2**20.
        indices, _ = generate_sparse_vector(FAILING_CRITERION)
        assert all(0 <= i < 2**20 for i in indices)

    def test_empty_text_returns_empty(self):
        assert generate_sparse_vector("") == ([], [])
        assert generate_sparse_vector("   !!!  ") == ([], [])


class TestDeterminism:
    """The core regression: identical input -> identical indices, regardless of
    process or PYTHONHASHSEED."""

    def test_uses_md5_not_builtin_hash(self):
        # With no fitted artifact, the fallback maps a token via md5 (stable
        # across processes), NOT builtin hash (salted per process). Pin the
        # exact index so a regression back to hash() fails loudly.
        indices, values = generate_sparse_vector("priorities")
        expected_index = int(hashlib.md5(b"priorities").hexdigest(), 16) % (2**20)
        assert indices == [expected_index]
        assert values == pytest.approx([math.log(2.0)])  # tf=1 -> 1*log(1+1/1)

    def test_indices_independent_of_pythonhashseed(self):
        # The real regression: run the encoder in two child processes with
        # different PYTHONHASHSEED and confirm identical indices. This fails on
        # the old builtin-hash() encoder and passes on the deterministic one.
        snippet = (
            "import sys;"
            f"sys.path.insert(0, {str(_PROJECT_ROOT)!r});"
            f"sys.path.insert(0, {str(_PROJECT_ROOT / 'vendor')!r});"
            "from kbase.vector.sync_embeddings import generate_sparse_vector;"
            "i,_=generate_sparse_vector('ranked priorities traceable to dialogue findings');"
            "print(','.join(map(str,i)))"
        )
        outs = []
        for seed in ("0", "1", "42"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            # Ensure the child does not pick up a stray fitted artifact.
            env.pop("BM25_ENCODER_PATH", None)
            env.pop("BM25_REQUIRE_FITTED", None)
            out = subprocess.check_output(
                [sys.executable, "-c", snippet], env=env, text=True
            ).strip()
            outs.append(out)
        assert outs[0] and outs[0] == outs[1] == outs[2], outs


class TestFittedEncoder:
    """When a fitted BM25 artifact is present, index and query share one vocab,
    so query indices are a subset of the source document's indices — the exact
    alignment the builtin-hash bug destroyed."""

    def test_query_indices_subset_of_doc_indices(self, monkeypatch, _reset_bm25_cache):
        from kbase.vector.bm25_encoder import BM25Encoder

        docs = [
            "The Global Fund CCM presents ranked priorities traceable to dialogue findings.",
            "Evidence based recommendation with a responsible actor and specific action.",
            "Poetry rubric evaluates rhythm meter and imagery in a stanza.",
        ]
        encoder = BM25Encoder().fit(docs, verbose=False)

        with tempfile.TemporaryDirectory() as tmp:
            art = os.path.join(tmp, "bm25_encoder.pkl")
            encoder.save(art)
            monkeypatch.setenv("BM25_ENCODER_PATH", art)
            monkeypatch.setenv("BM25_REQUIRE_FITTED", "1")

            doc_idx, _ = generate_sparse_vector(docs[0])
            q_idx, _ = generate_sparse_vector("ranked priorities dialogue findings")

            assert q_idx, "query sparse arm must be non-empty"
            assert set(q_idx).issubset(set(doc_idx))
            assert all(0 <= i < 2**20 for i in doc_idx + q_idx)
