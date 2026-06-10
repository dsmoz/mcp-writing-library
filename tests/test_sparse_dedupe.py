"""
Regression tests for sparse-vector index deduplication.

Bug: admin_add / contribute upserts intermittently failed with Qdrant 422
"[points[0].vector.?.indices: must be unique]". Root cause: distinct tokens
collided onto the same index via ``hash(token) % SPARSE_HASH_MOD`` in
generate_sparse_vector(); the salted hash() made it content- and run-dependent.
Fix: dedupe_sparse() collapses colliding indices and sums their values before
the sparse vector is returned (covering every collection that upserts).

The criterion text below is the exact string that failed twice in production
before succeeding on a shorter rephrase.
"""
import re

from kbase.vector import sync_embeddings
from kbase.vector.sync_embeddings import dedupe_sparse, generate_sparse_vector

FAILING_CRITERION = (
    "Presents ranked priorities, each structured as priority -> evidence -> "
    "specific recommendation -> responsible actor (Global Fund, CCM, "
    "government), traceable to dialogue findings."
)


def _strictly_increasing(xs) -> bool:
    return all(a < b for a, b in zip(xs, xs[1:]))


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

    def test_unique_even_under_forced_collisions(self, monkeypatch):
        # Shrink the index space so distinct tokens are guaranteed to collide,
        # making the regression deterministic regardless of PYTHONHASHSEED.
        monkeypatch.setattr(sync_embeddings, "SPARSE_HASH_MOD", 4)
        indices, values = generate_sparse_vector(FAILING_CRITERION)
        assert indices == sorted(set(indices))
        assert len(indices) == len(values)
        assert all(v > 0 for v in values)
        # No term-frequency mass is lost when colliding indices are merged.
        raw_tokens = re.findall(r"\b[a-z0-9]{2,}\b", FAILING_CRITERION.lower())
        assert sum(values) == float(len(raw_tokens))

    def test_result_is_valid_qdrant_sparse_vector(self):
        from qdrant_client.models import SparseVector

        indices, values = generate_sparse_vector(FAILING_CRITERION)
        sv = SparseVector(indices=indices, values=values)
        assert len(sv.indices) == len(set(sv.indices))
        assert len(sv.indices) == len(sv.values)

    def test_empty_text_returns_empty(self):
        assert generate_sparse_vector("") == ([], [])
        assert generate_sparse_vector("   !!!  ") == ([], [])
