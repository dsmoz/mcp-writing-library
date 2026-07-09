# Runbook: deterministic sparse encoder + full sparse re-index

## What was broken

The sparse (keyword / BM25) arm of hybrid search derived its indices from
Python's builtin `hash(token)`:

```python
# vendor/kbase/vector/sync_embeddings.py (old)
token_hash = hash(token) % SPARSE_HASH_MOD   # SPARSE_HASH_MOD = 2**20
```

CPython **salts `hash(str)` per process** via `PYTHONHASHSEED`. So the indices
computed when a document was **indexed** (one process) and the indices computed
for the same terms at **query** time (a different process) never agreed. The
query sparse vector's nonzero indices essentially never lined up with the stored
ones.

Both live paths used this exact function:

- index: `sync_indexing.py` → `generate_sparse_vectors_batch`
- query: `sync_search.py` → `generate_sparse_vector`

Every MCP tool searches through `sync_search.semantic_search`, so **the deployed
Qdrant hybrid sparse arm has effectively been returning empty on real queries —
hybrid silently collapsed to dense-only in production.** A diagnostic
(`scripts/which_encoder.py`) measured 0% index-set overlap between stored sparse
vectors and every encoder in the codebase, across `PYTHONHASHSEED` 0/1/42.

The stored sparse vectors are therefore **unreproducible garbage** and must be
regenerated. Dense vectors were always fine.

## The fix (code)

`generate_sparse_vector` is now deterministic. It delegates to
`hybrid_embeddings.get_sparse_embedding`, which uses:

1. a **fitted BM25 vocab encoder** when an artifact is present
   (`BM25_ENCODER_PATH`, default `.data/bm25_encoder.pkl`), else
2. a **deterministic md5-hashed TF fallback** (same 2**20 index space).

Both index and query call the one function, so both use the identical encoder
and their indices reproducibly align across processes. `dedupe_sparse` still
guards the Qdrant "indices must be unique" 422.

Two env vars (see `.env.example`):

| Var | Default | Meaning |
|-----|---------|---------|
| `BM25_ENCODER_PATH` | `.data/bm25_encoder.pkl` | Fitted encoder artifact location. |
| `BM25_REQUIRE_FITTED` | `0` | If `1`, refuse the md5 fallback and raise if the artifact is missing — prevents silent index/query encoder drift across a deploy. Set to `1` in production. |

## Why a re-index is mandatory

The encoder identity that produced the **stored** vectors must match the encoder
used at **query** time. Existing stored vectors came from the broken builtin-hash
encoder, so every writing collection must be re-indexed with the new encoder.
Only the `sparse` named vector is regenerated; dense vectors and payloads are
left untouched.

> ⚠️ **Ordering matters.** The md5 fallback and a fitted encoder occupy
> *different* index spaces. If you index with one and query with the other you
> reproduce the same "sparse arm returns nothing" outage. Deploy the SAME encoder
> everywhere, re-index, then serve queries.

## Procedure (Qdrant)

Chosen encoder: **fitted BM25** (per Danilo). Run from the repo root with the
target Qdrant env (`QDRANT_URL`, `QDRANT_API_KEY`) configured.

1. **Fit the shared encoder** over the union of every writing collection. A
   single shared vocab keeps cross-collection / cross-tenant sparse search
   coherent — do not fit per collection.
   ```bash
   uv run python scripts/train_bm25_encoder.py --output .data/bm25_encoder.pkl
   ```
   The script asserts `vocab_size <= 2**20` so the artifact fits both Qdrant and
   the Postgres `sparsevec(1048576)` mirror.

2. **Deploy the artifact to every writing-library instance** at the same
   `BM25_ENCODER_PATH` (e.g. bake into the image or mount on a Railway volume),
   and set `BM25_REQUIRE_FITTED=1`. Every instance must serve queries with the
   exact same `.pkl`.

3. **Re-index the sparse vectors** (idempotent; `--dry-run` first to preview):
   ```bash
   uv run python scripts/reindex_sparse.py --dry-run
   uv run python scripts/reindex_sparse.py
   ```
   This recomputes only the `sparse` named vector per point via
   `update_vectors` — dense + payload are preserved, no dense re-embed.

4. **Verify**:
   ```bash
   uv run python scripts/which_encoder.py
   ```
   Expected after re-index:
   - **self-consistency recall ≈ 100%** (stored ∩ recomputed / stored), vs ~0%
     before;
   - **text-query self-rank**: non-empty sparse arm and source passages
     `hit@10 ≈ 100%` on their own text.

   Exit code is non-zero if either check fails.

## Procedure (Postgres / pgvector mirror — Track-B)

Track-B migrated Danilo's tenant (`usr_93f07c15894b4877`) into the dsmoz-intel
Postgres project (`bwbghsnnrszdcmwqzjwv`), tables `writing_passages` etc., with a
`sparse_embedding sparsevec(1048576)` column. The migration **copied the same
garbage sparse data**, so the pg column must be regenerated with the identical
fitted encoder. The pg adapter (`pgvector_adapter.py`, in the Track-B migration
work) stores indices **1-based** and shifts query indices **+1**.

Regenerate the pg sparse column the same way:

1. Use the **same** `.data/bm25_encoder.pkl` fitted above (do not re-fit — the
   vocab index space must match Qdrant).
2. For each row, recompute `generate_sparse_vector(text)` on the row's stored
   text, then write `sparse_embedding` as a `sparsevec(1048576)` literal, applying
   the adapter's **+1** index shift (0-based encoder index `i` → pg index `i+1`;
   this is why the fit guard is `vocab_size <= 2**20`, keeping the shifted max
   `≤ 1048576`).
3. Verify a stored-vector self-query and a text-query on the pg side ranks the
   source passage highly, mirroring the Qdrant `which_encoder.py` checks.

> This repo does not contain `pgvector_adapter.py` or DB credentials; run the pg
> regeneration from the Track-B migration workspace with dsmoz-intel access, or
> hand it back to whoever owns that environment. The encoder artifact and
> `generate_sparse_vector` are the shared, authoritative pieces.

## Rollback

The change is additive and deterministic. To revert the *code* you would restore
the builtin-hash encoder, but that reintroduces the outage — don't. If a re-index
must be undone, dense search is unaffected throughout; the sparse arm simply
returns to its prior (broken) state. There is no destructive data step: dense
vectors and payloads are never modified.
