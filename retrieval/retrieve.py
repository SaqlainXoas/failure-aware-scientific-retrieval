"""First-stage retrieval: bm25_retrieve / dense_retrieve / hybrid_retrieve.

All three return the same schema: a flat list of row dicts
{"query_id": str, "doc_id": str, "rank": int, "score": float}, rank 1-indexed and
ascending, so callers (run.py, evaluate.py) never need pipeline-specific handling.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from retrieval.cache import cache_key, cache_path, corpus_fingerprint, load_or_compute

PREPROCESSING_VERSION = "v1"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Pinned explicitly rather than left to bm25s' defaults so the manifest can record them and
# a future bm25s release cannot silently change tokenization, which must stay identical across
# every BM25 run for the comparison to hold. These values *are* bm25s 0.3.x's defaults,
# so pinning them does not change any existing result or invalidate the cache.
BM25_TOKENIZER = {
    "lower": True,
    "token_pattern": r"(?u)\b\w\w+\b",
    "stopwords": "english",
    "stemmer": None,
}


def _load_bm25_index(path: Path):
    import bm25s

    index = bm25s.BM25.load(str(path), load_corpus=False, show_progress=False)
    doc_ids = json.loads((path / "doc_ids.json").read_text())
    return index, doc_ids


def _save_bm25_index(path: Path, value) -> None:
    index, doc_ids = value
    path.mkdir(parents=True, exist_ok=True)
    index.save(str(path), show_progress=False)
    (path / "doc_ids.json").write_text(json.dumps(doc_ids))


def bm25_retrieve(
    corpus: dict[str, str],
    queries: dict[str, str],
    top_k: int = 100,
    cache_dir: str | Path | None = None,
    force: bool = False,
    cache_hits: dict[str, bool] | None = None,
    params_out: dict[str, Any] | None = None,
) -> list[dict]:
    """BM25 (bm25s, library-default k1/b) over title+abstract corpus text; deterministic tokenization.

    `params_out`, when given, is filled with the scoring and tokenization parameters actually
    used, so the caller can record concrete values in the run manifest instead of an
    unresolvable "library default".
    """
    import bm25s

    doc_ids = sorted(corpus)
    query_ids = sorted(queries)

    def compute_index():
        texts = [corpus[doc_id] for doc_id in doc_ids]
        corpus_tokens = bm25s.tokenize(texts, show_progress=False, **BM25_TOKENIZER)
        index = bm25s.BM25()
        index.index(corpus_tokens, show_progress=False)
        return index, doc_ids

    if cache_dir is not None:
        key = cache_key(
            "bm25",
            corpus_fingerprint=corpus_fingerprint(corpus),
            bm25s_version=bm25s.__version__,
            preprocessing_version=PREPROCESSING_VERSION,
        )
        path = cache_path("bm25", key, cache_dir=cache_dir)
        (index, indexed_doc_ids), was_cached = load_or_compute(
            path, compute_index, force, _load_bm25_index, _save_bm25_index
        )
        if cache_hits is not None:
            cache_hits["bm25_index"] = was_cached
    else:
        index, indexed_doc_ids = compute_index()

    if params_out is not None:
        params_out.update(
            {
                "library": "bm25s",
                "library_version": bm25s.__version__,
                "k1": float(index.k1),
                "b": float(index.b),
                "delta": float(index.delta),
                "method": index.method,
                "idf_method": index.idf_method,
                "tokenizer": dict(BM25_TOKENIZER),
                "corpus_text_format": "<title>\\n<abstract>",
                "preprocessing_version": PREPROCESSING_VERSION,
            }
        )

    query_tokens = bm25s.tokenize(
        [queries[qid] for qid in query_ids], show_progress=False, **BM25_TOKENIZER
    )
    k = min(top_k, len(indexed_doc_ids))
    results = index.retrieve(query_tokens, corpus=indexed_doc_ids, k=k, show_progress=False)

    rows = []
    for i, query_id in enumerate(query_ids):
        for rank, (doc_id, score) in enumerate(
            zip(results.documents[i], results.scores[i]), start=1
        ):
            rows.append({"query_id": query_id, "doc_id": str(doc_id), "rank": rank, "score": float(score)})
    return rows


def _load_embeddings(path: Path):
    embeddings = np.load(path / "embeddings.npy")
    ids = json.loads((path / "ids.json").read_text())
    return embeddings, ids


def _save_embeddings(path: Path, value) -> None:
    embeddings, ids = value
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "embeddings.npy", embeddings)
    (path / "ids.json").write_text(json.dumps(ids))


def dense_retrieve(
    corpus: dict[str, str],
    queries: dict[str, str],
    top_k: int = 100,
    model_name: str = "BAAI/bge-small-en-v1.5",
    model_revision: str | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    cache_dir: str | Path | None = None,
    force: bool = False,
    cache_hits: dict[str, bool] | None = None,
) -> list[dict]:
    """Dense retrieval via BGE-small; query-instruction prefix on queries only; normalized cosine sim, exact matmul."""
    import sentence_transformers

    doc_ids = sorted(corpus)
    query_ids = sorted(queries)

    model = None

    def get_model():
        nonlocal model
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, revision=model_revision, device=device)
        return model

    def compute_corpus_embeddings():
        texts = [corpus[doc_id] for doc_id in doc_ids]
        embeddings = get_model().encode(
            texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False
        )
        return np.asarray(embeddings, dtype=np.float32), doc_ids

    def compute_query_embeddings():
        texts = [QUERY_INSTRUCTION + queries[qid] for qid in query_ids]
        embeddings = get_model().encode(
            texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False
        )
        return np.asarray(embeddings, dtype=np.float32), query_ids

    if cache_dir is not None:
        base_fields = dict(
            model_name=model_name,
            model_revision=model_revision,
            sentence_transformers_version=sentence_transformers.__version__,
            preprocessing_version=PREPROCESSING_VERSION,
        )
        corpus_key = cache_key(
            "dense_corpus", corpus_fingerprint=corpus_fingerprint(corpus), **base_fields
        )
        corpus_path = cache_path("dense_corpus", corpus_key, cache_dir=cache_dir)
        (corpus_embeddings, indexed_doc_ids), corpus_was_cached = load_or_compute(
            corpus_path, compute_corpus_embeddings, force, _load_embeddings, _save_embeddings
        )

        query_key = cache_key(
            "dense_query",
            query_fingerprint=corpus_fingerprint(queries),
            instruction=QUERY_INSTRUCTION,
            **base_fields,
        )
        query_path = cache_path("dense_query", query_key, cache_dir=cache_dir)
        (query_embeddings, indexed_query_ids), query_was_cached = load_or_compute(
            query_path, compute_query_embeddings, force, _load_embeddings, _save_embeddings
        )
        if cache_hits is not None:
            cache_hits["dense_corpus_embeddings"] = corpus_was_cached
            cache_hits["dense_query_embeddings"] = query_was_cached
    else:
        corpus_embeddings, indexed_doc_ids = compute_corpus_embeddings()
        query_embeddings, indexed_query_ids = compute_query_embeddings()

    similarity = query_embeddings @ corpus_embeddings.T
    k = min(top_k, len(indexed_doc_ids))
    top_indices = np.argsort(-similarity, axis=1, kind="stable")[:, :k]

    rows = []
    for i, query_id in enumerate(indexed_query_ids):
        for rank, doc_idx in enumerate(top_indices[i], start=1):
            rows.append(
                {
                    "query_id": query_id,
                    "doc_id": indexed_doc_ids[doc_idx],
                    "rank": rank,
                    "score": float(similarity[i, doc_idx]),
                }
            )
    return rows


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuses ranked doc_id lists by sum(1/(k+rank)); rank-based so score scales never need normalization.

    Returns (doc_id, fused_score) sorted by fused_score desc, ties broken by doc_id ascending
    for determinism.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _rows_by_query(rows: list[dict]) -> dict[str, list[str]]:
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    return {
        query_id: [row["doc_id"] for row in sorted(query_rows, key=lambda r: r["rank"])]
        for query_id, query_rows in by_query.items()
    }


def hybrid_retrieve(
    bm25_rows: list[dict], dense_rows: list[dict], k: int = 60, top_k: int | None = None
) -> list[dict]:
    """Reciprocal Rank Fusion of two top-100 rankings; rank-based, so raw BM25/dense score scales are irrelevant.

    The fused list is truncated to `top_k` so hybrid emits the same candidate depth as bm25 and
    dense. Depth has to match: the union of two 100-doc lists is up to 200 docs, and the Phase 4
    within-query score normalization would otherwise use a different denominator for this pipeline.
    """
    bm25_by_query = _rows_by_query(bm25_rows)
    dense_by_query = _rows_by_query(dense_rows)
    query_ids = sorted(set(bm25_by_query) | set(dense_by_query))

    rows = []
    for query_id in query_ids:
        fused = reciprocal_rank_fusion(
            [bm25_by_query.get(query_id, []), dense_by_query.get(query_id, [])], k=k
        )
        if top_k is not None:
            fused = fused[:top_k]
        for rank, (doc_id, score) in enumerate(fused, start=1):
            rows.append({"query_id": query_id, "doc_id": doc_id, "rank": rank, "score": score})
    return rows
