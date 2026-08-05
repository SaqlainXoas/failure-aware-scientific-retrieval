"""Top-`top_k` cross-encoder reranking of each query's own first-stage candidates."""

import json
from pathlib import Path

from retrieval.cache import cache_key, cache_path, corpus_fingerprint

PREPROCESSING_VERSION = "v1"


def _score_pairs(
    pairs: list[tuple[str, str]],
    model_name: str,
    model_revision: str | None,
    device: str,
    batch_size: int,
) -> list[float]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, revision=model_revision, device=device)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(s) for s in scores]


def _candidates_by_query(candidate_rows: list[dict], top_k: int) -> dict[str, list[str]]:
    by_query: dict[str, list[dict]] = {}
    for row in candidate_rows:
        if row["rank"] <= top_k:
            by_query.setdefault(row["query_id"], []).append(row)
    return {
        query_id: [row["doc_id"] for row in sorted(rows, key=lambda r: r["rank"])]
        for query_id, rows in by_query.items()
    }


def rerank(
    candidate_rows: list[dict],
    queries: dict[str, str],
    corpus: dict[str, str],
    top_k: int = 50,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    model_revision: str | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    cache_dir: str | Path | None = None,
    force: bool = False,
    cache_hits: dict[str, bool] | None = None,
) -> list[dict]:
    """Reranks each query's own top-`top_k` first-stage candidates via a frozen cross-encoder.

    Only ever scores and emits (query_id, doc_id) pairs already present in `candidate_rows` at
    rank <= top_k — this is what guarantees no leakage of documents outside a pipeline's own
    candidate set.
    """
    candidates = _candidates_by_query(candidate_rows, top_k)

    scores: dict[str, dict[str, float]] = {}
    cache_path_file: Path | None = None
    if cache_dir is not None:
        key = cache_key(
            "reranker_scores",
            model_name=model_name,
            model_revision=model_revision,
            corpus_fingerprint=corpus_fingerprint(corpus),
            query_fingerprint=corpus_fingerprint(queries),
            preprocessing_version=PREPROCESSING_VERSION,
        )
        cache_path_file = cache_path("reranker_scores", key, cache_dir=cache_dir) / "scores.json"
        if not force and cache_path_file.exists():
            scores = json.loads(cache_path_file.read_text())

    # A (query_id, doc_id) cross-encoder score doesn't depend on which pipeline surfaced the
    # candidate, so the cache is a pair-level dict merged incrementally across calls rather than
    # an all-or-nothing value — load_or_compute assumes one atomic cached value per call, which
    # doesn't fit a candidate set that differs per pipeline while remaining safely mergeable.
    missing_pairs: list[tuple[str, str]] = []
    for query_id, doc_ids in candidates.items():
        for doc_id in doc_ids:
            if force or doc_id not in scores.get(query_id, {}):
                missing_pairs.append((query_id, doc_id))

    if missing_pairs:
        pair_texts = [(queries[qid], corpus[did]) for qid, did in missing_pairs]
        computed = _score_pairs(pair_texts, model_name, model_revision, device, batch_size)
        for (query_id, doc_id), score in zip(missing_pairs, computed):
            scores.setdefault(query_id, {})[doc_id] = score

    if cache_path_file is not None and missing_pairs:
        cache_path_file.parent.mkdir(parents=True, exist_ok=True)
        cache_path_file.write_text(json.dumps(scores))

    if cache_hits is not None:
        cache_hits["reranker_pairs_computed"] = len(missing_pairs)

    rows = []
    for query_id, doc_ids in candidates.items():
        query_scores = scores[query_id]
        ranked = sorted(doc_ids, key=lambda doc_id: (-query_scores[doc_id], doc_id))
        for rank, doc_id in enumerate(ranked, start=1):
            rows.append(
                {"query_id": query_id, "doc_id": doc_id, "rank": rank, "score": query_scores[doc_id]}
            )
    return rows
