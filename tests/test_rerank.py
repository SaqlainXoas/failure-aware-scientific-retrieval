import pytest

import retrieval.rerank as rerank_module
from retrieval.rerank import rerank


def _fake_score_pairs(score_lookup):
    calls = []

    def _scorer(pairs, model_name, model_revision, device, batch_size):
        calls.append(len(pairs))
        return [score_lookup[pair] for pair in pairs]

    return _scorer, calls


def _patch_scorer(monkeypatch, score_lookup):
    scorer, calls = _fake_score_pairs(score_lookup)
    monkeypatch.setattr(rerank_module, "_score_pairs", scorer)
    return calls


def test_rerank_only_uses_given_candidates(monkeypatch):
    candidate_rows = [
        {"query_id": "q1", "doc_id": "a", "rank": 1, "score": 1.0},
        {"query_id": "q1", "doc_id": "b", "rank": 2, "score": 0.5},
    ]
    queries = {"q1": "query text"}
    corpus = {"a": "doc a", "b": "doc b", "e": "unseen doc"}
    score_lookup = {("query text", "doc a"): 0.9, ("query text", "doc b"): 0.1}
    _patch_scorer(monkeypatch, score_lookup)

    rows = rerank(candidate_rows, queries, corpus, top_k=50)

    assert {row["doc_id"] for row in rows} == {"a", "b"}


def test_rerank_respects_top_k_depth(monkeypatch):
    candidate_rows = [
        {"query_id": "q1", "doc_id": f"d{i}", "rank": i, "score": 1.0 / i} for i in range(1, 101)
    ]
    queries = {"q1": "query text"}
    corpus = {f"d{i}": f"doc {i}" for i in range(1, 101)}
    score_lookup = {("query text", f"doc {i}"): float(i) for i in range(1, 101)}
    _patch_scorer(monkeypatch, score_lookup)

    rows = rerank(candidate_rows, queries, corpus, top_k=50)

    assert {row["doc_id"] for row in rows} == {f"d{i}" for i in range(1, 51)}


def test_rerank_orders_by_score_desc_with_doc_id_tiebreak(monkeypatch):
    candidate_rows = [
        {"query_id": "q1", "doc_id": "b", "rank": 1, "score": 1.0},
        {"query_id": "q1", "doc_id": "a", "rank": 2, "score": 1.0},
        {"query_id": "q1", "doc_id": "c", "rank": 3, "score": 1.0},
    ]
    queries = {"q1": "query text"}
    corpus = {"a": "doc a", "b": "doc b", "c": "doc c"}
    score_lookup = {
        ("query text", "doc a"): 0.5,
        ("query text", "doc b"): 0.5,
        ("query text", "doc c"): 0.9,
    }
    _patch_scorer(monkeypatch, score_lookup)

    rows = rerank(candidate_rows, queries, corpus, top_k=50)

    ranked = sorted(rows, key=lambda r: r["rank"])
    assert [row["doc_id"] for row in ranked] == ["c", "a", "b"]
    assert [row["rank"] for row in ranked] == [1, 2, 3]


def test_rerank_schema(monkeypatch):
    candidate_rows = [
        {"query_id": "q1", "doc_id": "a", "rank": 1, "score": 1.0},
        {"query_id": "q1", "doc_id": "b", "rank": 2, "score": 0.5},
    ]
    queries = {"q1": "query text"}
    corpus = {"a": "doc a", "b": "doc b"}
    score_lookup = {("query text", "doc a"): 0.9, ("query text", "doc b"): 0.1}
    _patch_scorer(monkeypatch, score_lookup)

    rows = rerank(candidate_rows, queries, corpus, top_k=50)

    for row in rows:
        assert set(row) == {"query_id", "doc_id", "rank", "score"}
    ranked = sorted(rows, key=lambda r: r["rank"])
    assert [row["rank"] for row in ranked] == list(range(1, len(ranked) + 1))


def test_rerank_cache_reused_without_force(monkeypatch, tmp_path):
    candidate_rows = [
        {"query_id": "q1", "doc_id": "a", "rank": 1, "score": 1.0},
        {"query_id": "q1", "doc_id": "b", "rank": 2, "score": 0.5},
    ]
    queries = {"q1": "query text"}
    corpus = {"a": "doc a", "b": "doc b"}
    score_lookup = {("query text", "doc a"): 0.9, ("query text", "doc b"): 0.1}
    calls = _patch_scorer(monkeypatch, score_lookup)

    rerank(candidate_rows, queries, corpus, top_k=50, cache_dir=tmp_path)
    assert sum(calls) == 2

    calls.clear()
    rerank(candidate_rows, queries, corpus, top_k=50, cache_dir=tmp_path)
    assert sum(calls) == 0


def test_rerank_force_recomputes(monkeypatch, tmp_path):
    candidate_rows = [
        {"query_id": "q1", "doc_id": "a", "rank": 1, "score": 1.0},
        {"query_id": "q1", "doc_id": "b", "rank": 2, "score": 0.5},
    ]
    queries = {"q1": "query text"}
    corpus = {"a": "doc a", "b": "doc b"}
    score_lookup = {("query text", "doc a"): 0.9, ("query text", "doc b"): 0.1}
    calls = _patch_scorer(monkeypatch, score_lookup)

    rerank(candidate_rows, queries, corpus, top_k=50, cache_dir=tmp_path)
    calls.clear()

    rerank(candidate_rows, queries, corpus, top_k=50, cache_dir=tmp_path, force=True)
    assert sum(calls) == 2
