"""Integration smoke test required by plan §16: a tiny synthetic corpus run end to end through
retrieve -> rerank -> evaluate -> extract features -> calibrate.

The cross-encoder is mocked (plan §16: "Mock model scores rather than testing external model
quality"); BM25 is real because it is cheap and its tokenization is part of what we want
exercised. This covers the wiring in run.py, which is where the split and leakage rules actually
live — the unit tests only cover the pieces in isolation.
"""

import hashlib

import pytest

import retrieval.rerank as rerank_module
from retrieval.confidence import COMMON_FEATURES, HYBRID_FEATURES
from retrieval.evaluate import CATEGORIES, decomposition_metrics
from retrieval.rerank import rerank
from retrieval.retrieve import bm25_retrieve
from retrieval.run import (
    compute_metrics,
    compute_query_labels,
    exploratory_feature_names,
    per_query_metrics,
    run_calibration,
    run_confidence_features,
)

N_DOCS = 24
N_QUERIES = 12
RERANK_DEPTH = 5


def _synthetic_corpus():
    topics = ["cardiac", "immune", "genome", "neuron"]
    corpus = {}
    for i in range(N_DOCS):
        topic = topics[i % len(topics)]
        corpus[f"d{i}"] = f"{topic} study number {i}\nAn abstract about {topic} response and outcome {i}."
    return corpus


def _synthetic_queries():
    topics = ["cardiac", "immune", "genome", "neuron"]
    return {f"q{i}": f"{topics[i % len(topics)]} response outcome" for i in range(N_QUERIES)}


def _synthetic_qrels(corpus, queries):
    # Each query's gold doc is the same-topic doc at a matching offset; deterministic and
    # deliberately mixed so both successes and failures occur.
    qrels = {}
    for i, query_id in enumerate(sorted(queries)):
        qrels[query_id] = {f"d{(i * 5) % N_DOCS}": 1}
    return qrels


@pytest.fixture
def mocked_cross_encoder(monkeypatch):
    """Deterministic pseudo-scores from a hash of the pair — no model download, no network."""

    def fake_score_pairs(pairs, model_name, model_revision, device, batch_size):
        return [
            int.from_bytes(
                hashlib.sha256(f"{query}\0{document}".encode()).digest()[:4], "big"
            )
            / 2**32
            for query, document in pairs
        ]

    monkeypatch.setattr(rerank_module, "_score_pairs", fake_score_pairs)


def test_end_to_end_retrieve_rerank_evaluate_featurize_calibrate(mocked_cross_encoder):
    corpus = _synthetic_corpus()
    queries = _synthetic_queries()
    qrels = _synthetic_qrels(corpus, queries)

    # retrieve
    rows = bm25_retrieve(corpus, queries, top_k=10)
    assert {row["query_id"] for row in rows} == set(queries)
    assert all(set(row) == {"query_id", "doc_id", "rank", "score"} for row in rows)

    # rerank — only the pipeline's own top-RERANK_DEPTH candidates
    reranked = rerank(rows, queries, corpus, top_k=RERANK_DEPTH)
    first_stage_candidates = {
        (row["query_id"], row["doc_id"]) for row in rows if row["rank"] <= RERANK_DEPTH
    }
    assert {(row["query_id"], row["doc_id"]) for row in reranked} == first_stage_candidates

    # evaluate
    metrics = compute_metrics(rows, qrels)
    reranked_metrics = compute_metrics(reranked, qrels)
    assert metrics["n_queries"] == N_QUERIES
    assert 0.0 <= reranked_metrics["ndcg@10"] <= 1.0

    # failure/transition labels partition every query
    labels = compute_query_labels(rows, reranked, qrels, RERANK_DEPTH)
    assert len(labels) == N_QUERIES
    decomposition = decomposition_metrics([label["transition_label"] for label in labels])
    assert sum(decomposition["counts"].values()) == N_QUERIES
    assert decomposition["n_queries"] == N_QUERIES
    assert set(decomposition["counts"]) == set(CATEGORIES)

    # features
    features = run_confidence_features(rows, reranked, None, RERANK_DEPTH)
    assert set(features) == set(queries)
    for per_query in features.values():
        assert set(per_query) == set(COMMON_FEATURES)

    # per-query outputs line up for the query-level table / bootstrap
    assert set(per_query_metrics(rows, qrels)) == set(qrels)


def test_end_to_end_hybrid_reports_primary_and_exploratory_models(mocked_cross_encoder):
    corpus = _synthetic_corpus()
    all_queries = _synthetic_queries()
    qrels = _synthetic_qrels(corpus, all_queries)

    train_ids = sorted(all_queries)[:8]
    dev_ids = sorted(all_queries)[8:]
    assert set(train_ids).isdisjoint(dev_ids)

    def stage(query_ids):
        queries = {qid: all_queries[qid] for qid in query_ids}
        rows = bm25_retrieve(corpus, queries, top_k=10)
        # Stand in for the un-fused bm25/dense lists the hybrid overlap feature needs.
        raw_rows = {"bm25": rows, "dense": rows}
        reranked = rerank(rows, queries, corpus, top_k=RERANK_DEPTH)
        labels = compute_query_labels(rows, reranked, {q: qrels[q] for q in query_ids}, RERANK_DEPTH)
        features = run_confidence_features(rows, reranked, raw_rows, RERANK_DEPTH)
        return reranked, features, {label["query_id"]: label["final_success_10"] for label in labels}

    train_reranked, train_features, train_labels = stage(train_ids)
    dev_reranked, dev_features, dev_labels = stage(dev_ids)
    # Force both classes so AUROC/AUPRC are defined for this synthetic fixture.
    train_labels = {qid: (i % 2 == 0) for i, qid in enumerate(sorted(train_labels))}
    dev_labels = {qid: (i % 2 == 0) for i, qid in enumerate(sorted(dev_labels))}

    config = {
        "pipeline": "hybrid_rrf",
        "confidence_class_weight": None,
        "confidence_coverage_levels": [1.0, 0.8, 0.6],
    }
    calibration = run_calibration(
        config,
        train_features,
        train_labels,
        train_reranked,
        dev_features,
        dev_labels,
        {qid: "already_successful" for qid in dev_labels},
        dev_reranked,
    )

    # plan §7: the exploratory overlap model is reported *alongside* the common-feature model.
    assert calibration["primary_model"] == "calibrated"
    assert set(calibration["results"]) == {
        "raw_score",
        "raw_score_platt",
        "calibrated",
        "calibrated_hybrid_exploratory",
    }
    assert calibration["feature_names"]["calibrated"] == COMMON_FEATURES
    assert calibration["feature_names"]["calibrated_hybrid_exploratory"] == COMMON_FEATURES + HYBRID_FEATURES
    assert exploratory_feature_names("hybrid_rrf") == COMMON_FEATURES + HYBRID_FEATURES

    # every model produced a full metric set on the dev split
    for name, result in calibration["results"].items():
        assert result["confidence_metrics"]["n_queries"] == len(dev_labels)
        assert len(result["risk_coverage"]) == len(dev_labels)
        assert set(result["selective_results"]) == {"1.0", "0.8", "0.6"}

    # per-query predictions are persisted for the plan §10.4 paired bootstrap
    assert [p["query_id"] for p in calibration["predictions"]] == sorted(dev_labels)
    for prediction in calibration["predictions"]:
        assert set(prediction) == {
            "query_id",
            "raw_score",
            "raw_score_platt",
            "calibrated",
            "calibrated_hybrid_exploratory",
            "final_success_10",
            "transition_label",
        }
