"""Confidence feature extraction, raw-score baseline, and logistic calibration.

Predicts `final_success_10` (plan.md §9) from interpretable reranker-score and
first-stage score/rank features. The scaler and LogisticRegression are always fitted
together in one sklearn Pipeline on calibration-train only; calibration-dev is used
only to select display thresholds and report metrics, never to fit anything.
"""

import statistics
from typing import Any

FIRST_STAGE_FEATURES = [
    "first_stage_top1_score_norm",
    "first_stage_top1_top2_margin_norm",
    "first_stage_rerank_rank_correlation",
    "first_stage_top1_in_reranked_top3",
]
RERANK_FEATURES = [
    "reranker_top1_score",
    "reranker_top1_top2_margin",
    "reranker_top5_mean",
    "reranker_top5_std",
]
COMMON_FEATURES = FIRST_STAGE_FEATURES + RERANK_FEATURES
HYBRID_FEATURES = ["hybrid_bm25_dense_top10_overlap"]


def _sorted_by_rank(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: r["rank"])


def _rows_by_query(rows: list[dict]) -> dict[str, list[dict]]:
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    return {query_id: _sorted_by_rank(query_rows) for query_id, query_rows in by_query.items()}


def _reranker_features(reranked_docs: list[dict]) -> dict[str, float]:
    """Reranker score-shape features from a query's own reranked candidates (plan §9)."""
    scores = [row["score"] for row in reranked_docs]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) >= 2 else top1
    top5 = scores[:5]
    return {
        "reranker_top1_score": top1,
        "reranker_top1_top2_margin": (top1 - top2) if len(scores) >= 2 else 0.0,
        "reranker_top5_mean": statistics.fmean(top5) if top5 else 0.0,
        "reranker_top5_std": statistics.pstdev(top5) if len(top5) >= 2 else 0.0,
    }


def _first_stage_features(first_stage_docs: list[dict], reranked_docs: list[dict]) -> dict[str, float]:
    """First-stage score/rank features, within-query min-max normalized so BM25/cosine/RRF
    scales never need cross-pipeline comparison (plan §9). A constant candidate set (mx == mn)
    normalizes to the uninformative midpoint 0.5 rather than dividing by zero."""
    scores = [row["score"] for row in first_stage_docs]
    if scores:
        mn, mx = min(scores), max(scores)
        norm = (lambda s: (s - mn) / (mx - mn)) if mx > mn else (lambda s: 0.5)
    else:
        norm = lambda s: 0.0  # noqa: E731

    top1_norm = norm(scores[0]) if scores else 0.0
    top2_norm = norm(scores[1]) if len(scores) >= 2 else top1_norm
    margin_norm = (top1_norm - top2_norm) if len(scores) >= 2 else 0.0

    reranked_rank_by_doc = {row["doc_id"]: row["rank"] for row in reranked_docs}
    shared_doc_ids = [row["doc_id"] for row in first_stage_docs if row["doc_id"] in reranked_rank_by_doc]
    if len(shared_doc_ids) >= 2:
        first_stage_rank_by_doc = {row["doc_id"]: row["rank"] for row in first_stage_docs}
        first_ranks = [first_stage_rank_by_doc[doc_id] for doc_id in shared_doc_ids]
        rerank_ranks = [reranked_rank_by_doc[doc_id] for doc_id in shared_doc_ids]
        rank_correlation = _spearman(first_ranks, rerank_ranks)
    else:
        rank_correlation = 0.0

    reranked_top3 = {row["doc_id"] for row in reranked_docs[:3]}
    top1_in_reranked_top3 = 1.0 if first_stage_docs and first_stage_docs[0]["doc_id"] in reranked_top3 else 0.0

    return {
        "first_stage_top1_score_norm": top1_norm,
        "first_stage_top1_top2_margin_norm": margin_norm,
        "first_stage_rerank_rank_correlation": rank_correlation,
        "first_stage_top1_in_reranked_top3": top1_in_reranked_top3,
    }


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation. Inputs here are already ranks (unique ints), so this
    reduces to Pearson correlation over them; returns 0.0 for a degenerate (n<2) input."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5


def _hybrid_overlap_features(
    query_id: str, raw_rows: dict[str, list[dict]], rerank_depth: int
) -> dict[str, float]:
    bm25_top10 = {
        row["doc_id"] for row in raw_rows["bm25"] if row["query_id"] == query_id and row["rank"] <= 10
    }
    dense_top10 = {
        row["doc_id"] for row in raw_rows["dense"] if row["query_id"] == query_id and row["rank"] <= 10
    }
    overlap = len(bm25_top10 & dense_top10)
    return {"hybrid_bm25_dense_top10_overlap": overlap / 10.0}


def extract_features(
    first_stage_rows: list[dict],
    reranked_rows: list[dict],
    rerank_depth: int = 50,
    raw_rows: dict[str, list[dict]] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-query feature dict keyed by query_id (plan.md §9 common feature set).

    `raw_rows` (the un-fused `{"bm25": [...], "dense": [...]}` rows) is required only to
    add the hybrid-only overlap feature; omit it for bm25/dense_bge pipelines.
    """
    first_stage_by_query = _rows_by_query(first_stage_rows)
    reranked_by_query = _rows_by_query(reranked_rows)

    if raw_rows is not None:
        bm25_by_query = _rows_by_query(raw_rows["bm25"])
        dense_by_query = _rows_by_query(raw_rows["dense"])
        indexed_raw_rows = {
            "bm25": [row for rows in bm25_by_query.values() for row in rows],
            "dense": [row for rows in dense_by_query.values() for row in rows],
        }

    features_by_query: dict[str, dict[str, float]] = {}
    for query_id, first_stage_docs in first_stage_by_query.items():
        reranked_docs = reranked_by_query.get(query_id, [])
        features = {
            **_first_stage_features(first_stage_docs, reranked_docs),
            **_reranker_features(reranked_docs),
        }
        if raw_rows is not None:
            features.update(_hybrid_overlap_features(query_id, indexed_raw_rows, rerank_depth))
        features_by_query[query_id] = features
    return features_by_query


def raw_baseline_scores(reranked_rows: list[dict]) -> dict[str, float]:
    """Top-1 reranker score per query — the plan §9 raw-score baseline. Not a probability."""
    by_query = _rows_by_query(reranked_rows)
    return {query_id: docs[0]["score"] if docs else 0.0 for query_id, docs in by_query.items()}


def _feature_matrix(
    features_by_query: dict[str, dict[str, float]], feature_names: list[str], query_ids: list[str]
) -> list[list[float]]:
    return [[features_by_query[qid][name] for name in feature_names] for qid in query_ids]


def fit_calibrator(
    features_by_query: dict[str, dict[str, float]],
    labels_by_query: dict[str, bool],
    feature_names: list[str],
    class_weight: str | None = None,
):
    """Fits StandardScaler + LogisticRegression together in one Pipeline, on the data
    passed in only. class_weight is documented in analysis/experiment_log.md whenever
    'balanced' is used (plan §9 requires the imbalance decision to be recorded)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    query_ids = sorted(features_by_query)
    X = _feature_matrix(features_by_query, feature_names, query_ids)
    y = [labels_by_query[qid] for qid in query_ids]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=class_weight)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def predict_proba(calibrator, features_by_query: dict[str, dict[str, float]], feature_names: list[str]) -> dict[str, float]:
    query_ids = sorted(features_by_query)
    X = _feature_matrix(features_by_query, feature_names, query_ids)
    true_index = list(calibrator.classes_).index(True)
    probs = calibrator.predict_proba(X)[:, true_index]
    return {qid: float(p) for qid, p in zip(query_ids, probs)}


def select_thresholds(
    probs: dict[str, float], labels: dict[str, bool], coverage_levels: tuple[float, ...] = (1.0, 0.8, 0.6)
) -> dict[str, float]:
    """Score threshold per coverage level: keeping queries with score >= threshold covers
    (at least) that fraction of calibration-dev, selected on calibration-dev only."""
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    n = len(ranked)
    thresholds = {}
    for coverage in coverage_levels:
        n_keep = max(1, round(n * coverage)) if n else 0
        threshold = ranked[n_keep - 1][1] if n_keep else 0.0
        thresholds[str(coverage)] = threshold
    return thresholds


def confidence_metrics(probs: dict[str, float], labels: dict[str, bool]) -> dict[str, float | None]:
    """AUROC, AUPRC, Brier score. Returns None for AUROC/AUPRC if only one class is present
    (both are undefined in that case) rather than raising or silently returning a fake value.

    Brier score requires inputs in [0,1]; the raw reranker-score baseline is not a probability
    (plan §9), so its values are min-max normalized across this split only for the Brier
    computation. AUROC/AUPRC are unaffected — they are rank-based and already scale-free."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    query_ids = sorted(probs)
    y_true = [labels[qid] for qid in query_ids]
    y_score = [probs[qid] for qid in query_ids]
    single_class = len(set(y_true)) < 2

    if y_score and (min(y_score) < 0.0 or max(y_score) > 1.0):
        mn, mx = min(y_score), max(y_score)
        y_score_for_brier = [(s - mn) / (mx - mn) if mx > mn else 0.5 for s in y_score]
    else:
        y_score_for_brier = y_score

    return {
        "auroc": None if single_class else roc_auc_score(y_true, y_score),
        "auprc": None if single_class else average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score_for_brier),
        "n_queries": len(query_ids),
    }


def risk_coverage_curve(probs: dict[str, float], labels: dict[str, bool]) -> list[dict[str, float]]:
    """Points of (coverage, risk=1-selective_success_rate, threshold) as the confidence
    cutoff sweeps from most to least confident. query_id is not retained here (per-query
    correctness for a future bootstrap pass can be recomputed by joining probs/labels)."""
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    n = len(ranked)
    if n == 0:
        return []
    points = []
    successes = 0
    for i, (query_id, score) in enumerate(ranked, start=1):
        successes += 1 if labels[query_id] else 0
        points.append(
            {
                "coverage": i / n,
                "risk": 1.0 - successes / i,
                "threshold": score,
            }
        )
    return points


def selective_results_at_coverage(
    probs: dict[str, float], labels: dict[str, bool], coverage_levels: tuple[float, ...] = (1.0, 0.8, 0.6)
) -> dict[str, dict[str, float]]:
    """Success rate among the top-`coverage` fraction of queries by confidence, at each fixed level."""
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    n = len(ranked)
    results: dict[str, dict[str, float]] = {}
    for coverage in coverage_levels:
        n_keep = max(1, round(n * coverage)) if n else 0
        kept = ranked[:n_keep]
        successes = sum(1 for qid, _ in kept if labels[qid])
        results[str(coverage)] = {
            "n_kept": n_keep,
            "success_rate": successes / n_keep if n_keep else 0.0,
        }
    return results
