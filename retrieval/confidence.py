"""Confidence feature extraction, raw-score baseline, and logistic calibration.

Predicts `final_success_10` from interpretable reranker-score and first-stage features. The
scaler and LogisticRegression are fitted together in one sklearn Pipeline on calibration-train
only; calibration-dev selects display thresholds and reports metrics, and fits nothing.
"""

import statistics
from collections.abc import Callable
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


def confidence_feature_names(pipeline: str) -> list[str]:
    """The common feature set — the primary model for every pipeline, hybrid included."""
    del pipeline  # kept in the signature: the feature set is a per-pipeline question by design
    return COMMON_FEATURES


def exploratory_feature_names(pipeline: str) -> list[str] | None:
    """Common features plus the hybrid-only BM25/dense overlap, for hybrid_rrf only.

    This is one clearly labelled exploratory ablation, not a replacement for the common-feature
    comparison: it is fitted and reported *in addition to* the primary model, never instead
    of it."""
    return COMMON_FEATURES + HYBRID_FEATURES if pipeline == "hybrid_rrf" else None


def _validate_finite(value: float, name: str) -> float:
    import math

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _paired_query_ids(*values_by_query: dict[str, Any]) -> list[str]:
    if not values_by_query or not values_by_query[0]:
        raise ValueError("Paired bootstrap requires at least one query")
    expected = set(values_by_query[0])
    for values in values_by_query[1:]:
        if set(values) != expected:
            raise ValueError("Paired bootstrap inputs must contain identical query IDs")
    return sorted(expected)


def paired_query_bootstrap(
    query_ids: list[str],
    statistic_a: Callable[[list[str]], float],
    statistic_b: Callable[[list[str]], float],
    *,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired query-level percentile bootstrap using one shared resample for both sides.

    Query IDs, including repeated IDs introduced by sampling with replacement, are passed
    unchanged to both statistic functions. The reported difference is always B - A.
    """
    import numpy as np

    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("query_ids must be non-empty and unique before resampling")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    ordered_ids = sorted(query_ids)
    point_a = _validate_finite(statistic_a(ordered_ids), "point estimate A")
    point_b = _validate_finite(statistic_b(ordered_ids), "point estimate B")
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)

    for index in range(n_resamples):
        sampled_indices = rng.integers(0, len(ordered_ids), size=len(ordered_ids))
        sampled_ids = [ordered_ids[i] for i in sampled_indices]
        sample_a = _validate_finite(statistic_a(sampled_ids), "bootstrap estimate A")
        sample_b = _validate_finite(statistic_b(sampled_ids), "bootstrap estimate B")
        differences[index] = sample_b - sample_a

    if not np.isfinite(differences).all():
        raise ValueError("Bootstrap differences contain NaN or infinite values")
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.percentile(differences, [100.0 * tail, 100.0 * (1.0 - tail)])
    return {
        "point_estimate_a": point_a,
        "point_estimate_b": point_b,
        "difference": point_b - point_a,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_resamples": n_resamples,
        "seed": seed,
        "n_queries": len(ordered_ids),
    }


def bootstrap_mean_comparison(
    values_a: dict[str, float],
    values_b: dict[str, float],
    *,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired bootstrap for query-level mean metrics such as Recall@10 or success rate."""
    query_ids = _paired_query_ids(values_a, values_b)
    for query_id in query_ids:
        _validate_finite(values_a[query_id], f"side A value for {query_id}")
        _validate_finite(values_b[query_id], f"side B value for {query_id}")

    return paired_query_bootstrap(
        query_ids,
        lambda sampled: statistics.fmean(values_a[qid] for qid in sampled),
        lambda sampled: statistics.fmean(values_b[qid] for qid in sampled),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


def bootstrap_score_comparison(
    labels: dict[str, bool],
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    *,
    metric: str,
    side_a_is_probability: bool,
    side_b_is_probability: bool,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired AUROC, AUPRC, or Brier bootstrap for two confidence signals on the same queries.

    AUROC and AUPRC are rank-based, so they are valid for the un-normalized raw cross-encoder
    score. Brier comparisons require genuine probabilities on both sides — the raw score is
    never silently normalized or treated as a probability.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if metric not in {"auroc", "auprc", "brier"}:
        raise ValueError("metric must be 'auroc', 'auprc' or 'brier'")
    if metric == "brier" and not (side_a_is_probability and side_b_is_probability):
        raise ValueError("Brier bootstrap requires probability scores on both sides")

    query_ids = _paired_query_ids(labels, scores_a, scores_b)
    for query_id in query_ids:
        score_a = _validate_finite(scores_a[query_id], f"side A score for {query_id}")
        score_b = _validate_finite(scores_b[query_id], f"side B score for {query_id}")
        if metric == "brier" and not (0.0 <= score_a <= 1.0 and 0.0 <= score_b <= 1.0):
            raise ValueError("Brier bootstrap probabilities must lie in [0, 1]")

    def score(values: dict[str, float], sampled: list[str]) -> float:
        y_true = [bool(labels[qid]) for qid in sampled]
        y_score = [values[qid] for qid in sampled]
        if metric == "brier":
            return statistics.fmean(
                (float(probability) - float(target)) ** 2
                for probability, target in zip(y_score, y_true)
            )
        # A resample can be single-class; AUROC/AUPRC are undefined there, so fall back to the
        # degenerate-but-defined value rather than dropping the resample (which would bias the
        # interval toward whichever side survives more often).
        if not any(y_true):
            return 0.0
        if metric == "auroc":
            if all(y_true):
                return 0.5
            return float(roc_auc_score(y_true, y_score))
        return float(average_precision_score(y_true, y_score))

    return paired_query_bootstrap(
        query_ids,
        lambda sampled: score(scores_a, sampled),
        lambda sampled: score(scores_b, sampled),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


def _sorted_by_rank(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: r["rank"])


def _rows_by_query(rows: list[dict]) -> dict[str, list[dict]]:
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    return {query_id: _sorted_by_rank(query_rows) for query_id, query_rows in by_query.items()}


def _reranker_features(reranked_docs: list[dict]) -> dict[str, float]:
    """Reranker score-shape features from a query's own reranked candidates."""
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
    """First-stage score/rank features, min-max normalized within the query so BM25, cosine,
    and RRF scores never have to be compared across pipelines. A constant candidate set
    normalizes to 0.5 rather than dividing by zero."""
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
        first_ranks = _dense_ranks([first_stage_rank_by_doc[doc_id] for doc_id in shared_doc_ids])
        rerank_ranks = _dense_ranks([reranked_rank_by_doc[doc_id] for doc_id in shared_doc_ids])
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


def _dense_ranks(ranks: list[int]) -> list[float]:
    """Re-ranks values to a contiguous 1..n. Spearman is Pearson over *dense* ranks, so the
    original positions have to be re-densified whenever the shared document set is a subset of
    either list — otherwise the gaps distort the correlation."""
    order = sorted(range(len(ranks)), key=lambda i: ranks[i])
    dense = [0.0] * len(ranks)
    for position, index in enumerate(order, start=1):
        dense[index] = float(position)
    return dense


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation: Pearson correlation over dense ranks (see `_dense_ranks`).
    Returns 0.0 for a degenerate (n<2 or zero-variance) input."""
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
    bm25_docs: list[dict], dense_docs: list[dict], depth: int = 10
) -> dict[str, float]:
    """Fraction of the two retrievers' top-`depth` lists that agree — the hybrid-only
    exploratory feature. Reported as a fraction so it shares the [0,1] range of the other
    normalized features."""
    bm25_top = {row["doc_id"] for row in bm25_docs if row["rank"] <= depth}
    dense_top = {row["doc_id"] for row in dense_docs if row["rank"] <= depth}
    return {"hybrid_bm25_dense_top10_overlap": len(bm25_top & dense_top) / depth}


def extract_features(
    first_stage_rows: list[dict],
    reranked_rows: list[dict],
    rerank_depth: int = 50,
    raw_rows: dict[str, list[dict]] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-query feature dict keyed by query_id, over the common feature set.

    First-stage rows are truncated to `rerank_depth` first, so every feature sees exactly the
    candidate set the reranker saw. `raw_rows` is needed only for the hybrid overlap feature.
    """
    first_stage_by_query = _rows_by_query(first_stage_rows)
    reranked_by_query = _rows_by_query(reranked_rows)
    bm25_by_query = _rows_by_query(raw_rows["bm25"]) if raw_rows is not None else {}
    dense_by_query = _rows_by_query(raw_rows["dense"]) if raw_rows is not None else {}

    features_by_query: dict[str, dict[str, float]] = {}
    for query_id, all_first_stage_docs in first_stage_by_query.items():
        first_stage_docs = all_first_stage_docs[:rerank_depth]
        reranked_docs = reranked_by_query.get(query_id, [])
        features = {
            **_first_stage_features(first_stage_docs, reranked_docs),
            **_reranker_features(reranked_docs),
        }
        if raw_rows is not None:
            features.update(
                _hybrid_overlap_features(
                    bm25_by_query.get(query_id, []), dense_by_query.get(query_id, [])
                )
            )
        features_by_query[query_id] = features
    return features_by_query


def raw_baseline_scores(reranked_rows: list[dict]) -> dict[str, float]:
    """Top-1 reranker score per query — the raw-score baseline. Not a probability."""
    by_query = _rows_by_query(reranked_rows)
    return {query_id: docs[0]["score"] if docs else 0.0 for query_id, docs in by_query.items()}


def fit_raw_score_calibrator(scores: dict[str, float], labels_by_query: dict[str, bool]):
    """Platt scaling of the raw top-1 reranker score, fitted on calibration-train, so the
    baseline has a real probability to be scored with Brier. The transform is monotone, so the
    ranking metrics still measure the raw ordering."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    query_ids = sorted(scores)
    X = [[scores[qid]] for qid in query_ids]
    y = [labels_by_query[qid] for qid in query_ids]

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X, y)
    return pipeline


def apply_raw_score_calibrator(calibrator, scores: dict[str, float]) -> dict[str, float]:
    query_ids = sorted(scores)
    X = [[scores[qid]] for qid in query_ids]
    true_index = list(calibrator.classes_).index(True)
    probs = calibrator.predict_proba(X)[:, true_index]
    return {qid: float(p) for qid, p in zip(query_ids, probs)}


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
    'balanced' is used, since the class-imbalance decision has to stay on the record."""
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


CV_FOLDS = 5


def cross_validated_predictions(
    features_by_query: dict[str, dict[str, float]],
    labels_by_query: dict[str, bool],
    baseline_by_query: dict[str, float],
    feature_sets: dict[str, list[str]],
    class_weight: str | None = None,
    n_splits: int = CV_FOLDS,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Stratified K-fold out-of-fold predictions over the whole calibration set.

    The predeclared train/dev split leaves only ~20 failures to compare models on. Pooling
    out-of-fold predictions over all 809 calibration queries raises that to ~120 without
    weakening any leakage rule: every fold fits on its own training portion and predicts only
    held-out queries, thresholds are still selected on calibration/dev, and the test split is
    still untouched. An estimation-power change, not a model-selection one.
    """
    from sklearn.model_selection import StratifiedKFold

    query_ids = sorted(features_by_query)
    labels = [labels_by_query[qid] for qid in query_ids]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows_by_query: dict[str, dict[str, Any]] = {}
    for fold, (train_index, test_index) in enumerate(splitter.split(query_ids, labels)):
        train_ids = [query_ids[i] for i in train_index]
        held_out_ids = [query_ids[i] for i in test_index]

        train_labels = {qid: labels_by_query[qid] for qid in train_ids}
        train_features = {qid: features_by_query[qid] for qid in train_ids}
        held_out_features = {qid: features_by_query[qid] for qid in held_out_ids}
        held_out_baseline = {qid: baseline_by_query[qid] for qid in held_out_ids}

        platt = fit_raw_score_calibrator({qid: baseline_by_query[qid] for qid in train_ids}, train_labels)
        fold_scores = {"raw_score_platt": apply_raw_score_calibrator(platt, held_out_baseline)}
        for model, feature_names in feature_sets.items():
            calibrator = fit_calibrator(
                train_features, train_labels, feature_names, class_weight=class_weight
            )
            fold_scores[model] = predict_proba(calibrator, held_out_features, feature_names)

        for qid in held_out_ids:
            rows_by_query[qid] = {
                "query_id": qid,
                "fold": fold,
                "raw_score": baseline_by_query[qid],
                **{model: scores[qid] for model, scores in fold_scores.items()},
                "final_success_10": labels_by_query[qid],
            }

    return [rows_by_query[qid] for qid in query_ids]


def select_thresholds(
    probs: dict[str, float], coverage_levels: tuple[float, ...] = (1.0, 0.8, 0.6)
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


def confidence_metrics(
    probs: dict[str, float], labels: dict[str, bool], is_probability: bool = True
) -> dict[str, float | None]:
    """AUROC, AUPRC, Brier score. Returns None for AUROC/AUPRC if only one class is present
    (both are undefined in that case) rather than raising or silently returning a fake value.

    `is_probability=False` returns None for Brier rather than rescaling scores into a fake
    probability; use `fit_raw_score_calibrator` for a baseline Brier. `auprc_over_base_rate` is
    reported because AUPRC is bounded below by the positive-class rate, not by 0.5 — at this
    success rate a constant predictor already scores ~0.88, so bare AUPRC flatters the model."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    query_ids = sorted(probs)
    y_true = [labels[qid] for qid in query_ids]
    y_score = [probs[qid] for qid in query_ids]
    single_class = len(set(y_true)) < 2
    base_rate = (sum(y_true) / len(y_true)) if y_true else 0.0
    auprc = None if single_class else average_precision_score(y_true, y_score)

    return {
        "auroc": None if single_class else roc_auc_score(y_true, y_score),
        "auprc": auprc,
        "base_rate": base_rate,
        "auprc_over_base_rate": None if auprc is None else auprc - base_rate,
        "brier": brier_score_loss(y_true, y_score) if is_probability else None,
        "is_probability": is_probability,
        "n_queries": len(query_ids),
        "n_failures": len(y_true) - sum(y_true),
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
