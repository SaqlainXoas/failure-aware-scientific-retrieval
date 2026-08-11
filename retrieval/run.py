"""Thin entrypoint: retrieve -> rerank -> evaluate for a single pipeline config."""

import argparse
import json
import logging
from typing import Any

from retrieval.confidence import (
    CV_FOLDS,
    apply_raw_score_calibrator,
    confidence_feature_names,
    confidence_metrics,
    cross_validated_predictions,
    exploratory_feature_names,
    extract_features,
    fit_calibrator,
    fit_raw_score_calibrator,
    predict_proba,
    raw_baseline_scores,
    risk_coverage_curve,
    select_thresholds,
    selective_results_at_coverage,
)
from retrieval.data import load_config, load_scifact_split, resolve_device, resolve_split, setup_logging
from retrieval.evaluate import decomposition_metrics, evaluate_query, label_transition
from retrieval.rerank import rerank
from retrieval.retrieve import bm25_retrieve, dense_retrieve, hybrid_retrieve
from retrieval.runio import (
    EVAL_SPLIT,
    FIT_SPLIT,
    _capture_logs,
    build_manifest,
    build_query_results,
    write_run_dir,
)

logger = logging.getLogger(__name__)

CACHE_DIR = ".cache"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a retrieval pipeline end to end (retrieve, rerank, evaluate)."
    )
    parser.add_argument("--config", required=True, help="Path to a pipeline config YAML.")
    parser.add_argument("--split", default="calibration-dev", help="Query split to run on.")
    parser.add_argument("--force", action="store_true", help="Bypass cache and recompute.")
    return parser.parse_args(argv)


def seed_everything(seed: int | None) -> None:
    """Seeds Python/NumPy/Torch RNGs. Nothing here is stochastic today, so this changes no
    result; it exists so the seed recorded in the manifest is not a decorative field."""
    if seed is None:
        return
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_pipeline(
    config: dict[str, Any], data: dict[str, Any], device: str, force: bool
) -> tuple[list[dict], dict[str, bool], dict[str, list[dict]] | None, dict[str, Any]]:
    """Dispatches on config['pipeline'], returning (rows, cache hits, raw per-retriever rows,
    retrieval params). The raw rows exist only for hybrid_rrf, whose overlap confidence feature
    needs the two un-fused rankings."""
    pipeline = config["pipeline"]
    corpus, queries = data["corpus"], data["queries"]
    candidate_depth = config.get("candidate_depth", 100)
    cache_hits: dict[str, bool] = {}
    params: dict[str, Any] = {}

    def run_bm25() -> list[dict]:
        bm25_params: dict[str, Any] = {}
        rows = bm25_retrieve(
            corpus,
            queries,
            top_k=candidate_depth,
            cache_dir=CACHE_DIR,
            force=force,
            cache_hits=cache_hits,
            params_out=bm25_params,
        )
        params["bm25"] = bm25_params
        return rows

    def run_dense() -> list[dict]:
        return dense_retrieve(
            corpus,
            queries,
            top_k=candidate_depth,
            model_name=config["model_name"],
            model_revision=config.get("model_revision"),
            device=device,
            batch_size=config.get("batch_size", 32),
            cache_dir=CACHE_DIR,
            force=force,
            cache_hits=cache_hits,
        )

    if pipeline == "bm25":
        return run_bm25(), cache_hits, None, params
    if pipeline == "dense_bge":
        return run_dense(), cache_hits, None, params
    if pipeline == "hybrid_rrf":
        bm25_rows = run_bm25()
        dense_rows = run_dense()
        rows = hybrid_retrieve(
            bm25_rows, dense_rows, k=config.get("rrf_k", 60), top_k=candidate_depth
        )
        return rows, cache_hits, {"bm25": bm25_rows, "dense": dense_rows}, params
    raise ValueError(f"Unknown pipeline '{pipeline}'")


def per_query_metrics(rows: list[dict], qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """Ranking metrics per query — averaged into metrics.json and kept per-query in
    query_results.parquet so the paired bootstrap can resample at query level."""
    ranked_by_query: dict[str, list[dict]] = {}
    for row in rows:
        ranked_by_query.setdefault(row["query_id"], []).append(row)
    ranked_docs = {
        query_id: [r["doc_id"] for r in sorted(query_rows, key=lambda r: r["rank"])]
        for query_id, query_rows in ranked_by_query.items()
    }
    return {
        query_id: evaluate_query(ranked_docs.get(query_id, []), qrels_for_query)
        for query_id, qrels_for_query in qrels.items()
    }


def compute_metrics(rows: list[dict], qrels: dict[str, dict[str, int]]) -> dict[str, Any]:
    per_query = list(per_query_metrics(rows, qrels).values())
    n_queries = len(per_query)
    metric_keys = per_query[0].keys() if per_query else []
    averaged = {
        key: sum(m[key] for m in per_query) / n_queries if n_queries else 0.0
        for key in metric_keys
    }
    n_with_relevant = sum(
        1 for grades in qrels.values() if any(grade > 0 for grade in grades.values())
    )
    return {
        **averaged,
        "n_queries": n_queries,
        "pct_queries_with_relevant": n_with_relevant / n_queries if n_queries else 0.0,
    }


def run_reranking(
    config: dict[str, Any], data: dict[str, Any], first_stage_rows: list[dict], device: str, force: bool
) -> tuple[list[dict], dict[str, bool]]:
    """Reranks each query's own top rerank_depth first-stage candidates via the frozen cross-encoder."""
    cache_hits: dict[str, bool] = {}
    rows = rerank(
        first_stage_rows,
        data["queries"],
        data["corpus"],
        top_k=config.get("rerank_depth", 50),
        model_name=config["reranker_model"],
        model_revision=config.get("reranker_revision"),
        device=device,
        batch_size=config.get("reranker_batch_size", 32),
        cache_dir=CACHE_DIR,
        force=force,
        cache_hits=cache_hits,
    )
    return rows, cache_hits


def compute_query_labels(
    first_stage_rows: list[dict],
    reranked_rows: list[dict],
    qrels: dict[str, dict[str, int]],
    rerank_depth: int,
) -> list[dict]:
    """Per-query candidate/final success and transition label — the row-level source the
    decomposition table and later manual failure analysis both read from."""
    first_stage_by_query: dict[str, list[str]] = {}
    for row in sorted(first_stage_rows, key=lambda r: r["rank"]):
        first_stage_by_query.setdefault(row["query_id"], []).append(row["doc_id"])
    reranked_by_query: dict[str, list[str]] = {}
    for row in sorted(reranked_rows, key=lambda r: r["rank"]):
        reranked_by_query.setdefault(row["query_id"], []).append(row["doc_id"])

    labels = []
    for query_id, qrels_for_query in qrels.items():
        gold_doc_ids = {doc_id for doc_id, grade in qrels_for_query.items() if grade > 0}
        first_stage_ranked = first_stage_by_query.get(query_id, [])
        reranked_ranked = reranked_by_query.get(query_id, [])
        transition = label_transition(gold_doc_ids, first_stage_ranked, reranked_ranked, rerank_depth)
        labels.append(
            {
                "query_id": query_id,
                "candidate_success_50": bool(gold_doc_ids & set(first_stage_ranked[:rerank_depth])),
                "final_success_10": bool(gold_doc_ids & set(reranked_ranked[:10])),
                "transition_label": transition,
            }
        )
    return labels


def run_confidence_features(
    rows: list[dict],
    reranked_rows: list[dict],
    raw_rows: dict[str, list[dict]] | None,
    rerank_depth: int,
) -> dict[str, dict[str, float]]:
    return extract_features(rows, reranked_rows, rerank_depth, raw_rows=raw_rows)


def _score_model(
    scores: dict[str, float],
    labels: dict[str, bool],
    coverage_levels: tuple[float, ...],
    is_probability: bool,
) -> dict[str, Any]:
    return {
        "confidence_metrics": confidence_metrics(scores, labels, is_probability=is_probability),
        "thresholds": select_thresholds(scores, coverage_levels),
        "selective_results": selective_results_at_coverage(scores, labels, coverage_levels),
        "risk_coverage": risk_coverage_curve(scores, labels),
    }


def run_calibration(
    config: dict[str, Any],
    train_features: dict[str, dict[str, float]],
    train_labels: dict[str, bool],
    train_reranked_rows: list[dict],
    dev_features: dict[str, dict[str, float]],
    dev_labels: dict[str, bool],
    dev_transition_labels: dict[str, str],
    dev_reranked_rows: list[dict],
) -> dict[str, Any]:
    """Fits every model on calibration-train only; every threshold and metric below is computed
    on calibration-dev, never on train.

    Four models are reported side by side and never conflated: the raw top-1 reranker score
    (ranking metrics only — it is not a probability), the same score after train-fitted Platt
    scaling (which gives the baseline a comparable Brier), the common-feature calibrator
    (primary), and for hybrid the common+overlap ablation (exploratory).
    """
    pipeline = config["pipeline"]
    class_weight = config.get("confidence_class_weight")
    coverage_levels = tuple(config.get("confidence_coverage_levels", [1.0, 0.8, 0.6]))
    primary_features = confidence_feature_names(pipeline)
    exploratory_features = exploratory_feature_names(pipeline)

    train_baseline = raw_baseline_scores(train_reranked_rows)
    dev_baseline = raw_baseline_scores(dev_reranked_rows)

    platt = fit_raw_score_calibrator(train_baseline, train_labels)
    calibrator = fit_calibrator(train_features, train_labels, primary_features, class_weight=class_weight)

    dev_scores = {
        "raw_score": dev_baseline,
        "raw_score_platt": apply_raw_score_calibrator(platt, dev_baseline),
        "calibrated": predict_proba(calibrator, dev_features, primary_features),
    }
    estimators = {"raw_score_platt": platt, "calibrated": calibrator}
    feature_names = {"raw_score_platt": ["reranker_top1_score"], "calibrated": primary_features}

    if exploratory_features is not None:
        exploratory = fit_calibrator(
            train_features, train_labels, exploratory_features, class_weight=class_weight
        )
        dev_scores["calibrated_hybrid_exploratory"] = predict_proba(
            exploratory, dev_features, exploratory_features
        )
        estimators["calibrated_hybrid_exploratory"] = exploratory
        feature_names["calibrated_hybrid_exploratory"] = exploratory_features

    results = {
        name: _score_model(scores, dev_labels, coverage_levels, is_probability=(name != "raw_score"))
        for name, scores in dev_scores.items()
    }

    predictions = [
        {
            "query_id": query_id,
            **{name: scores[query_id] for name, scores in dev_scores.items()},
            "final_success_10": dev_labels[query_id],
            "transition_label": dev_transition_labels[query_id],
        }
        for query_id in sorted(dev_labels)
    ]

    feature_sets = {"calibrated": primary_features}
    if exploratory_features is not None:
        feature_sets["calibrated_hybrid_exploratory"] = exploratory_features
    # Everything above this line is the predeclared train/dev protocol and must never see a dev
    # query during fitting. The cross-validated estimate below deliberately pools train+dev, so
    # it is computed strictly after, and kept in its own key so the two can never be confused.
    n_splits = int(config.get("confidence_cv_folds", CV_FOLDS))
    cross_validated = (
        run_cross_validated_confidence(
            {**train_features, **dev_features},
            {**train_labels, **dev_labels},
            {**train_baseline, **dev_baseline},
            feature_sets,
            class_weight=class_weight,
            coverage_levels=coverage_levels,
            n_splits=n_splits,
            seed=int(config.get("seed", 42)),
        )
        if n_splits >= 2
        else None
    )

    return {
        "estimators": estimators,
        "primary_model": "calibrated",
        "exploratory_models": [n for n in dev_scores if n.endswith("_exploratory")],
        "feature_names": feature_names,
        "class_weight": class_weight,
        "fit_split": FIT_SPLIT,
        "eval_split": EVAL_SPLIT,
        "results": results,
        "predictions": predictions,
        "cross_validated": cross_validated,
    }


def run_cross_validated_confidence(
    features_by_query: dict[str, dict[str, float]],
    labels_by_query: dict[str, bool],
    baseline_by_query: dict[str, float],
    feature_sets: dict[str, list[str]],
    class_weight: str | None,
    coverage_levels: tuple[float, ...],
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    """Higher-powered secondary estimate of the raw-vs-calibrated comparison, reported
    *alongside* the predeclared train/dev result and never as a replacement for it."""
    predictions = cross_validated_predictions(
        features_by_query,
        labels_by_query,
        baseline_by_query,
        feature_sets,
        class_weight=class_weight,
        n_splits=n_splits,
        seed=seed,
    )
    model_names = ["raw_score", "raw_score_platt", *feature_sets]
    pooled = {
        name: {row["query_id"]: row[name] for row in predictions} for name in model_names
    }
    results = {
        name: _score_model(scores, labels_by_query, coverage_levels, is_probability=(name != "raw_score"))
        for name, scores in pooled.items()
    }
    return {
        "protocol": f"stratified {n_splits}-fold cross-validation over calibration-train + calibration-dev",
        "n_splits": n_splits,
        "seed": seed,
        "n_queries": len(predictions),
        "n_failures": sum(1 for row in predictions if not row["final_success_10"]),
        "results": results,
        "predictions": predictions,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()
    log_buffer = _capture_logs()
    config = load_config(args.config)
    seed_everything(config.get("seed"))
    logger.info("Loaded config from %s for split %s: %s", args.config, args.split, config)

    train_data = load_scifact_split("train")
    data = resolve_split(args.split, train_data, splits_dir=config.get("splits_dir", "splits"))
    device = resolve_device(config.get("device", "auto"))

    rows, cache_hits, raw_rows, retrieval_params = run_pipeline(config, data, device, args.force)
    metrics = compute_metrics(rows, data["qrels"])

    rerank_depth = config.get("rerank_depth", 50)
    reranked_rows, rerank_cache_hits = run_reranking(config, data, rows, device, args.force)
    reranked_metrics = compute_metrics(reranked_rows, data["qrels"])
    labels = compute_query_labels(rows, reranked_rows, data["qrels"], rerank_depth)
    decomposition = decomposition_metrics([label["transition_label"] for label in labels])
    features_by_query = run_confidence_features(rows, reranked_rows, raw_rows, rerank_depth)
    confidence_features = [{"query_id": qid, **feats} for qid, feats in sorted(features_by_query.items())]
    query_results = build_query_results(
        labels,
        per_query_metrics(rows, data["qrels"]),
        per_query_metrics(reranked_rows, data["qrels"]),
        features_by_query,
    )

    calibration = None
    if args.split == FIT_SPLIT:
        # Confidence models are fitted here and *only* here: this branch reads calibration-train
        # for fitting and calibration-dev for every threshold and metric. Nothing is ever fitted
        # on the split it is scored on.
        dev_data = resolve_split(EVAL_SPLIT, train_data, splits_dir=config.get("splits_dir", "splits"))
        dev_rows, dev_cache_hits, dev_raw_rows, _ = run_pipeline(config, dev_data, device, args.force)
        dev_reranked_rows, dev_rerank_cache_hits = run_reranking(config, dev_data, dev_rows, device, args.force)
        dev_labels_list = compute_query_labels(dev_rows, dev_reranked_rows, dev_data["qrels"], rerank_depth)
        dev_features_by_query = run_confidence_features(
            dev_rows, dev_reranked_rows, dev_raw_rows, rerank_depth
        )
        train_labels = {label["query_id"]: label["final_success_10"] for label in labels}
        dev_labels = {label["query_id"]: label["final_success_10"] for label in dev_labels_list}
        dev_transition_labels = {
            label["query_id"]: label["transition_label"] for label in dev_labels_list
        }
        calibration = run_calibration(
            config,
            features_by_query,
            train_labels,
            reranked_rows,
            dev_features_by_query,
            dev_labels,
            dev_transition_labels,
            dev_reranked_rows,
        )
        cache_hits = {**cache_hits, **rerank_cache_hits, **dev_cache_hits, **dev_rerank_cache_hits}
    else:
        cache_hits = {**cache_hits, **rerank_cache_hits}

    manifest = build_manifest(
        config,
        args.config,
        args.split,
        device,
        metrics["n_queries"],
        cache_hits,
        retrieval_params=retrieval_params,
        calibrated=calibration is not None,
    )

    logger.info("Metrics: %s", json.dumps(metrics, indent=2))
    logger.info("Reranked metrics: %s", json.dumps(reranked_metrics, indent=2))
    logger.info("Decomposition: %s", json.dumps(decomposition, indent=2))
    if calibration is not None:
        logger.info(
            "Confidence metrics (fit=%s, eval=%s): %s",
            calibration["fit_split"],
            calibration["eval_split"],
            json.dumps(
                {name: result["confidence_metrics"] for name, result in calibration["results"].items()},
                indent=2,
            ),
        )
        cross_validated = calibration["cross_validated"]
        logger.info(
            "Cross-validated confidence (%s, %s queries / %s failures): %s",
            cross_validated["protocol"],
            cross_validated["n_queries"],
            cross_validated["n_failures"],
            json.dumps(
                {
                    name: result["confidence_metrics"]
                    for name, result in cross_validated["results"].items()
                },
                indent=2,
            ),
        )

    run_dir = write_run_dir(
        config,
        manifest,
        rows,
        metrics,
        reranked_rows,
        reranked_metrics,
        labels,
        decomposition,
        confidence_features,
        query_results,
        calibration=calibration,
        logs_text=log_buffer.getvalue(),
    )
    logger.info("Wrote run directory %s", run_dir)


if __name__ == "__main__":
    main()
