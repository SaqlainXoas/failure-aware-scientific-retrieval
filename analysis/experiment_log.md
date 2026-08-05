# Experiment Log

## 2026-08-05 — Calibration split generation

- Decision: generated `splits/calibration_train.txt` and `splits/calibration_dev.txt` by taking only `beir/scifact/train` query IDs with at least one positive (`grade > 0`) qrel, sorting them, then `random.Random(seed=42).shuffle`, taking the first 20% (rounded) as dev.
- Reason: deterministic and reproducible from seed per plan §5; sorting before shuffling avoids depending on `ir_datasets`' internal iteration order across versions; restricting to evaluable queries ensures calibration only trains/evaluates on queries with a defined evidence label (queries with zero positive qrels can't produce a meaningful `final_success_10` target in later phases).
- Result: 809 `beir/scifact/train` queries total, all 809 had ≥1 positive qrel (0 excluded as unjudged/zero-positive) → 647 calibration_train / 162 calibration_dev (~80/20), 0 overlap. `beir/scifact/test` has 300 queries, 5183-doc corpus shared across both splits. Full counts recorded in `results/tables/dataset_stats.json`.
- Next action: `beir/scifact/test` remains untouched until retrieval implementations, configs, and confidence features are all locked, per plan §5.

## 2026-08-05 — First-stage retrieval implementation choices

- Decision: pinned `BAAI/bge-small-en-v1.5` to commit SHA `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` in `configs/dense_bge.yaml` / `configs/hybrid_rrf.yaml` (via `SentenceTransformer(..., revision=...)`) instead of always resolving `main`.
- Reason: guarantees the exact model weights stay fixed for reproducibility regardless of future changes to the Hub repo.
- Decision: duplicated shared fields (`dataset`, `seed`, `splits_dir`, `device`) into each pipeline config rather than building a base.yaml merge loader; `configs/base.yaml` is now reference documentation only, not loaded by code.
- Reason: keeps `load_config` trivial (plan §15 warns against building a general workflow engine) and makes each run's saved `config.yaml` fully self-contained for a later reviewer.
- Result: `bm25`, `dense_bge`, and `hybrid_rrf` all run successfully on `calibration-dev` (162 queries). Recall@50: BM25 0.873, dense 0.948, hybrid 0.966 — hybrid RRF beats both individual pipelines on candidate recall, consistent with H3 (hybrid complementarity). Full comparison table in `results/tables/first_stage_comparison.md`.
- Next action: Phase 3 adds cross-encoder reranking over these top-50 candidates and the candidate-set/reranking failure decomposition.

## 2026-08-05 — Confidence feature normalization and class weighting

- Decision: first-stage score/rank confidence features (`first_stage_top1_score_norm`, `first_stage_top1_top2_margin_norm`) use within-query min-max normalization over that query's own first-stage candidate scores, rather than any cross-query fitted scaler.
- Reason: BM25 raw scores, dense cosine similarity, and RRF fused scores have incompatible scales (plan §9); a per-query min-max transform makes the features comparable across pipelines without any fitting step, so there is no leakage risk from this normalization itself. A constant candidate set (max == min) normalizes to the uninformative midpoint 0.5 instead of dividing by zero.
- Decision: `confidence_class_weight: balanced` set in all three pipeline configs (`configs/{bm25,dense_bge,hybrid_rrf}.yaml`).
- Reason: `final_success_10` on `calibration-train` for BM25 is 535 True / 112 False (~83%/17%) — a material imbalance per plan §9's "use class weighting only if material" rule.
- Decision: the raw reranker-score baseline (a cross-encoder logit, not a probability) is min-max normalized across the evaluated split only when computing Brier score, so it lands in scikit-learn's required [0,1] range; AUROC/AUPRC are computed on the raw scores unchanged since both are rank-based and scale-free.
- Reason: plan §9 explicitly forbids calling the raw score a probability, but §10.3 still requires reporting Brier for the baseline alongside the calibrator — normalizing only for this one metric keeps that requirement satisfiable without claiming the baseline is calibrated.
- Result: BM25 `calibration-train` run, `class_weight="balanced"` — baseline AUROC 0.811 / AUPRC 0.955 / Brier 0.145 vs. calibrated AUROC 0.877 / AUPRC 0.977 / Brier 0.156 on `calibration-dev` (162 queries). AUROC/AUPRC improve over the baseline as expected, but Brier is worse than both the baseline and an unweighted fit (0.093) — `class_weight="balanced"` shifts the decision boundary to trade calibrated-probability quality for the minority (failure) class's recall, a known effect and not a bug. Reported honestly rather than reverting to whichever setting looks best. Full per-pipeline numbers in each run's `confidence_metrics.json`.
- Next action: Phase 5 will bootstrap the baseline-vs-calibrated AUPRC/Brier comparison (plan §10.4) and add the reliability diagram; `beir/scifact/test` remains untouched.
