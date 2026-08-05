# Experiment Log

## 2026-08-05 — Calibration split generation

- Decision: generated `splits/calibration_train.txt` and `splits/calibration_dev.txt` by taking only `beir/scifact/train` query IDs with at least one positive (`grade > 0`) qrel, sorting them, then `random.Random(seed=42).shuffle`, taking the first 20% (rounded) as dev.
- Reason: deterministic and reproducible from seed per plan §5; sorting before shuffling avoids depending on `ir_datasets`' internal iteration order across versions; restricting to evaluable queries ensures calibration only trains/evaluates on queries with a defined evidence label (queries with zero positive qrels can't produce a meaningful `final_success_10` target in later phases).
- Result: 809 `beir/scifact/train` queries total, all 809 had ≥1 positive qrel (0 excluded as unjudged/zero-positive) → 647 calibration_train / 162 calibration_dev (~80/20), 0 overlap. `beir/scifact/test` has 300 queries, 5183-doc corpus shared across both splits. Full counts recorded in `results/tables/dataset_stats.json`.
- Next action: `beir/scifact/test` remains untouched until retrieval implementations, configs, and confidence features are all locked, per plan §5.

## 2026-08-05 — First-stage retrieval implementation choices

- Decision: pinned `BAAI/bge-small-en-v1.5` to commit SHA `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` in `configs/dense_bge.yaml` / `configs/hybrid_rrf.yaml` (via `SentenceTransformer(..., revision=...)`) instead of always resolving `main`.
- Reason: guarantees the exact model weights stay fixed for reproducibility regardless of future changes to the Hub repo.
- Decision: fixed BM25 to the concrete `bm25s` settings used by the original runs: `k1=1.5`, `b=0.75`, `delta=0.5`, Lucene scoring/IDF, lowercase English-stopword tokenization with `(?u)\b\w\w+\b`, and no stemmer.
- Reason: these were the installed library defaults, but recording and applying them explicitly prevents a future `bm25s` release from silently changing the experiment.
- Decision: duplicated shared fields (`dataset`, `seed`, `splits_dir`, `device`) into each pipeline config rather than building a base.yaml merge loader; `configs/base.yaml` is now reference documentation only, not loaded by code.
- Reason: keeps `load_config` trivial (plan §15 warns against building a general workflow engine) and makes each run's saved `config.yaml` fully self-contained for a later reviewer.
- Result: `bm25`, `dense_bge`, and `hybrid_rrf` all run successfully on `calibration-dev` (162 queries). Recall@50: BM25 0.873, dense 0.948, hybrid 0.966 — hybrid RRF beats both individual pipelines on candidate recall, consistent with H3 (hybrid complementarity). Full comparison table in `results/tables/first_stage_comparison.md`.
- Decision: selected `hybrid_rrf` as the primary pipeline for detailed Phase 5 confidence analysis.
- Reason: the locked plan §5 rule selects the highest calibration-dev Recall@50; hybrid's 0.9660 exceeds dense's 0.9475 and BM25's 0.8735. This selection is fixed before any final-test evaluation.

## 2026-08-05 — Confidence feature normalization and class weighting

- Decision: first-stage score/rank confidence features (`first_stage_top1_score_norm`, `first_stage_top1_top2_margin_norm`) use within-query min-max normalization over the top-50 candidate set the reranker actually saw.
- Reason: BM25 raw scores, dense cosine similarity, and RRF fused scores have incompatible scales (plan §9). Candidate-scoped per-query normalization avoids a fitted cross-query transform and gives every pipeline the same denominator. A constant candidate set normalizes to the uninformative midpoint 0.5.
- Decision: `confidence_class_weight: balanced` set in all three pipeline configs (`configs/{bm25,dense_bge,hybrid_rrf}.yaml`).
- Reason: `final_success_10` on `calibration-train` for BM25 is 535 True / 112 False (~83%/17%) — a material imbalance per plan §9's "use class weighting only if material" rule.
- Decision: report the unmodified top reranker score as the raw ranking baseline for AUROC/AUPRC and withhold its Brier score; fit a separate one-feature Platt scaler on calibration/train to provide a genuine probability baseline for Brier.
- Reason: plan §9 forbids treating the raw cross-encoder logit as a probability. Train-only Platt scaling satisfies the plan's Brier comparison without fitting any transform on calibration/dev.
- Decision: `reranking_failure_rate` is denominated over all queries; the separately named `conditional_conversion_rate` is the opportunity-conditional statistic.
- Reason: this keeps candidate-set failure, reranking failure, and final success as an exact all-query partition while making conditional performance explicit.
- Result: on calibration/dev, the balanced common-feature models improve AUROC/AUPRC over the raw baseline for all three pipelines, but have worse Brier scores than the train-fitted Platt baselines. This mixed finding is retained rather than changing the predeclared class-weight choice after seeing dev performance.
- Next action: Phase 5 will bootstrap the baseline-vs-calibrated AUPRC/Brier comparison (plan §10.4) and add the reliability diagram; `beir/scifact/test` remains untouched.
