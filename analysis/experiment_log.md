# Experiment Log

Dated record of every methodological decision, written as it was made. Entries are kept verbatim,
including ones that record a result coming out worse than hoped.

`plan §N` in these entries cites the pre-registration document that fixed the dataset, splits,
models, failure taxonomy, leakage rules, and statistical tests before any result existed. That
document is not published; the README states the constraints it imposed, and this log is the
dated evidence they were followed.

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

## 2026-08-05 — Phase 0–4 alignment verification

- Decision: regenerated all three calibration/dev runs and all three calibration/train confidence runs from clean source commit `c1f1155e91d9125b392074374fd2c7141db7e5f0`.
- Result: retrieval, reranking, decomposition, and confidence metrics reproduced exactly; 115 default tests and all 7 real-SciFact integration tests passed. Tables 2–6 and six compact manifests were generated from saved runs, and every manifest records `git_dirty: false`.
- Result: hybrid remains the locked primary pipeline (calibration/dev Recall@50 `0.9660`). Its common-feature confidence model improves AUPRC from `0.9627` to `0.9748`, while Brier is worse than the train-fitted Platt baseline (`0.1659` vs. `0.0971`), so the Phase 5 analysis must treat calibration quality as a negative result rather than claiming uniform improvement.
- Next action: begin Phase 5 bootstrap intervals and structured qualitative analysis without opening `beir/scifact/test`.

## 2026-08-05 — Phase 5 uncertainty and structured failure analysis

- Decision: used deterministic paired query-level bootstrap on the 162 calibration/dev query IDs: 1,000 size-162 resamples with replacement, the same sampled IDs on both sides, 95% percentile intervals, difference `B - A`, seed 42.
- Decision: reliability uses ten fixed equal-width probability bins `[0.0, 0.1), …, [0.9, 1.0]`; probability 1.0 belongs to the last bin, empty bins are omitted, and every retained point reports its query count.
- Decision: selected the two highest-confidence hybrid errors first, then excluded them and sampled fixed confidence-spread quantiles within `no_opportunity` (3), `unchanged_failure` (3), rescue (2), and degradation (2), using query ID as the stable tie-break. This produced 12 unique cases before their text was inspected.
- Positive finding: hybrid's final top-10 success rate was `0.030864` above BM25, but its percentile interval touched zero (`[0.000000, 0.067901]`), so this is descriptive rather than decisive uncertainty evidence.
- Negative finding: dense reranking reduced nDCG@10 by `-0.043846` with an interval entirely below zero (`[-0.080770, -0.008949]`). The common-feature hybrid calibrator also had worse Brier than the train-fitted Platt baseline by `+0.068870` (`[0.027551, 0.108748]`).
- Mixed finding: Recall@10 changes for all rerankers and the hybrid raw-vs-calibrated AUPRC difference had intervals crossing zero. The manual cases show both genuine rescues and degradations, with recurring terminology mismatch, lexical distraction, incomplete candidate sets, and cross-encoder preference for literal title/phrase matches.
- Limitation: the analysis has only 162 calibration/dev queries, and SciFact qrels can make a highly topical non-gold abstract look like a retrieval error; confidence predicts annotated-evidence retrieval, not scientific truth.
- Result: all Phase 5 statistics, figures, and case selections use saved calibration/dev artifacts. The loader opened only `beir/scifact/train` to recover text for committed calibration/dev IDs; the final test split remains unopened for experimental evaluation.

## 2026-08-11 — Cross-validated confidence evaluation (higher-powered secondary analysis)

- Problem: the predeclared train/dev protocol evaluates confidence on 162 calibration/dev queries containing only ~20 failures. Every raw-vs-calibrated comparison had a bootstrap interval crossing zero, so H5 was neither confirmed nor refuted — the analysis was underpowered, not null.
- Decision: added a secondary estimate using stratified 5-fold cross-validation over all 809 calibration queries (calibration-train + calibration-dev pooled), seed 42, taking pooled out-of-fold predictions. Within each fold the scaler, LogisticRegression, and Platt baseline are fitted on that fold's training portion only and applied to held-out queries.
- Reason: this raises the failure count from ~20 to 117–137 depending on pipeline, which is what makes the comparison decisive. It is an estimation-power change, not a model-selection change: thresholds are still selected on calibration/dev by the predeclared rule, the train/dev result is still reported unchanged alongside, and `beir/scifact/test` remains untouched.
- Decision: report `base_rate` and `auprc_over_base_rate` alongside every AUPRC.
- Reason: AUPRC is bounded below by the positive-class rate, which is ~0.85 here. A bare AUPRC of 0.96 overstates the model; against the floor the real headroom captured is ~0.11.
- Positive finding (H5 confirmed for discrimination): the common-feature calibrator significantly outperforms the raw cross-encoder score on ranking metrics for all three pipelines, with every 95% interval excluding zero. AUROC `+0.065` (BM25, `[0.034, 0.098]`), `+0.083` (dense, `[0.039, 0.124]`), `+0.069` (hybrid, `[0.034, 0.105]`). AUPRC `+0.023` to `+0.025`, all intervals excluding zero.
- Negative finding (H5 refuted for calibration): the same calibrator produces significantly worse-calibrated probabilities than train-fitted Platt scaling of the raw score. Brier `+0.048` to `+0.058`, all three intervals entirely above zero. This is consistent with the predeclared `class_weight="balanced"` choice trading probability quality for minority-class recall, and is reported as a limitation rather than resolved by changing the setting after seeing results.
- Result: the honest summary of the confidence question is now "better at ranking which queries will fail, worse at stating the probability that one will" — a nuanced but decisive answer, replacing the previous underpowered null.
- Next action: `beir/scifact/test` remains unopened. An unweighted-calibrator ablation would test whether the Brier degradation is attributable to class weighting specifically, but has not been run.
