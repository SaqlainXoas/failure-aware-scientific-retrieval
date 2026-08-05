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
