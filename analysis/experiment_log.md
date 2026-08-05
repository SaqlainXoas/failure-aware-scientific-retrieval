# Experiment Log

## 2026-08-05 — Calibration split generation

- Decision: generated `splits/calibration_train.txt` and `splits/calibration_dev.txt` by taking only `beir/scifact/train` query IDs with at least one positive (`grade > 0`) qrel, sorting them, then `random.Random(seed=42).shuffle`, taking the first 20% (rounded) as dev.
- Reason: deterministic and reproducible from seed per plan §5; sorting before shuffling avoids depending on `ir_datasets`' internal iteration order across versions; restricting to evaluable queries ensures calibration only trains/evaluates on queries with a defined evidence label (queries with zero positive qrels can't produce a meaningful `final_success_10` target in later phases).
- Result: 809 `beir/scifact/train` queries total, all 809 had ≥1 positive qrel (0 excluded as unjudged/zero-positive) → 647 calibration_train / 162 calibration_dev (~80/20), 0 overlap. `beir/scifact/test` has 300 queries, 5183-doc corpus shared across both splits. Full counts recorded in `results/tables/dataset_stats.json`.
- Next action: `beir/scifact/test` remains untouched until retrieval implementations, configs, and confidence features are all locked, per plan §5.
