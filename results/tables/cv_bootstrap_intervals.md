# Cross-validated bootstrap intervals (higher-powered secondary analysis)

The predeclared train/dev protocol evaluates on 162 queries holding ~20 failures, too few to separate the confidence models. Pooling out-of-fold predictions over all calibration queries raises the failure count without touching the test split.

Protocol: stratified 5-fold cross-validation over calibration-train + calibration-dev. Difference: `B - A`, where A is the raw-score baseline and B the common-feature calibrator. 1000 paired resamples, 95% percentile intervals, seed 42.

| Comparison | Metric | A | B | Difference (B − A) | 95% CI | Excludes 0 | Queries | Failures |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| bm25_cv_auroc_raw_vs_common_calibrated | AUROC | 0.781 | 0.846 | +0.065 | [+0.034, +0.098] | yes | 809 | 137 |
| bm25_cv_auprc_raw_vs_common_calibrated | AUPRC | 0.936 | 0.962 | +0.025 | [+0.011, +0.043] | yes | 809 | 137 |
| bm25_cv_brier_platt_vs_common_calibrated | Brier | 0.115 | 0.163 | +0.048 | [+0.031, +0.065] | yes | 809 | 137 |
| dense_bge_cv_auroc_raw_vs_common_calibrated | AUROC | 0.757 | 0.840 | +0.083 | [+0.039, +0.124] | yes | 809 | 117 |
| dense_bge_cv_auprc_raw_vs_common_calibrated | AUPRC | 0.940 | 0.965 | +0.025 | [+0.010, +0.042] | yes | 809 | 117 |
| dense_bge_cv_brier_platt_vs_common_calibrated | Brier | 0.110 | 0.162 | +0.052 | [+0.032, +0.071] | yes | 809 | 117 |
| hybrid_rrf_cv_auroc_raw_vs_common_calibrated | AUROC | 0.758 | 0.827 | +0.069 | [+0.034, +0.105] | yes | 809 | 120 |
| hybrid_rrf_cv_auprc_raw_vs_common_calibrated | AUPRC | 0.939 | 0.962 | +0.023 | [+0.009, +0.039] | yes | 809 | 120 |
| hybrid_rrf_cv_brier_platt_vs_common_calibrated | Brier | 0.113 | 0.171 | +0.058 | [+0.039, +0.076] | yes | 809 | 120 |

Sources: `bm25` → `2026-08-17T095329.197391Z_bm25_calibration-train` @ `7d2b4aeb7a810d9d55d6b64c5502c5c157e596db`, `dense_bge` → `2026-08-17T095337.535188Z_dense_bge_calibration-train` @ `7d2b4aeb7a810d9d55d6b64c5502c5c157e596db`, `hybrid_rrf` → `2026-08-17T095349.389304Z_hybrid_rrf_calibration-train` @ `7d2b4aeb7a810d9d55d6b64c5502c5c157e596db`.
