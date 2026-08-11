# Cross-validated bootstrap intervals (higher-powered secondary analysis)

The predeclared train/dev protocol evaluates on 162 queries holding ~20 failures, too few to separate the confidence models. Pooling out-of-fold predictions over all calibration queries raises the failure count without touching the test split.

Protocol: stratified 5-fold cross-validation over calibration-train + calibration-dev. Difference: `B - A`. 1000 paired resamples, 95% percentile intervals, seed 42.

| comparison_id | metric | point_estimate_a | point_estimate_b | difference | ci_lower | ci_upper | excludes_zero | n_queries | n_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_cv_auroc_raw_vs_common_calibrated | auroc | 0.781337 | 0.846346 | 0.065009 | 0.033770 | 0.098201 | True | 809 | 137 |
| bm25_cv_auprc_raw_vs_common_calibrated | auprc | 0.936475 | 0.961686 | 0.025211 | 0.010949 | 0.042843 | True | 809 | 137 |
| bm25_cv_brier_platt_vs_common_calibrated | brier | 0.115203 | 0.163442 | 0.048239 | 0.031002 | 0.065253 | True | 809 | 137 |
| dense_bge_cv_auroc_raw_vs_common_calibrated | auroc | 0.756805 | 0.840250 | 0.083444 | 0.039029 | 0.123853 | True | 809 | 117 |
| dense_bge_cv_auprc_raw_vs_common_calibrated | auprc | 0.940375 | 0.965444 | 0.025069 | 0.009558 | 0.041891 | True | 809 | 117 |
| dense_bge_cv_brier_platt_vs_common_calibrated | brier | 0.110219 | 0.161721 | 0.051502 | 0.031892 | 0.071201 | True | 809 | 117 |
| hybrid_rrf_cv_auroc_raw_vs_common_calibrated | auroc | 0.757946 | 0.827443 | 0.069497 | 0.033575 | 0.105247 | True | 809 | 120 |
| hybrid_rrf_cv_auprc_raw_vs_common_calibrated | auprc | 0.939267 | 0.961851 | 0.022584 | 0.008596 | 0.039004 | True | 809 | 120 |
| hybrid_rrf_cv_brier_platt_vs_common_calibrated | brier | 0.113281 | 0.171248 | 0.057967 | 0.039111 | 0.075945 | True | 809 | 120 |

Sources: `bm25` → `2026-08-11T071021.766876Z_bm25_calibration-train` @ `58b849b35f849dd9738174d950f621580462c7a5`, `dense_bge` → `2026-08-11T071027.086483Z_dense_bge_calibration-train` @ `58b849b35f849dd9738174d950f621580462c7a5`, `hybrid_rrf` → `2026-08-11T071032.700346Z_hybrid_rrf_calibration-train` @ `58b849b35f849dd9738174d950f621580462c7a5`.
