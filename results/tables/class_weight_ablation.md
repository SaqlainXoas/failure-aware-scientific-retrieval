# Class-weight ablation (post-hoc)

**Status:** post-hoc ablation, specified after observing the calibration-set Brier result. Not used for model selection; the primary model remains the class-weighted calibrator, and no predeclared result was recomputed or replaced.

The manuscript attributes the risk model's worse-than-baseline Brier score to the predeclared class weighting, which up-weights the minority failure class and pulls predicted probabilities down. That attribution was an interpretation until this ablation measured it.

Protocol: stratified 5-fold cross-validation over calibration-train + calibration-dev. Difference: `B - A`, where B is always the unweighted refit of the common-feature model. Brier is a loss, so a *negative* difference means the unweighted model is better calibrated; AUROC is a score, so a negative difference there means discrimination was given up in exchange. 1000 paired resamples, 95% percentile intervals, seed 42.

| Comparison | Metric | A | B | Difference (B − A) | 95% CI | Excludes 0 | Queries | Failures |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| bm25_cv_brier_platt_vs_unweighted | Brier | 0.115 | 0.103 | -0.012 | [-0.019, -0.005] | yes | 809 | 137 |
| bm25_cv_brier_balanced_vs_unweighted | Brier | 0.163 | 0.103 | -0.060 | [-0.074, -0.046] | yes | 809 | 137 |
| bm25_cv_auroc_balanced_vs_unweighted | AUROC | 0.846 | 0.848 | +0.001 | [-0.001, +0.004] | no | 809 | 137 |
| dense_bge_cv_brier_platt_vs_unweighted | Brier | 0.110 | 0.096 | -0.015 | [-0.023, -0.006] | yes | 809 | 117 |
| dense_bge_cv_brier_balanced_vs_unweighted | Brier | 0.162 | 0.096 | -0.066 | [-0.081, -0.050] | yes | 809 | 117 |
| dense_bge_cv_auroc_balanced_vs_unweighted | AUROC | 0.840 | 0.842 | +0.002 | [-0.002, +0.005] | no | 809 | 117 |
| hybrid_rrf_cv_brier_platt_vs_unweighted | Brier | 0.113 | 0.102 | -0.012 | [-0.019, -0.004] | yes | 809 | 120 |
| hybrid_rrf_cv_brier_balanced_vs_unweighted | Brier | 0.171 | 0.102 | -0.070 | [-0.084, -0.054] | yes | 809 | 120 |
| hybrid_rrf_cv_auroc_balanced_vs_unweighted | AUROC | 0.827 | 0.827 | -0.001 | [-0.003, +0.002] | no | 809 | 120 |

Sources: `bm25` → `2026-08-17T095329.197391Z_bm25_calibration-train` @ `7d2b4aeb7a810d9d55d6b64c5502c5c157e596db`, `dense_bge` → `2026-08-17T095337.535188Z_dense_bge_calibration-train` @ `7d2b4aeb7a810d9d55d6b64c5502c5c157e596db`, `hybrid_rrf` → `2026-08-17T095349.389304Z_hybrid_rrf_calibration-train` @ `7d2b4aeb7a810d9d55d6b64c5502c5c157e596db`.
