# Held-out test split — paired bootstrap intervals

Single pre-registered evaluation of `beir/scifact/test`. Confidence models were fitted on calibration-train and never refitted; display thresholds come from calibration-dev. Difference: `B - A`.

1000 paired resamples, 95% percentile intervals, seed 42.

| Comparison | Metric | A | B | Difference (B − A) | 95% CI | Excludes 0 | Queries |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| bm25_recall_at_10_before_after | Recall@10 | 0.774 | 0.801 | +0.027 | [-0.005, +0.060] | no | 300 |
| bm25_ndcg_at_10_before_after | nDCG@10 | 0.662 | 0.684 | +0.022 | [-0.008, +0.052] | no | 300 |
| dense_bge_recall_at_10_before_after | Recall@10 | 0.836 | 0.823 | -0.013 | [-0.043, +0.018] | no | 300 |
| dense_bge_ndcg_at_10_before_after | nDCG@10 | 0.713 | 0.696 | -0.017 | [-0.043, +0.012] | no | 300 |
| hybrid_rrf_recall_at_10_before_after | Recall@10 | 0.822 | 0.829 | +0.007 | [-0.030, +0.041] | no | 300 |
| hybrid_rrf_ndcg_at_10_before_after | nDCG@10 | 0.700 | 0.696 | -0.004 | [-0.034, +0.024] | no | 300 |
| final_success_10_bm25_vs_dense_bge | Final success@10 | 0.817 | 0.837 | +0.020 | [-0.010, +0.050] | no | 300 |
| final_success_10_bm25_vs_hybrid_rrf | Final success@10 | 0.817 | 0.843 | +0.027 | [+0.003, +0.050] | yes | 300 |
| final_success_10_dense_bge_vs_hybrid_rrf | Final success@10 | 0.837 | 0.843 | +0.007 | [-0.010, +0.027] | no | 300 |
| bm25_auroc_raw_vs_common_calibrated | AUROC | 0.767 | 0.811 | +0.044 | [-0.005, +0.092] | no | 300 |
| bm25_auprc_raw_vs_common_calibrated | AUPRC | 0.937 | 0.951 | +0.014 | [-0.003, +0.032] | no | 300 |
| bm25_brier_raw_vs_common_calibrated | Brier | 0.131 | 0.179 | +0.048 | [+0.018, +0.078] | yes | 300 |
| dense_bge_auroc_raw_vs_common_calibrated | AUROC | 0.783 | 0.831 | +0.049 | [-0.017, +0.120] | no | 300 |
| dense_bge_auprc_raw_vs_common_calibrated | AUPRC | 0.947 | 0.960 | +0.014 | [-0.007, +0.036] | no | 300 |
| dense_bge_brier_raw_vs_common_calibrated | Brier | 0.113 | 0.166 | +0.054 | [+0.024, +0.084] | yes | 300 |
| hybrid_rrf_auroc_raw_vs_common_calibrated | AUROC | 0.750 | 0.784 | +0.034 | [-0.028, +0.102] | no | 300 |
| hybrid_rrf_auprc_raw_vs_common_calibrated | AUPRC | 0.942 | 0.949 | +0.007 | [-0.012, +0.027] | no | 300 |
| hybrid_rrf_brier_raw_vs_common_calibrated | Brier | 0.119 | 0.194 | +0.075 | [+0.043, +0.107] | yes | 300 |

Sources: `bm25` → `2026-08-11T090932.212943Z_bm25_test` @ `f1607e09ee6bd99ba9eee931b7daf7aefef5ac16`, `dense_bge` → `2026-08-11T090935.698926Z_dense_bge_test` @ `f1607e09ee6bd99ba9eee931b7daf7aefef5ac16`, `hybrid_rrf` → `2026-08-11T090939.130196Z_hybrid_rrf_test` @ `f1607e09ee6bd99ba9eee931b7daf7aefef5ac16`.
