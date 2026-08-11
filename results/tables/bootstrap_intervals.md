# Paired query-level bootstrap intervals

Split: `calibration-dev`. Difference: `B - A`. 1000 paired resamples, 95% percentile intervals, seed 42. Every row uses the same protocol, so the per-row resample count and seed live in the JSON rather than repeating here.

| Comparison | Metric | Side A | Side B | A | B | Difference (B − A) | 95% CI | Queries |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| bm25_recall_at_10_before_after | Recall@10 | bm25:first_stage | bm25:reranked | 0.824 | 0.840 | +0.015 | [-0.019, +0.049] | 162 |
| bm25_ndcg_at_10_before_after | nDCG@10 | bm25:first_stage | bm25:reranked | 0.715 | 0.714 | -0.001 | [-0.039, +0.036] | 162 |
| dense_bge_recall_at_10_before_after | Recall@10 | dense_bge:first_stage | dense_bge:reranked | 0.866 | 0.867 | +0.001 | [-0.037, +0.038] | 162 |
| dense_bge_ndcg_at_10_before_after | nDCG@10 | dense_bge:first_stage | dense_bge:reranked | 0.769 | 0.725 | -0.044 | [-0.081, -0.009] | 162 |
| hybrid_rrf_recall_at_10_before_after | Recall@10 | hybrid_rrf:first_stage | hybrid_rrf:reranked | 0.864 | 0.870 | +0.006 | [-0.040, +0.049] | 162 |
| hybrid_rrf_ndcg_at_10_before_after | nDCG@10 | hybrid_rrf:first_stage | hybrid_rrf:reranked | 0.751 | 0.726 | -0.026 | [-0.062, +0.010] | 162 |
| final_success_10_bm25_vs_dense_bge | Final success@10 | bm25:reranked_final_success_10 | dense_bge:reranked_final_success_10 | 0.846 | 0.870 | +0.025 | [-0.012, +0.068] | 162 |
| final_success_10_bm25_vs_hybrid_rrf | Final success@10 | bm25:reranked_final_success_10 | hybrid_rrf:reranked_final_success_10 | 0.846 | 0.877 | +0.031 | [+0.000, +0.068] | 162 |
| final_success_10_dense_bge_vs_hybrid_rrf | Final success@10 | dense_bge:reranked_final_success_10 | hybrid_rrf:reranked_final_success_10 | 0.870 | 0.877 | +0.006 | [-0.019, +0.031] | 162 |
| hybrid_auprc_raw_vs_common_calibrated | AUPRC | hybrid_rrf:raw_reranker_score | hybrid_rrf:common_feature_calibrated | 0.963 | 0.975 | +0.012 | [-0.009, +0.040] | 162 |
| hybrid_brier_platt_vs_common_calibrated | Brier | hybrid_rrf:train_fitted_platt | hybrid_rrf:common_feature_calibrated | 0.097 | 0.166 | +0.069 | [+0.028, +0.109] | 162 |

Sources: `bm25` → `2026-08-11T071020.275112Z_bm25_calibration-dev` @ `58b849b35f849dd9738174d950f621580462c7a5`, `dense_bge` → `2026-08-11T071024.303890Z_dense_bge_calibration-dev` @ `58b849b35f849dd9738174d950f621580462c7a5`, `hybrid_rrf` → `2026-08-11T071029.770459Z_hybrid_rrf_calibration-dev` @ `58b849b35f849dd9738174d950f621580462c7a5`, `hybrid_confidence` → `2026-08-11T071032.700346Z_hybrid_rrf_calibration-train` @ `58b849b35f849dd9738174d950f621580462c7a5`.
