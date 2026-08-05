# Paired query-level bootstrap intervals

Split: `calibration-dev`. Difference: `B - A`. 1000 paired resamples, 95% percentile intervals, seed 42.

| comparison_id | metric | side_a_label | side_b_label | point_estimate_a | point_estimate_b | difference | ci_lower | ci_upper | n_queries | n_resamples | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_recall_at_10_before_after | recall@10 | bm25:first_stage | bm25:reranked | 0.824074 | 0.839506 | 0.015432 | -0.018519 | 0.049460 | 162 | 1000 | 42 |
| bm25_ndcg_at_10_before_after | ndcg@10 | bm25:first_stage | bm25:reranked | 0.715084 | 0.714266 | -0.000819 | -0.038804 | 0.035819 | 162 | 1000 | 42 |
| dense_bge_recall_at_10_before_after | recall@10 | dense_bge:first_stage | dense_bge:reranked | 0.866255 | 0.867284 | 0.001029 | -0.037037 | 0.038066 | 162 | 1000 | 42 |
| dense_bge_ndcg_at_10_before_after | ndcg@10 | dense_bge:first_stage | dense_bge:reranked | 0.768918 | 0.725072 | -0.043846 | -0.080770 | -0.008949 | 162 | 1000 | 42 |
| hybrid_rrf_recall_at_10_before_after | recall@10 | hybrid_rrf:first_stage | hybrid_rrf:reranked | 0.864198 | 0.870370 | 0.006173 | -0.040123 | 0.049383 | 162 | 1000 | 42 |
| hybrid_rrf_ndcg_at_10_before_after | ndcg@10 | hybrid_rrf:first_stage | hybrid_rrf:reranked | 0.751239 | 0.725600 | -0.025639 | -0.062162 | 0.010206 | 162 | 1000 | 42 |
| final_success_10_bm25_vs_dense_bge | final_success_10 | bm25:reranked_final_success_10 | dense_bge:reranked_final_success_10 | 0.845679 | 0.870370 | 0.024691 | -0.012346 | 0.067901 | 162 | 1000 | 42 |
| final_success_10_bm25_vs_hybrid_rrf | final_success_10 | bm25:reranked_final_success_10 | hybrid_rrf:reranked_final_success_10 | 0.845679 | 0.876543 | 0.030864 | 0.000000 | 0.067901 | 162 | 1000 | 42 |
| final_success_10_dense_bge_vs_hybrid_rrf | final_success_10 | dense_bge:reranked_final_success_10 | hybrid_rrf:reranked_final_success_10 | 0.870370 | 0.876543 | 0.006173 | -0.018673 | 0.030864 | 162 | 1000 | 42 |
| hybrid_auprc_raw_vs_common_calibrated | auprc | hybrid_rrf:raw_reranker_score | hybrid_rrf:common_feature_calibrated | 0.962666 | 0.974803 | 0.012137 | -0.009031 | 0.039529 | 162 | 1000 | 42 |
| hybrid_brier_platt_vs_common_calibrated | brier | hybrid_rrf:train_fitted_platt | hybrid_rrf:common_feature_calibrated | 0.097053 | 0.165923 | 0.068870 | 0.027551 | 0.108748 | 162 | 1000 | 42 |

Sources: `bm25` → `2026-08-05T114728.602187Z_bm25_calibration-dev` @ `c1f1155e91d9125b392074374fd2c7141db7e5f0`, `dense_bge` → `2026-08-05T114740.008737Z_dense_bge_calibration-dev` @ `c1f1155e91d9125b392074374fd2c7141db7e5f0`, `hybrid_rrf` → `2026-08-05T114749.452090Z_hybrid_rrf_calibration-dev` @ `c1f1155e91d9125b392074374fd2c7141db7e5f0`, `hybrid_confidence` → `2026-08-05T114843.983136Z_hybrid_rrf_calibration-train` @ `c1f1155e91d9125b392074374fd2c7141db7e5f0`.
