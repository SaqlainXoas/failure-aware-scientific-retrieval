| Pipeline | Primary | Confidence model | Role | AUROC | AUPRC | Brier | Queries |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| bm25 | no | raw reranker score | ranking baseline | 0.811 | 0.955 | n/a | 162 |
| bm25 | no | raw score, Platt-scaled | probability baseline | 0.811 | 0.955 | 0.104 | 162 |
| bm25 | no | common-feature calibrator | primary | 0.874 | 0.976 | 0.157 | 162 |
| dense_bge | no | raw reranker score | ranking baseline | 0.814 | 0.963 | n/a | 162 |
| dense_bge | no | raw score, Platt-scaled | probability baseline | 0.814 | 0.963 | 0.098 | 162 |
| dense_bge | no | common-feature calibrator | primary | 0.881 | 0.979 | 0.152 | 162 |
| hybrid_rrf | yes | raw reranker score | ranking baseline | 0.804 | 0.963 | n/a | 162 |
| hybrid_rrf | yes | raw score, Platt-scaled | probability baseline | 0.804 | 0.963 | 0.097 | 162 |
| hybrid_rrf | yes | common-feature calibrator | primary | 0.851 | 0.975 | 0.166 | 162 |
| hybrid_rrf | yes | calibrator + BM25/dense overlap | exploratory ablation | 0.851 | 0.975 | 0.162 | 162 |
