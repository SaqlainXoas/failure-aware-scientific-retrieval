| Pipeline | Primary | Confidence model | Role | AUROC | AUPRC | Brier | Queries |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| bm25 | no | raw reranker score | ranking baseline | 0.767 | 0.937 | n/a | 300 |
| bm25 | no | raw score, Platt-scaled | probability baseline | 0.767 | 0.937 | 0.131 | 300 |
| bm25 | no | common-feature calibrator | primary | 0.811 | 0.951 | 0.179 | 300 |
| dense_bge | no | raw reranker score | ranking baseline | 0.783 | 0.947 | n/a | 300 |
| dense_bge | no | raw score, Platt-scaled | probability baseline | 0.783 | 0.947 | 0.113 | 300 |
| dense_bge | no | common-feature calibrator | primary | 0.831 | 0.960 | 0.166 | 300 |
| hybrid_rrf | yes | raw reranker score | ranking baseline | 0.750 | 0.942 | n/a | 300 |
| hybrid_rrf | yes | raw score, Platt-scaled | probability baseline | 0.750 | 0.942 | 0.119 | 300 |
| hybrid_rrf | yes | common-feature calibrator | primary | 0.784 | 0.949 | 0.194 | 300 |
| hybrid_rrf | yes | calibrator + BM25/dense overlap | exploratory ablation | 0.788 | 0.950 | 0.192 | 300 |
