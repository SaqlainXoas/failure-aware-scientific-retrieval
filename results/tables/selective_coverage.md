| Pipeline | Primary | Confidence model | Role | Success @100% | Kept @100% | Success @80% | Kept @80% | Success @60% | Kept @60% |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | no | raw reranker score | ranking baseline | 0.846 | 162 | 0.923 | 130 | 0.938 | 97 |
| bm25 | no | raw score, Platt-scaled | probability baseline | 0.846 | 162 | 0.923 | 130 | 0.938 | 97 |
| bm25 | no | common-feature calibrator | primary | 0.846 | 162 | 0.908 | 130 | 0.979 | 97 |
| dense_bge | no | raw reranker score | ranking baseline | 0.870 | 162 | 0.938 | 130 | 0.959 | 97 |
| dense_bge | no | raw score, Platt-scaled | probability baseline | 0.870 | 162 | 0.938 | 130 | 0.959 | 97 |
| dense_bge | no | common-feature calibrator | primary | 0.870 | 162 | 0.946 | 130 | 0.979 | 97 |
| hybrid_rrf | yes | raw reranker score | ranking baseline | 0.877 | 162 | 0.946 | 130 | 0.948 | 97 |
| hybrid_rrf | yes | raw score, Platt-scaled | probability baseline | 0.877 | 162 | 0.946 | 130 | 0.948 | 97 |
| hybrid_rrf | yes | common-feature calibrator | primary | 0.877 | 162 | 0.938 | 130 | 0.979 | 97 |
| hybrid_rrf | yes | calibrator + BM25/dense overlap | exploratory ablation | 0.877 | 162 | 0.931 | 130 | 0.979 | 97 |
