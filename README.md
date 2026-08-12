# Failure-aware retrieval on SciFact

Separating candidate-set failures from reranking failures in a two-stage scientific-claim
retrieval pipeline, and testing whether failure can be predicted at query time.

When a retrieval pipeline fails to surface the right scientific abstract, there are two
possibilities: the abstract never entered the candidate set, or it entered and the reranker did
not put it near the top. These need opposite fixes, and a single Recall@10 or nDCG@10 number
does not distinguish them. This study measures the split directly across BM25, a dense
bi-encoder, and Reciprocal Rank Fusion of the two, each followed by the same frozen
cross-encoder.

Everything was developed on 809 calibration queries. The 300-query SciFact test split was opened
**once**, at the end, with every model and threshold already fixed. Both sets of numbers appear
below.

1. **Hybrid retrieval has the best candidate recall.** Recall@50 of 0.966 on calibration and
   0.944 held-out, against 0.925 for dense and 0.869 for BM25 held-out.
2. **As first-stage recall improves, the bottleneck moves to the reranker.** 76% of BM25's
   failures are queries where the gold abstract was never retrieved; for hybrid that falls to
   25% (held-out: 67% to 34%). Better retrieval relocates failures more than it removes them.
3. **Cross-encoder reranking does not reliably improve final ranking quality here.** On held-out
   data no before/after difference is significant for any pipeline. The stronger claim visible on
   calibration data, that reranking significantly *degrades* dense nDCG@10, did not replicate.
4. **An 8-feature logistic calibrator ranks failure risk better than the raw reranker score but
   states worse probabilities.** The Brier degradation is significant everywhere, including
   held-out (+0.048 to +0.075). The AUROC advantage is consistent in direction everywhere and
   significant only in the higher-powered cross-validated estimate.

In short, the calibrator is better at ranking which queries will fail and worse at saying how
likely one is to fail.

---

## The decomposition

Every query is assigned exactly one outcome, so failures partition cleanly:

- `candidate_success_50`: is a gold abstract anywhere in the top-50 candidates the reranker sees?
- `final_success_10`: is a gold abstract in the final top 10?

A **candidate-set failure** is a query the reranker never had a chance on. A **reranking
failure** is one where the evidence was retrieved and then not ranked.

![Candidate-set vs. reranking failures](results/figures/failure_breakdown.png)

BM25 fails more often overall but for a recoverable reason. Hybrid fails less often, and what
remains is mostly the reranker's doing.

![What reranking changed](results/figures/reranking_transitions.png)

Reranking rescues 4–7 queries per pipeline and degrades 3–6. Those roughly cancel, which is why
the aggregate metrics barely move.

## Results

### First-stage retrieval

([full table](results/tables/first_stage_comparison.md))

| Pipeline | Recall@10 | Recall@50 | nDCG@10 |
| --- | ---: | ---: | ---: |
| bm25 | 0.824 | 0.873 | 0.715 |
| dense_bge | 0.866 | 0.948 | 0.769 |
| hybrid_rrf | 0.864 | **0.966** | 0.751 |

Fusion buys candidate recall rather than top-10 precision: hybrid has the best Recall@50 and the
second-best nDCG@10.

### Where the failures come from

([full table](results/tables/failure_decomposition.md))

| Pipeline | Candidate-set failure | Reranking failure | Final success | Failures from candidate set |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 11.7% | 3.7% | 84.6% | 76.0% |
| dense_bge | 4.9% | 8.0% | 87.0% | 38.1% |
| hybrid_rrf | 3.1% | 9.3% | **87.7%** | 25.0% |

Final success rises by only 3 points from BM25 to hybrid, but the composition of the remaining
failures inverts. Improving the retriever further would buy little; the reranker is now the
binding constraint.

### Effect of reranking

([full table](results/tables/rerank_deltas.md))

| Pipeline | Recall@10 before → after | nDCG@10 before → after | nDCG@10 Δ |
| --- | ---: | ---: | ---: |
| bm25 | 0.824 → 0.840 | 0.715 → 0.714 | −0.001 |
| dense_bge | 0.866 → 0.867 | 0.769 → 0.725 | **−0.044** |
| hybrid_rrf | 0.864 → 0.870 | 0.751 → 0.726 | −0.026 |

A general-domain MS MARCO cross-encoder applied to scientific claims is not a free improvement.
On calibration data only the dense drop has an interval excluding zero ([bootstrap
intervals](results/tables/bootstrap_intervals.md)), and that drop **did not replicate** on the
held-out split, where the same comparison is −0.017 with an interval crossing zero. What survives
in both is the weaker claim: no pipeline shows a significant gain from reranking.

### Predicting failure

([full table](results/tables/cv_bootstrap_intervals.md))

| Pipeline | Metric | Raw score | Calibrator | Difference | 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| bm25 | AUROC | 0.781 | 0.846 | +0.065 | [+0.034, +0.098] |
| dense_bge | AUROC | 0.757 | 0.840 | +0.083 | [+0.039, +0.124] |
| hybrid_rrf | AUROC | 0.758 | 0.827 | +0.069 | [+0.034, +0.105] |
| bm25 | Brier | 0.115 | 0.163 | +0.048 | [+0.031, +0.065] |
| dense_bge | Brier | 0.110 | 0.162 | +0.052 | [+0.032, +0.071] |
| hybrid_rrf | Brier | 0.113 | 0.171 | +0.058 | [+0.039, +0.076] |

Lower Brier is better, so the model that wins on AUROC loses on calibration, consistently and
significantly. This traces to the predeclared `class_weight="balanced"` setting, which trades
probability quality for minority-class recall. It was left in place rather than tuned away after
the fact.

![Hybrid reliability](results/figures/hybrid_reliability.png)

The calibrator's curve sits above the diagonal almost everywhere, meaning it systematically
under-states success. That is what a worse Brier at better AUROC looks like.

The panels hold a different number of points because the baseline never predicts below 0.3, so
its first three bins are empty, while the calibrator uses the full range. That wider spread is
why it ranks failures better and why its probabilities overshoot.

## Held-out evaluation

The 300-query SciFact test split was opened once, after every retriever, feature, model, and
selection rule was locked. Nothing was refitted: the calibrator is the calibration-train model,
and the abstention thresholds are the ones chosen on calibration-dev.
([full table](results/tables/final_test_bootstrap.md))

| | Calibration | Held-out | Replicated? |
| --- | --- | --- | --- |
| Best Recall@50 | hybrid, 0.966 | hybrid, 0.944 | yes; the primary-pipeline rule holds out of sample |
| Failures from candidate set | 76% → 25% | 67% → 34% | yes; same ordering and direction |
| Hybrid vs BM25 final success | +0.031, CI touches 0 | **+0.027, CI [+0.003, +0.050]** | yes, and stronger held-out |
| Reranking hurts dense nDCG@10 | −0.044, CI [−0.081, −0.009] | −0.017, CI [−0.043, +0.012] | **no** |
| Calibrator Brier worse | +0.048 to +0.058 | **+0.048 to +0.075** | yes; significant on all three |
| Calibrator AUROC better | +0.065 to +0.083 (CV) | +0.034 to +0.049, CIs cross 0 | direction yes, power no |

The failure-decomposition result and the calibration weakness both survive. The claim that
reranking actively degrades dense retrieval did not; it was a calibration-set effect, and the
sections above were weakened to match rather than left as the more interesting sentence.

Held-out data also answers a question the calibration split could not. Carrying the
calibration-dev abstention threshold over unchanged, the calibrator lands much closer to its
target coverage than the raw score does: at a 60% target it keeps 58.0% of queries against the
raw score's 48.3%. Its probability scale transfers across splits even though its Brier is worse,
which is what makes it the more usable of the two for routing.

## Method

**Data.** SciFact via `ir_datasets` (`beir/scifact`), a 5,183-abstract corpus. The 809 training
queries are split 80/20 by seed 42 into 647 calibration-train and 162 calibration-dev. The
300-query `beir/scifact/test` split was held out through all development and evaluated once, at
the end.

**Retrievers**, all frozen, none fine-tuned, top-100 candidates each:

- **BM25** via `bm25s` (k1 = 1.5, b = 0.75, δ = 0.5, Lucene scoring, English stopwords, no stemmer)
- **Dense** via `BAAI/bge-small-en-v1.5` at revision `5c38ec7`, with the model card's query
  instruction applied to queries only, cosine over normalized embeddings
- **Hybrid** via Reciprocal Rank Fusion of the two, k = 60, not tuned

**Reranker.** `cross-encoder/ms-marco-MiniLM-L6-v2` at revision `c5ee24c`, over each pipeline's
own top-50 candidates. It can only reorder what its own retriever supplied, which is the
constraint that makes the failure decomposition meaningful.

**Failure taxonomy.** Each query gets one of five transition labels: `already_successful`,
`rescued_by_reranker`, `degraded_by_reranker`, `unchanged_failure`, `no_opportunity`. They
partition the query set, and the saved artifacts carry the raw counts so that is checkable.

**Confidence.** Eight features, all computed from a query's own retrieval, none derived from
relevance labels: four first-stage (normalized top-1 score, top1–top2 margin, first-stage/rerank
rank correlation, whether the first-stage top-1 survives into the reranked top 3) and four
reranker-side (top-1 score, top1–top2 margin, top-5 mean and standard deviation). A
StandardScaler and LogisticRegression predict `final_success_10`. Two baselines are reported
beside it and never conflated: the raw reranker score for ranking metrics, and a train-fitted
Platt scaling of that score for Brier.

**Statistics.** Paired query-level bootstrap, 1,000 resamples, 95% percentile intervals, seed 42,
on a fixed list of comparisons. Because 162 dev queries hold only ~20 failures, the confidence
comparison is *also* reported over pooled out-of-fold predictions from stratified 5-fold
cross-validation on all 809 calibration queries (~120 failures). Both are reported; the
cross-validated version is labelled a secondary analysis and never replaces the predeclared one.

![Hybrid risk-coverage](results/figures/hybrid_risk_coverage.png)

## Pre-registration and leakage control

The failure definitions, the leakage rules, the metrics, and the list of statistical comparisons
were written down before the first run, as was the rule for choosing the primary pipeline:
highest calibration-dev Recall@50, which selected hybrid.

The scaler and the logistic regression are fitted on calibration-train only. Thresholds are
selected on calibration-dev only. No feature is derived from relevance judgments. The
calibration loader cannot reach held-out data: `resolve_split` rejects `test` and a unit test
asserts it, so opening that split required calling a separately named function, once,
deliberately, and it is recorded in the log.

Nothing was changed in response to the held-out results. Where a held-out result contradicted a
calibration-set result, the calibration-set claim was weakened rather than the reverse.

Where a result came out badly, as with reranking and with the calibrator's Brier score, the
setting was left as predeclared and the result reported.
[`analysis/experiment_log.md`](analysis/experiment_log.md) is the dated record of every such
decision, including the ones made after seeing bad numbers.

## Limitations

- **One dataset.** SciFact is small (5,183 abstracts, 162 calibration-dev and 300 held-out
  queries). Nothing here shows the pattern holds elsewhere.
- **Off-the-shelf models, no tuning.** The reranker is trained on MS MARCO web passages. Its
  weak showing on scientific claims is evidence about *that transfer*, not about cross-encoders.
- **Small failure counts.** 162 dev queries hold ~20 failures, which is why the cross-validated
  secondary analysis exists. Treat the dev-only intervals as underpowered rather than null.
- **Qrels mark annotated evidence, not truth.** A topically excellent non-gold abstract counts
  as a failure here. Several such cases appear in [the case study](analysis/failure_cases.md).
- **Confidence predicts retrieval, not correctness.** The target is "did we retrieve the
  annotated evidence", not "is the claim true".
- **The held-out split is spent.** It was evaluated once and will not be used again, so further
  changes to this system cannot be validated the same way.
- **300 held-out queries hold ~50 failures**, which is why the AUROC comparison is directionally
  consistent there but not individually significant.

## Reproducing

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+. The corpus downloads automatically
via `ir_datasets` on first run (~9 MB cached locally).

```bash
uv sync
```

`runs/` is not committed, since the runs come to ~31 MB of per-query artifacts, so they have to
be regenerated before the tables and figures can be rebuilt:

```bash
uv run python -m retrieval.run --config configs/bm25.yaml --split calibration-dev
```

Repeat for `dense_bge.yaml` and `hybrid_rrf.yaml`, then for `--split calibration-train`, which
additionally fits the confidence models on train and evaluates them on dev. To reproduce the
held-out section, run each config once more with `--split test`:

```bash
uv run python -m retrieval.run --config configs/hybrid_rrf.yaml --split test
```

That path loads `beir/scifact/test` through a separate loader and logs a warning; it refits
nothing and reuses the calibration-train models. Then rebuild the artifacts:

```bash
uv run python -m retrieval.tables
```

That rewrites everything in `results/` from the saved runs: tables, figures, manifests, and the
failure case study. Held-out tables are written only if test runs exist. Nothing in `results/` is
ever hand-edited. Tests:

```bash
uv run pytest
```

On Apple Silicon the dense and reranking stages use MPS automatically. Corpus embeddings, the
BM25 index, and reranker scores are cached in `.cache/`, so only the first run of each pipeline
pays the model cost. A cold BM25 calibration-dev run takes about a minute on an M-series laptop,
and the calibration-train runs are four times the queries.

## Longer write-up

[`paper/report.md`](paper/report.md) is a 4–6 page version of this study with the full method,
related work, the qualitative failure analysis, and the reasoning behind each choice. See
[`paper/README.md`](paper/README.md) for the one-line pandoc build.

## Layout

```
retrieval/
├── data.py        SciFact loading, the seed-42 split, device selection
├── retrieve.py    bm25_retrieve / dense_retrieve / hybrid_retrieve
├── rerank.py      top-50 cross-encoder reranking
├── evaluate.py    ranking metrics, failure and transition labels
├── confidence.py  features, calibration, paired bootstrap
├── analysis.py    bootstrap intervals and figures
├── cases.py       deterministic failure-case selection and rendering
├── plots.py       the five committed figures
├── tables.py      result tables from saved runs
├── run.py         pipeline entrypoint
└── runio.py       run-directory persistence and manifests

results/tables/    14 tables as canonical .json, 12 with a .md view for reading
results/figures/   5 figures, with figure_provenance.json naming their source runs
results/manifests/ the manifest of every run backing a committed number
analysis/          experiment log, and 12 hand-annotated failure cases
splits/            the committed calibration query IDs
paper/             the longer write-up
```

Every committed number traces to a run directory, a git commit, and a model revision through
`results/manifests/`.

## License

MIT. See [LICENSE](LICENSE).
