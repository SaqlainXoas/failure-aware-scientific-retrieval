# Contribution summary

*Diagnosing Candidate-Set and Reranking Failures in Scientific Claim Retrieval: A Controlled
SciFact Case Study* — Saqlain Ahmed, Independent Researcher.

## Research question

When a two-stage retrieval pipeline fails, which stage was responsible? Either the first stage
never surfaced the evidence document — in which case the reranker was irrelevant to the outcome
— or it surfaced it and the reranker failed to rank it. A single Recall@10 or nDCG@10 number
cannot tell these apart. The paper measures the split directly, and asks whether the resulting
failures can be predicted at query time from retrieval-internal signals alone.

## Why it matters

The two failure types demand opposite engineering responses, and aggregate metrics hide which
one a system is actually limited by. In this case study the aggregate view and the decomposed
view point to different conclusions: hybrid fusion looks like a modest three-point improvement,
but the decomposition shows it nearly eliminates one failure mode while a second grows to
replace it. Knowing which stage binds is what tells you where further effort would pay.

## What was evaluated

SciFact via BEIR (5,183 abstracts). Three frozen first stages — BM25 (`bm25s`), a dense
bi-encoder (`BAAI/bge-small-en-v1.5`), and their Reciprocal Rank Fusion (k = 60, untuned) — each
at candidate depth 100, each reranked by the same frozen `cross-encoder/ms-marco-MiniLM-L6-v2`
over its own top 50. Nothing was fine-tuned. Development used 809 training claims split 80/20
(seed 42) into 647 calibration-train and 162 calibration-dev queries. The 300-query test split
was opened once, at the end, with every model, feature, threshold and selection rule already
fixed. Uncertainty is paired query-level percentile bootstrap (1,000 resamples, 95%, seed 42)
over a comparison list fixed in advance.

## Main findings

1. **Improved candidate recall relocates failures more than it removes them.** From BM25 to
   hybrid, candidate-set failures fall from 11.7% to 3.1% of queries, but final top-10 success
   rises only from 84.6% to 87.7%; the share of remaining failures that originate in the
   candidate set falls from 76% to 25% (held out: 67% to 34%). The reranking stage becomes the
   binding constraint.
2. **The evaluated cross-encoder does not reliably improve final ranking here.** Rescues and
   degradations are of similar magnitude, and no held-out before/after difference is
   significant for any pipeline.
3. **Failure is predictable, but the pre-specified model's probabilities are not trustworthy.**
   An eight-feature logistic failure-risk model beats the raw reranker score at ordering queries
   by risk (cross-validated AUROC +0.065 to +0.083, all intervals excluding zero) and is
   significantly worse-calibrated (Brier +0.048 to +0.058, replicated held out at +0.048 to
   +0.075). A post-hoc ablation locates that miscalibration in the pre-specified
   `class_weight="balanced"`: refitting the same features unweighted improves Brier by −0.060 to
   −0.070, beats even the Platt baseline, and leaves AUROC unchanged. The weighting cost
   probability quality and bought no discrimination. It is calibration-data only, so the
   unweighted refit is an explanation of the pre-specified result, not a validated replacement.
4. **The abstention cutoff transferred more consistently than the raw score's.** At a 60%
   coverage target the risk model's calibration-dev cutoff kept 58.0% of held-out queries; the
   raw score's kept 48.3%. This is a statement about the score distribution, not about
   probability calibration.
5. **One calibration-set finding did not replicate.** A significant degradation of dense
   nDCG@10 by reranking (−0.044, CI [−0.081, −0.009]) became −0.017 with an interval crossing
   zero on held-out data. It is reported as not replicating rather than dropped.

## What is genuinely distinctive

Not the pipeline — it is entirely standard and deliberately so. What is distinctive is the
protocol: an exhaustive failure taxonomy in which every query gets exactly one label so
candidate-set and reranking failures partition cleanly; failure discrimination and probability
calibration evaluated and reported as separate properties; and a single, once-only held-out
evaluation under choices fixed beforehand, with the non-replicating result reported. Two of the
three headline results are negative.

## Major limitations

One small dataset, so nothing here shows the composition shift generalizes. About 20 failures in
the development split and about 50 held out, which is why a cross-validated secondary analysis
exists and why the held-out AUROC comparison is directionally consistent but not individually
significant. All models frozen and general-domain, so the reranking result is evidence about one
specific MS MARCO → SciFact transfer. The attribution of the Brier degradation to class weighting
is now measured, but on calibration data only — the better-calibrated unweighted refit has no
out-of-sample evaluation, and no other risk-model hyperparameter was varied. SciFact
qrels mark annotated evidence, not scientific truth. And the test split is now spent: no
untouched validation data remains for future changes to this system.

## Questions for the professor

1. Are the six stated contributions scientifically defensible as written, or should any be
   demoted to an observation?
2. Is the positioning against Al-Joofi et al. (2026) and SciRet (2026) accurate and fair?
3. Are the statistical interpretations — especially "underpowered, not null" and the
   threshold-transfer wording — appropriate?
4. Is this an arXiv-only preprint, a workshop paper, a short conference paper, or a journal
   submission?
5. Which venue and template should be targeted?
6. Would additional experiments be essential for peer-reviewed acceptance, and if so, which
   ones, given that the SciFact test split is spent?
7. Any recommendation on affiliation or authorship?
