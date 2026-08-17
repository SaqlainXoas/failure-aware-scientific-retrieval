# Manuscript: Diagnosing Candidate-Set and Reranking Failures in Scientific Claim Retrieval

Venue-neutral LaTeX manuscript prepared for scientific review. It is a reframing of
[`../report.md`](../report.md) — the same experiments, the same committed numbers, restructured
and re-worded for an academic reader. Neither the report nor any result artifact was modified
to produce it.

## Files

| File | What it is |
| --- | --- |
| `main.tex` | The manuscript. Standard `article` class, A4, 11pt, single column. |
| `references.bib` | BibTeX database. Every entry was checked against Crossref, the arXiv API, the ACL Anthology, PMLR or JMLR on 2026-08-12. |
| `figures/` | Copies of the four figures used, taken unmodified from `../../results/figures/`. |
| `manuscript.pdf` | Compiled output. |
| `contribution-summary.md` | One-page summary for the reviewing professor. |
| `professor-review-notes.md` | The specific questions being asked of the reviewer. |
| `arxiv-preparation-checklist.md` | What must happen before any arXiv submission. |

## Compiling

With a standard TeX distribution (TeX Live, MacTeX):

```bash
latexmk -pdf main.tex
```

Or without `latexmk`:

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Or with [Tectonic](https://tectonic-typesetting.github.io/), which fetches its own packages and
runs the BibTeX passes automatically (this is what was used to build the committed PDF):

```bash
tectonic -X compile main.tex
```

All three write `main.pdf`; the deliverable copy is `manuscript.pdf`:

```bash
cp main.pdf manuscript.pdf
```

Build by-products (`main.aux`, `main.bbl`, `main.log`, `main.out`, `main.pdf`) are not intended
to be committed and have been removed from this directory. One exception applies at submission
time: arXiv wants `main.bbl` shipped with the source, so regenerate it with a full `latexmk` run
before packaging (see [`arxiv-preparation-checklist.md`](arxiv-preparation-checklist.md)).

The committed PDF is 17 pages: 15 pages of main content and 2 pages of references.

## Compile-time switches

Both live near the top of `main.tex`.

- **Anonymous mode.** Change `\anonymousfalse` to `\anonymoustrue` to replace the author block
  with "Anonymous Author(s) / Affiliation withheld for review". Nothing else in the document
  identifies the author.
- **Affiliation.** `\authorname`, `\authoraffiliation` and `\authoremail` are defined once and
  used only in the author block. The affiliation is currently `Independent Researcher`; no
  institutional affiliation is claimed anywhere in the manuscript.
- **Repository availability.** `\repopublicfalse` is the current setting, because the project's
  Git remote was a private repository when the manuscript was written. The reproducibility
  statement therefore describes the artifacts without asserting public availability. Once the
  repository is public — or archived with a DOI — set `\repopublictrue` and update `\repourl`.

## Where the numbers come from

Every number in the manuscript was read from a committed artifact, not recomputed:

| Manuscript element | Source |
| --- | --- |
| Table 1 (first stage, both splits) | `results/tables/first_stage_comparison.{json,md}`, `results/tables/final_test_first_stage.{json,md}` |
| Table 2 (failure decomposition, both splits) | `results/tables/failure_decomposition.{json,md}`, `results/tables/final_test_failure_decomposition.{json,md}` |
| Table 3 (reranking deltas with CIs) | `results/tables/rerank_deltas.md`, `results/tables/bootstrap_intervals.{json,md}`, `results/tables/final_test_bootstrap.{json,md}` |
| Table 4, top panel (risk model, cross-validated) | `results/tables/cv_bootstrap_intervals.{json,md}` |
| Table 4, bottom panel (risk model, held out) | `results/tables/final_test_confidence.md`, `results/tables/final_test_bootstrap.{json,md}` |
| Table 5 (post-hoc class-weight ablation) | `results/tables/class_weight_ablation.{json,md}` |
| Table 6 (replication summary) | the tables above, plus `analysis/experiment_log.md` |
| Table 7 (threshold transfer) | `runs/…_hybrid_rrf_test/threshold_transfer.json` |
| Dev-protocol values quoted in §4.4 | `results/tables/confidence_comparison.{json,md}`, `results/tables/selective_coverage.md` |
| AUPRC floor (0.852 / 0.088 / 0.110) | `runs/…_hybrid_rrf_calibration-train/confidence_cv_metrics.json` |
| Dataset counts (§3.1) | `results/tables/dataset_stats.json` |
| Figures 1–4 | `results/figures/`, provenance in `results/figures/figure_provenance.json` |
| Transition counts in Figure 2 and §4.3 | `results/figures/reranking_transitions.png`, `results/tables/failure_decomposition.json` |
| Qualitative cases (§5) | `analysis/failure_cases.md`, `results/tables/failure_case_selection.json` |
| Leakage-control claims (§3.7) | `tests/test_splits.py`, `tests/test_calibration_leakage.py` |

The figure labelled *common-feature calibrator* in `figures/hybrid_reliability.png` and
`figures/hybrid_risk_coverage.png` is the model the manuscript calls the **failure-risk model**.
The figures were not regenerated or relabelled; the correspondence is stated in the captions and
in §1 of the manuscript so that the artifact names stay traceable.
