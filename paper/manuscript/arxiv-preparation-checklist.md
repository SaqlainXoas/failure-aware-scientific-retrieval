# arXiv preparation checklist

Nothing here has been done yet. This is the list of things that must be true before the
manuscript is submitted anywhere, and specifically before it is posted to arXiv. Items marked
**blocking** must be resolved by a human; the rest are mechanical.

## 1. Final human review — **blocking**

- [ ] Read the compiled PDF end to end. The manuscript was drafted with AI assistance; every
      claim, hedge and number needs a human sign-off before it carries your name.
- [ ] Confirm that no claim is stronger than the evidence: no state-of-the-art claim, no
      generalization beyond SciFact, no assertion that the risk model is well calibrated, no
      claim that threshold transfer implies probability calibration.
- [ ] Confirm the negative and non-replicating results are still stated plainly (Sections 5.3,
      5.5, 8).
- [ ] Get the professor's review back and incorporate it before posting. Posting to arXiv is
      effectively permanent — versions can be added, but nothing is removed.

## 2. Citation verification — **blocking**

Metadata for every entry in `references.bib` was checked on 2026-08-12 against Crossref, the
arXiv API, the ACL Anthology, PMLR and JMLR. Re-verify anything that could have changed:

- [ ] `chifu2025qpplimits` (arXiv:2504.01101) — preprint at time of writing. Check whether it
      has since appeared at a venue and cite the published version if so.
- [ ] `apurba2026sciret` (arXiv:2608.03860) — preprint at time of writing, posted 2026-08-04.
      Same check.
- [ ] `aljoofi2026multistage` — verify volume/issue/article number against the MDPI page
      (Crossref reports *Applied Sciences* 16(10):4813, 2026, DOI `10.3390/app16104813`).
- [ ] `robertson2009bm25` — Crossref reports *Foundations and Trends in IR* 4(1–2):1–174; some
      papers cite this as 3(4):333–389. Pick one and be consistent.
- [ ] `shtok2012querydrift` — ACM lists this as TOIS 30(2), Article 11; the bib entry uses the
      Crossref page range 1–35. Harmonize if the target venue's style expects article numbers.
- [ ] Confirm every reference in the bibliography is actually cited in the text and vice versa.

## 3. Author identity and ORCID

- [ ] Register an ORCID and add it to the author block (no ORCID is present today).
- [ ] Confirm the author name spelling and the contact email in `main.tex`
      (`\authorname`, `\authoremail`).
- [ ] Confirm the affiliation. It currently reads `Independent Researcher`. Do not claim an
      institutional affiliation unless it has been approved by that institution.
- [ ] Make sure the arXiv account owner name matches the manuscript author.

## 4. Endorsement — **blocking for a first-time cs.IR submission**

- [ ] arXiv requires endorsement for authors without a submission history in the category. Check
      the account's endorsement status early; obtaining an endorsement takes time.
- [ ] If an endorsement is needed, the reviewing professor is the natural person to ask — but ask
      explicitly, and separately from the review request.

## 5. Category selection

- [ ] Primary: **cs.IR** (Information Retrieval) — provisional.
- [ ] Consider cross-listing **cs.CL** (the dataset and the claim-verification framing) and
      possibly **cs.LG** (the calibration and selective-prediction analysis).
- [ ] Confirm the choice with the professor; the primary category affects who sees it.

## 6. LaTeX source packaging

- [ ] arXiv wants the source, not just a PDF. Upload `main.tex`, `references.bib`, the compiled
      `main.bbl`, and the `figures/` directory.
- [ ] **Include `main.bbl`.** arXiv runs BibTeX only if it can; shipping the `.bbl` avoids a
      failed build. Generate it with a full `latexmk -pdf main.tex` run and do not delete it.
- [ ] Remove build by-products (`.aux`, `.log`, `.out`, `.blg`, `.fdb_latexmk`, `.fls`).
- [ ] Confirm no absolute local filesystem paths appear anywhere in the source (figure paths are
      relative: `figures/…`).
- [ ] Confirm the package list is arXiv-safe: `geometry`, `microtype`, `amsmath`, `amssymb`,
      `graphicx`, `booktabs`, `array`, `caption`, `natbib`, `xcolor`, `hyperref`, `lmodern`,
      `url`, `fontenc`, `inputenc`. No `minted`, no `-shell-escape`, no external font files.
- [ ] Confirm filenames are arXiv-compatible: ASCII, no spaces (`main.tex`, `references.bib`,
      `figures/failure_breakdown.png`, etc.).

## 7. Figures

- [ ] Four figures are included, all PNG, all copied unmodified from `results/figures/`:
      `failure_breakdown.png`, `reranking_transitions.png`, `hybrid_reliability.png`,
      `hybrid_risk_coverage.png`.
- [ ] Check legibility at the printed size in the compiled PDF — the figures are scaled to
      66–76% of the text width, and axis labels should still be readable on paper.
- [ ] Confirm the reliability and risk-coverage captions still explain that the series labelled
      *common-feature calibrator* is the failure-risk model.
- [ ] Total source size is well under arXiv's limit, but re-check after any figure changes.

## 8. Licence selection

- [ ] Choose an arXiv licence. The repository is MIT-licensed; that covers the code, not the
      manuscript. CC BY 4.0 is the usual choice for a preprint intended to be reusable; the
      arXiv non-exclusive licence is the more conservative default.
- [ ] Confirm the choice does not conflict with any venue you may later submit to (most are
      fine with CC BY preprints; check the specific venue's policy).

## 9. Repository link — **blocking**

- [ ] The project repository (`github.com/SaqlainXoas/failure-aware-scientific-retrieval`) is
      currently **private**. The reproducibility statement deliberately makes no
      public-availability claim while that is true.
- [ ] Make it public, or archive a snapshot (e.g. Zenodo) and use the DOI.
- [ ] Then set `\repopublictrue` in `main.tex`, check `\repourl`, and remove the `TODO` comment
      block above it.
- [ ] Re-read the reproducibility statement after flipping the switch and confirm it reads
      correctly.

## 10. AI-assistance disclosure — **decision required**

- [ ] The manuscript currently carries **no** AI-assistance statement. A draft section existed
      and was removed at the author's direction; the paper now ends at the Reproducibility
      Statement.
- [ ] Check the target venue's policy before submitting. Several venues and journals now require
      an explicit statement, sometimes with mandated phrasing in a mandated location, and
      submitting without one where it is required can be grounds for desk rejection.
- [ ] arXiv has no dedicated disclosure field, so if a statement is wanted it goes in the
      manuscript body.

## 11. Final PDF inspection

- [ ] Compile from a clean directory and confirm no errors and no undefined references.
- [ ] Confirm the PDF text layer is searchable (it is not an image-only PDF).
- [ ] Check every hyperlink and DOI resolves.
- [ ] Search the compiled text for `TODO`, `XXX`, `FIXME`, placeholder names, and any absolute
      path — there should be none.
- [ ] Confirm there is no venue name, no copyright footer and no publication-status line
      anywhere in the document.
- [ ] Re-check every number against `results/tables/*.json` one final time.
