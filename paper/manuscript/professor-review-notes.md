# Notes for the reviewing professor

Thank you for reading this. The manuscript is `manuscript.pdf` in this directory; a one-page
summary is in [`contribution-summary.md`](contribution-summary.md). Everything is
venue-neutral — no venue template has been applied yet, deliberately, because the choice of
venue is one of the questions below.

## What I am asking

1. **Are the contributions scientifically defensible?**
   Section 1 lists six. The ones I am least sure about are (3), that improving candidate recall
   shifts failure composition without an equally large improvement in final success, and (6),
   the leakage-conscious protocol. Is (3) a finding or a restatement of arithmetic that follows
   from the recall numbers? Is (6) a contribution at all, or simply the minimum standard?

2. **Is the related-work positioning accurate?**
   Section 2 distinguishes this work from Al-Joofi, Sagheer and Hamdoun (*Applied Sciences*,
   2026), who also study multi-stage retrieval with RRF and cross-encoder reranking on SciFact,
   and from the SciRet preprint (arXiv:2608.03860, 2026), which independently reports an
   MS MARCO cross-encoder reducing precision on a scientific corpus. My reranking result points
   the opposite way to Al-Joofi et al.'s, and I have written that as "not directly comparable"
   rather than as a contradiction, because the encoders, candidate depths and indexing units
   differ and their absolute numbers are far from mine. Is that the right call, or am I
   under-engaging with a genuine disagreement? Is there work I have missed that makes the
   candidate-set/reranking decomposition less novel than I present it?

3. **Are the statistical interpretations appropriate?**
   Three places concern me specifically.
   - I describe the pre-specified development-set comparison as *underpowered rather than null*
     and add a cross-validated secondary analysis (pooled out-of-fold predictions over all 809
     calibration queries) to raise the failure count from ~20 to ~120. The primary result is
     still reported unchanged alongside. Is that an acceptable use of a secondary analysis, or
     does it read as moving the goalposts after an inconclusive result?
   - No multiplicity correction is applied across the pre-specified comparison list. I state
     this in Section 3.6 and in the limitations. Should intervals be adjusted?
   - Section 4.6 reports that the risk model's abstention cutoff transferred more consistently
     across splits than the raw score's. I have deliberately *not* framed this as evidence that
     the probability scale is stable, since the Brier results show it is not. Is that
     distinction drawn clearly enough?

4. **Is this better suited to arXiv only, a workshop, a short conference paper, or a journal?**
   My own guess is that it is a workshop or short-paper contribution rather than a full
   conference or journal paper: one small dataset, no new method, and three of the results are
   negative or non-replicating. I would rather be told to aim lower and be right than pad it.

5. **Which venue and template should be used?**
   The manuscript is written so it can be retargeted without rewriting: arXiv cs.IR preprint,
   ECIR short paper (Springer LNCS), an ACL/SDP-style workshop paper, an IEEE conference paper,
   or TMLR. If you have a preference, I will reformat to that template.

6. **Would additional experiments be essential for peer-reviewed acceptance?**
   The obvious candidates are (a) a second dataset to test whether the composition shift
   generalizes and (b) a domain-adapted or larger reranker to separate "this cross-encoder" from
   "cross-encoders". Both are real work. Which, if either, is a prerequisite rather than future
   work?
   The third candidate, an unweighted-risk-model ablation, **has since been run** and is
   reported in Section 4.4. It confirmed that the Brier degradation comes from the pre-specified
   class weighting: refitting the same features unweighted improves Brier on every pipeline, beats
   the Platt baseline it previously lost to, and leaves AUROC unchanged. I kept the pre-specified
   weighted model as the primary result and report the refit as a post-hoc explanation, since the
   test split was already spent and it therefore has no out-of-sample evaluation. I would value
   your view on whether that is the right call, or whether a reader would expect the
   better-calibrated configuration to be promoted despite lacking held-out validation.
   One constraint worth flagging: **the SciFact test split is spent.** It was evaluated once,
   by design, so any new experiment on this pipeline cannot be validated against untouched
   SciFact held-out data. A second dataset may therefore be the only clean way to add
   confirmatory evidence.

7. **Do you recommend any affiliation or authorship change?**
   The manuscript currently lists me as an independent researcher with no institutional
   affiliation. I have deliberately not claimed a university affiliation and have not added
   anyone as a coauthor. If you think either should change, that is entirely your call —
   `\authoraffiliation` in `main.tex` is a single macro, and there is an anonymous-mode switch
   for blind submission.

## Things I have deliberately not claimed

These may be worth pushing back on if you think I have been too conservative — or not
conservative enough.

- No state-of-the-art claim, and no claim that hybrid retrieval is generally superior.
- No claim that the decomposition generalizes beyond SciFact.
- No claim that cross-encoders fail in general; the result is scoped to one frozen MS MARCO
  cross-encoder transferred to scientific claims.
- No claim that the failure-risk model is well calibrated — the paper says the opposite.
- The word *preregistration* is not used. The protocol document that fixed the design was
  written before any experiment ran, but it is internal and not publicly verifiable, so the
  manuscript says "pre-specified protocol" and "documented before experiment execution". If you
  think publishing that document as supplementary material would strengthen the claim, I can do
  that.

## Open items I could not resolve myself

1. **Repository visibility.** The project's Git remote
   (`github.com/SaqlainXoas/failure-aware-scientific-retrieval`) is currently **private**. The
   reproducibility statement therefore describes the artifacts without asserting public
   availability, and `main.tex` has a `\repopublic` switch plus a source comment marking this.
   Before submission the repository needs to be public or archived (Zenodo DOI), and the switch
   flipped.
2. **Run provenance detail.** The manifests record a Git commit and a working-tree-clean flag
   per run. The calibration runs backing Sections 4.1–4.4 record a clean tree; the three
   held-out runs record a dirty tree (the working copy carried uncommitted table/report edits at
   that moment, and the run code itself was at commit `f1607e0`). The manuscript states the
   clean-tree fact for the calibration runs only, and does not claim it for the held-out runs.
   Tell me if you would rather it were stated explicitly either way.
3. **AI-assistance disclosure.** The manuscript contains no AI-assistance statement. A draft
   section existed and was removed at my direction, so the paper now ends at the Reproducibility
   Statement. Worth knowing before we pick a venue: several venues and journals now require an
   explicit statement, sometimes with mandated wording in a mandated place. If the target venue
   requires one, it has to go back in before submission. Tell me whether you think it belongs
   there regardless.
4. **Author metadata.** No ORCID is included yet. I will register one before any submission.
5. **Length versus format.** The brief for this draft asked for single-column, 11pt, A4 and
   roughly 7--10 pages of main content. Those two constraints pull against each other: the
   manuscript runs 15 pages of main content plus 2 of references in single-column 11pt, which is
   about 7--8 pages of a typical two-column proceedings template. I chose not to cut further,
   because the remaining material is method detail, uncertainty reporting and the honest
   caveats. If you want it genuinely shorter, tell me what to drop --- my own candidates are the
   qualitative section (§5) and the threshold-transfer subsection (§4.6).

## What was *not* changed to produce this manuscript

No code, configuration, split file, result table, figure, manifest, experiment log, root README
or the existing `paper/report.md` was modified. No experiment was re-run and no number was
recomputed. Every figure in `figures/` is a byte-for-byte copy of the committed original. The
manuscript is a reframing of existing, fixed results.
