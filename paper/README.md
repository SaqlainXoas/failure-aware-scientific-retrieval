# Report

[`report.md`](report.md) is the source of truth — a 4–6 page write-up of the study, longer than
the project README and self-contained enough to read on its own. It references the figures in
`../results/figures/` rather than copying them, so it cannot drift from the committed results.

Build outputs are not committed; regenerate them with:

```bash
pandoc paper/report.md --standalone --embed-resources --toc --number-sections --resource-path=paper:. -o paper/report.html
```

That works with pandoc alone and inlines the figures, so the HTML is a single portable file. For
a PDF, pandoc needs a TeX engine (`brew install --cask basictex`, or `cargo install tectonic`):

```bash
pandoc paper/report.md --toc --number-sections --resource-path=paper:. -V geometry:margin=1in -o paper/report.pdf
```

Every number in the report comes from `../results/tables/*.json`. If a result is ever
regenerated, re-check the report against those files before rebuilding.
