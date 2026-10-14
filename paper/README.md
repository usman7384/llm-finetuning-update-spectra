# Paper

TMLR submission source. The style files (`tmlr.sty`, `tmlr.bst`,
`fancyhdr.sty`, `math_commands.tex`) are the official ones, unmodified; their
license is in `TMLR-STYLE-LICENSE`.

## Building

On Overleaf: zip this folder (`main.tex` at the top level) and upload it as a
new project. Locally, with MiKTeX or TeX Live: pdflatex, bibtex, then pdflatex
until the labels are stable (this document needs three passes after bibtex).

## Versions

`main.tex` loads `\usepackage{tmlr}`, the anonymous submission version: no
author block, "Under review as submission to TMLR" header. Keep it that way
for OpenReview, and never link the repository or the preprint from it.

- **arXiv / preprint:** `uv run python scripts/build_arxiv.py --out <dir>`
  builds the package from a copy of this folder with
  `\usepackage[preprint]{tmlr}` (author block shown, header removed), includes
  the generated `main.bbl`, and checks that it compiles with pdflatex alone, as
  arXiv does. The `\author` block already carries the name, email and
  affiliation. Two optional edits for the preprint are off by default; see the
  script's `--help`.
- **Camera-ready:** `\usepackage[accepted]{tmlr}`, then set `\month`, `\year`
  and the OpenReview link in `main.tex`, and put the repository link in the
  Reproducibility paragraph of `sections/discussion.tex`.

## Layout

```
main.tex          preamble, notation macros, section order
sections/         one file per section, appendix last
figures/          vector PDFs from scripts/paper_figures.py
main.bib          references
```

Every number in the text comes from the CSVs in `results/` (the top-level
README maps each table and figure to its script and file), and the figures are
regenerated with `uv run python scripts/paper_figures.py`.
