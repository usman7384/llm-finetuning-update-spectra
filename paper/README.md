# Paper

LaTeX source of the paper, posted as
[arXiv:2610.22438](https://arxiv.org/abs/2610.22438). It uses the TMLR style;
the style files (`tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `math_commands.tex`)
are the official ones, unmodified, and their license is in
`TMLR-STYLE-LICENSE`.

## Building

On Overleaf: zip this folder (`main.tex` at the top level) and upload it as a
new project. Locally, with MiKTeX or TeX Live: pdflatex, bibtex, then pdflatex
until the labels are stable (this document needs three passes after bibtex).

## Versions

`main.tex` loads `\usepackage{tmlr}`, the anonymous version: no author block,
and an "Under review as submission to TMLR" header.

The arXiv version is built with

```bash
uv run python scripts/build_arxiv.py --out <dir> --pdf-metadata --reword-reproducibility
```

It copies this folder, switches to `\usepackage[preprint]{tmlr}` (author block
shown, header removed), includes the generated `main.bbl`, and checks that the
package compiles with pdflatex alone, as arXiv does. The `\author` block
already carries the name, email and affiliation. Upload the zip it writes as
the source; for a new version, upload it as a replacement of 2610.22438.

The Reproducibility paragraph at the end of `sections/discussion.tex` still
says the code is withheld. Before the next arXiv version, replace that clause
with a link to this repository and drop `--reword-reproducibility`.

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
