# Paper

TMLR submission source. The style files (`tmlr.sty`, `tmlr.bst`,
`fancyhdr.sty`, `math_commands.tex`) are the official ones, unmodified; their
license is in `TMLR-STYLE-LICENSE`.

## Building

On Overleaf: zip this folder and upload it as a new project, main file
`main.tex`. Locally: `latexmk -pdf main.tex`.

## Versions

`main.tex` loads `\usepackage{tmlr}`, which is the anonymous submission
version: no author names, "Under review" header. Keep it that way for
OpenReview, and don't add the repository link.

- **arXiv / preprint:** switch to `\usepackage[preprint]{tmlr}`, fill in the
  email in `\author`, and add the code link in the conclusion.
- **Camera-ready:** `\usepackage[accepted]{tmlr}`, plus `\month`, `\year` and
  the OpenReview link.

## Layout

```
main.tex          preamble, notation macros, section order
sections/         one file per section, appendix last
figures/          vector PDFs from scripts/paper_figures.py
main.bib          references
```

Every number in the text comes from the CSVs in `results/`, and the figures
are regenerated with `uv run python scripts/paper_figures.py`. `docs/report.md`
has the longer write-up the paper was cut down from.
