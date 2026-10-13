"""Build the arXiv preprint package from the TMLR submission sources in paper/.

The submission sources are never modified. The script works on a copy:

  1. copy paper/ to a temporary work directory;
  2. in the copy's main.tex, replace ``\\usepackage{tmlr}`` with
     ``\\usepackage[preprint]{tmlr}`` -- this is the only source change by
     default. The option reveals the author block and drops the "Under review
     as submission to TMLR" running header; nothing else in the layout moves;
  3. compile with pdflatex, bibtex, then pdflatex until the labels are stable
     (this document needs three passes after bibtex) to produce main.bbl;
  4. assemble the arXiv package (main.tex, main.bbl, math_commands.tex,
     tmlr.sty, tmlr.bst, fancyhdr.sty, TMLR-STYLE-LICENSE, sections/*.tex,
     figures/*.pdf and, unless --no-bib, main.bib) as a flat zip with
     main.tex at the top level;
  5. unpack the zip into a clean directory and compile it the way arXiv does,
     with pdflatex only (at least three passes, no BibTeX). Fail if any
     citation or reference is undefined, if labels have not converged, or if
     the page count differs from the full build;
  6. write arxiv_source.zip and arxiv_preview.pdf (the PDF from step 5) to the
     --out directory and print a short verification summary.

Two further source edits exist for the preprint but are OFF by default and
must be asked for explicitly:

  --reword-reproducibility   the Reproducibility paragraph ends with "they are
                             kept out of the anonymous submission", which reads
                             oddly in a non-anonymous preprint; this swaps that
                             clause for "they are withheld while the paper is
                             under review" (see REPRODUCIBILITY_SUBSTITUTION);
  --pdf-metadata             add a \\hypersetup{pdftitle, pdfauthor} line after
                             \\usepackage{hyperref} so the PDF's Title/Author
                             fields are filled (they are empty otherwise).

Usage:

    uv run python scripts/build_arxiv.py --out <output directory>

Requires MiKTeX (miktex-pdftex.exe, miktex-bibtex.exe) or a TeX Live
pdflatex/bibtex on PATH. The MiKTeX bin directory is taken from --miktex-bin,
then the MIKTEX_BIN environment variable, then the default per-user install
location.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_DIR = REPO_ROOT / "paper"
DEFAULT_MIKTEX_BIN = Path(
    os.environ.get(
        "MIKTEX_BIN",
        Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64",
    )
)

# The one change that makes the preprint. Must match exactly one
# uncommented line in main.tex.
STYLE_LINE_SUBMISSION = r"\usepackage{tmlr}"
STYLE_LINE_PREPRINT = r"\usepackage[preprint]{tmlr}"

# ---------------------------------------------------------------------------
# Optional edits, OFF by default. Each is an exact-string substitution that
# must match exactly once, so a changed source cannot be silently half-edited.
# ---------------------------------------------------------------------------
REPRODUCIBILITY_FILE = "sections/discussion.tex"
REPRODUCIBILITY_SUBSTITUTION = (
    "they are kept out of the anonymous submission.",
    "they are withheld while the paper is under review.",
)

HYPERREF_LINE = r"\usepackage{hyperref}"

# Files that go into the package, relative to the paper directory.
PACKAGE_FILES = [
    "main.tex",
    "main.bbl",  # generated in step 3
    "math_commands.tex",
    "tmlr.sty",
    "tmlr.bst",
    "fancyhdr.sty",
    "TMLR-STYLE-LICENSE",
]
PACKAGE_GLOBS = ["sections/*.tex", "figures/*.pdf"]
OPTIONAL_BIB = "main.bib"

ZIP_NAME = "arxiv_source.zip"
PREVIEW_NAME = "arxiv_preview.pdf"
MAX_PACKAGE_BYTES = 50 * 1024 * 1024  # arXiv's hard limit


class BuildError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# TeX toolchain
# ---------------------------------------------------------------------------
class Tex:
    """Locates pdflatex/bibtex (MiKTeX first, then PATH) and runs them."""

    def __init__(self, miktex_bin: Path):
        self.pdftotext: list[str] | None = None
        self.pdfinfo: list[str] | None = None
        mt_pdftex = miktex_bin / "miktex-pdftex.exe"
        mt_bibtex = miktex_bin / "miktex-bibtex.exe"
        if mt_pdftex.exists() and mt_bibtex.exists():
            self.pdflatex = [str(mt_pdftex), "-undump=pdflatex"]
            self.bibtex = [str(mt_bibtex)]
            for name, attr in (("miktex-pdftotext.exe", "pdftotext"),
                               ("miktex-pdfinfo.exe", "pdfinfo")):
                if (miktex_bin / name).exists():
                    setattr(self, attr, [str(miktex_bin / name)])
        else:
            pdflatex = shutil.which("pdflatex")
            bibtex = shutil.which("bibtex")
            if not (pdflatex and bibtex):
                raise BuildError(
                    f"no TeX found: neither {mt_pdftex} nor pdflatex/bibtex on PATH"
                )
            self.pdflatex = [pdflatex]
            self.bibtex = [bibtex]
        for name, attr in (("pdftotext", "pdftotext"), ("pdfinfo", "pdfinfo")):
            if getattr(self, attr) is None and shutil.which(name):
                setattr(self, attr, [shutil.which(name)])

    def _run(self, cmd: list[str], cwd: Path, log_name: str) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=900,
        )
        (cwd / log_name).write_text(proc.stdout + proc.stderr, encoding="utf-8")
        return proc

    def run_pdflatex(self, cwd: Path, pass_no: int) -> str:
        cmd = self.pdflatex + ["-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        proc = self._run(cmd, cwd, f"_pdflatex_{pass_no}.txt")
        log = (cwd / "main.log").read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0 or "Output written on main.pdf" not in log:
            tail = "\n".join(proc.stdout.splitlines()[-30:])
            raise BuildError(f"pdflatex pass {pass_no} failed in {cwd}:\n{tail}")
        return log

    def run_bibtex(self, cwd: Path) -> None:
        env = dict(os.environ)
        # Build is in place, so cwd already holds main.bib and tmlr.bst; the
        # variables only matter if someone points the build elsewhere.
        env["BIBINPUTS"] = str(cwd) + os.pathsep + env.get("BIBINPUTS", "")
        env["BSTINPUTS"] = str(cwd) + os.pathsep + env.get("BSTINPUTS", "")
        proc = subprocess.run(
            self.bibtex + ["main"], cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env, timeout=300,
        )
        (cwd / "_bibtex.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        blg = (cwd / "main.blg").read_text(encoding="utf-8", errors="replace") \
            if (cwd / "main.blg").exists() else ""
        if proc.returncode != 0 or not (cwd / "main.bbl").exists():
            raise BuildError(f"bibtex failed in {cwd}:\n{proc.stdout}\n{blg[-2000:]}")
        errors = [ln for ln in blg.splitlines()
                  if ln.startswith("I couldn't") or "error message" in ln
                  or ln.startswith("Warning--")]
        if errors:
            print("  bibtex notes:")
            for ln in errors:
                print("    " + ln)

    def text(self, pdf: Path, first: int | None = None, last: int | None = None) -> str | None:
        if not self.pdftotext:
            return None
        cmd = list(self.pdftotext)
        if first is not None:
            cmd += ["-f", str(first)]
        if last is not None:
            cmd += ["-l", str(last)]
        cmd += [str(pdf), "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=120)
        return proc.stdout if proc.returncode == 0 else None

    def info(self, pdf: Path) -> dict[str, str]:
        if not self.pdfinfo:
            return {}
        proc = subprocess.run(self.pdfinfo + [str(pdf)], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=60)
        out: dict[str, str] = {}
        for ln in proc.stdout.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                out[k.strip()] = v.strip()
        return out


# ---------------------------------------------------------------------------
# Log inspection
# ---------------------------------------------------------------------------
PAGES_RE = re.compile(r"Output written on main\.pdf \((\d+) pages?")
UNDEFINED_RES = [
    re.compile(r"LaTeX Warning: There were undefined (references|citations)"),
    re.compile(r"LaTeX Warning: Citation .* undefined"),
    re.compile(r"LaTeX Warning: Reference .* undefined"),
    re.compile(r"LaTeX Warning: Label\(s\) may have changed"),
    re.compile(r"Package natbib Warning: Citation .* undefined"),
    re.compile(r"^! ", re.M),
]


RERUN_RE = re.compile(r"Rerun to get|Label\(s\) may have changed|rerunfilecheck Warning")
MIN_PASSES = 3  # what arXiv runs; fewer is never accepted here
MAX_PASSES = 6


def page_count(log: str) -> int:
    m = PAGES_RE.search(log)
    if not m:
        raise BuildError("could not find page count in main.log")
    return int(m.group(1))


def pdflatex_until_stable(tex: "Tex", cwd: Path, first_pass: int = 1) -> tuple[str, int]:
    """Run pdflatex at least MIN_PASSES times (counting from first_pass) and
    then until the log no longer asks for a rerun. Returns (log, passes run)."""
    pass_no = first_pass
    while True:
        log = tex.run_pdflatex(cwd, pass_no)
        done = pass_no - first_pass + 1
        if done >= MIN_PASSES and not RERUN_RE.search(log):
            return log, done
        if done >= MAX_PASSES:
            raise BuildError(f"labels did not converge after {done} pdflatex passes in {cwd}")
        pass_no += 1


def unresolved(log: str) -> list[str]:
    hits = []
    for rx in UNDEFINED_RES:
        hits += [m.group(0) for m in rx.finditer(log)]
    return hits


# ---------------------------------------------------------------------------
# Source edits (on the copy only)
# ---------------------------------------------------------------------------
def substitute_once(path: Path, old: str, new: str, what: str) -> None:
    data = path.read_bytes().decode("utf-8")
    # Ignore commented-out copies of the line when counting.
    live = "\n".join(ln for ln in data.splitlines() if not ln.lstrip().startswith("%"))
    n = live.count(old)
    if n != 1:
        raise BuildError(f"{what}: expected exactly one occurrence of {old!r} in "
                         f"{path.name}, found {n}")
    out_lines = []
    done = False
    for ln in data.splitlines(keepends=True):
        if not done and not ln.lstrip().startswith("%") and old in ln:
            ln = ln.replace(old, new, 1)
            done = True
        out_lines.append(ln)
    path.write_bytes("".join(out_lines).encode("utf-8"))


def title_and_author(main_tex: str) -> tuple[str, str]:
    m = re.search(r"\\title\{(.*?)\}\s*$", main_tex, re.S | re.M)
    if not m:
        raise BuildError("could not parse \\title{...} from main.tex")
    title = re.sub(r"\s+", " ", m.group(1).replace("\\\\", " ")).strip()
    m = re.search(r"\\name\s+(.*?)\s+\\email", main_tex, re.S)
    if not m:
        raise BuildError("could not parse \\name ... \\email from main.tex")
    author = re.sub(r"\s+", " ", m.group(1)).strip()
    return title, author


def prepare_sources(src: Path, reword_reproducibility: bool, pdf_metadata: bool) -> None:
    main_tex = src / "main.tex"
    substitute_once(main_tex, STYLE_LINE_SUBMISSION, STYLE_LINE_PREPRINT, "style option")
    if reword_reproducibility:
        old, new = REPRODUCIBILITY_SUBSTITUTION
        substitute_once(src / REPRODUCIBILITY_FILE, old, new, "reproducibility rewording")
    if pdf_metadata:
        title, author = title_and_author(main_tex.read_text(encoding="utf-8"))
        hypersetup = (f"{HYPERREF_LINE}\n"
                      f"\\hypersetup{{pdftitle={{{title}}},pdfauthor={{{author}}}}}")
        substitute_once(main_tex, HYPERREF_LINE, hypersetup, "pdf metadata")


# ---------------------------------------------------------------------------
# Static checks on the package contents
# ---------------------------------------------------------------------------
ABS_PATH_RE = re.compile(
    r"\\(?:input|include|includegraphics)(?:\[[^\]]*\])?\{\s*(?:[A-Za-z]:[\\/]|/|~)")


def static_checks(src: Path, files: list[Path]) -> list[str]:
    problems = []
    for f in files:
        rel = f.relative_to(src).as_posix()
        if not rel.isascii():
            problems.append(f"non-ASCII file name: {rel}")
        if f.suffix == ".tex":
            text = f.read_text(encoding="utf-8", errors="replace")
            for m in ABS_PATH_RE.finditer(text):
                problems.append(f"absolute path in {rel}: {m.group(0)}")
        if f.suffix == ".pdf":
            data = f.read_bytes()
            if b"/Type3" in data or b"FontType 3" in data:
                problems.append(f"Type 3 font in figure {rel}")
    return problems


def package_files(src: Path, include_bib: bool) -> list[Path]:
    names = list(PACKAGE_FILES) + ([OPTIONAL_BIB] if include_bib else [])
    files = [src / n for n in names]
    for pattern in PACKAGE_GLOBS:
        files += sorted(src.glob(pattern))
    missing = [f for f in files if not f.exists()]
    if missing:
        raise BuildError("missing package files: " + ", ".join(str(m) for m in missing))
    return files


def write_zip(zip_path: Path, src: Path, files: list[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(files, key=lambda p: p.relative_to(src).as_posix()):
            zf.write(f, f.relative_to(src).as_posix())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--out", required=True, type=Path,
                    help="directory that receives arxiv_source.zip and arxiv_preview.pdf")
    ap.add_argument("--paper", type=Path, default=DEFAULT_PAPER_DIR,
                    help=f"submission sources (default: {DEFAULT_PAPER_DIR})")
    ap.add_argument("--miktex-bin", type=Path, default=DEFAULT_MIKTEX_BIN,
                    help="MiKTeX bin directory holding miktex-pdftex.exe and miktex-bibtex.exe")
    ap.add_argument("--work", type=Path, default=None,
                    help="work directory (default: a fresh temp directory, removed afterwards)")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the work directory for inspection")
    ap.add_argument("--no-bib", action="store_true",
                    help="leave main.bib out of the package (arXiv ignores it when main.bbl exists)")
    ap.add_argument("--reword-reproducibility", action="store_true",
                    help="OPTIONAL EDIT, off by default: reword the 'anonymous submission' clause")
    ap.add_argument("--pdf-metadata", action="store_true",
                    help="OPTIONAL EDIT, off by default: set PDF Title/Author via \\hypersetup")
    args = ap.parse_args(argv)

    paper = args.paper.resolve()
    if not (paper / "main.tex").exists():
        raise BuildError(f"{paper} does not contain main.tex")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    tex = Tex(args.miktex_bin)

    if args.work:
        work = args.work.resolve()
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix="arxiv_build_"))
        cleanup = not args.keep_work

    try:
        # 1. copy
        src = work / "src"
        shutil.copytree(paper, src, ignore=shutil.ignore_patterns(
            "*.aux", "*.bbl", "*.blg", "*.log", "*.out", "*.pdf.bak",
            "*.synctex.gz", "*.fdb_latexmk", "*.fls", "main.pdf", "__pycache__"))
        print(f"copied {paper} -> {src}")

        # 2. edit the copy
        prepare_sources(src, args.reword_reproducibility, args.pdf_metadata)
        edits = [STYLE_LINE_SUBMISSION + " -> " + STYLE_LINE_PREPRINT]
        if args.reword_reproducibility:
            edits.append("reproducibility clause reworded")
        if args.pdf_metadata:
            edits.append("\\hypersetup{pdftitle,pdfauthor} added")
        print("source edits: " + "; ".join(edits))

        # 3. full build
        print("full build: pdflatex, bibtex, then pdflatex until labels are stable")
        tex.run_pdflatex(src, 1)
        tex.run_bibtex(src)
        full_log, n_full = pdflatex_until_stable(tex, src, first_pass=2)
        full_pages = page_count(full_log)
        bad = unresolved(full_log)
        if bad:
            raise BuildError("full build has unresolved items:\n  " + "\n  ".join(bad))
        print(f"  ok, {full_pages} pages after 1 + {n_full} pdflatex passes")

        # 4. package
        files = package_files(src, include_bib=not args.no_bib)
        problems = static_checks(src, files)
        if problems:
            raise BuildError("package checks failed:\n  " + "\n  ".join(problems))
        zip_path = work / ZIP_NAME
        write_zip(zip_path, src, files)
        size = zip_path.stat().st_size
        if size > MAX_PACKAGE_BYTES:
            raise BuildError(f"package is {size} bytes, over arXiv's 50 MB limit")
        print(f"package: {len(files)} files, {size / 1024:.0f} KB")
        for f in files:
            print("  " + f.relative_to(src).as_posix())

        # 5. arXiv-style check build: pdflatex only, from the zip contents
        check = work / "check"
        check.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(check)
        print("check build from the zip: pdflatex only (at least three passes), no bibtex")
        check_log, n_check = pdflatex_until_stable(tex, check)
        check_pages = page_count(check_log)
        bad = unresolved(check_log)
        if bad:
            raise BuildError("pdflatex-only build has unresolved items:\n  " + "\n  ".join(bad))
        if check_pages != full_pages:
            raise BuildError(f"page count differs: full build {full_pages}, "
                             f"pdflatex-only build {check_pages}")
        print(f"  ok, {check_pages} pages after {n_check} passes, "
              "no undefined citations or references")

        pdf = check / "main.pdf"
        text_all = tex.text(pdf)
        if text_all is not None:
            _, author = title_and_author((src / "main.tex").read_text(encoding="utf-8"))
            email_m = re.search(r"\\email\s+(\S+)", (src / "main.tex").read_text(encoding="utf-8"))
            email = email_m.group(1) if email_m else ""
            page1 = tex.text(pdf, 1, 1) or ""
            findings = []
            if "Under review" in text_all:
                findings.append("'Under review' still appears in the PDF")
            if "Anonymous authors" in text_all:
                findings.append("'Anonymous authors' still appears in the PDF")
            if author not in page1:
                findings.append(f"author name {author!r} not found on page 1")
            if email and email not in page1:
                findings.append(f"email {email!r} not found on page 1")
            if "[?]" in text_all or re.search(r"\(\?\)", text_all):
                findings.append("a '?' placeholder appears in the PDF text")
            if findings:
                raise BuildError("PDF text checks failed:\n  " + "\n  ".join(findings))
            print(f"  page 1 shows {author} <{email}>; no 'Under review', no '?' placeholders")
        else:
            print("  (pdftotext not available; text checks skipped)")
        info = tex.info(pdf)
        if info:
            print(f"  PDF metadata: Title={info.get('Title', '')!r} "
                  f"Author={info.get('Author', '')!r}")

        # 6. deliverables
        shutil.copy2(zip_path, out / ZIP_NAME)
        shutil.copy2(pdf, out / PREVIEW_NAME)
        print(f"wrote {out / ZIP_NAME}")
        print(f"wrote {out / PREVIEW_NAME}")
        if not cleanup:
            print(f"work directory kept: {work}")
        return 0
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
