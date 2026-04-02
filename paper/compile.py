#!/usr/bin/env python3
"""
compile.py — Build all ELAPSE journal variants to PDF.

Usage:
    python compile.py              # compile all
    python compile.py AMM          # compile one target
    python compile.py AMM CNSNS    # compile selected targets

Source:  paper/tex/
Output:  paper/pdf/
Build:   paper/build/
"""

import subprocess, sys, os, shutil
from pathlib import Path

HERE     = Path(__file__).parent.resolve()
TEX_DIR  = HERE / "tex"
PDF_DIR  = HERE / "pdf"
BUILD_DIR = HERE / "build"
PDF_DIR.mkdir(exist_ok=True)
BUILD_DIR.mkdir(exist_ok=True)

TARGETS = {
    "AMM":      "ELAPSE_AMM.tex",
    "CNSNS":    "ELAPSE_CNSNS.tex",
    "PhysicaA": "ELAPSE_PhysicaA.tex",
    "CSF":      "ELAPSE_CSF.tex",
    "IEEE":     "ELAPSE_ieee.tex",
    "changes":  "changes_summary.tex",
}

BUILD_EXTS = {".aux", ".log", ".bbl", ".blg", ".out", ".spl", ".toc", ".lof"}

def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True)
    out = (r.stdout + r.stderr).decode("latin-1")
    return r.returncode == 0, out

def compile_target(name, tex_file):
    stem = Path(tex_file).stem
    print(f"[{name}] Compiling...", end=" ", flush=True)
    run(["pdflatex", "-interaction=nonstopmode", tex_file], TEX_DIR)
    run(["bibtex", stem], TEX_DIR)
    run(["pdflatex", "-interaction=nonstopmode", tex_file], TEX_DIR)
    ok, log = run(["pdflatex", "-interaction=nonstopmode", tex_file], TEX_DIR)

    pdf_src = TEX_DIR / f"{stem}.pdf"
    if pdf_src.exists():
        shutil.move(str(pdf_src), str(PDF_DIR / f"{stem}.pdf"))
        print(f"OK  →  pdf/{stem}.pdf")
    else:
        print("FAILED")
        for l in log.splitlines()[-30:]:
            if "error" in l.lower() or l.startswith("!"):
                print(f"  {l}")
        return False

    for ext in BUILD_EXTS:
        f = TEX_DIR / f"{stem}{ext}"
        if f.exists():
            shutil.move(str(f), str(BUILD_DIR / f"{stem}{ext}"))
    return True

if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(TARGETS.keys())
    unknown = [t for t in requested if t not in TARGETS]
    if unknown:
        print(f"Unknown targets: {unknown}. Valid: {list(TARGETS.keys())}")
        sys.exit(1)
    results = {n: compile_target(n, TARGETS[n]) for n in requested}
    print("\n--- Summary ---")
    for n, ok in results.items():
        print(f"  {n:12s}  {'OK' if ok else 'FAILED'}")
