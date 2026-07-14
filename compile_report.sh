#!/bin/bash
# Build the paper. The eLife version is the source of truth; the bioRxiv version is
# ported from it by Report/biorxiv/port_from_elife.py.
#
#   ./compile_report.sh          # eLife (default)
#   ./compile_report.sh biorxiv  # bioRxiv port
#
# eLife needs XeLaTeX; the bioRxiv template needs pdfLaTeX.
set -e

case "${1:-elife}" in
  elife)
    cd project5_symmetry/Report/elife/
    xelatex -interaction=nonstopmode main_best.tex
    bibtex main_best
    xelatex -interaction=nonstopmode main_best.tex
    xelatex -interaction=nonstopmode main_best.tex
    ;;
  biorxiv)
    cd project5_symmetry/Report/biorxiv/
    pdflatex -interaction=nonstopmode main.tex
    bibtex main
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
    ;;
  *)
    echo "usage: $0 [elife|biorxiv]" >&2
    exit 1
    ;;
esac
