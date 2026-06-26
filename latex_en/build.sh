#!/usr/bin/env bash
# Build latex_en/main.tex. Usage: run  ./build.sh  in this directory.
# Dependencies: pdfLaTeX + BibTeX.

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "Error: pdflatex not found. Install MacTeX / TeX Live and retry." >&2
    exit 1
fi

# Copy the figure from docs/ if it is not here yet.
if [ ! -f eight_theorems_toymodel.png ] && [ -f ../docs/eight_theorems_toymodel.png ]; then
    cp ../docs/eight_theorems_toymodel.png .
fi

echo "[1/4] pdflatex pass 1 (generate .aux) ..."
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "[2/4] bibtex (resolve references) ..."
bibtex main >/dev/null || true
echo "[3/4] pdflatex pass 2 (write citations) ..."
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "[4/4] pdflatex pass 3 (resolve cross-references) ..."
pdflatex -interaction=nonstopmode main.tex >/dev/null

echo "Done. Output: $(pwd)/main.pdf"
