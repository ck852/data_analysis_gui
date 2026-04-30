#!/bin/sh
# build-paper.sh
# Builds a generic academic PDF from paper.md + paper.bib
#
# Prerequisites (install once):
#   - Pandoc >= 2.11  (https://pandoc.org/installing.html)
#   - A TeX distribution: TeX Live (Linux/macOS) or MiKTeX (Windows)
#     On Ubuntu/Debian:  sudo apt install pandoc texlive-latex-extra texlive-fonts-recommended texlive-xetex latexmk
#     On macOS (brew):   brew install pandoc && brew install --cask mactex-no-gui
#     On Windows:        choco install pandoc miktex
#
# Usage:
#   chmod +x build-paper.sh
#   ./build-paper.sh
#
# Or with Docker (no local install needed):
#   docker run --rm --entrypoint sh -v "%cd%:/data" -w /data pandoc/extra:latest build-paper.sh

set -eu

# Input files
PAPER="paper.md"
BIB="paper.bib"
LUA_FILTER="scholarly-metadata.lua"
OUTPUT="paper.pdf"

# Verify prerequisites
if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc is not installed. See header of this script for install instructions."
  exit 1
fi

if [ ! -f "$PAPER" ]; then
  echo "ERROR: $PAPER not found in $(pwd)"
  exit 1
fi

if [ ! -f "$BIB" ]; then
  echo "WARNING: $BIB not found. Bibliography will not be rendered."
fi

# Build PDF
echo "Building $OUTPUT ..."

FILTER_ARG=""
if [ -f "$LUA_FILTER" ]; then
  FILTER_ARG="--lua-filter=$LUA_FILTER"
fi

pandoc "$PAPER" \
  --from=markdown+yaml_metadata_block \
  --to=pdf \
  --pdf-engine=xelatex \
  --citeproc \
  --bibliography="$BIB" \
  $FILTER_ARG \
  --number-sections \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V citecolor=blue \
  -V documentclass=article \
  -V header-includes='\usepackage{float}\floatplacement{figure}{H}' \
  -o "$OUTPUT"

echo "Done: $OUTPUT"
