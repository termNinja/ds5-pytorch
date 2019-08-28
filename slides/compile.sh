#! /usr/bin/env bash
# pandoc -t beamer pytorch-intro.md  --pdf-engine=xelatex -V theme:metropolis -o pytorch-intro.pdf
pandoc \
    -t beamer pytorch-intro.md \
    -V theme:metropolis \
    -o pytorch-intro.pdf  \
    --filter pandoc-citeproc \
    --bibliography=literature.bib