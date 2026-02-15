# arXiv Submission Package

This folder contains the arXiv-ready source package for:

**PIPE-RDF: An LLM-Assisted Pipeline for Enterprise RDF Benchmarking**

## Metadata in this version

- Author: Suraj Ranganath
- Affiliation: UC San Deigo
- Email: suranganath@ucsd
- Abstract code link: https://github.com/suraj-ranganath/PIPE-RDF

## Files included

- `acl_latex.tex`
- `references.bib`
- `acl.sty`
- `acl_natbib.bst`
- `figures/*.png`

## Build locally

```bash
cd arxiv
pdflatex acl_latex.tex
bibtex acl_latex
pdflatex acl_latex.tex
pdflatex acl_latex.tex
```
