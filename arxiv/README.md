# arXiv Submission Package

This folder contains the arXiv-ready source package for:

**PIPE-RDF: Execution-Grounded Generation of Schema-Specific NL--SPARQL Benchmarks**

## Metadata in this version

- Author: Suraj Ranganath
- Affiliation: UC San Diego
- Email: suranganath@ucsd.edu
- Public code link: https://github.com/suraj-ranganath/PIPE-RDF

## Files included

- `acl_latex.tex`
- `references.bib`
- `acl_latex.bbl`
- `acl.sty`
- `acl_natbib.bst`
- `figures/*.png`
- `PIPE-RDF-arxiv-submission.zip`

## Build locally

```bash
cd arxiv
latexmk -pdf -interaction=nonstopmode -halt-on-error acl_latex.tex
```
