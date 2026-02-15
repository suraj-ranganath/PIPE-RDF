# PIPE-RDF (Anonymous Submission Package)

This branch is the anonymized, minimal ACL double-blind submission package for PIPE-RDF.

## Included Files

- `paper_acl2026_industry/acl_latex.tex`
- `paper_acl2026_industry/references.bib`
- `paper_acl2026_industry/acl.sty`
- `paper_acl2026_industry/acl_natbib.bst`
- Referenced figures under `paper_acl2026_industry/figures/`

## Build

From repository root:

```bash
cd paper_acl2026_industry
pdflatex acl_latex.tex
bibtex acl_latex
pdflatex acl_latex.tex
pdflatex acl_latex.tex
```
