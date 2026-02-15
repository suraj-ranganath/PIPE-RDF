# PIPE-RDF

PIPE-RDF is an LLM-assisted pipeline for building schema-specific NL-SPARQL benchmarks for enterprise RDF graphs.

## Abstract

Enterprises rely on RDF knowledge graphs and SPARQL to expose operational data through natural language interfaces, yet public KGQA benchmarks do not reflect proprietary schemas, prefixes, or query distributions. PIPE-RDF is a three-phase pipeline that constructs schema-specific NL-SPARQL benchmarks using reverse querying, category-balanced template generation, retrieval-augmented prompting, deduplication, and execution-based validation with repair. In our fixed-schema company-location slice, PIPE-RDF produces a balanced benchmark across nine categories and logs operational metrics to support model evaluation and deployment planning.

## Branch Purpose

- `main`: Public/arXiv/community-facing branch (code + docs + non-anonymous paper source).
- `pre-submission`: Snapshot of the working project before submission packaging (large artifacts ignored).
- `paper-ready`: Minimal anonymous branch for double-blind ACL submission assets only.

## Repository Layout

- `pipekg/`: Core pipeline modules.
- `scripts/`: End-to-end, smoke test, indexing, and utility scripts.
- `configs/`: Run configurations (smoke and full runs).
- `paper_acl2026_industry/`: Paper source and figures.
- `db_setup.md`: Local GraphDB setup instructions.

## Setup

1. Create and activate a Python environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.
```bash
pip install -r requirements.txt
```

3. Configure environment variables.
```bash
cp .env.example .env
```

4. Start GraphDB and load data.
- Follow `db_setup.md` for Docker setup and repository loading.
- Set `SPARQL_ENDPOINT_URL` in `.env` (example: `http://localhost:7200/repositories/spb_company_mini_slice`).

5. Ensure local LLM backends are available (if using Ollama defaults).
```bash
ollama pull qwen3:4b-instruct
ollama pull bge-m3:latest
```

## Verify Environment

Run preflight checks:

```bash
python scripts/preflight_check.py --config configs/smoke_test.yaml
```

Quick endpoint check:

```bash
python scripts/verify_endpoint.py
```

## Run PIPE-RDF

Smoke run:

```bash
python scripts/run_pipeline_ollama.py --config configs/smoke_test.yaml
```

Full run:

```bash
python scripts/run_pipeline_ollama.py --config configs/full_run.yaml
```

## Paper

- Public paper source: `paper_acl2026_industry/acl_latex.tex`
- References: `paper_acl2026_industry/references.bib`
