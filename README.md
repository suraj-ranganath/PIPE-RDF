# PIPE-RDF

## Paper Title

**PIPE-RDF: An LLM-Assisted Pipeline for Enterprise RDF Benchmarking**

## Abstract

Enterprises rely on RDF knowledge graphs and SPARQL to expose operational data through natural language interfaces, yet public KGQA benchmarks do not reflect proprietary schemas, prefixes, or query distributions. PIPE-RDF is a three-phase pipeline that constructs schema-specific NL-SPARQL benchmarks using reverse querying, category-balanced template generation, retrieval-augmented prompting, deduplication, and execution-based validation with repair. The pipeline is designed to produce evaluation-ready benchmark artifacts and operational metrics for real-world deployment planning.

## Repository Overview

- `pipekg/`: Core pipeline implementation.
- `scripts/`: Execution scripts for setup checks, generation, and utilities.
- `configs/`: Smoke/full run configuration files.
- `db_setup.md`: Local triple-store setup guide.

## Setup

1. Create and activate a virtual environment.

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

4. Update `.env` with your SPARQL endpoint and model settings.

- `SPARQL_ENDPOINT_URL` should point to your running endpoint.
- If using OpenAI, set `OPENAI_API_KEY`.
- If using Ollama, ensure local models are available.

5. Prepare and verify your RDF endpoint.

```bash
python scripts/verify_endpoint.py
```

## Run

Smoke run:

```bash
python scripts/run_pipeline_ollama.py --config configs/smoke_test.yaml
```

Full run:

```bash
python scripts/run_pipeline_ollama.py --config configs/full_run.yaml
```

## Preflight Check

Run the environment preflight before full experiments:

```bash
python scripts/preflight_check.py --config configs/smoke_test.yaml
```
