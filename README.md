# PIPE-RDF

PIPE-RDF builds schema-specific natural-language-to-SPARQL benchmarks for RDF knowledge graphs. It grounds generation in the target graph, balances query categories, validates every SPARQL query by parsing and execution, and writes reproducible run artifacts for benchmark construction and downstream NL-to-SPARQL evaluation.

This branch is the public code branch. Paper drafts, submission packages, and anonymous-review source files are intentionally not kept here.

## Features

- Reverse-query grounding so generated questions are answerable on the target graph.
- Category-balanced generation across generic, counting, comparative, superlative, ordinal, multi-hop, intersection, difference, and yes/no query types.
- GraphDB/SPARQL validation with strict parse, execution, answer-shape, and category-form checks.
- Binding-bank sampling and batched label/type lookup to avoid repeated expensive SPARQL calls during large runs.
- LLM backends through Ollama or OpenAI-compatible endpoints such as vLLM.
- Local embeddings through `sentence-transformers` with `BAAI/bge-m3`.
- Utilities for schema profiling, benchmark summarization, semantic LLM judging, and downstream utility evaluation.

## Repository Layout

- `pipekg/`: Core pipeline modules.
- `configs/`: Smoke, full-run, ARR-scale, cross-model, and top-up run configurations.
- `scripts/`: GraphDB, vLLM, generation, audit, evaluation, and summarization scripts.
- `db_setup.md`: GraphDB setup and loading notes.
- `AGENTS.md`: Operational instructions for Codex agents working in this repository.

Generated datasets, logs, GraphDB data, local model outputs, and paper/submission workspaces are ignored by Git.

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment file:

```bash
cp .env.example .env
```

Set the SPARQL endpoint for your GraphDB repository:

```bash
SPARQL_ENDPOINT_URL=http://localhost:7200/repositories/spb_1m
```

For GraphDB setup and data loading, see `db_setup.md`.

## LLM Providers

For Ollama:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:4b-instruct
OLLAMA_EMBED_MODEL=bge-m3:latest
```

For vLLM or another OpenAI-compatible server:

```bash
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_CHAT_MODEL=Qwen/Qwen3.5-4B
OPENAI_API_KEY=EMPTY
EMBED_PROVIDER=sentence_transformers
LOCAL_EMBED_MODEL=BAAI/bge-m3
```

Smoke-test an OpenAI-compatible endpoint:

```bash
python scripts/vllm_smoke.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-4B
```

## GraphDB Operations

On `ds-serv6`, the helper script manages GraphDB lifecycle and health checks:

```bash
bash scripts/ds_serv6_graphdb.sh status
bash scripts/ds_serv6_graphdb.sh health
bash scripts/ds_serv6_graphdb.sh restart
```

For local development, start GraphDB using the instructions in `db_setup.md`, then verify the configured endpoint:

```bash
python scripts/verify_endpoint.py
```

## Preflight Checks

Run a config-level preflight before launching an experiment:

```bash
python scripts/preflight_check.py --config configs/smoke_test.yaml
```

## Running PIPE-RDF

Smoke run:

```bash
python scripts/run_pipeline_ollama.py --config configs/smoke_test.yaml
```

ARR-scale Schema C run:

```bash
python scripts/run_pipeline_ollama.py --config configs/arr_schema_c_200.yaml
```

ARR-scale SPB run:

```bash
python scripts/run_pipeline_ollama.py --config configs/arr_spb_full_200.yaml
```

Cross-model probes:

```bash
python scripts/run_pipeline_ollama.py --config configs/arr_cross_model_schema_c_50.yaml
python scripts/run_pipeline_ollama.py --config configs/arr_cross_model_spb_full_50.yaml
```

On `ds-serv6`, use tmux-backed helper scripts for longer runs:

```bash
bash scripts/ds_serv6_run_arr_experiments.sh
```

## Analysis Utilities

Profile a schema:

```bash
python scripts/profile_schema.py --config configs/arr_spb_full_200.yaml
```

Summarize a benchmark artifact:

```bash
python scripts/summarize_benchmark_artifact.py --input path/to/phase3.jsonl
```

Sample a semantic-audit packet:

```bash
python scripts/sample_semantic_audit.py --input path/to/phase3.jsonl --output audit_packet.csv
```

Run dual LLM semantic judges:

```bash
python scripts/evaluate_semantic_llm_judges.py \
  --input audit_packet.csv \
  --output-dir artifacts/llm_semantic_audit/run_name \
  --judge openai \
  --judge xai
```

Run downstream utility evaluation:

```bash
python scripts/evaluate_downstream_utility.py --help
```

## Reproducibility

- Keep `.env` local and never commit API keys or GraphDB credentials.
- Keep generated artifacts under ignored directories such as `artifacts/`, `experiments/`, or `results/`.
- Record the exact config, model, endpoint, GraphDB repository, and run manifest for each reported experiment.
- Keep paper drafts and submission packages outside `main`; use dedicated paper/review branches for those.
