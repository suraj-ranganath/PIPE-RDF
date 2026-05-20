# PIPE-RDF

PIPE-RDF builds schema-specific natural-language-to-SPARQL benchmarks for RDF knowledge graphs. It generates executable question/query pairs by grounding templates in graph bindings, balancing query categories, retrieving category-relevant examples, deduplicating outputs, and validating every SPARQL query through parse and execution checks.

The current public snapshot supports the ARR-scale revision of:

> PIPE-RDF: Execution-Grounded Generation of Schema-Specific NL-SPARQL Benchmarks

## What This Repository Contains

- A three-phase benchmark generation pipeline for NL-SPARQL data.
- Reverse-query grounding so generated questions are answerable on the target graph.
- Category-balanced generation across nine KGQA-style categories: generic, counting, comparative, superlative, ordinal, multi-hop, intersection, difference, and yes/no.
- GraphDB/SPARQL execution validation with repair and run manifests.
- Open model serving through Ollama or OpenAI-compatible endpoints such as vLLM.
- Local sentence-transformer embeddings with `BAAI/bge-m3`.
- Utilities for schema profiling, binding-bank construction, benchmark summarization, downstream utility evaluation, and dual LLM semantic judging.

## Current Evaluation Scope

The ARR-scale configuration targets two RDF schemas and 3,600 Phase-3 benchmark records:

- Schema C company-location slice: 9 categories x 200 records.
- LDBC SPB graph: 9 categories x 200 records.
- Primary generator: `Qwen/Qwen3.5-4B` through vLLM.
- Robustness probes: `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, and `Qwen/Qwen3.5-9B`.
- Semantic quality check: a 216-record stratified audit judged by two strong LLM judges.
- Downstream utility check: schema-only zero-shot vs category-RAG prompting.

Generated datasets, logs, GraphDB data, and model outputs are intentionally ignored by Git. Commit only code, configs, documentation, selected paper-ready figures, and reproducible summaries.

## Repository Layout

- `pipekg/`: Core pipeline modules.
- `configs/`: Smoke, full-run, ARR-scale, and top-up run configurations.
- `scripts/`: Experiment, GraphDB, vLLM, audit, utility, and summarization scripts.
- `paper_acl2026_industry/`: Legacy public paper source from the earlier ACL Industry submission.
- `knowledge_base/`: Legacy planning and related-work notes.
- `db_setup.md`: Local GraphDB setup notes.
- `AGENTS.md`: Operational instructions for Codex agents working in this repository.

## Branches

- `main`: Public branch for released code, public-facing documentation, and community use.
- `paper-ready`: Anonymous reviewer branch. It should contain the current code/configs but no author-identifying paper metadata or public links.
- `arr-revision`: Working branch for ARR paper text, submission packaging, revision notes, and arXiv mirroring.
- `pre-submission`: Legacy arXiv/pre-submission snapshot.
- `feat/arr-eval-scaling`: Development branch for ARR-scale code changes before they are promoted to `main`.

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

Configure GraphDB:

```bash
SPARQL_ENDPOINT_URL=http://localhost:7200/repositories/spb_1m
```

For local GraphDB setup, use `db_setup.md`. On `ds-serv6`, use:

```bash
bash scripts/ds_serv6_graphdb.sh status
bash scripts/ds_serv6_graphdb.sh health
```

If GraphDB or a SPARQL endpoint becomes unresponsive during a run, restart and health-check the service before continuing:

```bash
bash scripts/ds_serv6_graphdb.sh restart
bash scripts/ds_serv6_graphdb.sh health
```

## LLM Providers

PIPE-RDF supports the original Ollama path and OpenAI-compatible endpoints.

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

Example vLLM setup for `ds-serv6`:

```bash
bash scripts/ds_serv6_setup_vllm.sh
python scripts/vllm_smoke.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-4B
```

## Preflight Checks

Run a config-level preflight before launching an experiment:

```bash
python scripts/preflight_check.py --config configs/smoke_test.yaml
```

Verify the configured endpoint:

```bash
python scripts/verify_endpoint.py
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

On `ds-serv6`, the helper script coordinates GraphDB, vLLM endpoints, and tmux-backed jobs:

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

Sample a semantic audit packet:

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

Run the downstream utility evaluation:

```bash
python scripts/evaluate_downstream_utility.py --help
```

## Reproducibility Notes

- Keep `.env` local and never commit API keys or GraphDB credentials.
- Keep generated artifacts under ignored directories such as `artifacts/`, `experiments/`, or `results/`.
- Record the exact config, model, endpoint, GraphDB repository, and run manifest for each reported experiment.
- For anonymous review, use the `paper-ready` branch rather than `main`.
