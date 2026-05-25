# PIPE-RDF: Execution-Grounded Generation of Schema-Specific NL-SPARQL Benchmarks

This is the anonymous reviewer branch for the ARR submission.

PIPE-RDF builds schema-specific natural-language-to-SPARQL benchmarks for RDF knowledge graphs. It grounds benchmark generation in the target graph through reverse querying, category-balanced templates, retrieval-augmented prompting, deduplication, and parse/execution validation.

This branch contains the current code and configurations needed to reproduce the reviewer-facing experiments. Author-identifying paper metadata, public links, and non-anonymous project notes are intentionally omitted from this branch.

## Abstract

Natural-language access to RDF knowledge graphs depends on evaluation sets whose queries actually run on the graph being tested. Existing KGQA benchmarks provide useful shared tasks, but their schemas, namespaces, predicates, and query distributions often differ from the graphs used in practice. We introduce PIPE-RDF, an execution-grounded workflow for building schema-specific natural-language/SPARQL benchmarks from a target RDF graph. PIPE-RDF starts with reverse queries that populate deterministic binding banks, uses category-aware retrieval to supply schema-matched examples, and applies controlled LLM generation to produce candidate question-query pairs. Each candidate is accepted only after passing predicate and type checks, deduplication, parsing, execution, answer-shape validation, and non-empty-result checks. Across a compact company-location schema and a 25M-triple LDBC Semantic Publishing Benchmark graph, PIPE-RDF produces 3,600 balanced pairs across nine query categories with no parse, execution, or empty-answer failures in the released artifacts. Cross-model probes, a dual LLM-judge audit, and downstream prompting experiments indicate that the resulting benchmarks are robust, semantically aligned, and useful as schema-matched examples for NL-SPARQL evaluation.

## Contents

- `pipekg/`: Core benchmark generation pipeline.
- `configs/`: Smoke, full-run, ARR-scale, cross-model, and top-up configurations.
- `scripts/`: GraphDB, vLLM, generation, audit, utility-evaluation, and summarization scripts.
- `db_setup.md`: Local GraphDB setup notes.
- `AGENTS.md`: Operational instructions for agents running the experiments.

## Experiment Scope

The ARR-scale configuration targets:

- Two RDF schemas: a controlled company-location schema and an LDBC SPB graph.
- Nine categories: generic, counting, comparative, superlative, ordinal, multi-hop, intersection, difference, and yes/no.
- 200 Phase-3 records per category per schema for the main runs.
- Qwen3.5 open models served through vLLM for generation and robustness probes.
- A stratified LLM-judge semantic audit and a downstream utility evaluation.

Generated datasets, logs, model outputs, and local GraphDB data are not committed. They should remain under ignored artifact directories.

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Set the SPARQL endpoint for the active GraphDB repository:

```bash
SPARQL_ENDPOINT_URL=http://localhost:7200/repositories/spb_1m
```

## LLM Providers

For Ollama:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:4b-instruct
OLLAMA_EMBED_MODEL=bge-m3:latest
```

For vLLM or another OpenAI-compatible endpoint:

```bash
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_CHAT_MODEL=Qwen/Qwen3.5-4B
OPENAI_API_KEY=EMPTY
EMBED_PROVIDER=sentence_transformers
LOCAL_EMBED_MODEL=BAAI/bge-m3
```

## GraphDB

For local setup, see `db_setup.md`.

For the remote experiment server, use:

```bash
bash scripts/ds_serv6_graphdb.sh status
bash scripts/ds_serv6_graphdb.sh health
```

If GraphDB or a SPARQL endpoint becomes unresponsive, restart and health-check it before resuming runs:

```bash
bash scripts/ds_serv6_graphdb.sh restart
bash scripts/ds_serv6_graphdb.sh health
```

## Preflight

Run preflight checks before full experiments:

```bash
python scripts/preflight_check.py --config configs/smoke_test.yaml
python scripts/verify_endpoint.py
```

## Generation Runs

Smoke run:

```bash
python scripts/run_pipeline_ollama.py --config configs/smoke_test.yaml
```

Main Schema C run:

```bash
python scripts/run_pipeline_ollama.py --config configs/arr_schema_c_200.yaml
```

Main SPB run:

```bash
python scripts/run_pipeline_ollama.py --config configs/arr_spb_full_200.yaml
```

Cross-model probes:

```bash
python scripts/run_pipeline_ollama.py --config configs/arr_cross_model_schema_c_50.yaml
python scripts/run_pipeline_ollama.py --config configs/arr_cross_model_spb_full_50.yaml
```

## Evaluation Utilities

Summarize benchmark artifacts:

```bash
python scripts/summarize_benchmark_artifact.py --input path/to/phase3.jsonl
```

Sample semantic-audit records:

```bash
python scripts/sample_semantic_audit.py --input path/to/phase3.jsonl --output audit_packet.csv
```

Run the dual LLM semantic judges:

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

## Reviewer Notes

- This branch is intended for anonymous review.
- Do not commit API keys, generated logs, model outputs, or GraphDB data.
- Use `main` only for the public, non-anonymous code release after review.
