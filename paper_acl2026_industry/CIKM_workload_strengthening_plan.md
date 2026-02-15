# PIPE-RDF Strengthening Plan (Inspired by CIKM 2024)

Reference paper:
- Agarwal et al., *Generating Cross-model Analytics Workloads Using LLMs* (CIKM 2024), DOI: https://doi.org/10.1145/3627673.3679932
- PDF mirror used for review: https://adalabucsd.github.io/publication/cikm24-autoquery/cikm24-autoquery.pdf

## What to take away immediately

1. Separate **generation quality** from **model quality** via cross-model runs on the exact same workload generator.
2. Report both **query string accuracy** and **execution accuracy** (the paper shows these can diverge).
3. Add **strategy-level coverage and failure analysis** (operator/logic strategy buckets, not only category buckets).
4. Treat workload generation as a two-stage pipeline:
   - reverse-query grounding
   - NL-question generation over grounded templates

## Experiments to run next

## E1. Cross-model robustness (highest priority)
- Goal: show pipeline reliability independent of one LLM.
- Run PIPE-RDF generation with at least 3 models (small, medium, frontier/local-vs-api).
- Report per-category and per-strategy:
  - parse validity
  - execution success
  - repair rate
  - empty-result rate
  - generation latency/cost

## E2. Strategy-level error taxonomy
- Extend current logs with explicit strategy tags per query:
  - join-heavy
  - aggregation
  - order/rank
  - negation
  - boolean ASK
- Build confusion/error table by strategy:
  - syntax error
  - schema violation
  - semantic mismatch
  - empty due to sparsity

## E3. Semantic-vs-execution validity audit
- Keep execution validation, but add larger semantic audit:
  - increase from 45 to >=100 sampled pairs
  - two annotators
  - report agreement + disagreement types
- This addresses the same risk highlighted in CIKM: executable query does not always mean semantically correct.

## E4. Query rewriting and normalization ablation
- Compare:
  - no rewrite
  - lightweight normalization
  - full repair+rewrite
- Measure impact on execution and semantic correctness per strategy.

## E5. Retrieval ablations for workload realism
- Compare top-k in {0, 1, 2, 4} and with/without retrieval leakage filtering.
- Report effect on diversity (templates/entities), and on strategy failure rates.

## Reporting upgrades for the paper

1. Add a dedicated table: **Cross-model generation robustness**.
2. Add a figure: **Strategy coverage + strategy failure heatmap**.
3. Add one subsection: **Execution-valid but semantically wrong cases** with 3-5 concrete examples.
4. Add a compact **cost/quality Pareto plot** across models.
5. Keep figure narrative explicit: schema graph + strategy matrix + two-stage generation flow.

## Implementation mapping in this repo

- New figure generator (already added):
  - `scripts/generate_reference_style_figures.py`
- Cross-model runner (implemented):
  - `scripts/run_cross_model_experiments.py`
  - Example:
    - `python scripts/run_cross_model_experiments.py --config configs/smoke_test.yaml --models qwen3:4b-instruct llama3.1:8b-instruct`
- Automatic strategy tagging + error heatmaps (implemented):
  - `scripts/generate_strategy_analysis.py`
  - Auto-invoked at the end of `scripts/run_pipeline_ollama.py`
  - Phase-3-only analysis command:
    - `python scripts/generate_strategy_analysis.py --run-id 20260201_223633_platinum_run --input-jsonl artifacts/runs/20260201_223633_platinum_run/data/benchmark_phase3.jsonl --label phase3`
- Paper figure assets (already regenerated):
  - `paper_acl2026_industry/figures/pipeline_architecture.png`
  - `paper_acl2026_industry/figures/schema_ontology.png`
- Existing metrics script to extend for E2/E4/E5:
  - `scripts/generate_figures_from_logs.py`
- Paper text locations already updated for CIKM citation and framing:
  - `paper_acl2026_industry/acl_latex.tex`
  - `paper_acl2026_industry/references.bib`
