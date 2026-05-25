# ARR Submission Form Helper

Use this as a copy/paste aid while filling OpenReview. Keep the ARR paper PDF itself anonymous.

## Paper Metadata

Title:

PIPE-RDF: Execution-Grounded Generation of Schema-Specific NL-SPARQL Benchmarks

Paper type:

Long paper

Recommended area:

Resources and Evaluation

Alternate areas/keywords:

Question Answering; Semantics; Information Extraction; Knowledge Graphs; Text-to-SPARQL; Benchmark Construction

Abstract:

Natural-language access to RDF knowledge graphs depends on evaluation sets whose queries actually run on the graph being tested. Existing KGQA benchmarks provide useful shared tasks, but their schemas, namespaces, predicates, and query distributions often differ from the graphs used in practice. We introduce PIPE-RDF, an execution-grounded workflow for building schema-specific natural-language/SPARQL benchmarks from a target RDF graph. PIPE-RDF starts with reverse queries that populate deterministic binding banks, uses category-aware retrieval to supply schema-matched examples, and applies controlled LLM generation to produce candidate question-query pairs. Each candidate is accepted only after passing predicate and type checks, deduplication, parsing, execution, answer-shape validation, and non-empty-result checks. Across a compact company-location schema and a 25M-triple LDBC Semantic Publishing Benchmark graph, PIPE-RDF produces 3,600 balanced pairs across nine query categories with no parse, execution, or empty-answer failures in the released artifacts. Cross-model probes, a dual LLM-judge audit, and downstream prompting experiments indicate that the resulting benchmarks are robust, semantically aligned, and useful as schema-matched examples for NL-SPARQL evaluation.

## Resubmission Fields

Previous submission:

ACL 2026 Industry Track submission 527, "PIPE-RDF: An LLM-Assisted Pipeline for Enterprise RDF Benchmarking".

Previous URL:

Fill with the previous OpenReview forum URL if the ACL 2026 Industry Track submission has one. I could not infer the OpenReview URL from the local files.

Previous PDF:

Upload the previous submission PDF if the ARR form requires it. The old arXiv PDF is not a substitute if OpenReview specifically asks for the previous reviewed submission PDF.

Response PDF:

`arr_submission/PIPE_RDF_ARR_resubmission_notes.pdf`

Reviewer/editor reassignment request:

Leave blank unless you want new reviewers. The prepared response notes say the prior reviews were useful and constructive.

Existing non-anonymous preprints:

`https://arxiv.org/abs/2602.18497`

Preferred venue:

Use the current ACL 2026 / ARR venue acronym shown in OpenReview, if available.

## Upload Files

Paper PDF:

`arr_submission/PIPE_RDF_ARR_review.pdf`

Response PDF:

`arr_submission/PIPE_RDF_ARR_resubmission_notes.pdf`

Optional anonymous source/software archive:

`arr_submission/PIPE_RDF_ARR_anonymous_source.zip`

## Responsible NLP Checklist Notes

Use prose answers rather than bare yes/no where the form asks for details.

- Data: the evaluated graphs are public RDF data/benchmark graphs; the paper discusses public-source licensing and private-graph data governance in Ethical Considerations.
- Code/artifacts: the review PDF uses an anonymous code link; final public code can be linked after review.
- Models: generation uses Qwen3.5-2B/4B/9B via vLLM; semantic judging uses GPT-5-mini and Grok-4.20-0309 non-reasoning; embeddings use BAAI/bge-m3.
- Human subjects: none.
- Annotation: semantic validation is performed by two LLM judges with the full prompt included in the appendix.
- Validation diagnostics: the paper reports strict accepted-artifact checks, natural-language diversity, SPARQL structural distributions, and same/different lexical-shape checks.
- Compute: experiments ran on GPU resources with GraphDB 10.6.3; the paper reports runtime/latency diagnostics.
- Use of generative AI: LLMs are part of the studied method and evaluation, and their roles are explicitly documented in the paper.
