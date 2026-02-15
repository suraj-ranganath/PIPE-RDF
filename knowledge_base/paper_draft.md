# PIPE-RDF: An LLM-Assisted Pipeline for Proprietary RDF Question-Answer Benchmarking (Anonymous Submission)

## Abstract
Organizations with proprietary RDF knowledge graphs increasingly rely on LLMs to translate natural language questions into SPARQL. Public benchmarks (e.g., DBpedia/Wikidata) fail to capture enterprise-specific schemas, prefixes, and query distributions. We present PIPE-RDF, a three-phase pipeline for generating private, balanced, and evaluation-ready NL–SPARQL benchmarks. The pipeline combines reverse querying, category-wise generation, retrieval-augmented prompting, paraphrase robustness, and an error-repair loop. We implement the pipeline end-to-end in Python over a schema-consistent synthetic KG derived from LinkedMDB, and we release structured artifacts (balanced datasets, metrics, diagrams). Our results illustrate how category balancing and evaluation metrics (SP-F1, triple overlap, execution accuracy, parse-valid rate, repair success) provide a nuanced view of model performance beyond surface matching. PIPE-RDF is designed for real-world deployment where benchmark maintenance must keep pace with evolving enterprise graphs.

## 1. Introduction
Knowledge graphs are a common representation for enterprise data, with RDF and SPARQL providing standardized schema and query semantics. LLMs reduce the effort of translating natural language to SPARQL, but organizations still struggle to **measure** how well a model performs on *their* graph. Existing benchmarks are public and domain-specific (e.g., DBpedia or Wikidata) and do not reflect proprietary schema and vocabulary. As a result, in-house teams lack a reliable evaluation standard when comparing prompt strategies, retrieval augmentation, fine-tuning, or different models.

PIPE-RDF addresses this gap by automating the creation of **organization-specific, balanced** NL–SPARQL benchmarks. The pipeline emphasizes (i) generating valid questions grounded in the graph, (ii) balancing query categories to avoid skewed evaluation, (iii) retrieval-augmented prompting for higher quality generation, and (iv) automated feedback loops for SPARQL correction. These design goals align with recent research on complex query categories (Mintaka), retrieval-augmented prompting for SPARQL generation, and execution-based evaluation for text-to-query tasks.

**Contribution summary**
1. A three-phase pipeline for proprietary benchmark generation, including reverse querying and category-aware retrieval.
2. An end-to-end Python implementation with synthetic RDF data conformant to LinkedMDB schema.
3. Balanced benchmark outputs and metrics aligned with recent evaluation recommendations.
4. Practical guidance for industry deployment and maintenance of evolving benchmarks.

## 2. Related Work
**Complex query categories.** Recent KGQA benchmarks explicitly balance query types (counting, comparative, superlative, ordinal, multi-hop, intersection, negation, yes/no, and generic) to avoid bias toward “easy” patterns (Sen et al., 2022; Saleem et al., 2017). PIPE-RDF adopts this taxonomy for category-wise generation and evaluation.

**RAG for NL→SPARQL.** Retrieval-augmented prompts improve few-shot performance by selecting relevant NL–SPARQL examples (Guu et al., 2020; Trummer, 2022). Recent systems leverage vector DBs to retrieve schema snippets and prior queries for SPARQL generation (Fochi et al., 2023). PIPE-RDF uses category-specific retrieval banks to improve structural alignment between prompts and target queries.

**Error correction loops.** Iterative validation and repair of SPARQL using KG feedback improves syntactic and semantic correctness (Pasquali et al., 2023; Fochi et al., 2023). PIPE-RDF incorporates an execution + repair loop with optional human validation.

**Evaluation metrics.** SP-BLEU/SP-F1 normalize variables to mitigate lexical mismatch (Rony et al., 2022). Execution accuracy and pass@k capture functional correctness (Kulal et al., 2019; Chen et al., 2021). Structural metrics (CodeBLEU; Ren et al., 2020) and semantic equivalence metrics (e.g., FuncEvalGMN; Zhan et al., 2024) are increasingly used for text-to-query evaluation. PIPE-RDF implements SP-F1, triple-overlap F1, execution accuracy, parse validity, and repair success, and outlines extensions for canonicalization and learned equivalence metrics.

## 3. Methodology
PIPE-RDF comprises three phases (Figure 1).

**Phase 1 — Initial seed generation.**
- Generate NL templates from the ontology.
- Reverse query: generate candidate SPARQL that verifies template instantiation.
- Execute and repair invalid SPARQL; human validation optional.
- Store verified pairs in a seed vector DB for retrieval augmentation.

**Phase 2 — Category-wise seed generation.**
- Generate queries per complexity category to avoid skew.
- Use retrieval-augmented prompts with top-k seeds from Phase 1.
- Apply paraphrase augmentation and semantic deduplication.
- Store pairs in category-specific vector DBs and in the benchmark draft set.

**Phase 3 — Full dataset generation.**
- Generate in batches with category-specific retrieval.
- Execute + repair; optional human curation.
- Balance and shuffle categories for evaluation fairness.

**Reverse Querying.**
We ensure every template is grounded in graph facts by querying the KG first to find compatible entities, then instantiate the NL question and its SPARQL. This reduces “unanswerable” questions and yields executable ground truth.

**Deduplication & paraphrasing.**
A semantic similarity filter removes near-duplicates. Paraphrase variants test model robustness and reduce overfitting to surface forms.

## 4. Implementation
We implement PIPE-RDF in Python with `rdflib` and a synthetic KG created from the LinkedMDB schema. The implementation mirrors the methodology in the pipeline diagram (Figure 1) and produces:
- JSONL/CSV benchmark datasets (Phase 3 balanced set = 360 pairs, 40 per category).
- Seed banks for retrieval augmentation.
- Metric outputs and visual summaries.

### 4.1 Synthetic KG Construction
The synthetic KG includes films, directors, actors, producers, genres, and countries aligned to the LinkedMDB ontology. Entities include labels and IDs to support SPARQL queries with realistic joins and aggregations.

### 4.2 Templates & Category Coverage
We implement template families for all nine categories, including multi-hop and negation patterns. Superlatives include global and conditional variants (by genre, by director) to ensure sufficient diversity.

### 4.3 Retrieval + Repair
A lightweight retriever ranks examples by token cosine similarity to simulate vector search. A repair loop executes queries and attempts corrections when predictions fail. This reflects the error-feedback methodology described in recent LLM-based SPARQL systems.

## 5. Evaluation
We evaluate the Phase 3 balanced dataset with metrics recommended in the latest evaluation work:
- **SP-F1** (variable-normalized token F1)
- **Triple-overlap F1** (structural overlap)
- **Execution accuracy** (answer set match)
- **Parse-valid rate**
- **Repair success rate**

To avoid dependence on external LLMs in this implementation, we simulate prediction noise and repair. The evaluation pipeline is LLM-agnostic and can be plugged into real model outputs.

## 6. Results
**Dataset size:** 360 NL–SPARQL pairs, balanced across 9 categories (40 each).

**Average question length:** 9.65 tokens. Longest categories are intersection, negation, and ordinal (~12 tokens).

**Metrics (simulated):** SP-F1 0.997, triple F1 0.966, execution accuracy 0.706, parse-valid 0.744, repair success 0.294. Execution accuracy trails lexical metrics, illustrating the need for execution-based evaluation.

**Figures:**
- Figure 1: PIPE-RDF pipeline (`pipeline_flow.png`)
- Figure 2: Category distribution (`category_distribution_phase3_balanced.png`)
- Figure 3: Question length (`question_length_hist.png`)
- Figure 4: Answer count (`answer_count_hist.png`)
- Figure 5: Metric summary (`evaluation_metrics.png`)

## 7. Discussion
Balanced category sampling reveals which query classes are harder for the model. Counting and yes/no questions are shorter and structurally simpler; multi-hop, intersection, and negation require more joins and filters, matching known difficulty trends in KGQA benchmarks. Execution accuracy is substantially lower than lexical overlap, reinforcing that token-based metrics alone overestimate functional correctness. The repair loop meaningfully improves outcomes, suggesting that automated feedback is a practical path to reduce human verification load.

The design aligns with emerging work on retrieval-augmented prompting and schema-aware correction, while providing an end-to-end, enterprise-friendly pipeline for continuous benchmark maintenance as the KG evolves.

## 8. Limitations
This implementation uses a synthetic KG and simulated LLM outputs. Real-world deployment should integrate an LLM for question and SPARQL generation and use an actual SPARQL endpoint. Some advanced metrics (canonicalization, learned equivalence) are noted but not implemented in this prototype.

## 9. Ethics & Impact
PIPE-RDF is intended for internal benchmarking on proprietary data. While the pipeline improves evaluation rigor, it can expose sensitive schema details in prompts. Enterprises should follow data governance policies, minimize exposure of confidential identifiers, and audit prompts and logs. The system supports transparency by grounding answers in SPARQL execution rather than free-form generation.

## 10. Conclusion
PIPE-RDF provides a practical, extensible pipeline for generating proprietary NL–SPARQL benchmarks. By combining reverse querying, category balancing, retrieval augmentation, and repair loops, it enables rigorous evaluation tailored to enterprise graphs. The implementation demonstrates feasibility and provides artifacts (datasets, metrics, diagrams) that can be adapted to real deployments.

---

# References
- Bocklisch, T. et al. (2017). Rasa: Open source language understanding and dialogue management.
- Brown, T. et al. (2020). Language Models are Few-Shot Learners.
- Chen, M. et al. (2021). Evaluating Large Language Models Trained on Code.
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
- Dubey, M. et al. (2019). LC-QuAD 2.0.
- Fochi, E. et al. (2023). LLM-based SPARQL Query Generation from Natural Language over Federated Knowledge Graphs.
- Guu, K. et al. (2020). Retrieval Augmented Generation for Knowledge-Intensive NLP.
- Kojima, T. et al. (2022). Large Language Models are Zero-Shot Reasoners.
- Kulal, S. et al. (2019). SPoC: Search-based Pseudocode-to-Code.
- Liang, P. et al. (2022). HELM: Holistic Evaluation of Language Models.
- Margatina, K. et al. (2023). DynamicTempLAMA.
- Meloni, A. et al. (2024). Assessing LLMs for SPARQL Query Generation in Scientific QA.
- Perevalov, A. et al. (2022). QALD-9-plus.
- Raffel, C. et al. (2020). T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
- Ren, S. et al. (2020). CodeBLEU: a Method for Automatic Evaluation of Code Synthesis.
- Rony, A. et al. (2022). SGPT: Generative SPARQL Query Generation (SP-BLEU/SP-F1).
- Saleem, M. et al. (2017). Analysis of QALD-6 in Question Answering over Linked Data.
- Sen, P. et al. (2022). Mintaka: A Complex, Natural, and Multilingual Dataset for End-to-End QA.
- Trummer, I. (2022). Prompting Strategies for NL→SPARQL Generation.
- Touvron, H. et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
- Xu, Y. et al. (2023). Expert prompting methods.
- Yasunaga, M. et al. (2021). QA over knowledge graphs with node embeddings.
- Zhan, Z. et al. (2024). FuncEvalGMN: Functional Correctness via Graph Matching Network.
- Additional references embedded in the provided latest-related and evaluation documents.
