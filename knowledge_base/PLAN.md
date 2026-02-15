# PLAN

## Goal
Create a **company-location mini-slice** (Schema C) of the SPB graph (5,000 companies) into a new GraphDB repo (`spb_company_mini_slice`), then adapt the existing PIPE-KG pipeline to generate a **balanced question–SPARQL benchmark** over that fixed schema, aligned with the PIPE-RDF methodology and the latest related/evaluation notes.

## Inputs to Use
- `PIPE-RDF.txt` (methodology: 3 phases, reverse querying, RAG, repair loop, human-in-the-loop, category balance)
- `Latest_related_work.txt` (category taxonomy, RAG, error correction)
- `Latest_evaluation_work.txt` (evaluation metrics and logging extensions)
- `methodology.png` (pipeline flow to mirror in diagrams)
- `db_setup.md` (GraphDB setup + endpoints)
- Existing pipeline code under `pipekg/` and `scripts/`

## Plan of Work
1. **Review current implementation + constraints**
   - Read `pipekg/pipeline_ollama.py`, `scripts/run_pipeline_ollama.py`, and `pipekg/prompts.py` to confirm the 3-phase flow and reverse-querying behavior.
   - Note current schema discovery and whitelist logic (top-k predicates/types) and identify where to override with a **fixed Schema C**.

2. **Define Schema C (company-location mini-slice)**
   - Confirm available predicates/types in `spb_1m` with targeted SPARQL probes:
     - `dbo:Company` count, sample `dbo:location`, `dbo:industry`, `dbo:keyPerson`, `dbo:foundingYear`, `dbo:numberOfEmployees`
     - `gn:Feature`/location labels + `geo:lat`/`geo:long`
     - `foaf:Person` labels/names for key people
   - Lock the schema to a **small, explicit predicate/type list** to avoid out-of-schema LLM generations.

3. **Generate the 5k company mini-slice**
   - Use `scripts/create_company_mini_slice.py` to extract:
     - Company core facts + labels + Schema C predicates
     - Linked locations/industries/key people + minimal labels
   - Output TTL to `artifacts/slices/spb_company_mini_slice.ttl`.
   - Verify TTL size and sample triples (spot-check format).

4. **Load mini-slice into GraphDB**
   - Create (or re-create) repo `spb_company_mini_slice` if needed.
   - Load TTL via `scripts/load_graphdb.py` into the new repo.
   - Validate with quick SPARQL queries (company count, sample predicates).

5. **Adapt pipeline to fixed Schema C**
   - Add config options to **override**:
     - SPARQL endpoint (mini-slice repo)
     - Schema summary (static text with allowed predicates/types)
     - Predicate/type whitelist (explicit Schema C list)
     - Slot type hints (company/location/person/industry)
   - Ensure prompts **only** use Schema C predicates and avoid DBpedia-style hallucinations.
   - Keep reverse queries simple (no ORDER BY/GROUP BY/aggregates) and **LIMIT 5**.
   - Keep paraphrasing disabled for now (as requested).

6. **Iterative validation (feedback loop)**
   - After each pipeline adjustment, run **small SPARQL probes** on the mini-slice to verify:
     - Predicates exist and return rows
     - Templates are instantiable
     - Reverse queries produce bindings for all slots
   - Use the logs to identify failure modes and tighten constraints.

7. **Smoke test + stabilization**
   - Run `smoke_test.yaml` against `spb_company_mini_slice`.
   - Confirm Phase 1 and Phase 2 produce **non-empty seeds**.
   - Confirm SPARQL parse/exec success is > 0 and results are non-empty.

8. **Prepare for full experiment**
   - Provide stable run instructions and expected runtime for a full run on the mini-slice.
   - Ensure artifacts and logs are saved per run for later analysis and paper figures.

## Deliverables
- `artifacts/slices/spb_company_mini_slice.ttl` (mini-slice data)
- GraphDB repo `spb_company_mini_slice`
- Updated pipeline config + schema overrides for Schema C
- Successful smoke test run with non-empty logs and seeds

## Validation Checklist
- Mini-slice repo responds to SPARQL queries without timeouts
- Schema summary/whitelist matches Schema C predicates/types
- Phase 1 and Phase 2 produce seed pairs (non-empty)
- Reverse queries return valid bindings and simple SPARQL
