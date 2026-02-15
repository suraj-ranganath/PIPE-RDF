# Builder Agent Runbook: Set up LDBC SPB RDF Graph Locally (MacBook Pro M4, 48GB RAM) for PIPE-RDF (1M triples)

## Goal
Create a **queryable RDF graph** (schema + instance data) locally for PIPE-RDF.  
We are **NOT** using SPB’s benchmark workload; we only need:
- Ontology/schema (TTL)
- ~**1,000,000 triples/quads** of instance data
- A local **SPARQL 1.1 endpoint** to execute queries during reverse-querying and evaluation

## Target constraints / limits
- Target instance data: **~1M generated triples/quads** (plus additional ontology/reference triples)
- Disk: minimum **10 GB free**, recommended **20–30 GB free**
- RAM: 48 GB total; allocate **8–16 GB** to the triple store; leave the rest for OS + pipeline
- Run everything locally on macOS (Apple Silicon / arm64). Avoid x86 emulation if possible.

---

## Step 0 — Choose a local triple store
Preferred for minimal friction on Apple Silicon:
- **GraphDB Free** via Docker (arm64-native)

Alternative:
- Virtuoso via Docker
- Fuseki (run as a local Java server)

This runbook assumes **GraphDB via Docker**.

---

## Step 1 — Install prerequisites (macOS)
1. Install Homebrew if needed.
2. Install Java and Ant:
   - Java: Temurin/OpenJDK 17 is recommended.
   - Ant: required to build SPB from source.

Commands:
- `brew install ant`
- `brew install --cask temurin`  (or another OpenJDK 17 distribution)
- Verify:
  - `java -version`  (should be 17.x)
  - `ant -version`

Also install:
- Docker Desktop for Mac (Apple Silicon build)
- Verify: `docker version`

---

## Step 2 — Get SPB (LDBC Semantic Publishing Benchmark v2)
### Option A (preferred): download release
1. Download the latest release zip/tar of **ldbc_spb_bm_2.0** from GitHub.
2. Unpack it.
3. Confirm a `dist/` folder exists OR proceed to Option B.

### Option B: build from source (if no usable `dist/`)
1. Clone repo:
   - `git clone https://github.com/ldbc/ldbc_spb_bm_2.0.git`
   - `cd ldbc_spb_bm_2.0`
2. Build querymix (either one is fine; we mainly need generator artifacts in `dist/`):
   - `ant build-basic-querymix`
3. Confirm output:
   - `dist/` directory exists
   - contains a `semantic_publishing_benchmark*.jar`
   - contains `test.properties` and data directories

**Limitation:** SPB build tooling assumes Java/Ant are correctly installed. If `ant` fails, fix Java/Ant first.

---

## Step 3 — Configure SPB to generate ~1M triples/quads
We want instance data generation output as **N-Quads** for easy loading.

1. Go to:
   - `cd dist/`
2. Edit `test.properties`:
   - Set **datasetSize** to ~1M (this controls generated CreativeWorks data size)
   - Set `generateCreativeWorksFormat` to `NQUADS` (or equivalent enum supported)
   - Set `generatedTriplesPerFile` to something manageable, e.g. `250000`
   - Set `creativeWorksPath` to a local output folder, e.g. `./generated_data`

3. Enable only the phases you need.
   Recommended minimal phases:
   - `loadOntologies` = true (optional; you can also load ontology manually)
   - `loadReferenceDatasets` = true (optional)
   - `generateCreativeWorks` = true
   Disable:
   - `runBenchmark`
   - `warmUp`
   - any vendor-specific benchmark phases

**Limitation:** “1M triples” typically applies to generated instance data; total triples loaded in the store will be a bit higher once you include ontology/reference triples.

---

## Step 4 — Generate the RDF data (N-Quads output)
From `dist/` run:
- `java -Xmx8G -jar semantic_publishing_benchmark-*.jar test.properties`

Expected result:
- `creativeWorksPath` contains one or more `.nq` (N-Quads) files
- Ontology/reference files exist in SPB’s provided `data/` directories (typically TTL)

**Limitation:** If generation is slow, reduce datasetSize further for initial debugging, then return to 1M.

---

## Step 5 — Start GraphDB locally (Docker)
1. Start GraphDB container (example):
- `docker run -d --name graphdb -p 7200:7200 -e GDB_HEAP_SIZE=8g ontotext/graphdb:latest`

Notes:
- Use **8g** heap initially. If imports are slow, raise to 12–16g.
- Keep at least ~16–24 GB RAM free for OS + your pipeline processes.

2. Open:
- http://localhost:7200

**Limitation:** Some GraphDB images/tags may differ in env var naming. If heap variable isn’t honored, use GraphDB’s documented env var or Docker memory limits. Do not exceed ~16g heap on a laptop to avoid OS pressure.

---

## Step 6 — Create a repository in GraphDB
In GraphDB UI:
1. Setup → Repositories → Create new repository
2. Choose a ruleset appropriate for RDF Schema/OWL if needed (RDFS/OWL-Horst), or choose “no inference” for a pure benchmark baseline.
3. Name it, e.g. `spb_1m`

**Recommendation:** Start with **no inference** for controlled evaluation; inference can change answers and complicate benchmark ground truth.

---

## Step 7 — Load ontology + reference + instance data
You need to load three types of data:
1. Ontologies (schema) — TTL/RDF files from SPB `data/ontologies/`
2. Reference datasets (if used) — TTL/RDF from SPB `data/datasets/` (or similar)
3. Generated CreativeWorks — N-Quads from `creativeWorksPath`

Load method (UI):
- Import → Server files (or Upload RDF files)
- Load ontologies first, then reference datasets, then generated N-Quads.

Load method (automated):
- If GraphDB supports REST import in your setup, implement a script that POSTs files to the repository import endpoint.
- Keep logs of import success/failure.

**Limitation:** Importing large N-Quads in one shot may be slow. Use multiple files (controlled by `generatedTriplesPerFile`) to avoid timeouts.

---

## Step 8 — Verify the graph is queryable (smoke tests)
Run these in GraphDB SPARQL view against `spb_1m`:

1) Count triples:
```sparql
SELECT (COUNT(*) AS ?triples) WHERE { ?s ?p ?o }
````

2. Top predicates:

```sparql
SELECT ?p (COUNT(*) AS ?c)
WHERE { ?s ?p ?o }
GROUP BY ?p
ORDER BY DESC(?c)
LIMIT 20
```

3. Top rdf:types:

```sparql
SELECT ?type (COUNT(*) AS ?c)
WHERE { ?s a ?type }
GROUP BY ?type
ORDER BY DESC(?c)
LIMIT 20
```

Record:

* total triple count
* import time
* any query latency observations

---

## Step 9 — Expose endpoints for PIPE-RDF

PIPE-RDF needs at minimum a **SPARQL query endpoint**.

For GraphDB, typically:

* Query endpoint: `http://localhost:7200/repositories/spb_1m`

If PIPE-RDF uses updates (optional):

* Update endpoint: `http://localhost:7200/repositories/spb_1m/statements`

Store these in your pipeline config.

---

## Step 10 — Operational limitations / gotchas (must enforce)

1. **Do not rely on labels** for “proprietary-like” behavior:

   * Prefer using IDs/IRIs in templates and reverse-querying.
2. **Keep inference OFF** unless explicitly needed:

   * Inference changes answer sets and complicates evaluation reproducibility.
3. **Dataset size is approximate**:

   * Generated instance triples ≈ 1M, but total loaded includes schema/reference.
4. **Resource management**:

   * If the store becomes unstable, reduce heap or reduce dataset size.
   * Ensure at least 20 GB free disk to avoid corruption during import.
5. **Architecture**:

   * Use arm64-native Docker images where possible; avoid `--platform linux/amd64` emulation.
6. **Reproducibility**:

   * Version-pin SPB commit/release tag and GraphDB image tag.
   * Save `test.properties` used for generation alongside your benchmark artifacts.

---

## Deliverables to produce (agent output)

* `dist/test.properties` used (archived)
* Generated N-Quads files path
* Triple store choice + version/tag
* Repository name and endpoints
* Smoke-test query results (counts + top predicates/types)
* Resource footprint summary (disk used, heap used)

## ADD-ON: Schema coverage enforcement (required)

### After loading ontology + reference + generated data, compute coverage stats
Run and save outputs (CSV) for these queries:

1) Predicate distribution:
```sparql
SELECT ?p (COUNT(*) AS ?triples)
WHERE { ?s ?p ?o }
GROUP BY ?p
ORDER BY DESC(?triples)
````

2. Class/type distribution:

```sparql
SELECT ?t (COUNT(*) AS ?instances)
WHERE { ?s a ?t }
GROUP BY ?t
ORDER BY DESC(?instances)
```

3. Class→predicate usage (data-driven domain coverage):

```sparql
SELECT ?t ?p (COUNT(*) AS ?c)
WHERE {
  ?s a ?t .
  ?s ?p ?o .
  FILTER(?p != rdf:type)
}
GROUP BY ?t ?p
ORDER BY DESC(?c)
```

### Define acceptance thresholds (fail setup if unmet)

* Total distinct predicates >= 50 (or >= 70 if SPB exposes more at this scale)
* Total distinct rdf:types >= 20 (or >= 30 if SPB exposes more at this scale)
* No single predicate accounts for > 40% of all triples
* Long-tail presence: at least 20 predicates have >= 100 triples AND at least 50 predicates have >= 10 triples
  (If these are impossible at 1M, relax slightly, but keep fixed once chosen.)

Save:

* total triples, distinct predicates, distinct types
* top-20 predicates + counts
* top-20 types + counts

### If coverage fails: enforce diversity without exceeding ~1M total triples

Do a multi-batch union approach:

1. Regenerate data as K independent batches with different seeds/config (K=3–5):

   * each batch datasetSize ≈ 250k–400k triples (so total ≈ 1M)
2. Load all batches into the same repository (or merge N-Quads before load)
3. Re-run coverage stats; repeat once if still failing

### If specific schema elements remain missing/rare: targeted injection (last resort)

* Identify missing/rare predicates/classes from coverage stats
* Create a small SPARQL INSERT script to add 50–200 synthetic instances per missing/rare element
* Insert into a separate named graph `:schema_coverage_injections` so it’s auditable and removable

### Archive coverage artifacts (required)

Store alongside the project:

* coverage CSV outputs
* generation config(s) / seeds / test.properties for each batch
* summary JSON: {total_triples, distinct_predicates, distinct_types, max_predicate_share, long_tail_counts}