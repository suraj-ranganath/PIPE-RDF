# GraphDB Setup

PIPE-RDF expects a SPARQL 1.1 endpoint backed by an RDF repository. The experiments in this repo use GraphDB, but the pipeline only needs a query endpoint that can parse and execute the generated SPARQL.

## Local GraphDB

Start GraphDB with Docker:

```bash
docker run -d \
  --name graphdb \
  -p 7200:7200 \
  -e GDB_HEAP_SIZE=8g \
  ontotext/graphdb:latest
```

Open the UI at:

```text
http://localhost:7200
```

Create a repository with inference disabled for reproducible benchmark answers. Common repository names used by the configs are:

- `spb_company_mini_slice`
- `spb_1m`

Set the query endpoint in `.env`:

```bash
SPARQL_ENDPOINT_URL=http://localhost:7200/repositories/spb_1m
```

## Loading Data

Load ontology/reference files first, then instance files. For large SPB data, split generated N-Quads into manageable files before import to avoid upload timeouts.

Useful smoke queries:

```sparql
SELECT (COUNT(*) AS ?triples)
WHERE { ?s ?p ?o }
```

```sparql
SELECT ?p (COUNT(*) AS ?count)
WHERE { ?s ?p ?o }
GROUP BY ?p
ORDER BY DESC(?count)
LIMIT 20
```

```sparql
SELECT ?type (COUNT(*) AS ?count)
WHERE { ?s a ?type }
GROUP BY ?type
ORDER BY DESC(?count)
LIMIT 20
```

## Verifying PIPE-RDF Access

After loading data, run:

```bash
python scripts/verify_endpoint.py
python scripts/preflight_check.py --config configs/smoke_test.yaml
```

## ds-serv6 Operations

For remote runs on `ds-serv6`, use the repository helper instead of manually managing the service:

```bash
bash scripts/ds_serv6_graphdb.sh status
bash scripts/ds_serv6_graphdb.sh health
bash scripts/ds_serv6_graphdb.sh restart
```

If GraphDB becomes slow or unresponsive during a PIPE-RDF run, restart it and rerun health checks before continuing.

## Scaling Notes

- Keep inference disabled unless a specific experiment requires it.
- Avoid `ORDER BY RAND()` on large repositories; PIPE-RDF uses deterministic binding banks plus client-side shuffling for large runs.
- Batch label/type metadata lookups with `VALUES` clauses when adding new templates or scripts.
- Allocate more heap for large SPB runs on servers with enough RAM. The `ds-serv6` helper is configured for larger heap sizes than a laptop should use.
