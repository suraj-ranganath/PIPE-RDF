from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipekg.runtime import apply_run_config
from pipekg.schema_summary import build_schema_summary
from pipekg.settings import get_settings
from pipekg.sparql_client import SparqlClient


def load_run_config(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Run config must be a YAML mapping")
    return data


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def query_rows(client: SparqlClient, query: str) -> list[dict[str, str]]:
    return client.query(query).rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile an RDF/SPARQL endpoint for ARR scaling experiments.")
    parser.add_argument("--config", required=True, help="Run config with sparql_endpoint_url and schema options.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to artifacts/schema_profiles/<config-stem>.")
    parser.add_argument("--top-k", type=int, default=250)
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    settings = apply_run_config(get_settings(), cfg)
    if not settings.sparql_endpoint_url:
        raise SystemExit("SPARQL endpoint is not configured.")

    sparql_params = {"infer": "false"} if not cfg.get("sparql_infer", False) else {}
    timeout = int(cfg.get("schema_timeout_sec", 120))
    client = SparqlClient(settings.sparql_endpoint_url, timeout=timeout, params=sparql_params)
    out_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/schema_profiles") / Path(args.config).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = query_rows(client, "SELECT (COUNT(*) AS ?triples) WHERE { ?s ?p ?o }")
    distinct_pred_rows = query_rows(client, "SELECT (COUNT(DISTINCT ?p) AS ?predicates) WHERE { ?s ?p ?o }")
    distinct_type_rows = query_rows(client, "SELECT (COUNT(DISTINCT ?t) AS ?types) WHERE { ?s a ?t }")

    predicates = query_rows(
        client,
        f"""
SELECT ?p (COUNT(*) AS ?triples)
WHERE {{ ?s ?p ?o }}
GROUP BY ?p
ORDER BY DESC(?triples)
LIMIT {args.top_k}
""",
    )
    types = query_rows(
        client,
        f"""
SELECT ?t (COUNT(*) AS ?instances)
WHERE {{ ?s a ?t }}
GROUP BY ?t
ORDER BY DESC(?instances)
LIMIT {args.top_k}
""",
    )
    class_predicates = query_rows(
        client,
        f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?t ?p (COUNT(*) AS ?triples)
WHERE {{
  ?s a ?t .
  ?s ?p ?o .
  FILTER(?p != rdf:type)
}}
GROUP BY ?t ?p
ORDER BY DESC(?triples)
LIMIT {args.top_k * 5}
""",
    )

    write_rows(out_dir / "predicates.csv", ["p", "triples"], predicates)
    write_rows(out_dir / "types.csv", ["t", "instances"], types)
    write_rows(out_dir / "class_predicates.csv", ["t", "p", "triples"], class_predicates)

    total_triples = int(total_rows[0].get("triples", 0)) if total_rows else 0
    predicate_count = int(distinct_pred_rows[0].get("predicates", 0)) if distinct_pred_rows else 0
    type_count = int(distinct_type_rows[0].get("types", 0)) if distinct_type_rows else 0
    top_predicate_share = 0.0
    if total_triples and predicates:
        top_predicate_share = int(predicates[0].get("triples", 0)) / total_triples

    summary = {
        "config": args.config,
        "endpoint": settings.sparql_endpoint_url,
        "total_triples": total_triples,
        "distinct_predicates": predicate_count,
        "distinct_types": type_count,
        "top_predicate_share": top_predicate_share,
        "long_tail_predicates_ge_100": sum(1 for r in predicates if int(r.get("triples", 0)) >= 100),
        "long_tail_predicates_ge_10": sum(1 for r in predicates if int(r.get("triples", 0)) >= 10),
        "artifacts": {
            "predicates_csv": str(out_dir / "predicates.csv"),
            "types_csv": str(out_dir / "types.csv"),
            "class_predicates_csv": str(out_dir / "class_predicates.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    schema_summary = build_schema_summary(client, top_k=min(args.top_k, 120), include_sample=True)
    (out_dir / "schema_summary.txt").write_text(schema_summary, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
