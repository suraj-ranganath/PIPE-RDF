from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipekg.settings import get_settings
from pipekg.sparql_client import SparqlClient


def main() -> None:
    settings = get_settings()
    if not settings.sparql_endpoint_url:
        raise SystemExit("SPARQL_ENDPOINT_URL is not set")

    client = SparqlClient(settings.sparql_endpoint_url)

    queries = {
        "triple_count": "SELECT (COUNT(*) AS ?triples) WHERE { ?s ?p ?o }",
        "top_predicates": """
            SELECT ?p (COUNT(*) AS ?c)
            WHERE { ?s ?p ?o }
            GROUP BY ?p
            ORDER BY DESC(?c)
            LIMIT 20
        """,
        "top_types": """
            SELECT ?type (COUNT(*) AS ?c)
            WHERE { ?s a ?type }
            GROUP BY ?type
            ORDER BY DESC(?c)
            LIMIT 20
        """,
    }

    for name, query in queries.items():
        result = client.query(query)
        print(f"\n[{name}] {result.elapsed_ms:.1f} ms")
        for row in result.rows:
            print(row)


if __name__ == "__main__":
    main()
