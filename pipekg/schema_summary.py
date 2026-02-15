from __future__ import annotations

from typing import Dict, List

from .sparql_client import SparqlClient


def build_schema_summary(client: SparqlClient, top_k: int = 10, include_sample: bool = False) -> str:
    queries = {
        "predicates": f"""
        SELECT ?p (COUNT(*) AS ?c)
        WHERE {{ ?s ?p ?o }}
        GROUP BY ?p
        ORDER BY DESC(?c)
        LIMIT {top_k}
        """,
        "predicates_distinct": f"""
        SELECT DISTINCT ?p
        WHERE {{ ?s ?p ?o }}
        LIMIT {top_k}
        """,
        "types": f"""
        SELECT ?type (COUNT(?s) AS ?c)
        WHERE {{ ?s a ?type }}
        GROUP BY ?type
        ORDER BY DESC(?c)
        LIMIT {top_k}
        """,
        "types_distinct": f"""
        SELECT DISTINCT ?type
        WHERE {{ ?s a ?type }}
        LIMIT {top_k}
        """,
    }

    parts: List[str] = []
    try:
        res = client.query(queries["predicates"])
        preds = [f"{r['p']} ({r['c']})" for r in res.rows if r.get("p")]
        parts.append("Top predicates: " + "; ".join(preds))
    except Exception:
        try:
            res = client.query(queries["predicates_distinct"])
            preds = [r["p"] for r in res.rows if r.get("p")]
            parts.append("Predicates (distinct): " + "; ".join(preds))
        except Exception:
            pass

    try:
        res = client.query(queries["types"])
        types = [f"{r['type']} ({r['c']})" for r in res.rows if r.get("type")]
        parts.append("Top types: " + "; ".join(types))
    except Exception:
        try:
            res = client.query(queries["types_distinct"])
            types = [r["type"] for r in res.rows if r.get("type")]
            parts.append("Types (distinct): " + "; ".join(types))
        except Exception:
            pass

    if include_sample:
        try:
            res = client.query(
                """
            SELECT ?s ?p ?o
            WHERE { ?s ?p ?o }
            LIMIT 10
            """
            )
            triples = [f"{r['s']} {r['p']} {r['o']}" for r in res.rows]
            parts.append("Sample triples: " + " | ".join(triples))
        except Exception:
            pass

    return "\n".join(parts)


def build_schema_whitelist(client: SparqlClient, top_k: int = 25) -> Dict[str, List[str]]:
    predicates_query = f"""
    SELECT ?p (COUNT(*) AS ?c)
    WHERE {{ ?s ?p ?o }}
    GROUP BY ?p
    ORDER BY DESC(?c)
    LIMIT {top_k}
    """
    types_query = f"""
    SELECT ?type (COUNT(?s) AS ?c)
    WHERE {{ ?s a ?type }}
    GROUP BY ?type
    ORDER BY DESC(?c)
    LIMIT {top_k}
    """
    predicates = []
    types = []
    try:
        res = client.query(predicates_query)
        predicates = [r["p"] for r in res.rows if r.get("p")]
    except Exception:
        try:
            res = client.query(
                f"""
                SELECT DISTINCT ?p
                WHERE {{ ?s ?p ?o }}
                LIMIT {top_k}
                """
            )
            predicates = [r["p"] for r in res.rows if r.get("p")]
        except Exception:
            pass
    try:
        res = client.query(types_query)
        types = [r["type"] for r in res.rows if r.get("type")]
    except Exception:
        try:
            res = client.query(
                f"""
                SELECT DISTINCT ?type
                WHERE {{ ?s a ?type }}
                LIMIT {top_k}
                """
            )
            types = [r["type"] for r in res.rows if r.get("type")]
        except Exception:
            pass
    return {"predicates": predicates, "types": types}
