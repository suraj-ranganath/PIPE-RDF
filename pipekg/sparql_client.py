from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import time

from SPARQLWrapper import SPARQLWrapper, JSON, POST


@dataclass
class QueryResult:
    rows: List[Dict[str, str]]
    boolean: bool | None
    elapsed_ms: float


class SparqlClient:
    def __init__(self, endpoint_url: str, timeout: int = 60, params: Optional[Dict[str, str]] = None) -> None:
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.params = params or {}

    def query(self, sparql: str) -> QueryResult:
        client = SPARQLWrapper(self.endpoint_url)
        client.setReturnFormat(JSON)
        client.setMethod(POST)
        client.setTimeout(self.timeout)
        for key, value in self.params.items():
            client.addCustomParameter(key, value)
        client.setQuery(sparql)
        start = time.time()
        try:
            data = client.query().convert()
        except Exception as exc:
            raise RuntimeError(f"SPARQL query failed: {exc}")
        elapsed = (time.time() - start) * 1000

        if "boolean" in data:
            return QueryResult(rows=[], boolean=bool(data["boolean"]), elapsed_ms=elapsed)

        rows = []
        for binding in data.get("results", {}).get("bindings", []):
            row = {k: v.get("value", "") for k, v in binding.items()}
            rows.append(row)
        return QueryResult(rows=rows, boolean=None, elapsed_ms=elapsed)
