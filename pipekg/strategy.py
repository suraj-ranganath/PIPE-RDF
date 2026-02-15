from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Set


DEFAULT_STRATEGY_ORDER = [
    "JOIN",
    "FILTER",
    "COUNT",
    "ORDER",
    "NEGATION",
    "ASK",
    "RAG",
]

DEFAULT_ERROR_BUCKET_ORDER = [
    "parse_error",
    "endpoint_error",
    "empty_result",
    "answer_type_mismatch",
    "exec_failure",
    "other_error",
]


def _extract_body(sparql: str) -> str:
    if "{" not in sparql or "}" not in sparql:
        return sparql
    return sparql.split("{", 1)[1].rsplit("}", 1)[0]


def _triple_lines(body: str) -> List[str]:
    lines = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith(("FILTER", "OPTIONAL", "VALUES", "BIND", "UNION", "MINUS", "#")):
            continue
        if line in ("{", "}"):
            continue
        # Keep likely triple pattern lines.
        if " " in line:
            lines.append(line)
    return lines


def _variable_counter(text: str) -> Counter:
    vars_found = re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", text)
    return Counter(vars_found)


def infer_query_strategies(
    sparql: str,
    retrieved_examples: Iterable[Dict[str, object]] | None = None,
) -> List[str]:
    if not sparql:
        return []
    upper = re.sub(r"\s+", " ", sparql.upper()).strip()
    body = _extract_body(sparql)
    triples = _triple_lines(body)
    var_counts = _variable_counter("\n".join(triples))

    tags: Set[str] = set()

    if re.search(r"\bASK\b", upper):
        tags.add("ASK")
    if "FILTER" in upper:
        tags.add("FILTER")
    if "COUNT(" in upper:
        tags.add("COUNT")
    if "ORDER BY" in upper:
        tags.add("ORDER")
    if (
        "NOT EXISTS" in upper
        or "MINUS" in upper
        or "FILTER(!" in upper
        or "FILTER ( !" in upper
        or "FILTER(!BOUND" in upper
    ):
        tags.add("NEGATION")
    if len(triples) >= 2:
        # JOIN if at least one variable appears in multiple triple patterns.
        if any(c >= 2 for c in var_counts.values()):
            tags.add("JOIN")
    if retrieved_examples:
        tags.add("RAG")

    ordered = [s for s in DEFAULT_STRATEGY_ORDER if s in tags]
    leftovers = sorted(tags.difference(DEFAULT_STRATEGY_ORDER))
    return ordered + leftovers


def classify_error_bucket(record: Dict[str, object]) -> str:
    err_type = (record.get("error_type") or "").strip()
    err_msg = (record.get("error") or "").strip().lower()
    parse_valid = bool(record.get("parse_valid", False))
    exec_success = bool(record.get("exec_success", False))
    answer_count = int(record.get("answer_count", 0) or 0)

    if err_type in DEFAULT_ERROR_BUCKET_ORDER:
        return err_type
    if not parse_valid:
        return "parse_error"
    if exec_success and answer_count == 0:
        return "empty_result"
    if not exec_success:
        if "timeout" in err_msg or "connection" in err_msg or "endpoint" in err_msg:
            return "endpoint_error"
        return "exec_failure"
    if err_type:
        return "other_error"
    return "none"
