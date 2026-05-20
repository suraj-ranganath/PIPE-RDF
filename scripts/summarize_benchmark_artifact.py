from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PREDICATE_RE = re.compile(r"\b(?:rdf|rdfs|foaf|dbo|dc|dcterms|spb|gn|geo):[A-Za-z_][\w-]*\b")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if "^^" in text:
        text = text.split("^^", 1)[0].strip('"')
    try:
        return float(text)
    except ValueError:
        return None


def row_has_zero_count(row: dict[str, object]) -> bool:
    if str(row.get("category", "")).lower() != "counting":
        return False
    answers = row.get("answers") or []
    if not isinstance(answers, list) or not answers:
        return False
    return any(as_number(answer) == 0 for answer in answers)


def is_ask_query(row: dict[str, object]) -> bool:
    query = str(row.get("sparql", "")).upper()
    return "ASK" in query and "SELECT" not in query


def has_ordinal_construct(row: dict[str, object]) -> bool:
    query = str(row.get("sparql", "")).upper()
    return "ORDER BY" in query and "OFFSET" in query and "LIMIT 1" in query


def predicates_for_query(sparql: str) -> list[str]:
    predicates: set[str] = set()
    for line in sparql.splitlines():
        stripped = line.strip().rstrip(".").strip()
        upper = stripped.upper()
        if not stripped or upper.startswith(("PREFIX", "SELECT", "ASK", "WHERE", "VALUES", "FILTER", "ORDER", "LIMIT", "OPTIONAL")):
            continue
        if stripped in {"{", "}"}:
            continue
        parts = stripped.split()
        if stripped.startswith(";") and parts:
            candidate = parts[0].lstrip(";")
        elif len(parts) >= 3 and (parts[0].startswith("?") or parts[0].startswith("<")):
            candidate = parts[1]
        else:
            continue
        if candidate == "a":
            candidate = "rdf:type"
        if PREDICATE_RE.fullmatch(candidate):
            predicates.add(candidate)
    return sorted(predicates)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    categories = Counter(str(row.get("category", "unknown")) for row in rows)
    predicates = Counter()
    for row in rows:
        predicates.update(predicates_for_query(str(row.get("sparql", ""))))
    total = len(rows)
    summary = {
        "total": total,
        "categories": dict(sorted(categories.items())),
        "min_category_count": min(categories.values()) if categories else 0,
        "max_category_count": max(categories.values()) if categories else 0,
        "balanced": len(set(categories.values())) <= 1 if categories else False,
        "parse_fail": sum(not bool(row.get("parse_valid", False)) for row in rows),
        "exec_fail": sum(not bool(row.get("exec_success", False)) for row in rows),
        "empty_answers": sum(int(row.get("answer_count", len(row.get("answers") or [])) or 0) == 0 for row in rows),
        "error_type_nonempty": sum(bool(row.get("error_type")) for row in rows),
        "repairs": sum(int(row.get("repair_attempts", 0) or 0) > 0 for row in rows),
        "zero_count_records": sum(row_has_zero_count(row) for row in rows),
        "ask_nonboolean_records": sum(
            is_ask_query(row) and str(row.get("category")) not in {"yesno", "comparative", "difference"}
            for row in rows
        ),
        "ordinal_without_ranked_offset": sum(
            str(row.get("category")) == "ordinal" and not has_ordinal_construct(row)
            for row in rows
        ),
        "predicate_inventory": dict(sorted(predicates.items())),
        "latency_ms": {
            "llm_total": sum(float(row.get("llm_latency_ms", 0.0) or 0.0) for row in rows),
            "sparql_total": sum(float(row.get("sparql_exec_ms", 0.0) or 0.0) for row in rows),
            "question_total": sum(float(row.get("question_latency_ms", 0.0) or 0.0) for row in rows),
        },
    }
    if total:
        summary["latency_ms"]["llm_mean"] = summary["latency_ms"]["llm_total"] / total
        summary["latency_ms"]["sparql_mean"] = summary["latency_ms"]["sparql_total"] / total
        summary["latency_ms"]["question_mean"] = summary["latency_ms"]["question_total"] / total
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PIPE-RDF benchmark JSONL artifact.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    summary = summarize(load_jsonl(input_path))
    summary["input_jsonl"] = str(input_path)
    rendered = json.dumps(summary, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
