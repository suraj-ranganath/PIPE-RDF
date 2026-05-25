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


def ask_answer(row: dict[str, object]) -> bool | None:
    if not is_ask_query(row):
        return None
    answers = row.get("answers") or []
    if not isinstance(answers, list) or not answers:
        return None
    value = str(answers[0]).strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    return None


def has_ordinal_construct(row: dict[str, object]) -> bool:
    query = str(row.get("sparql", "")).upper()
    return "ORDER BY" in query and "OFFSET" in query and "LIMIT 1" in query


def same_different_shape_issue(row: dict[str, object]) -> str:
    """Catch lexical same/different intent mismatches for boolean comparisons.

    This is intentionally conservative.  It does not flag entity-identity
    filters such as ?company1 != ?company2, which are correct when two distinct
    entities share the same industry/location/person/year.  It only flags
    cases where the question asks for sameness but the query compares the
    compared value variables with inequality, or vice versa.
    """
    category = str(row.get("category", "")).lower()
    if category not in {"comparative", "difference"}:
        return ""
    question = str(row.get("question", "")).lower()
    sparql = re.sub(r"\s+", " ", str(row.get("sparql", "")).lower())
    value_pairs = {
        "industry": (("same industry", "different industries", "different industry"), ("industry1", "industry2")),
        "location": (("same location", "different locations", "different location"), ("loc1", "loc2", "location1", "location2")),
        "year": (("same year", "same founding year", "different years", "different founding years"), ("year1", "year2")),
        "employees": (("same employee count", "same number of employees", "different employee count"), ("count1", "count2", "employees1", "employees2")),
        "person": (("same key person", "share a key person", "different key person"), ("person1", "person2")),
    }
    for label, (markers, vars_) in value_pairs.items():
        same_markers = [m for m in markers if m.startswith("same") or m.startswith("share")]
        different_markers = [m for m in markers if m.startswith("different")]
        value_neq = any(f"?{vars_[i]} != ?{vars_[j]}" in sparql for i in range(len(vars_)) for j in range(len(vars_)) if i != j)
        value_eq = any(f"?{vars_[i]} = ?{vars_[j]}" in sparql for i in range(len(vars_)) for j in range(len(vars_)) if i != j)
        if any(marker in question for marker in same_markers) and value_neq:
            return f"same_{label}_question_uses_value_inequality"
        if any(marker in question for marker in different_markers) and not (value_neq or "not exists" in sparql or "minus" in sparql):
            return f"different_{label}_question_without_value_difference"
        if category == "difference" and any(marker in question for marker in same_markers) and not value_eq:
            return f"difference_category_same_{label}_wording"
    return ""


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
    ask_rows = [row for row in rows if is_ask_query(row)]
    ask_values = Counter()
    ask_by_category: dict[str, Counter[str]] = {}
    for row in ask_rows:
        category = str(row.get("category", "unknown"))
        value = ask_answer(row)
        label = "true" if value is True else "false" if value is False else "unknown"
        ask_values[label] += 1
        ask_by_category.setdefault(category, Counter())[label] += 1
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
        "ask_answers": {
            "total": len(ask_rows),
            "true": ask_values.get("true", 0),
            "false": ask_values.get("false", 0),
            "unknown": ask_values.get("unknown", 0),
            "false_rate": ask_values.get("false", 0) / len(ask_rows) if ask_rows else 0.0,
            "by_category": {
                category: dict(sorted(counter.items()))
                for category, counter in sorted(ask_by_category.items())
            },
        },
        "ordinal_without_ranked_offset": sum(
            str(row.get("category")) == "ordinal" and not has_ordinal_construct(row)
            for row in rows
        ),
        "same_different_shape_issues": sum(bool(same_different_shape_issue(row)) for row in rows),
        "same_different_shape_issue_types": dict(
            sorted(Counter(issue for row in rows if (issue := same_different_shape_issue(row))).items())
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
