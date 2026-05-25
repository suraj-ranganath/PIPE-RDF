from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
TRIPLE_RE = re.compile(r"^\s*(?!PREFIX\b|FILTER\b|BIND\b|VALUES\b|OPTIONAL\b|UNION\b|SELECT\b|ASK\b|WHERE\b|ORDER\b|LIMIT\b|OFFSET\b|GROUP\b|HAVING\b)([^#{}]+)\.\s*$", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def distinct_ngram_ratio(token_lists: list[list[str]], n: int) -> float:
    total = 0
    seen: set[tuple[str, ...]] = set()
    for toks in token_lists:
        grams = [tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))]
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else 0.0


def structural_features(sparql: str) -> dict[str, object]:
    upper = sparql.upper()
    triple_patterns = 0
    property_path = False
    for line in sparql.splitlines():
        stripped = line.strip()
        if stripped.startswith("?") or re.match(r"^(?:[A-Za-z_][\w-]*:|<)", stripped):
            if TRIPLE_RE.match(stripped):
                triple_patterns += 1
                parts = stripped.rstrip(".").split()
                if len(parts) >= 3:
                    pred = parts[1]
                    if pred == "a":
                        pred = ""
                    if pred.startswith("<") and ">" in pred:
                        pred = pred[pred.index(">") + 1 :]
                    property_path = property_path or any(marker in pred for marker in ("/", "|", "+", "*", "?"))
    return {
        "triple_patterns": triple_patterns,
        "has_filter": "FILTER" in upper,
        "has_count": "COUNT" in upper,
        "has_order": "ORDER BY" in upper,
        "has_ask": re.search(r"\bASK\b", upper) is not None,
        "has_distinct": "DISTINCT" in upper,
        "has_values": "VALUES" in upper,
        "has_optional": "OPTIONAL" in upper,
        "has_union": "UNION" in upper,
        "has_not_exists": "NOT EXISTS" in upper,
        "has_named_graph": re.search(r"\bGRAPH\s+[\?<]", upper) is not None,
        "has_property_path": property_path,
    }


def summarize_rows(schema: str, category: str, rows: list[dict[str, object]]) -> dict[str, object]:
    question_tokens = [tokens(str(row.get("question", ""))) for row in rows]
    features = [structural_features(str(row.get("sparql", ""))) for row in rows]
    n = len(rows)
    denom = n or 1
    return {
        "schema": schema,
        "category": category,
        "n": n,
        "unique_question_pct": 100.0 * len({str(row.get("question", "")) for row in rows}) / denom,
        "mean_question_tokens": mean([len(toks) for toks in question_tokens]) if question_tokens else 0.0,
        "distinct_1": distinct_ngram_ratio(question_tokens, 1),
        "distinct_2": distinct_ngram_ratio(question_tokens, 2),
        "mean_ast_nodes": mean([float(row.get("ast_node_count") or 0) for row in rows]) if rows else 0.0,
        "mean_ast_depth": mean([float(row.get("ast_max_depth") or 0) for row in rows]) if rows else 0.0,
        "mean_triple_patterns": mean([int(feat["triple_patterns"]) for feat in features]) if features else 0.0,
        "values_pct": 100.0 * sum(bool(feat["has_values"]) for feat in features) / denom,
        "filter_pct": 100.0 * sum(bool(feat["has_filter"]) for feat in features) / denom,
        "count_pct": 100.0 * sum(bool(feat["has_count"]) for feat in features) / denom,
        "order_pct": 100.0 * sum(bool(feat["has_order"]) for feat in features) / denom,
        "ask_pct": 100.0 * sum(bool(feat["has_ask"]) for feat in features) / denom,
        "distinct_pct": 100.0 * sum(bool(feat["has_distinct"]) for feat in features) / denom,
        "optional_pct": 100.0 * sum(bool(feat["has_optional"]) for feat in features) / denom,
        "union_pct": 100.0 * sum(bool(feat["has_union"]) for feat in features) / denom,
        "not_exists_pct": 100.0 * sum(bool(feat["has_not_exists"]) for feat in features) / denom,
        "named_graph_pct": 100.0 * sum(bool(feat["has_named_graph"]) for feat in features) / denom,
        "property_path_pct": 100.0 * sum(bool(feat["has_property_path"]) for feat in features) / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NL diversity and SPARQL structure for accepted benchmark JSONL files.")
    parser.add_argument("--input", action="append", required=True, help="schema_name:path/to/benchmark.jsonl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    schema_rows: dict[str, list[dict[str, object]]] = {}
    for spec in args.input:
        if ":" not in spec:
            raise SystemExit(f"Input must be schema:path, got {spec}")
        schema, path_text = spec.split(":", 1)
        rows = load_jsonl(Path(path_text))
        schema_rows[schema] = rows
        for row in rows:
            row = dict(row)
            row["_schema"] = schema
            all_rows.append(row)

    summaries: list[dict[str, object]] = []
    for schema, rows in schema_rows.items():
        summaries.append(summarize_rows(schema, "ALL", rows))
        by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_category[str(row.get("category", "unknown"))].append(row)
        for category in sorted(by_category):
            summaries.append(summarize_rows(schema, category, by_category[category]))

    combined_by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        combined_by_category[str(row.get("category", "unknown"))].append(row)
    for category in sorted(combined_by_category):
        summaries.append(summarize_rows("combined", category, combined_by_category[category]))

    csv_path = out_dir / "linguistic_structural_diagnostics.csv"
    json_path = out_dir / "linguistic_structural_diagnostics.json"
    fieldnames = list(summaries[0].keys()) if summaries else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
