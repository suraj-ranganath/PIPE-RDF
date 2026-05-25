from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_input(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Input must be schema_name=path: {item}")
    name, path = item.split("=", 1)
    return name.strip(), Path(path.strip())


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")


def write_instructions(path: Path, records: int, per_category: int, inputs: list[str]) -> None:
    text = f"""# PIPE-RDF Semantic Audit Instructions

This packet contains {records} stratified examples sampled from:

{chr(10).join(f"- {item}" for item in inputs)}

Sampling target: {per_category} examples per category per schema.

## Task

For each row, judge whether the natural-language question and SPARQL query are semantically aligned. Use the answer preview only as supporting evidence; the primary judgment is whether the query correctly operationalizes the question on the given schema.

## Required Fields

Use `yes` or `no` for each binary field.

- `intent_query_match`: the query asks for the same relation, comparison, count, set operation, or yes/no condition as the question.
- `entity_binding_correct`: concrete entities in the question are correctly bound in the query.
- `answer_type_correct`: the selected output type matches the question, such as entity, boolean, count, date, or literal.
- `category_construct_correct`: the SPARQL construct matches the intended category, such as `ASK` for yes/no, `COUNT` for counting, ordering/limit for superlatives and ordinals, and set logic for intersection or difference.
- `overall_pass`: `yes` only when all required semantic checks pass and no material issue is present.

## Error Types

When `overall_pass=no`, choose one primary `error_type`:

- `intent_mismatch`
- `entity_binding`
- `answer_type`
- `category_construct`
- `syntax_or_execution`
- `unsupported_or_ambiguous_question`
- `other`

Use `notes` for short free-text explanation, especially for borderline cases.
"""
    path.write_text(text, encoding="utf-8")


ANNOTATION_FIELDS = [
    "intent_query_match",
    "entity_binding_correct",
    "answer_type_correct",
    "category_construct_correct",
    "overall_pass",
    "error_type",
    "notes",
]


def annotator_fieldnames(prefix: str) -> list[str]:
    base = [
        "audit_id",
        "schema",
        "category",
        "question",
        "sparql",
        "answers",
        "source_answer_count",
        "source_exec_success",
        "source_parse_valid",
    ]
    return base + [f"{prefix}_{field}" for field in ANNOTATION_FIELDS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified semantic-audit packet.")
    parser.add_argument("--input", action="append", required=True, help="schema_name=benchmark_phase3.jsonl")
    parser.add_argument("--per-category", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/audits/arr_semantic_audit")
    parser.add_argument("--allow-short-category", action="store_true", help="Sample fewer rows if a category is short.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled: list[dict[str, object]] = []
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for raw in args.input:
        schema, path = parse_input(raw)
        rows = load_jsonl(path)
        by_category: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_category.setdefault(str(row.get("category", "unknown")), []).append(row)
        for category in sorted(by_category):
            candidates = list(by_category[category])
            counts[schema][category] = len(candidates)
            if len(candidates) < args.per_category and not args.allow_short_category:
                raise ValueError(
                    f"{schema}:{category} has {len(candidates)} records, "
                    f"below requested per-category sample {args.per_category}"
                )
            rng.shuffle(candidates)
            for local_idx, row in enumerate(candidates[: args.per_category], start=1):
                audit_id = f"{slug(schema)}_{slug(category)}_{local_idx:02d}"
                sampled.append(
                    {
                        "audit_id": audit_id,
                        "schema": schema,
                        "category": category,
                        "question": row.get("question", ""),
                        "sparql": row.get("sparql", ""),
                        "answers": json.dumps(row.get("answers", []), ensure_ascii=False),
                        "source_answer_count": row.get("answer_count", ""),
                        "source_exec_success": row.get("exec_success", ""),
                        "source_parse_valid": row.get("parse_valid", ""),
                        "annotator1_intent_query_match": "",
                        "annotator1_entity_binding_correct": "",
                        "annotator1_answer_type_correct": "",
                        "annotator1_category_construct_correct": "",
                        "annotator1_overall_pass": "",
                        "annotator1_error_type": "",
                        "annotator1_notes": "",
                        "annotator2_intent_query_match": "",
                        "annotator2_entity_binding_correct": "",
                        "annotator2_answer_type_correct": "",
                        "annotator2_category_construct_correct": "",
                        "annotator2_overall_pass": "",
                        "annotator2_error_type": "",
                        "annotator2_notes": "",
                        "adjudicated_intent_query_match": "",
                        "adjudicated_entity_binding_correct": "",
                        "adjudicated_answer_type_correct": "",
                        "adjudicated_category_construct_correct": "",
                        "adjudicated_overall_pass": "",
                        "adjudicated_error_type": "",
                        "adjudication_notes": "",
                    }
                )

    csv_path = out_dir / "semantic_audit_packet.csv"
    jsonl_path = out_dir / "semantic_audit_packet.jsonl"
    fieldnames = list(sampled[0].keys()) if sampled else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for annotator in ("annotator1", "annotator2"):
        template_path = out_dir / f"{annotator}_template.csv"
        fields = annotator_fieldnames(annotator)
        with template_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in sampled:
                writer.writerow({field: row.get(field, "") for field in fields})
    write_instructions(out_dir / "semantic_audit_instructions.md", len(sampled), args.per_category, args.input)

    manifest = {
        "inputs": args.input,
        "per_category": args.per_category,
        "seed": args.seed,
        "records": len(sampled),
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
        "annotator_templates": [
            str(out_dir / "annotator1_template.csv"),
            str(out_dir / "annotator2_template.csv"),
        ],
        "instructions": str(out_dir / "semantic_audit_instructions.md"),
        "available_counts": counts,
        "annotation_schema": {
            "binary_values": "yes | no",
            "binary_fields": ANNOTATION_FIELDS[:-2],
            "error_type": (
                "intent_mismatch | entity_binding | answer_type | category_construct | "
                "syntax_or_execution | unsupported_or_ambiguous_question | other"
            ),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
