from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


BINARY_FIELDS = [
    "intent_query_match",
    "entity_binding_correct",
    "answer_type_correct",
    "category_construct_correct",
    "overall_pass",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_binary(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"yes", "y", "true", "1", "pass", "passed"}:
        return "yes"
    if raw in {"no", "n", "false", "0", "fail", "failed"}:
        return "no"
    return ""


def cohen_kappa(pairs: Iterable[tuple[str, str]]) -> dict[str, float | int | None]:
    materialized = [(a, b) for a, b in pairs if a and b]
    n = len(materialized)
    if n == 0:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None, "kappa": None}
    agree = sum(1 for a, b in materialized if a == b)
    labels = sorted({value for pair in materialized for value in pair})
    a_counts = Counter(a for a, _ in materialized)
    b_counts = Counter(b for _, b in materialized)
    observed = agree / n
    expected = sum((a_counts[label] / n) * (b_counts[label] / n) for label in labels)
    if expected == 1.0:
        kappa = 1.0 if observed == 1.0 else None
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return {
        "n": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
    }


def by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        audit_id = str(row.get("audit_id", "")).strip()
        if not audit_id:
            raise ValueError("Every annotation row must include audit_id")
        if audit_id in out:
            raise ValueError(f"Duplicate audit_id in annotation file: {audit_id}")
        out[audit_id] = row
    return out


def annotation_value(row: dict[str, str], prefix: str, field: str) -> str:
    return normalize_binary(row.get(f"{prefix}_{field}", row.get(field, "")))


def error_value(row: dict[str, str], prefix: str) -> str:
    return str(row.get(f"{prefix}_error_type", row.get("error_type", ""))).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute semantic-audit agreement and Cohen's kappa.")
    parser.add_argument("--annotator1", required=True, help="CSV with audit_id and annotator1_* columns, or bare annotation columns.")
    parser.add_argument("--annotator2", required=True, help="CSV with audit_id and annotator2_* columns, or bare annotation columns.")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    ann1 = by_id(read_csv(Path(args.annotator1)))
    ann2 = by_id(read_csv(Path(args.annotator2)))
    common_ids = sorted(set(ann1) & set(ann2))
    if not common_ids:
        raise ValueError("No overlapping audit_id values between annotation files")

    missing = {
        "annotator1_only": sorted(set(ann1) - set(ann2)),
        "annotator2_only": sorted(set(ann2) - set(ann1)),
    }
    field_agreement = {}
    disagreements: list[dict[str, str]] = []
    for field in BINARY_FIELDS:
        pairs = []
        for audit_id in common_ids:
            a = annotation_value(ann1[audit_id], "annotator1", field)
            b = annotation_value(ann2[audit_id], "annotator2", field)
            pairs.append((a, b))
            if a and b and a != b:
                disagreements.append({"audit_id": audit_id, "field": field, "annotator1": a, "annotator2": b})
        field_agreement[field] = cohen_kappa(pairs)

    error_pairs = [
        (error_value(ann1[audit_id], "annotator1"), error_value(ann2[audit_id], "annotator2"))
        for audit_id in common_ids
        if normalize_binary(ann1[audit_id].get("annotator1_overall_pass", ann1[audit_id].get("overall_pass", ""))) == "no"
        or normalize_binary(ann2[audit_id].get("annotator2_overall_pass", ann2[audit_id].get("overall_pass", ""))) == "no"
    ]
    error_agreement = cohen_kappa(error_pairs)

    summary = {
        "annotator1": args.annotator1,
        "annotator2": args.annotator2,
        "common_records": len(common_ids),
        "missing": missing,
        "binary_field_agreement": field_agreement,
        "error_type_agreement_on_any_failure": error_agreement,
        "disagreements": disagreements,
    }
    rendered = json.dumps(summary, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
