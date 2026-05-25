from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


METRICS = [
    "parse_valid",
    "exec_success",
    "exact_answer_match",
    "answer_f1",
    "sp_f1",
    "triple_f1",
    "predicate_f1",
    "sketch_similarity",
]


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1.0
    if text in {"false", "no", ""}:
        return 0.0
    return float(text)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: list[float], *, seed: int, draws: int = 5000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(draws):
        samples.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    samples.sort()
    lo = samples[int(0.025 * (draws - 1))]
    hi = samples[int(0.975 * (draws - 1))]
    return (lo, hi)


def sign_test_p_value(values: list[float]) -> float:
    positives = sum(1 for v in values if v > 0)
    negatives = sum(1 for v in values if v < 0)
    n = positives + negatives
    if n == 0:
        return 1.0
    k = min(positives, negatives)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def load_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {"n": len(rows)}
    for metric in METRICS:
        out[metric] = mean([as_float(row, metric) for row in rows])
    return out


def infer_run_id(path: Path) -> str:
    return path.parent.name


def infer_schema_model(run_id: str) -> tuple[str, str]:
    schema = "Schema C" if "schema_c" in run_id else "SPB-full" if "spb" in run_id else run_id
    if "qwen35_2b" in run_id:
        model = "Qwen3.5-2B"
    elif "qwen35_4b" in run_id:
        model = "Qwen3.5-4B"
    elif "qwen35_9b" in run_id:
        model = "Qwen3.5-9B"
    else:
        model = "unknown"
    return schema, model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize downstream utility CSVs by category and paired RAG-vs-zero-shot deltas."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="utility_eval_results.csv files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for input_name in args.inputs:
        path = Path(input_name)
        run_id = infer_run_id(path)
        schema, model = infer_schema_model(run_id)
        for row in load_csv(path):
            row = dict(row)
            row["run_id"] = run_id
            row["schema_name"] = schema
            row["model_name"] = model
            rows.append(row)

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["schema_name"],
                row["model_name"],
                row.get("condition", ""),
                row.get("category", ""),
            )
        ].append(row)

    by_category = []
    for (schema, model, condition, category), items in sorted(grouped.items()):
        summary = summarize_rows(items)
        summary.update(
            {
                "schema": schema,
                "model": model,
                "condition": condition,
                "category": category,
            }
        )
        by_category.append(summary)

    paired: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["schema_name"],
            row["model_name"],
            row.get("category", ""),
            row.get("question", ""),
        )
        paired[key][row.get("condition", "")] = row

    paired_by_schema_model: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    paired_by_category: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for (schema, model, category, question), pair in paired.items():
        if "zero_shot" not in pair or "category_rag" not in pair:
            continue
        zero = pair["zero_shot"]
        rag = pair["category_rag"]
        delta = {
            "schema": schema,
            "model": model,
            "category": category,
            "question": question,
            "answer_f1_delta": as_float(rag, "answer_f1") - as_float(zero, "answer_f1"),
            "sp_f1_delta": as_float(rag, "sp_f1") - as_float(zero, "sp_f1"),
            "exec_success_delta": as_float(rag, "exec_success") - as_float(zero, "exec_success"),
            "parse_valid_delta": as_float(rag, "parse_valid") - as_float(zero, "parse_valid"),
        }
        paired_by_schema_model[(schema, model)].append(delta)
        paired_by_category[(schema, model, category)].append(delta)

    deltas_overall = []
    for (schema, model), items in sorted(paired_by_schema_model.items()):
        values = [float(item["answer_f1_delta"]) for item in items]
        lo, hi = bootstrap_ci(values, seed=args.seed)
        deltas_overall.append(
            {
                "schema": schema,
                "model": model,
                "n_pairs": len(items),
                "answer_f1_delta": mean(values),
                "answer_f1_delta_ci95_low": lo,
                "answer_f1_delta_ci95_high": hi,
                "answer_f1_delta_sign_test_p": sign_test_p_value(values),
                "sp_f1_delta": mean([float(item["sp_f1_delta"]) for item in items]),
                "exec_success_delta": mean([float(item["exec_success_delta"]) for item in items]),
                "parse_valid_delta": mean([float(item["parse_valid_delta"]) for item in items]),
            }
        )

    deltas_category = []
    for (schema, model, category), items in sorted(paired_by_category.items()):
        values = [float(item["answer_f1_delta"]) for item in items]
        lo, hi = bootstrap_ci(values, seed=args.seed, draws=3000)
        deltas_category.append(
            {
                "schema": schema,
                "model": model,
                "category": category,
                "n_pairs": len(items),
                "answer_f1_delta": mean(values),
                "answer_f1_delta_ci95_low": lo,
                "answer_f1_delta_ci95_high": hi,
                "answer_f1_delta_sign_test_p": sign_test_p_value(values),
                "sp_f1_delta": mean([float(item["sp_f1_delta"]) for item in items]),
                "exec_success_delta": mean([float(item["exec_success_delta"]) for item in items]),
                "parse_valid_delta": mean([float(item["parse_valid_delta"]) for item in items]),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "downstream_by_category_summary.json").write_text(
        json.dumps(
            {
                "by_category": by_category,
                "paired_deltas_overall": deltas_overall,
                "paired_deltas_by_category": deltas_category,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with (out_dir / "downstream_by_category.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["schema", "model", "condition", "category", "n", *METRICS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in by_category:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with (out_dir / "downstream_paired_deltas.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "schema",
            "model",
            "category",
            "n_pairs",
            "answer_f1_delta",
            "answer_f1_delta_ci95_low",
            "answer_f1_delta_ci95_high",
            "answer_f1_delta_sign_test_p",
            "sp_f1_delta",
            "exec_success_delta",
            "parse_valid_delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in deltas_category:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(json.dumps({"paired_deltas_overall": deltas_overall}, indent=2))


if __name__ == "__main__":
    main()
