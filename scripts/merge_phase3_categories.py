import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CSV_FIELDS = [
    "id",
    "category",
    "question",
    "sparql",
    "answers",
    "exec_success",
    "parse_valid",
    "error",
    "error_type",
    "repair_attempts",
    "llm_latency_ms",
    "question_latency_ms",
    "sparql_exec_ms",
    "answer_count",
    "result_hash",
]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for idx, record in enumerate(records, start=1):
            row = {field: record.get(field) for field in CSV_FIELDS}
            row["id"] = idx
            row["answers"] = json.dumps(record.get("answers", []), ensure_ascii=False)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace selected Phase-3 categories from a top-up run.")
    parser.add_argument("--base", required=True, type=Path, help="Base benchmark_phase3.jsonl")
    parser.add_argument("--replacement", required=True, type=Path, help="Replacement benchmark_phase3.jsonl")
    parser.add_argument("--categories", required=True, help="Comma-separated categories to replace")
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    categories = {item.strip() for item in args.categories.split(",") if item.strip()}
    if not categories:
        raise SystemExit("No categories provided")

    base_records = read_jsonl(args.base)
    replacement_records = [r for r in read_jsonl(args.replacement) if r.get("category") in categories]
    replacement_counts = Counter(r.get("category") for r in replacement_records)
    missing = [cat for cat in sorted(categories) if replacement_counts.get(cat, 0) == 0]
    if missing:
        raise SystemExit(f"Replacement file has no records for: {', '.join(missing)}")

    merged = [r for r in base_records if r.get("category") not in categories]
    merged.extend(replacement_records)
    write_jsonl(args.out_jsonl, merged)
    if args.out_csv:
        write_csv(args.out_csv, merged)

    counts = Counter(r.get("category") for r in merged)
    summary = {
        "base": str(args.base),
        "replacement": str(args.replacement),
        "replaced_categories": sorted(categories),
        "base_counts": dict(sorted(Counter(r.get("category") for r in base_records).items())),
        "replacement_counts": dict(sorted(replacement_counts.items())),
        "merged_counts": dict(sorted(counts.items())),
        "total": len(merged),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
