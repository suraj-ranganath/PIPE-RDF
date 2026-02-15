from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipekg.strategy import (
    DEFAULT_ERROR_BUCKET_ORDER,
    DEFAULT_STRATEGY_ORDER,
    classify_error_bucket,
    infer_query_strategies,
)
from pipekg.plot_style import apply_publication_style

apply_publication_style()


def _save_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    path: Path,
    cmap: str = "YlGnBu",
    annotate_as_pct: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8.5, len(col_labels) * 1.12), max(5.2, len(row_labels) * 0.56)))
    vmax = 1.0 if annotate_as_pct else max(1e-9, float(np.max(matrix)))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, pad=10, weight="semibold")
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="#E2E8F0", linestyle="-", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if annotate_as_pct:
                txt = f"{val * 100:.0f}%"
            else:
                txt = f"{val:.0f}"
            txt_color = "white" if val >= 0.55 * vmax else "#0F172A"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate_as_pct:
        cbar.set_label("Rate")
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    else:
        cbar.set_label("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_records(path: Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_rate(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="", help="Run ID under artifacts/runs/<run-id>")
    parser.add_argument(
        "--input-jsonl",
        default="",
        help="Optional explicit input JSONL path. Overrides default log path selection.",
    )
    parser.add_argument(
        "--phase",
        default="",
        help="Optional phase filter (e.g., phase1, phase2, phase3).",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional output label suffix (e.g., phase3).",
    )
    args = parser.parse_args()

    if args.run_id:
        run_dir = Path("artifacts/runs") / args.run_id
        log_path = run_dir / "pipeline_records.jsonl"
        fig_dir = run_dir / "figures"
        analysis_dir = run_dir / "analysis"
    else:
        run_dir = Path("artifacts")
        log_path = Path("artifacts/logs/pipeline_records.jsonl")
        fig_dir = Path("artifacts/figures")
        analysis_dir = Path("artifacts/analysis")
    if args.input_jsonl:
        log_path = Path(args.input_jsonl)

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    records = _load_records(log_path)
    if not records:
        print("No records in log file.")
        return
    if args.phase:
        phase_filter = args.phase.strip().lower()
        records = [r for r in records if str(r.get("phase", "")).strip().lower() == phase_filter]
        if not records:
            print(f"No records found for phase={args.phase}")
            return

    analysis_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    label = args.label.strip().lower()
    phase_label = args.phase.strip().lower()
    if label:
        suffix_key = label
    elif phase_label:
        suffix_key = phase_label
    else:
        suffix_key = ""
    phase_suffix = f"_{suffix_key}" if suffix_key else ""

    all_strategies = set(DEFAULT_STRATEGY_ORDER)
    categories = sorted({str(r.get("category", "unknown")) for r in records})

    tagged_records = []
    by_category = defaultdict(list)
    by_strategy = defaultdict(list)
    for r in records:
        strategies = infer_query_strategies(
            str(r.get("sparql", "")),
            retrieved_examples=r.get("retrieved_examples") or [],
        )
        for s in strategies:
            all_strategies.add(s)
        err_bucket = classify_error_bucket(r)
        tagged = dict(r)
        tagged["strategy_tags"] = strategies
        tagged["error_bucket"] = err_bucket
        tagged_records.append(tagged)
        by_category[str(r.get("category", "unknown"))].append(tagged)
        for s in strategies:
            by_strategy[s].append(tagged)

    strategy_order = [s for s in DEFAULT_STRATEGY_ORDER if s in all_strategies] + sorted(
        all_strategies.difference(DEFAULT_STRATEGY_ORDER)
    )

    # Coverage matrix: category x strategy (usage rate within category).
    coverage = np.zeros((len(categories), len(strategy_order)), dtype=float)
    for i, cat in enumerate(categories):
        rows = by_category.get(cat, [])
        denom = len(rows)
        if denom == 0:
            continue
        strat_counts = defaultdict(int)
        for row in rows:
            for s in row.get("strategy_tags", []):
                strat_counts[s] += 1
        for j, s in enumerate(strategy_order):
            coverage[i, j] = strat_counts[s] / denom

    _save_heatmap(
        coverage,
        categories,
        strategy_order,
        "Strategy Coverage by Category",
        fig_dir / f"strategy_coverage_heatmap{phase_suffix}.png",
        cmap="YlGnBu",
        annotate_as_pct=True,
    )

    # Error matrix: strategy x error bucket (error rate within strategy-tagged rows).
    error_buckets = list(DEFAULT_ERROR_BUCKET_ORDER)
    error_matrix = np.zeros((len(strategy_order), len(error_buckets)), dtype=float)
    error_counts_matrix = np.zeros((len(strategy_order), len(error_buckets)), dtype=float)
    strategy_totals = {}
    for i, s in enumerate(strategy_order):
        rows = by_strategy.get(s, [])
        denom = len(rows)
        strategy_totals[s] = denom
        bucket_counts = defaultdict(int)
        for row in rows:
            b = row.get("error_bucket", "none")
            if b in error_buckets:
                bucket_counts[b] += 1
        for j, b in enumerate(error_buckets):
            error_counts_matrix[i, j] = bucket_counts[b]
            error_matrix[i, j] = _safe_rate(bucket_counts[b], denom)

    _save_heatmap(
        error_matrix,
        strategy_order,
        error_buckets,
        "Error Rate by Strategy",
        fig_dir / f"strategy_error_heatmap{phase_suffix}.png",
        cmap="OrRd",
        annotate_as_pct=True,
    )

    # Save tagged records for downstream analysis.
    tagged_path = analysis_dir / f"pipeline_records_with_strategies{phase_suffix}.jsonl"
    with tagged_path.open("w", encoding="utf-8") as f:
        for row in tagged_records:
            f.write(json.dumps(row) + "\n")

    # CSV export for strategy/error matrix.
    matrix_csv = analysis_dir / f"strategy_error_matrix{phase_suffix}.csv"
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "error_bucket",
                "count",
                "rate",
                "strategy_total",
            ],
        )
        writer.writeheader()
        for i, strategy in enumerate(strategy_order):
            for j, bucket in enumerate(error_buckets):
                writer.writerow(
                    {
                        "strategy": strategy,
                        "error_bucket": bucket,
                        "count": int(error_counts_matrix[i, j]),
                        "rate": float(error_matrix[i, j]),
                        "strategy_total": int(strategy_totals[strategy]),
                    }
                )

    coverage_csv = analysis_dir / f"strategy_coverage_matrix{phase_suffix}.csv"
    with coverage_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "strategy", "rate", "count", "category_total"],
        )
        writer.writeheader()
        for i, cat in enumerate(categories):
            cat_total = len(by_category.get(cat, []))
            for j, strategy in enumerate(strategy_order):
                count = int(round(coverage[i, j] * cat_total))
                writer.writerow(
                    {
                        "category": cat,
                        "strategy": strategy,
                        "rate": float(coverage[i, j]),
                        "count": count,
                        "category_total": cat_total,
                    }
                )

    per_strategy = {}
    for s in strategy_order:
        rows = by_strategy.get(s, [])
        n = len(rows)
        if n == 0:
            continue
        exec_ok = sum(1 for r in rows if bool(r.get("exec_success", False)))
        parse_ok = sum(1 for r in rows if bool(r.get("parse_valid", False)))
        err_counts = defaultdict(int)
        for r in rows:
            err_counts[str(r.get("error_bucket", "none"))] += 1
        non_none_errors = sum(v for k, v in err_counts.items() if k != "none")
        per_strategy[s] = {
            "records": n,
            "exec_success_rate": _safe_rate(exec_ok, n),
            "parse_valid_rate": _safe_rate(parse_ok, n),
            "error_rate": _safe_rate(non_none_errors, n),
            "error_counts": dict(err_counts),
            "error_rates": {k: _safe_rate(v, n) for k, v in err_counts.items()},
        }

    summary = {
        "run_id": args.run_id or "",
        "phase_filter": args.phase or "",
        "records": len(records),
        "categories": categories,
        "strategy_order": strategy_order,
        "error_buckets": error_buckets,
        "per_strategy": per_strategy,
        "artifacts": {
            "tagged_records": str(tagged_path),
            "strategy_error_matrix_csv": str(matrix_csv),
            "strategy_coverage_matrix_csv": str(coverage_csv),
            "strategy_coverage_heatmap_png": str(fig_dir / f"strategy_coverage_heatmap{phase_suffix}.png"),
            "strategy_error_heatmap_png": str(fig_dir / f"strategy_error_heatmap{phase_suffix}.png"),
        },
    }
    summary_path = analysis_dir / f"strategy_summary{phase_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Strategy analysis complete: {summary_path}")


if __name__ == "__main__":
    main()
