from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _safe_div(numer: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return numer / denom


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "model"


def _load_yaml(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _write_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _detect_new_run_dir(runs_root: Path, before_names: set[str]) -> Path:
    after = [p for p in runs_root.iterdir() if p.is_dir()]
    new_dirs = [p for p in after if p.name not in before_names]
    if not new_dirs:
        raise RuntimeError("Could not detect a new run directory under artifacts/runs")
    new_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return new_dirs[0]


def _save_grouped_rate_plot(rows: List[Dict[str, object]], out_path: Path) -> None:
    models = [str(r["model"]) for r in rows]
    exec_rates = [float(r["phase3_exec_rate"]) * 100 for r in rows]
    parse_rates = [float(r["phase3_parse_rate"]) * 100 for r in rows]
    x = np.arange(len(models))
    w = 0.36

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.4), 5))
    ax.bar(x - w / 2, exec_rates, width=w, label="Exec Success %", color="#1D4ED8")
    ax.bar(x + w / 2, parse_rates, width=w, label="Parse Valid %", color="#0EA5E9")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Cross-model Phase 3 Robustness")
    ax.legend()
    for i, v in enumerate(exec_rates):
        ax.text(i - w / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(parse_rates):
        ax.text(i + w / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_strategy_heatmap(strategy_rows: List[Dict[str, object]], out_path: Path) -> None:
    if not strategy_rows:
        return
    models = sorted({str(r["model"]) for r in strategy_rows})
    strategies = sorted({str(r["strategy"]) for r in strategy_rows})
    matrix = np.zeros((len(models), len(strategies)), dtype=float)
    model_idx = {m: i for i, m in enumerate(models)}
    strat_idx = {s: j for j, s in enumerate(strategies)}
    for row in strategy_rows:
        i = model_idx[str(row["model"])]
        j = strat_idx[str(row["strategy"])]
        matrix[i, j] = float(row["error_rate"])

    fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.2), max(4, len(models) * 0.8)))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=max(1e-9, float(np.max(matrix))))
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=35, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Cross-model Strategy Error Rate")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j] * 100:.0f}%", ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Error rate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def _parse_models(raw_items: List[str]) -> List[str]:
    models = []
    for item in raw_items:
        for part in item.split(","):
            p = part.strip()
            if p:
                models.append(p)
    deduped = []
    seen = set()
    for m in models:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base YAML config file")
    parser.add_argument("--models", nargs="+", required=True, help="Chat models to evaluate (space or comma separated)")
    parser.add_argument("--embed-model", default="", help="Optional embedding model override")
    parser.add_argument("--run-prefix", default="crossmodel", help="Run-name prefix")
    parser.add_argument("--output-dir", default="", help="Output directory (default: artifacts/cross_model/<timestamp>)")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to use")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands, do not execute runs")
    args = parser.parse_args()

    base_config_path = Path(args.config)
    base_cfg = _load_yaml(base_config_path)
    models = _parse_models(args.models)
    if not models:
        raise SystemExit("No models provided")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/cross_model") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_dir / "configs"
    runs_root = Path("artifacts/runs")
    runs_root.mkdir(parents=True, exist_ok=True)

    results = []
    strategy_rows = []
    planned_cmds = []

    for model in models:
        slug = _slugify(model)
        model_cfg = dict(base_cfg)
        model_cfg.setdefault("models", {})
        model_cfg["models"] = dict(model_cfg["models"])
        model_cfg["models"]["chat"] = model
        if args.embed_model:
            model_cfg["models"]["embed"] = args.embed_model

        cfg_path = cfg_dir / f"{slug}.yaml"
        _write_yaml(cfg_path, model_cfg)

        run_name = f"{args.run_prefix}_{slug}"
        preflight_cmd = [args.python_bin, "scripts/preflight_check.py", "--config", str(cfg_path)]
        run_cmd = [args.python_bin, "scripts/run_pipeline_ollama.py", "--config", str(cfg_path), "--run-name", run_name]

        planned_cmds.append({"model": model, "preflight": preflight_cmd, "run": run_cmd})
        if args.dry_run:
            continue

        if not args.skip_preflight:
            subprocess.run(preflight_cmd, check=True)

        before_names = {p.name for p in runs_root.iterdir() if p.is_dir()}
        subprocess.run(run_cmd, check=True)
        run_dir = _detect_new_run_dir(runs_root, before_names)
        run_id = run_dir.name

        # Ensure strategy analysis exists for this run.
        strat_cmd = [args.python_bin, "scripts/generate_strategy_analysis.py", "--run-id", run_id]
        subprocess.run(strat_cmd, check=False)

        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Missing run summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        phase3 = summary.get("phase3", {})
        total = int(phase3.get("total", 0) or 0)
        exec_ok = int(phase3.get("exec_ok", 0) or 0)
        parse_ok = int(phase3.get("parse_ok", 0) or 0)
        repairs = int(phase3.get("repairs", 0) or 0)
        empty = int(phase3.get("empty", 0) or 0)
        parse_err = int(phase3.get("parse_err", 0) or 0)
        endpoint_err = int(phase3.get("endpoint_err", 0) or 0)

        row = {
            "model": model,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "phase3_total": total,
            "phase3_exec_rate": _safe_div(exec_ok, total),
            "phase3_parse_rate": _safe_div(parse_ok, total),
            "phase3_repair_rate": _safe_div(repairs, total),
            "phase3_empty_rate": _safe_div(empty, total),
            "phase3_parse_error_rate": _safe_div(parse_err, total),
            "phase3_endpoint_error_rate": _safe_div(endpoint_err, total),
            "phase3_avg_llm_ms": float(phase3.get("avg_llm_ms", 0.0) or 0.0),
            "phase3_avg_exec_ms": float(phase3.get("avg_exec_ms", 0.0) or 0.0),
            "phase3_avg_q_ms": float(phase3.get("avg_q_ms", 0.0) or 0.0),
            "runtime_sec": float(summary.get("runtime_sec", 0.0) or 0.0),
        }
        results.append(row)

        strategy_summary_path = run_dir / "analysis" / "strategy_summary.json"
        if strategy_summary_path.exists():
            strat_summary = json.loads(strategy_summary_path.read_text(encoding="utf-8"))
            for strategy, info in (strat_summary.get("per_strategy") or {}).items():
                strategy_rows.append(
                    {
                        "model": model,
                        "run_id": run_id,
                        "strategy": strategy,
                        "records": int(info.get("records", 0) or 0),
                        "exec_success_rate": float(info.get("exec_success_rate", 0.0) or 0.0),
                        "parse_valid_rate": float(info.get("parse_valid_rate", 0.0) or 0.0),
                        "error_rate": float(info.get("error_rate", 0.0) or 0.0),
                    }
                )

    if args.dry_run:
        dry_manifest = {"base_config": str(base_config_path), "models": models, "planned": planned_cmds}
        (out_dir / "dry_run_plan.json").write_text(json.dumps(dry_manifest, indent=2), encoding="utf-8")
        print(json.dumps(dry_manifest, indent=2))
        return

    # Save comparison tables.
    comparison_csv = out_dir / "model_comparison.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "run_id",
                "run_dir",
                "phase3_total",
                "phase3_exec_rate",
                "phase3_parse_rate",
                "phase3_repair_rate",
                "phase3_empty_rate",
                "phase3_parse_error_rate",
                "phase3_endpoint_error_rate",
                "phase3_avg_llm_ms",
                "phase3_avg_exec_ms",
                "phase3_avg_q_ms",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    strategy_csv = out_dir / "strategy_comparison.csv"
    with strategy_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "run_id",
                "strategy",
                "records",
                "exec_success_rate",
                "parse_valid_rate",
                "error_rate",
            ],
        )
        writer.writeheader()
        for row in strategy_rows:
            writer.writerow(row)

    # Save plots.
    if results:
        _save_grouped_rate_plot(results, out_dir / "cross_model_phase3_rates.png")
    if strategy_rows:
        _save_strategy_heatmap(strategy_rows, out_dir / "cross_model_strategy_error_heatmap.png")

    manifest = {
        "base_config": str(base_config_path),
        "models": models,
        "embed_model_override": args.embed_model,
        "results": results,
        "strategy_rows": strategy_rows,
        "artifacts": {
            "comparison_csv": str(comparison_csv),
            "strategy_csv": str(strategy_csv),
            "phase3_rates_plot": str(out_dir / "cross_model_phase3_rates.png"),
            "strategy_error_heatmap": str(out_dir / "cross_model_strategy_error_heatmap.png"),
        },
    }
    (out_dir / "cross_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Cross-model experiment complete. Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
