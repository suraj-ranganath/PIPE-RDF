from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def phase(summary: dict[str, object], name: str) -> dict[str, object]:
    value = summary.get(name, {})
    return value if isinstance(value, dict) else {}


def numeric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def add_phase_totals(target: dict[str, float], source: dict[str, object], name: str) -> None:
    source_phase = phase(source, name)
    prefix = f"{name}_"
    target[prefix + "accepted"] += numeric(source_phase.get("total"))
    target[prefix + "repairs"] += numeric(source_phase.get("repairs"))
    target[prefix + "exec_ok"] += numeric(source_phase.get("exec_ok"))
    target[prefix + "parse_ok"] += numeric(source_phase.get("parse_ok"))
    target[prefix + "llm_ms_weighted"] += numeric(source_phase.get("avg_llm_ms")) * numeric(source_phase.get("total"))
    target[prefix + "exec_ms_weighted"] += numeric(source_phase.get("avg_exec_ms")) * numeric(source_phase.get("total"))


def summarize_group(name: str, paths: list[Path], final_records: int) -> dict[str, object]:
    totals: dict[str, float] = {
        "runtime_sec": 0.0,
        "phase1_accepted": 0.0,
        "phase1_repairs": 0.0,
        "phase1_exec_ok": 0.0,
        "phase1_parse_ok": 0.0,
        "phase1_llm_ms_weighted": 0.0,
        "phase1_exec_ms_weighted": 0.0,
        "phase2_accepted": 0.0,
        "phase2_repairs": 0.0,
        "phase2_exec_ok": 0.0,
        "phase2_parse_ok": 0.0,
        "phase2_llm_ms_weighted": 0.0,
        "phase2_exec_ms_weighted": 0.0,
        "phase3_accepted": 0.0,
        "phase3_repairs": 0.0,
        "phase3_exec_ok": 0.0,
        "phase3_parse_ok": 0.0,
        "phase3_llm_ms_weighted": 0.0,
        "phase3_exec_ms_weighted": 0.0,
    }
    run_ids = []
    for path in paths:
        summary = load(path)
        run_ids.append(str(summary.get("run_id") or path.parent.name))
        totals["runtime_sec"] += numeric(summary.get("runtime_sec"))
        for phase_name in ("phase1", "phase2", "phase3"):
            add_phase_totals(totals, summary, phase_name)

    def weighted_mean(phase_name: str, key: str) -> float:
        count = totals[f"{phase_name}_accepted"]
        return totals[f"{phase_name}_{key}_weighted"] / count if count else 0.0

    phase3_generated = int(totals["phase3_accepted"])
    return {
        "schema": name,
        "run_ids": run_ids,
        "runs": len(paths),
        "final_retained_phase3_records": final_records,
        "generated_phase3_records_before_merge": phase3_generated,
        "runtime_sec": totals["runtime_sec"],
        "runtime_hours": totals["runtime_sec"] / 3600 if totals["runtime_sec"] else 0.0,
        "retained_phase3_per_minute": final_records / (totals["runtime_sec"] / 60) if totals["runtime_sec"] else 0.0,
        "generated_phase3_per_minute": phase3_generated / (totals["runtime_sec"] / 60) if totals["runtime_sec"] else 0.0,
        "phase1_accepted": int(totals["phase1_accepted"]),
        "phase2_accepted": int(totals["phase2_accepted"]),
        "phase3_accepted": phase3_generated,
        "all_phase_repairs": int(totals["phase1_repairs"] + totals["phase2_repairs"] + totals["phase3_repairs"]),
        "phase3_repairs": int(totals["phase3_repairs"]),
        "phase3_avg_llm_ms": weighted_mean("phase3", "llm_ms"),
        "phase3_avg_exec_ms": weighted_mean("phase3", "exec_ms"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PIPE-RDF operational efficiency from run_summary.json files.")
    parser.add_argument("--group", nargs="+", action="append", required=True, metavar="ITEM")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summaries = []
    for group in args.group:
        if len(group) < 3:
            raise SystemExit("--group requires NAME FINAL_RECORDS PATH [PATH ...]")
        name = group[0]
        final_records = int(group[1])
        paths = [Path(item) for item in group[2:]]
        summaries.append(summarize_group(name, paths, final_records))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {"groups": summaries}
    (out_dir / "operational_efficiency_summary.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
