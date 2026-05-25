from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


ATTEMPT_RE = re.compile(r"\[(phase\d):([^\]]+)\].*?\s(OK|FAIL)\s\|")


def parse_log(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"attempts": 0, "ok": 0, "fail": 0})
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = ATTEMPT_RE.search(line)
        if not match:
            continue
        phase, category, status = match.groups()
        bucket = counts[(phase, category)]
        bucket["attempts"] += 1
        bucket["ok" if status == "OK" else "fail"] += 1
    return counts


def summarize_group(name: str, paths: list[Path]) -> dict[str, object]:
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"attempts": 0, "ok": 0, "fail": 0})
    for path in paths:
        for key, values in parse_log(path).items():
            totals[key]["attempts"] += values["attempts"]
            totals[key]["ok"] += values["ok"]
            totals[key]["fail"] += values["fail"]
    by_phase_category = {
        f"{phase}:{category}": values
        for (phase, category), values in sorted(totals.items())
    }
    by_phase: dict[str, dict[str, int | float]] = {}
    for (phase, _category), values in totals.items():
        bucket = by_phase.setdefault(phase, {"attempts": 0, "ok": 0, "fail": 0, "ok_rate": 0.0})
        bucket["attempts"] += values["attempts"]
        bucket["ok"] += values["ok"]
        bucket["fail"] += values["fail"]
    for values in by_phase.values():
        attempts = int(values["attempts"])
        values["ok_rate"] = float(values["ok"]) / attempts if attempts else 0.0
    return {
        "schema": name,
        "logs": [str(path) for path in paths],
        "by_phase": by_phase,
        "by_phase_category": by_phase_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Count materialized PIPE-RDF generation attempts from run logs.")
    parser.add_argument("--group", nargs="+", action="append", required=True, metavar="ITEM")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    groups = []
    for group in args.group:
        if len(group) < 2:
            raise SystemExit("--group requires NAME PATH [PATH ...]")
        groups.append(summarize_group(group[0], [Path(item) for item in group[1:]]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {"groups": groups}
    (out_dir / "run_log_attempt_summary.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
