from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI


BINARY_FIELDS = [
    "intent_query_match",
    "entity_binding_correct",
    "answer_type_correct",
    "category_construct_correct",
    "overall_pass",
]

ERROR_TYPES = {
    "none",
    "intent_mismatch",
    "entity_binding",
    "answer_type",
    "category_construct",
    "syntax_or_execution",
    "unsupported_or_ambiguous_question",
    "other",
}

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of NL-SPARQL benchmark quality.

Your task is to judge whether a natural-language question and its SPARQL query are semantically aligned for the stated RDF schema and query category. Focus on meaning, bindings, answer type, and category construct. Do not reward a query merely because it parses or executes. Use the answer preview only as supporting evidence; the primary decision is whether the SPARQL query correctly operationalizes the question on the given schema.

Return JSON only. Do not include markdown, prose, or code fences."""

JUDGE_USER_PROMPT_TEMPLATE = """Evaluate this PIPE-RDF NL-SPARQL benchmark record.

Audit ID: {audit_id}
Schema: {schema}
Category: {category}

Natural-language question:
{question}

SPARQL query:
```sparql
{sparql}
```

Answer preview JSON:
{answers}

Pipeline checks:
- parse_valid: {source_parse_valid}
- execution_success: {source_exec_success}
- answer_count: {source_answer_count}

Rubric:
1. intent_query_match: yes iff the query asks for the same relation, comparison, count, set operation, ranking, or yes/no condition as the question.
2. entity_binding_correct: yes iff concrete entities or literals in the question are correctly represented by VALUES clauses, IRIs, filters, or equivalent query constraints. If the question has no concrete entity, mark yes only if the query does not introduce a contradictory concrete entity.
3. answer_type_correct: yes iff the selected output includes the answer type requested by the question, such as entity, literal, boolean, count, date/year, or ranked item. Supporting columns are acceptable: for example, a superlative query may return both the winning entity and the numeric value used for ranking.
4. category_construct_correct: yes iff the SPARQL construct matches the intended category. Examples: ASK for yes/no and boolean comparative/difference questions; COUNT(DISTINCT ...) for counting; ORDER BY with LIMIT/OFFSET for superlative or ordinal; connected multi-triple joins through variables for multi-hop; conjunction for intersection; exclusion or inequality for difference. For multi-hop, direct schema predicates are valid when they are the schema path that operationalizes the question; do not require an unrelated ontology path.
5. overall_pass: yes iff all required semantic checks pass and there is no material issue.

When overall_pass is no, choose exactly one primary error_type from:
intent_mismatch, entity_binding, answer_type, category_construct, syntax_or_execution, unsupported_or_ambiguous_question, other.
When overall_pass is yes, use error_type none.

Judgment conventions:
- Treat canonical IRIs and aliases as correct when they denote the same entity named in the question.
- Do not mark answer_type_correct no only because the query returns an auxiliary label, year, count, or numeric value in addition to the requested entity.
- Do not mark category_construct_correct no only because the query uses the schema's direct predicate names; judge whether the variables and joins express the intended category under the given schema.

Return exactly this JSON object:
{{
  "intent_query_match": "yes or no",
  "entity_binding_correct": "yes or no",
  "answer_type_correct": "yes or no",
  "category_construct_correct": "yes or no",
  "overall_pass": "yes or no",
  "error_type": "none or one allowed error type",
  "confidence": 0.0,
  "notes": "one short sentence"
}}"""


@dataclass(frozen=True)
class JudgeSpec:
    provider: str
    model: str
    base_url: str
    api_key_env: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.provider}_{slug(self.model)}"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_records(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [{k: "" if v is None else str(v) for k, v in row.items()} for row in read_jsonl(path)]
    return read_csv(path)


def normalize_binary(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"yes", "y", "true", "1", "pass", "passed"}:
        return "yes"
    if raw in {"no", "n", "false", "0", "fail", "failed"}:
        return "no"
    return "no"


def normalize_error_type(value: object, overall_pass: str) -> str:
    raw = str(value or "").strip().lower()
    if overall_pass == "yes":
        return "none"
    if raw in ERROR_TYPES and raw != "none":
        return raw
    return "other"


def clamp_confidence(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def normalize_judgment(raw: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for field in BINARY_FIELDS:
        out[field] = normalize_binary(raw.get(field))
    out["error_type"] = normalize_error_type(raw.get("error_type"), str(out["overall_pass"]))
    out["confidence"] = clamp_confidence(raw.get("confidence"))
    out["notes"] = str(raw.get("notes", "")).strip().replace("\n", " ")[:500]
    return out


def extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Judge response was not a JSON object")
    return parsed


def get_api_key(spec: JudgeSpec) -> str:
    for env_name in spec.api_key_env:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    joined = ", ".join(spec.api_key_env)
    raise RuntimeError(f"Missing API key for {spec.provider}; set one of: {joined}")


def make_client(spec: JudgeSpec) -> OpenAI:
    kwargs = {"api_key": get_api_key(spec)}
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
    return OpenAI(**kwargs)


def completion_attempts(spec: JudgeSpec, messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, object]]:
    base: dict[str, object] = {
        "model": spec.model,
        "messages": messages,
        "timeout": 120,
    }
    if spec.provider == "openai":
        token_key = "max_completion_tokens"
    else:
        token_key = "max_tokens"
    return [
        {**base, token_key: max_tokens, "temperature": 0.0, "response_format": {"type": "json_object"}},
        {**base, token_key: max_tokens, "response_format": {"type": "json_object"}},
        {**base, token_key: max_tokens, "temperature": 0.0},
        {**base, token_key: max_tokens},
        {**base},
    ]


def call_judge(
    client: OpenAI,
    spec: JudgeSpec,
    record: dict[str, str],
    max_tokens: int,
    retries: int,
) -> dict[str, object]:
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        audit_id=record.get("audit_id", ""),
        schema=record.get("schema", ""),
        category=record.get("category", ""),
        question=record.get("question", ""),
        sparql=record.get("sparql", ""),
        answers=record.get("answers", "[]"),
        source_parse_valid=record.get("source_parse_valid", ""),
        source_exec_success=record.get("source_exec_success", ""),
        source_answer_count=record.get("source_answer_count", ""),
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    last_error = ""
    for attempt_idx in range(retries):
        for kwargs in completion_attempts(spec, messages, max_tokens):
            try:
                resp = client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                judgment = normalize_judgment(extract_json_object(content))
                return {
                    "audit_id": record.get("audit_id", ""),
                    "schema": record.get("schema", ""),
                    "category": record.get("category", ""),
                    "judge_provider": spec.provider,
                    "judge_model": spec.model,
                    **judgment,
                    "raw_response": content,
                    "call_error": "",
                }
            except Exception as exc:  # noqa: BLE001 - continue across provider-specific API quirks
                last_error = str(exc)
        time.sleep(min(2**attempt_idx, 8))
    return {
        "audit_id": record.get("audit_id", ""),
        "schema": record.get("schema", ""),
        "category": record.get("category", ""),
        "judge_provider": spec.provider,
        "judge_model": spec.model,
        "intent_query_match": "no",
        "entity_binding_correct": "no",
        "answer_type_correct": "no",
        "category_construct_correct": "no",
        "overall_pass": "no",
        "error_type": "other",
        "confidence": 0.0,
        "notes": "Judge call failed.",
        "raw_response": "",
        "call_error": last_error[:1000],
    }


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


def load_existing_judgments(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    return {str(row["audit_id"]): row for row in read_jsonl(path)}


def judge_records(
    records: list[dict[str, str]],
    spec: JudgeSpec,
    out_dir: Path,
    workers: int,
    max_tokens: int,
    retries: int,
    resume: bool,
) -> list[dict[str, object]]:
    out_jsonl = out_dir / f"{spec.key}_judgments.jsonl"
    existing = load_existing_judgments(out_jsonl) if resume else {}
    pending = [record for record in records if str(record.get("audit_id", "")) not in existing]
    client = make_client(spec)
    results_by_id = dict(existing)

    if pending:
        print(f"[{spec.provider}:{spec.model}] judging {len(pending)} records", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(call_judge, client, spec, record, max_tokens, retries): record
            for record in pending
        }
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            results_by_id[str(row["audit_id"])] = row
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                print(f"[{spec.provider}:{spec.model}] {completed}/{len(pending)} completed", flush=True)

    ordered = [results_by_id[str(record.get("audit_id", ""))] for record in records if str(record.get("audit_id", "")) in results_by_id]
    write_jsonl(out_jsonl, ordered)
    write_csv(out_dir / f"{spec.key}_judgments.csv", ordered)
    return ordered


def build_adjudicated(records: list[dict[str, str]], judgment_sets: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    by_judge = {
        judge_key: {str(row["audit_id"]): row for row in rows}
        for judge_key, rows in judgment_sets.items()
    }
    out = []
    for record in records:
        audit_id = str(record.get("audit_id", ""))
        row: dict[str, object] = {
            "audit_id": audit_id,
            "schema": record.get("schema", ""),
            "category": record.get("category", ""),
            "question": record.get("question", ""),
        }
        available = []
        for judge_key, rows_by_id in by_judge.items():
            judgment = rows_by_id.get(audit_id)
            if not judgment:
                continue
            available.append(judgment)
            prefix = judge_key
            for field in BINARY_FIELDS + ["error_type", "confidence", "notes", "call_error"]:
                row[f"{prefix}_{field}"] = judgment.get(field, "")

        for field in BINARY_FIELDS:
            values = [str(j.get(field, "")) for j in available if str(j.get(field, ""))]
            row[f"consensus_{field}"] = "yes" if values and all(v == "yes" for v in values) else "no"
            row[f"agreement_{field}"] = "yes" if values and len(set(values)) == 1 else "no"

        failing_errors = [
            str(j.get("error_type", "other"))
            for j in available
            if str(j.get("overall_pass", "")) == "no"
        ]
        row["consensus_error_type"] = "none" if row["consensus_overall_pass"] == "yes" else "|".join(sorted(set(failing_errors or ["other"])))
        row["judge_count"] = len(available)
        out.append(row)
    return out


def summarize(records: list[dict[str, str]], judgment_sets: dict[str, list[dict[str, object]]], adjudicated: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "records": len(records),
        "judges": {},
        "inter_judge_agreement": {},
        "consensus": {},
    }
    for judge_key, rows in judgment_sets.items():
        n = len(rows) or 1
        summary["judges"][judge_key] = {
            "n": len(rows),
            "call_errors": sum(1 for row in rows if row.get("call_error")),
            "field_pass_rates": {
                field: sum(str(row.get(field)) == "yes" for row in rows) / n
                for field in BINARY_FIELDS
            },
            "error_type_counts": dict(Counter(str(row.get("error_type", "other")) for row in rows)),
            "by_schema_category_overall_pass": {},
        }
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row.get("schema", "")), str(row.get("category", "")))].append(row)
        for (schema, category), items in grouped.items():
            denom = len(items) or 1
            summary["judges"][judge_key]["by_schema_category_overall_pass"][f"{schema}:{category}"] = {
                "n": len(items),
                "pass_rate": sum(str(row.get("overall_pass")) == "yes" for row in items) / denom,
            }

    judge_keys = list(judgment_sets)
    if len(judge_keys) == 2:
        left = {str(row["audit_id"]): row for row in judgment_sets[judge_keys[0]]}
        right = {str(row["audit_id"]): row for row in judgment_sets[judge_keys[1]]}
        common_ids = sorted(set(left) & set(right))
        for field in BINARY_FIELDS:
            summary["inter_judge_agreement"][field] = cohen_kappa(
                (str(left[audit_id].get(field, "")), str(right[audit_id].get(field, "")))
                for audit_id in common_ids
            )

    denom = len(adjudicated) or 1
    summary["consensus"] = {
        "n": len(adjudicated),
        "overall_pass_rate": sum(str(row.get("consensus_overall_pass")) == "yes" for row in adjudicated) / denom,
        "overall_agreement_rate": sum(str(row.get("agreement_overall_pass")) == "yes" for row in adjudicated) / denom,
        "error_type_counts": dict(Counter(str(row.get("consensus_error_type", "other")) for row in adjudicated)),
        "by_schema_category": {},
    }
    grouped_adj: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in adjudicated:
        grouped_adj[(str(row.get("schema", "")), str(row.get("category", "")))].append(row)
    for (schema, category), items in grouped_adj.items():
        n = len(items) or 1
        summary["consensus"]["by_schema_category"][f"{schema}:{category}"] = {
            "n": len(items),
            "pass_rate": sum(str(row.get("consensus_overall_pass")) == "yes" for row in items) / n,
            "agreement_rate": sum(str(row.get("agreement_overall_pass")) == "yes" for row in items) / n,
        }
    return summary


def write_prompt(path: Path) -> None:
    text = f"""# PIPE-RDF LLM Semantic Judge Prompt

## System Prompt

{JUDGE_SYSTEM_PROMPT}

## User Prompt Template

{JUDGE_USER_PROMPT_TEMPLATE}
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dual LLM semantic judges over a PIPE-RDF audit packet.")
    parser.add_argument("--input", required=True, help="Semantic audit packet CSV or JSONL.")
    parser.add_argument("--output-dir", default="artifacts/llm_semantic_audit/arr_20260520")
    parser.add_argument("--judge", action="append", choices=["openai", "xai"], help="Judge provider to run. Omit to run both.")
    parser.add_argument("--openai-model", default="gpt-5-mini")
    parser.add_argument("--xai-model", default="grok-4.20-0309-non-reasoning")
    parser.add_argument("--xai-base-url", default=os.getenv("XAI_BASE_URL", os.getenv("XAI_API_BASE", "https://api.x.ai/v1")))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--audit-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true", help="Skip API calls and summarize existing judge outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_prompt(out_dir / "llm_judge_prompt.md")

    records = load_records(Path(args.input))
    if args.audit_id:
        wanted = set(args.audit_id)
        records = [row for row in records if row.get("audit_id") in wanted]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("No input records selected")

    providers = args.judge or ["openai", "xai"]
    specs = []
    if "openai" in providers:
        specs.append(JudgeSpec("openai", args.openai_model, os.getenv("OPENAI_BASE_URL", ""), ("OPENAI_API_KEY",)))
    if "xai" in providers:
        specs.append(JudgeSpec("xai", args.xai_model, args.xai_base_url, ("XAI_API_KEY", "GROK_API_KEY")))

    judgment_sets: dict[str, list[dict[str, object]]] = {}
    for spec in specs:
        out_jsonl = out_dir / f"{spec.key}_judgments.jsonl"
        if args.merge_only:
            if not out_jsonl.exists():
                print(f"[merge-only] missing {out_jsonl}; skipping {spec.key}", flush=True)
                continue
            rows = list(load_existing_judgments(out_jsonl).values())
        else:
            rows = judge_records(
                records=records,
                spec=spec,
                out_dir=out_dir,
                workers=args.workers,
                max_tokens=args.max_tokens,
                retries=args.retries,
                resume=args.resume,
            )
        judgment_sets[spec.key] = rows
    if not judgment_sets:
        raise ValueError("No judge outputs available to summarize")

    adjudicated = build_adjudicated(records, judgment_sets)
    write_jsonl(out_dir / "llm_semantic_audit_adjudicated.jsonl", adjudicated)
    write_csv(out_dir / "llm_semantic_audit_adjudicated.csv", adjudicated)
    summary = summarize(records, judgment_sets, adjudicated)
    (out_dir / "llm_semantic_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
