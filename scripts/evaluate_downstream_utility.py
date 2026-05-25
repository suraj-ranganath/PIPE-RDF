from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipekg.evaluation import (
    answer_set_scores,
    parse_valid_sparql_detail,
    predicate_coverage,
    sketch_similarity,
    token_f1,
    triple_f1,
)
from pipekg.runtime import apply_run_config, build_llm
from pipekg.settings import get_settings
from pipekg.sparql_client import SparqlClient
from pipekg.vector_store import FaissStore


DEFAULT_PREFIXES = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX spb: <http://www.ldbcouncil.org/spb#>
PREFIX gn: <http://www.geonames.org/ontology#>
PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
""".strip()


def load_run_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Run config must be a YAML mapping")
    return data


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_sparql(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) >= 3 else parts[-1]
    text = re.sub(r"(?i)^sparql", "", text.strip()).strip()
    for token in ("PREFIX", "SELECT", "ASK"):
        idx = text.upper().find(token)
        if idx != -1:
            return text[idx:].strip()
    return text


def sanitize_predicted_sparql(sparql: str) -> str:
    sparql = re.sub(r"(?is)\(\s*SPARQLQuery\s*\)", "", sparql)
    sparql = re.sub(
        r"(?is)SELECT\s+COUNT\s*\(\s*([^\)]+)\s*\)",
        r"SELECT (COUNT(\1) AS ?count)",
        sparql,
    )
    count_alias = re.search(r"(?is)AS\s+\?count\s*\)", sparql)
    if count_alias:
        where = sparql[count_alias.end():]
        if re.search(r"(?<!AS\s)\?count\b", where, flags=re.IGNORECASE):
            sparql = re.sub(r"(?is)AS\s+\?count\s*\)", "AS ?answer_count)", sparql, count=1)
    return sparql.strip()


def get_schema_summary(cfg: dict[str, object]) -> str:
    if str(cfg.get("schema_summary_text", "")).strip():
        return str(cfg["schema_summary_text"])
    if str(cfg.get("schema_summary_path", "")).strip():
        return Path(str(cfg["schema_summary_path"])).read_text(encoding="utf-8")
    return "Use only predicates and types visible in the retrieved examples and endpoint profile."


def build_store(llm, records: list[dict[str, object]]) -> FaissStore | None:
    if not records:
        return None
    texts = [str(r.get("question", "")) for r in records]
    embeddings = np.array(llm.embed_texts(texts), dtype="float32")
    metadata = [
        {
            "question": str(r.get("question", "")),
            "sparql": str(r.get("sparql", "")),
            "category": str(r.get("category", "")),
        }
        for r in records
    ]
    return FaissStore.build(embeddings, metadata)


def embed_record_questions(llm, records: list[dict[str, object]]) -> dict[int, list[float]]:
    if not records:
        return {}
    texts = [str(r.get("question", "")) for r in records]
    embeddings = llm.embed_texts(texts)
    return {id(record): embedding for record, embedding in zip(records, embeddings)}


def build_store_from_embeddings(
    records: list[dict[str, object]],
    record_embeddings: dict[int, list[float]],
) -> FaissStore | None:
    if not records:
        return None
    embeddings = np.array([record_embeddings[id(r)] for r in records], dtype="float32")
    metadata = [
        {
            "question": str(r.get("question", "")),
            "sparql": str(r.get("sparql", "")),
            "category": str(r.get("category", "")),
        }
        for r in records
    ]
    return FaissStore.build(embeddings, metadata)


def retrieve(llm, stores: dict[str, FaissStore | None], question: str, category: str, k: int) -> list[dict[str, str]]:
    store = stores.get(category)
    if store is None:
        return []
    emb = np.array(llm.embed_texts([question]), dtype="float32")
    return store.search_with_scores(emb, k=k)[0]


def random_examples(
    rng: random.Random,
    records: list[dict[str, object]],
    k: int,
) -> list[dict[str, str]]:
    if not records or k <= 0:
        return []
    sample = rng.sample(records, k=min(k, len(records)))
    return [
        {
            "question": str(row.get("question", "")),
            "sparql": str(row.get("sparql", "")),
            "category": str(row.get("category", "")),
            "score": 0.0,
        }
        for row in sample
    ]


def prompt_for_question(prefixes: str, schema: str, question: str, examples: list[dict[str, str]]) -> tuple[str, str]:
    example_text = "None"
    if examples:
        rendered = []
        for idx, ex in enumerate(examples, start=1):
            rendered.append(f"Example {idx} question: {ex['question']}\nExample {idx} SPARQL:\n{ex['sparql']}")
        example_text = "\n\n".join(rendered)
    system = "You are an expert SPARQL engineer. Return only a SPARQL query."
    user = f"""Prefixes:
{prefixes}

Schema summary:
{schema}

Retrieved examples:
{example_text}

Question:
{question}

Rules:
- Use only the provided prefixes, predicates, and classes.
- Prefer VALUES clauses when the question names a concrete entity.
- For yes/no questions, return ASK.
- For counting questions, return COUNT(DISTINCT ...).
- Return only SPARQL."""
    return system, user


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["category"]))].append(row)
    out: dict[str, object] = {"overall": {}, "by_condition_category": {}}
    for key, items in grouped.items():
        condition, category = key
        n = len(items) or 1
        out["by_condition_category"][f"{condition}:{category}"] = {
            "n": len(items),
            "parse_valid": sum(bool(r["parse_valid"]) for r in items) / n,
            "exec_success": sum(bool(r["exec_success"]) for r in items) / n,
            "exact_answer_match": sum(bool(r["exact_answer_match"]) for r in items) / n,
            "answer_f1": sum(float(r["answer_f1"]) for r in items) / n,
            "sp_f1": sum(float(r["sp_f1"]) for r in items) / n,
            "triple_f1": sum(float(r["triple_f1"]) for r in items) / n,
            "predicate_f1": sum(float(r["predicate_f1"]) for r in items) / n,
            "sketch_similarity": sum(float(r["sketch_similarity"]) for r in items) / n,
        }
    for condition in sorted({str(r["condition"]) for r in rows}):
        items = [r for r in rows if str(r["condition"]) == condition]
        n = len(items) or 1
        out["overall"][condition] = {
            "n": len(items),
            "parse_valid": sum(bool(r["parse_valid"]) for r in items) / n,
            "exec_success": sum(bool(r["exec_success"]) for r in items) / n,
            "exact_answer_match": sum(bool(r["exact_answer_match"]) for r in items) / n,
            "answer_f1": sum(float(r["answer_f1"]) for r in items) / n,
            "sp_f1": sum(float(r["sp_f1"]) for r in items) / n,
            "triple_f1": sum(float(r["triple_f1"]) for r in items) / n,
            "predicate_f1": sum(float(r["predicate_f1"]) for r in items) / n,
            "sketch_similarity": sum(float(r["sketch_similarity"]) for r in items) / n,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated benchmark utility for NL-to-SPARQL prompting.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--schema-name", default="schema")
    parser.add_argument("--test-per-category", type=int, default=20)
    parser.add_argument("--retrieval-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["zero_shot", "category_rag"],
        choices=["zero_shot", "category_rag", "cross_category_rag", "random_same_schema"],
        help=(
            "Prompt settings to evaluate. cross_category_rag retrieves from the same schema "
            "but excludes the target category; random_same_schema samples same-schema examples "
            "without retrieval."
        ),
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    settings = apply_run_config(get_settings(), cfg)
    llm = build_llm(settings)
    sparql = SparqlClient(
        settings.sparql_endpoint_url,
        timeout=int(cfg.get("sparql_timeout_sec", 120)),
        params={"infer": "false"} if not cfg.get("sparql_infer", False) else {},
    )
    prefixes = str(cfg.get("prefixes", "")).strip() or DEFAULT_PREFIXES
    schema_summary = get_schema_summary(cfg)

    rng = random.Random(args.seed)
    records = load_jsonl(Path(args.input_jsonl))
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        by_category[str(row.get("category", "unknown"))].append(row)

    train_records: list[dict[str, object]] = []
    test_records: list[dict[str, object]] = []
    for category, items in by_category.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        test_records.extend(shuffled[: args.test_per_category])
        train_records.extend(shuffled[args.test_per_category :])

    needs_retrieval = any(c in args.conditions for c in ("category_rag", "cross_category_rag"))
    train_embeddings = embed_record_questions(llm, train_records) if needs_retrieval else {}
    stores: dict[str, FaissStore | None] = {}
    cross_category_stores: dict[str, FaissStore | None] = {}
    if "category_rag" in args.conditions:
        stores = {
            category: build_store_from_embeddings(
                [r for r in train_records if str(r.get("category")) == category],
                train_embeddings,
            )
            for category in by_category
        }
    if "cross_category_rag" in args.conditions:
        cross_category_stores = {
            category: build_store_from_embeddings(
                [r for r in train_records if str(r.get("category")) != category],
                train_embeddings,
            )
            for category in by_category
        }

    out_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/downstream_utility") / args.schema_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []
    total_prompts = len(test_records) * len(args.conditions)
    completed_prompts = 0

    for record in test_records:
        question = str(record.get("question", ""))
        category = str(record.get("category", "unknown"))
        gold_sparql = str(record.get("sparql", ""))
        gold_answers = [str(a) for a in (record.get("answers") or [])]
        for condition in args.conditions:
            examples = []
            if condition == "category_rag":
                examples = retrieve(llm, stores, question, category, args.retrieval_k)
            elif condition == "cross_category_rag":
                examples = retrieve(llm, cross_category_stores, question, category, args.retrieval_k)
            elif condition == "random_same_schema":
                examples = random_examples(rng, train_records, args.retrieval_k)
            system, user = prompt_for_question(prefixes, schema_summary, question, examples)
            completed_prompts += 1
            print(
                f"[{args.schema_name}] {completed_prompts}/{total_prompts} "
                f"{condition}:{category}",
                flush=True,
            )
            raw = llm.chat(system=system, user=user, temperature=0.0, max_tokens=768)
            pred_sparql = sanitize_predicted_sparql(extract_sparql(raw))
            if "PREFIX" not in pred_sparql.upper():
                pred_sparql = prefixes + "\n" + pred_sparql
            parse_valid, parse_error = parse_valid_sparql_detail(pred_sparql)
            pred_answers: list[str] = []
            exec_success = False
            exec_error = ""
            if parse_valid:
                try:
                    res = sparql.query(pred_sparql)
                    exec_success = True
                    pred_answers = [str(res.boolean)] if res.boolean is not None else [
                        str(v) for row in res.rows for v in row.values()
                    ]
                except Exception as exc:
                    exec_error = str(exc)
            precision, recall, answer_f1 = answer_set_scores(pred_answers, gold_answers)
            pred_norm = re.sub(r"\s+", " ", pred_sparql)
            gold_norm = re.sub(r"\s+", " ", gold_sparql)
            result_rows.append(
                {
                    "schema": args.schema_name,
                    "condition": condition,
                    "category": category,
                    "question": question,
                    "parse_valid": parse_valid,
                    "parse_error": parse_error or "",
                    "exec_success": exec_success,
                    "exec_error": exec_error,
                    "exact_answer_match": set(pred_answers) == set(gold_answers),
                    "answer_precision": precision,
                    "answer_recall": recall,
                    "answer_f1": answer_f1,
                    "sp_f1": token_f1(pred_norm, gold_norm),
                    "triple_f1": triple_f1(pred_sparql, gold_sparql),
                    "predicate_f1": predicate_coverage(pred_sparql, gold_sparql),
                    "sketch_similarity": sketch_similarity(pred_sparql, gold_sparql),
                    "retrieved_examples": json.dumps(examples),
                    "pred_sparql": pred_sparql,
                    "gold_sparql": gold_sparql,
                    "pred_answers": json.dumps(pred_answers),
                    "gold_answers": json.dumps(gold_answers),
                }
            )

    jsonl_path = out_dir / "utility_eval_results.jsonl"
    csv_path = out_dir / "utility_eval_results.csv"
    summary_path = out_dir / "utility_eval_summary.json"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if not result_rows:
            raise SystemExit("No utility-evaluation rows were produced")
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)
    summary = summarize(result_rows)
    summary["schema"] = args.schema_name
    summary["input_jsonl"] = args.input_jsonl
    summary["config"] = args.config
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
