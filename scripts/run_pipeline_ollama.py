import csv
import json
import random
import re
import time
import logging
import urllib.parse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipekg.config import (
    TARGET_PER_CATEGORY,
    REPAIR_ATTEMPTS,
    PHASE1_TEMPLATES_PER_CATEGORY,
    PHASE1_SEEDS_PER_TEMPLATE,
    PHASE2_SEEDS_PER_CATEGORY,
    RETRIEVAL_TOP_K,
    DUP_SIM_THRESHOLD,
    CATEGORIES,
)
from pipekg.settings import get_settings
from pipekg.llm import LLMClient
from pipekg.runtime import apply_run_config, build_llm
from pipekg.sparql_client import SparqlClient
import argparse
import hashlib
import subprocess
from datetime import datetime
from pipekg.pipeline_ollama import (
    GenerationRecord,
    OllamaPipeline,
    build_faiss_index,
    retrieve_examples,
    append_record_jsonl,
)
from pipekg.evaluation import parse_valid_sparql_detail, validate_answer_type
from pipekg.ast_utils import ast_stats
from pipekg.figures_extra import bar_by_category
from pipekg.logging_utils import result_set_hash
from pipekg.utils import tokenize, jaccard
from pipekg.schema_summary import build_schema_summary, build_schema_whitelist
from pipekg.logger import get_logger

def build_endpoint_url(base_url: str, infer: bool) -> str:
    if infer:
        return base_url
    if "infer=" in base_url:
        return base_url
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}infer=false"


def load_run_config(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Run config must be a YAML mapping")
    return data


def load_profile_values(path: str, limit: int | None = None) -> list[str]:
    values: list[str] = []
    if not path:
        return values
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample else False
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if row and row[0].strip():
                values.append(row[0].strip())
            if limit and len(values) >= limit:
                break
    return values


def add_example_to_store(llm: LLMClient, store, question: str, sparql: str, category: str):
    embedding = np.array(llm.embed_texts([question]), dtype="float32")
    example = {"question": question, "sparql": sparql, "category": category}
    if store is None:
        return build_faiss_index(llm, [example])
    store.add(embedding, [example])
    return store


def is_duplicate(llm: LLMClient, store, question: str, threshold: float) -> bool:
    if store is None or not getattr(store, "metadata", None):
        return False
    q_tokens = tokenize(question)
    for ex in store.metadata:
        ex_q = ex.get("question", "")
        if not ex_q:
            continue
        sim = jaccard(q_tokens, tokenize(ex_q))
        if sim >= threshold:
            return True
    return False


def is_external_uri(uri: str) -> bool:
    return "wikidata.org" in uri


def is_bad_type(type_uri: str) -> bool:
    bad = {
        "http://www.w3.org/2002/07/owl#Class",
        "http://www.w3.org/2002/07/owl#Thing",
        "http://www.w3.org/2000/01/rdf-schema#Class",
    }
    return type_uri in bad


def is_iri(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def row_key(row, slots):
    if not slots:
        return ()
    return tuple(str(row.get(slot)) for slot in slots)


def select_valid_row(rows, slots, pipeline, max_checks: int, logger=None, used_keys=None, entity_diversity=None, category=None, target_per_cat=None) -> dict | None:
    if not rows:
        return None
    sample = rows[:]
    random.shuffle(sample)
    stats = {"missing": 0, "external": 0, "bad_type": 0, "bad_label": 0, "duplicate": 0, "self_compare": 0, "diversity_cap": 0}
    for row in sample[: max_checks]:
        valid = True
        key = None
        if used_keys is not None:
            key = row_key(row, slots)
            if key in used_keys:
                stats["duplicate"] += 1
                continue
        # Self-comparison filter: reject rows where the same entity fills multiple slots
        if has_self_comparison(row, slots):
            stats["self_compare"] += 1
            continue
        if len(slots) > 1:
            base_map = {}
            for slot in slots:
                base = re.sub(r"\\d+$", "", slot)
                base_map.setdefault(base, []).append(slot)
            for base, slot_list in base_map.items():
                if len(slot_list) > 1:
                    vals = [row.get(s) for s in slot_list]
                    if any(v is None for v in vals) or len(set(vals)) < len(vals):
                        valid = False
                        break
            if not valid:
                stats["duplicate"] += 1
                continue
        for slot in slots:
            val = row.get(slot)
            if not val or not isinstance(val, str):
                stats["missing"] += 1
                valid = False
                break
            if is_iri(val):
                if is_external_uri(val):
                    stats["external"] += 1
                    valid = False
                    break
                type_uri = pipeline.entity_type_for_uri(val)
                if type_uri and is_bad_type(type_uri):
                    stats["bad_type"] += 1
                    valid = False
                    break
                label = pipeline.label_for_uri(val)
                if not label or "{" in label or "}" in label:
                    stats["bad_label"] += 1
                    valid = False
                    break
            else:
                if "{" in val or "}" in val:
                    stats["bad_label"] += 1
                    valid = False
                    break
        if valid:
            # Entity diversity cap check
            if entity_diversity is not None and category and target_per_cat:
                entity_uris = [row.get(s) for s in slots if row.get(s) and is_iri(row.get(s, ""))]
                if entity_diversity.would_exceed_cap(category, entity_uris, target_per_cat):
                    stats["diversity_cap"] += 1
                    continue
            if used_keys is not None and key is not None:
                used_keys.add(key)
            return row
    if logger:
        logger.debug("No valid row found | slots=%s stats=%s", slots, stats)
    return None


def log_row_sample(rows, phase: str, cat: str, logger=None) -> None:
    if not logger or not rows:
        return
    sample = rows[0]
    if isinstance(sample, dict):
        preview = {k: sample.get(k) for k in list(sample.keys())[:4]}
        logger.debug("Reverse query rows sample (%s:%s): %s", phase, cat, preview)


def dedupe_templates(templates):
    seen = set()
    unique = []
    for item in templates:
        template = item.get("template")
        if not template or template in seen:
            continue
        seen.add(template)
        unique.append(item)
    return unique


def normalize_template_structure(template: str) -> str:
    """Normalize a template by replacing slot values with generic placeholders
    so that structurally identical templates are detected as duplicates."""
    norm = re.sub(r"\{[^}]+\}", "<SLOT>", template)
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    return norm


def structural_dedupe_templates(templates):
    """Deduplicate templates by slot-normalized structure, not just exact string."""
    seen_structures = set()
    unique = []
    for item in templates:
        template = item.get("template", "")
        if not template:
            continue
        structure = normalize_template_structure(template)
        if structure in seen_structures:
            continue
        seen_structures.add(structure)
        unique.append(item)
    return unique


def priority_templates_for_category(category: str) -> list[dict[str, object]]:
    """Deterministic high-yield templates for categories needing many rows."""
    if category == "generic":
        return [
            {"template": "Where is {company} located?", "slots": ["company"]},
            {"template": "What industry is {company} in?", "slots": ["company"]},
            {"template": "Who is a key person at {company}?", "slots": ["company"]},
            {"template": "What is the founding year of {company}?", "slots": ["company"]},
            {"template": "How many employees does {company} have?", "slots": ["company"]},
        ]
    if category == "counting":
        return [
            {"template": "How many companies are in {industry}?", "slots": ["industry"]},
            {"template": "How many companies are located in {location}?", "slots": ["location"]},
            {"template": "How many companies have key person {person}?", "slots": ["person"]},
            {"template": "How many key people are associated with {company}?", "slots": ["company"]},
            {"template": "How many companies were founded in {year}?", "slots": ["year"]},
        ]
    if category == "comparative":
        return [
            {"template": "Are {company1} and {company2} in the same industry?", "slots": ["company1", "company2"]},
            {"template": "Are {company1} and {company2} located in the same location?", "slots": ["company1", "company2"]},
            {"template": "Were {company1} and {company2} founded in the same year?", "slots": ["company1", "company2"]},
            {"template": "Do {company1} and {company2} share a key person?", "slots": ["company1", "company2"]},
            {"template": "Do {company1} and {company2} have the same employee count?", "slots": ["company1", "company2"]},
        ]
    if category == "difference":
        return [
            {"template": "Are {company1} and {company2} in different industries?", "slots": ["company1", "company2"]},
            {"template": "Do {company1} and {company2} have different locations?", "slots": ["company1", "company2"]},
            {"template": "Are {company1} and {company2} located in different locations?", "slots": ["company1", "company2"]},
            {"template": "Do {company1} and {company2} operate in different industries?", "slots": ["company1", "company2"]},
            {"template": "Were {company1} and {company2} founded in different years?", "slots": ["company1", "company2"]},
        ]
    if category == "multi-hop":
        return [
            {"template": "Which company is in {industry} and located in {location}?", "slots": ["industry", "location"]},
            {"template": "Which company has key person {person} and is located in {location}?", "slots": ["person", "location"]},
            {"template": "Which company in {industry} has key person {person}?", "slots": ["industry", "person"]},
        ]
    if category == "intersection":
        return [
            {"template": "Which company operates in {industry} and is located in {location}?", "slots": ["industry", "location"]},
            {"template": "Which company in {industry} has key person {person}?", "slots": ["industry", "person"]},
            {"template": "Which company located in {location} has key person {person}?", "slots": ["location", "person"]},
        ]
    if category == "superlative":
        return [
            {"template": "Which company in {industry} has the most employees?", "slots": ["industry"]},
            {"template": "Which company in {industry} has the fewest employees?", "slots": ["industry"]},
            {"template": "Which company in {industry} was founded earliest?", "slots": ["industry"]},
            {"template": "Which company in {industry} was founded most recently?", "slots": ["industry"]},
            {"template": "Which company in {location} has the most employees?", "slots": ["location"]},
            {"template": "Which company in {location} has the fewest employees?", "slots": ["location"]},
            {"template": "Which company in {location} was founded earliest?", "slots": ["location"]},
            {"template": "Which company in {location} was founded most recently?", "slots": ["location"]},
            {"template": "Which company in {industry} has the most key people?", "slots": ["industry"]},
            {"template": "Which company in {location} has the most key people?", "slots": ["location"]},
            {"template": "Which company has the most employees?", "slots": []},
            {"template": "Which company has the fewest employees?", "slots": []},
            {"template": "Which company was founded earliest?", "slots": []},
            {"template": "Which company was founded most recently?", "slots": []},
        ]
    if category == "ordinal":
        return [
            {"template": "Which company has the {rank} most employees?", "slots": ["rank"]},
            {"template": "Which company has the {rank} fewest employees?", "slots": ["rank"]},
            {"template": "Which company was founded {rank} earliest?", "slots": ["rank"]},
            {"template": "Which company was founded {rank} most recently?", "slots": ["rank"]},
            {"template": "Which company in {industry} has the {rank} most employees?", "slots": ["industry", "rank"]},
            {"template": "Which company in {industry} has the {rank} fewest employees?", "slots": ["industry", "rank"]},
            {"template": "Which company in {industry} was founded {rank} earliest?", "slots": ["industry", "rank"]},
            {"template": "Which company in {industry} was founded {rank} most recently?", "slots": ["industry", "rank"]},
            {"template": "Which company in {location} has the {rank} most employees?", "slots": ["location", "rank"]},
            {"template": "Which company in {location} has the {rank} fewest employees?", "slots": ["location", "rank"]},
        ]
    if category == "yesno":
        return [
            {"template": "Is {company} located in {location}?", "slots": ["company", "location"]},
            {"template": "Is {company} based in {location}?", "slots": ["company", "location"]},
            {"template": "Is {company} in the {industry} industry?", "slots": ["company", "industry"]},
            {"template": "Does {company} operate in the {industry} industry?", "slots": ["company", "industry"]},
            {"template": "Is {person} a key person at {company}?", "slots": ["person", "company"]},
        ]
    return []


def merge_priority_templates(category: str, templates: list[dict[str, object]]) -> list[dict[str, object]]:
    priority = priority_templates_for_category(category)
    if not priority:
        return filter_templates_for_category(category, templates)
    merged = dedupe_templates(priority + list(templates))
    if category == "superlative":
        slot_templates = [item for item in merged if item.get("slots")]
        slotless_templates = [item for item in merged if not item.get("slots")]
        return filter_templates_for_category(category, slot_templates + slotless_templates)
    return filter_templates_for_category(category, merged)


def filter_templates_for_category(category: str, templates: list[dict[str, object]]) -> list[dict[str, object]]:
    """Drop category-mismatched templates before reverse querying.

    LLMs occasionally return a fluent template for the wrong category (for
    example, a generic founding-year lookup under the ordinal category). The
    deterministic priority templates are sufficient for the ARR runs, and this
    guard keeps fallback LLM templates from weakening construct coverage.
    """
    filtered: list[dict[str, object]] = []
    for item in templates:
        template = str(item.get("template", ""))
        lower = template.lower()
        slots = set(item.get("slots") or [])
        keep = True
        if category == "counting":
            keep = ("count" in lower or "how many" in lower) and bool(slots)
        elif category == "comparative":
            keep = bool({"company1", "company2"} <= slots) and any(
                marker in lower
                for marker in ("same", "share", "both", "similar", "equal")
            )
        elif category == "difference":
            keep = bool({"company1", "company2"} <= slots) and any(
                marker in lower
                for marker in ("different", "differ", "not the same")
            )
        elif category == "superlative":
            keep = any(
                marker in lower
                for marker in ("most", "fewest", "least", "earliest", "latest", "recently", "highest", "lowest")
            )
        elif category == "ordinal":
            keep = "rank" in slots and any(
                marker in lower
                for marker in ("{rank}", "second", "third", "fourth", "fifth")
            )
        elif category in {"multi-hop", "intersection"}:
            keep = len(slots) >= 2 and any(marker in lower for marker in (" and ", " with ", " in "))
        elif category == "yesno":
            keep = lower.startswith(("is ", "are ", "does ", "do ", "was ", "were "))
        if keep:
            filtered.append(item)
    return filtered


class EntityDiversityTracker:
    """Tracks entity frequency per category to enforce diversity caps."""

    def __init__(self, max_entity_pct: float = 0.15):
        self.max_entity_pct = max_entity_pct
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._totals: dict[str, int] = defaultdict(int)

    def would_exceed_cap(self, category: str, entity_uris: list[str], target_per_cat: int) -> bool:
        """Check if adding these entities would push any entity above the cap."""
        cap = max(2, int(target_per_cat * self.max_entity_pct))
        for uri in entity_uris:
            if not uri:
                continue
            current = self._counts[category].get(uri, 0)
            if current + 1 > cap:
                return True
        return False

    def record(self, category: str, entity_uris: list[str]) -> None:
        """Record that these entities were used in a record."""
        for uri in entity_uris:
            if not uri:
                continue
            self._counts[category][uri] += 1
        self._totals[category] += 1


def has_self_comparison(row: dict, slots: list[str]) -> bool:
    """Check if a multi-slot row compares an entity with itself."""
    if len(slots) < 2:
        return False
    # Group slots by base name (e.g., company1, company2 -> company)
    base_groups: dict[str, list[str]] = {}
    for slot in slots:
        base = re.sub(r"\d+$", "", slot)
        base_groups.setdefault(base, []).append(slot)
    for base, slot_list in base_groups.items():
        if len(slot_list) < 2:
            continue
        vals = [row.get(s) for s in slot_list]
        if any(v is None for v in vals):
            continue
        if len(set(vals)) < len(vals):
            return True
    return False


def filter_retrieval_leakage(examples: list[dict], question: str, threshold: float = 0.99) -> list[dict]:
    """Filter out retrieved examples that are exact or near-exact matches
    to the current question (retrieval self-reference)."""
    filtered = []
    for ex in examples:
        score = ex.get("score", 0.0)
        if score >= threshold:
            continue
        # Also check exact question match
        if ex.get("question", "").strip().lower() == question.strip().lower():
            continue
        filtered.append(ex)
    return filtered


def bounded_reverse_query(sparql: str, limit: int, offset: int = 0) -> str:
    """Build a deterministic binding-bank query without ORDER BY RAND()."""
    text = sparql
    text = re.sub(r"(?im)^\s*ORDER\s+BY\b.*$", "", text)
    text = re.sub(r"(?im)^\s*LIMIT\s+\d+\b.*$", "", text)
    text = re.sub(r"(?im)^\s*OFFSET\s+\d+\b.*$", "", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        upper = ln.strip().upper()
        if upper.startswith(("ORDER BY", "LIMIT", "OFFSET")):
            continue
        cleaned.append(ln)
    cleaned.append(f"LIMIT {limit}")
    if offset > 0:
        cleaned.append(f"OFFSET {offset}")
    return "\n".join(cleaned)


def select_vars_cover_slots(sparql: str, slots) -> bool:
    upper = sparql.upper()
    if "SELECT" not in upper or "WHERE" not in upper:
        return False
    start = upper.find("SELECT") + len("SELECT")
    end = upper.find("WHERE")
    select_clause = sparql[start:end]
    if "*" in select_clause:
        return True
    select_clause = select_clause.replace("DISTINCT", "").replace("REDUCED", "")
    for slot in slots:
        if f"?{slot}" not in select_clause:
            return False
    return True


def is_simple_reverse_query(sparql: str, category: str | None = None) -> bool:
    upper = sparql.upper()
    banned = ["ORDER BY", "OPTIONAL", "UNION", "SUBSELECT", "AVG(", "MIN(", "MAX(", "SUM("]
    if category != "ordinal":
        banned.extend(["GROUP BY", "HAVING", "COUNT("])
    return not any(tok in upper for tok in banned)


def body_contains_slots(sparql: str, slots) -> bool:
    if "{" not in sparql or "}" not in sparql:
        return False
    body = sparql.split("{", 1)[1].rsplit("}", 1)[0]
    for slot in slots:
        if f"?{slot}" not in body:
            return False
    return True


def extract_predicates(sparql: str, prefix_map: dict) -> list[str]:
    preds = []
    if "{" not in sparql or "}" not in sparql:
        return preds
    body = sparql.split("{", 1)[1].rsplit("}", 1)[0]
    for line in body.split("."):
        seg = line.strip()
        if not seg:
            continue
        upper = seg.upper()
        if upper.startswith(("FILTER", "BIND", "VALUES", "OPTIONAL", "UNION")):
            continue
        parts = seg.split()
        if len(parts) < 3:
            continue
        pred = parts[1]
        if pred == "a":
            preds.append("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            continue
        if pred.startswith("<") and pred.endswith(">"):
            preds.append(pred[1:-1])
            continue
        if ":" in pred:
            prefix, local = pred.split(":", 1)
            if prefix in prefix_map:
                preds.append(prefix_map[prefix] + local)
            else:
                preds.append(pred)
    return preds


def extract_types(sparql: str, prefix_map: dict) -> list[str]:
    types = []
    if "{" not in sparql or "}" not in sparql:
        return types
    body = sparql.split("{", 1)[1].rsplit("}", 1)[0]
    for line in body.split("."):
        seg = line.strip()
        if not seg:
            continue
        upper = seg.upper()
        if upper.startswith(("FILTER", "BIND", "VALUES", "OPTIONAL", "UNION")):
            continue
        parts = seg.split()
        if len(parts) < 3:
            continue
        pred = parts[1]
        if pred != "a":
            continue
        obj = parts[2]
        if obj.startswith("<") and obj.endswith(">"):
            types.append(obj[1:-1])
            continue
        if ":" in obj:
            prefix, local = obj.split(":", 1)
            if prefix in prefix_map:
                types.append(prefix_map[prefix] + local)
            else:
                types.append(obj)
    return types


def to_prefixed(uri: str, prefix_map: dict) -> str:
    for prefix, base in prefix_map.items():
        if uri.startswith(base):
            return f"{prefix}:{uri[len(base):]}"
    return uri


def parse_prefix_map(prefixes_text: str) -> dict:
    prefix_map = {}
    for line in prefixes_text.splitlines():
        line = line.strip()
        if not line or not line.lower().startswith("prefix"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        prefix = parts[1].rstrip(":")
        iri = parts[2].strip("<>")
        prefix_map[prefix] = iri
    return prefix_map


def expand_qname(value: str, prefix_map: dict) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if ":" in value:
        prefix, local = value.split(":", 1)
        base = prefix_map.get(prefix)
        if base:
            return base + local
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="", help="Optional run name label")
    parser.add_argument("--config", default="", help="Path to YAML config (e.g., configs/smoke_test.yaml)")
    args = parser.parse_args()

    run_start = time.time()
    settings = get_settings()
    cfg = load_run_config(args.config)
    apply_run_config(settings, cfg)
    if not settings.sparql_endpoint_url:
        raise SystemExit("SPARQL_ENDPOINT_URL is not set")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = args.run_name.strip().replace(" ", "_")
    run_id = f"{timestamp}_{run_label}" if run_label else timestamp
    run_dir = Path("artifacts/runs") / run_id
    log_path = run_dir / "pipeline_records.jsonl"
    data_dir = run_dir / "data"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    logger = get_logger("pipekg.run", log_file=run_dir / "run.log", level=logging.DEBUG)
    logger.info("Starting PIPE-KG run")
    logger.info("Run ID: %s", run_id)
    logger.info("SPARQL endpoint: %s", settings.sparql_endpoint_url)
    logger.info("LLM provider: %s", settings.llm_provider)
    if settings.llm_provider == "ollama":
        logger.info("Ollama base: %s", settings.ollama_base_url)
        logger.info("Models: chat=%s embed=%s", settings.ollama_chat_model, settings.ollama_embed_model)
    else:
        logger.info("OpenAI-compatible base: %s", settings.openai_base_url)
        logger.info("Models: chat=%s embed=%s", settings.openai_chat_model, settings.openai_embed_model)
    logger.info(
        "Embedding provider: %s local_model=%s device=%s",
        settings.embed_provider or settings.llm_provider,
        settings.local_embed_model,
        settings.local_embed_device or "auto",
    )
    if args.config:
        logger.info("Run config: %s", args.config)
    logger.info("Logs: %s", log_path)
    logger.info("Data: %s", data_dir)

    sizes = cfg.get("sizes", {})
    phase1_templates_per_category = sizes.get("phase1_templates_per_category", PHASE1_TEMPLATES_PER_CATEGORY)
    phase1_seeds_per_template = sizes.get("phase1_seeds_per_template", PHASE1_SEEDS_PER_TEMPLATE)
    phase2_seeds_per_category = sizes.get("phase2_seeds_per_category", PHASE2_SEEDS_PER_CATEGORY)
    target_per_category = sizes.get("target_per_category", TARGET_PER_CATEGORY)
    retrieval_top_k = cfg.get("retrieval_top_k", RETRIEVAL_TOP_K)
    repair_attempts = cfg.get("repair_attempts", REPAIR_ATTEMPTS)
    dup_sim_threshold = cfg.get("dup_sim_threshold", DUP_SIM_THRESHOLD)
    categories = cfg.get("categories", CATEGORIES)
    template_candidates = cfg.get("template_candidates", 1)
    reverse_query_timeout_sec = cfg.get("reverse_query_timeout_sec", 120)
    reverse_query_limit = cfg.get("reverse_query_limit", 85)
    max_attempts_factor = cfg.get("max_attempts_factor", 5)
    max_row_checks = cfg.get("max_row_checks", 20)
    predicate_whitelist_topk = cfg.get("predicate_whitelist_topk", 200)
    schema_summary_topk = cfg.get("schema_summary_topk", 10)
    schema_timeout_sec = cfg.get("schema_timeout_sec", 15)
    sparql_timeout_sec = cfg.get("sparql_timeout_sec", 60)
    sparql_infer = cfg.get("sparql_infer", False)
    generated_query_limit = cfg.get("generated_query_limit")
    binding_bank_target_rows = int(cfg.get("binding_bank_target_rows", 1000))
    binding_bank_query_limit = int(cfg.get("binding_bank_query_limit", min(500, max(100, binding_bank_target_rows))))
    binding_bank_offset_stride = int(cfg.get("binding_bank_offset_stride", binding_bank_query_limit))
    binding_bank_max_queries = int(
        cfg.get(
            "binding_bank_max_queries",
            max(1, (binding_bank_target_rows + binding_bank_query_limit - 1) // binding_bank_query_limit),
        )
    )
    binding_bank_extend_queries = int(cfg.get("binding_bank_extend_queries", 1))
    binding_bank_batch_size = int(cfg.get("binding_bank_batch_size", 200))
    binding_bank_dir = data_dir / "binding_banks"
    binding_bank_dir.mkdir(parents=True, exist_ok=True)

    sparql_params = {}
    if not sparql_infer:
        sparql_params["infer"] = "false"
        logger.info("SPARQL inference disabled (infer=false)")
    logger.info("SPARQL endpoint (base): %s", settings.sparql_endpoint_url)
    logger.info("SPARQL timeouts: query=%ss schema=%ss", sparql_timeout_sec, schema_timeout_sec)
    if generated_query_limit:
        logger.info("Generated query LIMIT cap: %s", generated_query_limit)
    logger.info("Reverse query LIMIT cap: %s", reverse_query_limit)
    logger.info(
        "Binding banks: target_rows=%s query_limit=%s offset_stride=%s max_queries=%s extend_queries=%s batch_size=%s",
        binding_bank_target_rows,
        binding_bank_query_limit,
        binding_bank_offset_stride,
        binding_bank_max_queries,
        binding_bank_extend_queries,
        binding_bank_batch_size,
    )

    llm = build_llm(settings)
    sparql = SparqlClient(settings.sparql_endpoint_url, timeout=sparql_timeout_sec, params=sparql_params)
    schema_client = SparqlClient(settings.sparql_endpoint_url, timeout=schema_timeout_sec, params=sparql_params)
    logger.info(
        "Sizes: phase1_templates_per_category=%s phase1_seeds_per_template=%s phase2_seeds_per_category=%s target_per_category=%s",
        phase1_templates_per_category,
        phase1_seeds_per_template,
        phase2_seeds_per_category,
        target_per_category,
    )
    total_expected = (
        phase1_templates_per_category * phase1_seeds_per_template * len(categories)
        + phase2_seeds_per_category * len(categories)
        + target_per_category * len(categories)
    )
    overall_done = 0

    def overall_pct() -> float:
        if total_expected <= 0:
            return 0.0
        return (overall_done / total_expected) * 100.0

    default_prefixes = """
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

    prefixes = cfg.get("prefixes", "").strip()
    if not prefixes and cfg.get("prefixes_path"):
        prefixes = Path(cfg["prefixes_path"]).read_text(encoding="utf-8").strip()
    if not prefixes:
        prefixes = default_prefixes
    prefix_map = parse_prefix_map(prefixes)

    logger.info("Building schema summary (top_k=%d)...", schema_summary_topk)
    t_schema = time.time()
    schema_summary = ""
    schema_summary_text = cfg.get("schema_summary_text", "").strip()
    schema_summary_path = cfg.get("schema_summary_path", "").strip()
    if schema_summary_text:
        schema_summary = schema_summary_text
    elif schema_summary_path:
        schema_summary = Path(schema_summary_path).read_text(encoding="utf-8")
    else:
        try:
            schema_summary = build_schema_summary(schema_client, top_k=schema_summary_topk)
        except Exception as exc:
            logger.warning("Schema summary query failed: %s", exc)
    if schema_summary:
        for prefix, base in prefix_map.items():
            schema_summary = schema_summary.replace(base, f"{prefix}:")
    logger.info("Schema summary built in %.1fs", time.time() - t_schema)
    allowed_predicates_cfg = list(cfg.get("allowed_predicates") or [])
    allowed_types_cfg = list(cfg.get("allowed_types") or [])
    allowed_predicates_path = cfg.get("allowed_predicates_path", "").strip()
    allowed_types_path = cfg.get("allowed_types_path", "").strip()
    if allowed_predicates_path:
        allowed_predicates_cfg.extend(load_profile_values(allowed_predicates_path, predicate_whitelist_topk))
    if allowed_types_path:
        allowed_types_cfg.extend(load_profile_values(allowed_types_path, predicate_whitelist_topk))
    whitelist = {"predicates": [], "types": []}
    if allowed_predicates_cfg or allowed_types_cfg:
        allowed_predicates = {expand_qname(p, prefix_map) for p in allowed_predicates_cfg}
        allowed_types = list(dict.fromkeys(expand_qname(t, prefix_map) for t in allowed_types_cfg))
        whitelist["predicates"] = sorted(allowed_predicates)
        whitelist["types"] = allowed_types
        logger.info("Using config-provided predicate/type whitelist")
    else:
        logger.info("Building predicate/type whitelist (top_k=%d)...", predicate_whitelist_topk)
        t_white = time.time()
        try:
            whitelist = build_schema_whitelist(schema_client, top_k=predicate_whitelist_topk)
        except Exception as exc:
            logger.warning("Whitelist query failed: %s", exc)
        logger.info("Whitelist built in %.1fs", time.time() - t_white)
        allowed_predicates = set(whitelist.get("predicates", []))
        allowed_types = whitelist.get("types", [])

    whitelist_preds = whitelist.get("predicates") or []
    primary_predicates = whitelist_preds[:schema_summary_topk] if whitelist_preds else []
    use_common_allowed = cfg.get("use_common_allowed", True)
    if allowed_predicates_cfg or allowed_types_cfg:
        use_common_allowed = False
    common_allowed = {
        "http://dbpedia.org/ontology/birthPlace",
        "http://dbpedia.org/ontology/birthDate",
        "http://www.w3.org/2003/01/geo/wgs84_pos#lat",
        "http://www.w3.org/2003/01/geo/wgs84_pos#long",
        "http://www.geonames.org/ontology#countryCode",
        "http://www.geonames.org/ontology#featureClass",
        "http://www.geonames.org/ontology#featureCode",
        "http://www.geonames.org/ontology#parentFeature",
        "http://www.geonames.org/ontology#parentCountry",
        "http://www.geonames.org/ontology#name",
        "http://xmlns.com/foaf/0.1/name",
        "http://xmlns.com/foaf/0.1/givenName",
        "http://xmlns.com/foaf/0.1/surname",
        "http://www.ldbcouncil.org/spb#prefLabel",
        "http://www.ldbcouncil.org/spb#hasRDFRank",
        "http://purl.org/dc/elements/1.1/description",
    }
    allowed_predicates.update(
        {
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://www.w3.org/2000/01/rdf-schema#label",
        }
    )
    if use_common_allowed:
        allowed_predicates.update(common_allowed)
    if primary_predicates:
        extra = []
        if use_common_allowed:
            extra.extend(list(common_allowed))
        extra.extend(
            [
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://www.w3.org/2000/01/rdf-schema#label",
            ]
        )
        primary_predicates = list(dict.fromkeys(primary_predicates + extra))
    else:
        base = []
        if use_common_allowed:
            base.extend(list(common_allowed))
        base.extend(
            [
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://www.w3.org/2000/01/rdf-schema#label",
            ]
        )
        primary_predicates = list(dict.fromkeys(base))
    def is_noisy_type(t: str) -> bool:
        if "owl#" in t or "rdf-schema" in t or "rdf-syntax-ns" in t:
            return True
        if "/class/yago/" in t:
            return True
        if t.endswith("#Class") or t.endswith("#Code"):
            return True
        return False

    allowed_types = [t for t in allowed_types if not is_noisy_type(t)]
    if allowed_types:
        logger.debug("Filtered types count=%d", len(allowed_types))

    preferred_types = []
    preferred_types_cfg = cfg.get("preferred_types") or []
    if preferred_types_cfg:
        preferred_types = [
            expand_qname(t, prefix_map)
            for t in preferred_types_cfg
            if not is_noisy_type(expand_qname(t, prefix_map))
        ]
    elif cfg.get("skip_preferred_types", False):
        logger.info("Skipping live preferred-type query by config")
    else:
        try:
            res = schema_client.query(
                """
SELECT ?type (COUNT(?s) AS ?c)
WHERE { ?s a ?type }
GROUP BY ?type
ORDER BY DESC(?c)
LIMIT 10
"""
            )
            for row in res.rows:
                t = row.get("type")
                if t and not is_noisy_type(t):
                    preferred_types.append(t)
        except Exception as exc:
            logger.debug("Preferred types query failed: %s", exc)
            try:
                res = schema_client.query(
                    """
SELECT DISTINCT ?type
WHERE { ?s a ?type }
LIMIT 10
"""
                )
                for row in res.rows:
                    t = row.get("type")
                    if t and not is_noisy_type(t):
                        preferred_types.append(t)
            except Exception as exc2:
                logger.debug("Preferred types DISTINCT query failed: %s", exc2)
    if allowed_predicates:
        allowed_pred_list = sorted(allowed_predicates)
        if primary_predicates:
            schema_summary += "\nPreferred predicates (use these for templates/queries): " + ", ".join(
                to_prefixed(p, prefix_map) for p in primary_predicates[:30]
            )
        schema_summary += "\nAllowed predicates (use only these when possible): " + ", ".join(
            to_prefixed(p, prefix_map) for p in allowed_pred_list[:80]
        )
    if allowed_types:
        schema_summary += "\nAllowed types: " + ", ".join(
            to_prefixed(t, prefix_map) for t in allowed_types[:50]
        )
    if preferred_types:
        schema_summary += "\nPreferred instance types: " + ", ".join(
            to_prefixed(t, prefix_map) for t in preferred_types[:8]
        )

    def infer_slot_type_hints(types_list: list[str]) -> dict[str, str]:
        prefs = {
            "person": [
                "http://xmlns.com/foaf/0.1/Person",
                "http://dbpedia.org/ontology/Person",
                "http://schema.org/Person",
            ],
            "location": [
                "http://www.geonames.org/ontology#Feature",
                "http://dbpedia.org/ontology/Place",
                "http://schema.org/Place",
            ],
            "company": [
                "http://dbpedia.org/ontology/Company",
                "http://schema.org/Organization",
                "http://xmlns.com/foaf/0.1/Organization",
            ],
            "organization": [
                "http://schema.org/Organization",
                "http://dbpedia.org/ontology/Organisation",
                "http://xmlns.com/foaf/0.1/Organization",
            ],
            "event": [
                "http://dbpedia.org/ontology/Event",
                "http://schema.org/Event",
            ],
            "feature": ["http://www.geonames.org/ontology#Feature"],
        }
        hints = {}
        for slot, candidates in prefs.items():
            chosen = None
            for c in candidates:
                if c in types_list:
                    chosen = c
                    break
            if not chosen and not types_list:
                chosen = candidates[0]
            if chosen:
                hints[slot] = chosen
        return hints

    slot_type_hints = {}
    if cfg.get("slot_type_hints"):
        slot_type_hints = {
            k: expand_qname(v, prefix_map) for k, v in cfg.get("slot_type_hints", {}).items()
        }
    else:
        slot_type_hints = infer_slot_type_hints(allowed_types)
    if slot_type_hints:
        schema_summary += "\nSlot type hints (use for slots): " + ", ".join(
            f"{k}={to_prefixed(v, prefix_map)}" for k, v in slot_type_hints.items()
        )
    if schema_summary:
        (run_dir / "schema_summary.txt").write_text(schema_summary, encoding="utf-8")
    logger.info(
        "Whitelist predicates=%d types=%d (topk=%d)",
        len(allowed_predicates),
        len(allowed_types),
        predicate_whitelist_topk,
    )
    if not allowed_predicates:
        logger.warning("Predicate whitelist is empty; reverse-query filtering disabled.")
    else:
        logger.debug(
            "Allowed predicates sample: %s",
            [to_prefixed(p, prefix_map) for p in sorted(allowed_predicates)[:30]],
        )
    if allowed_types:
        logger.debug(
            "Allowed types sample: %s",
            [to_prefixed(t, prefix_map) for t in allowed_types[:20]],
        )

    pipeline = OllamaPipeline(
        llm,
        sparql,
        prefixes,
        schema_summary,
        slot_type_hints=slot_type_hints,
        logger=logger,
        reverse_query_timeout_sec=reverse_query_timeout_sec,
        generated_query_limit=generated_query_limit,
    )
    avoid_templates = {cat: [] for cat in categories}
    predicate_cache: dict[str, bool] = {}
    reverse_row_cache: dict[str, dict[str, object]] = {}
    template_reverse_cache: dict[tuple[str, tuple[str, ...]], str] = {}
    seen_filled = defaultdict(set)
    phase1_templates_by_cat: dict[str, list[dict[str, object]]] = {}
    phase2_templates_by_cat: dict[str, list[dict[str, object]]] = {}
    phase2_template_seen: dict[str, set[str]] = {cat: set() for cat in categories}
    entity_diversity = EntityDiversityTracker(max_entity_pct=cfg.get("max_entity_pct", 0.15))
    template_structure_tracker: dict[str, set[str]] = {cat: set() for cat in categories}

    def predicate_exists(uri: str) -> bool:
        if uri in predicate_cache:
            return predicate_cache[uri]
        if is_external_uri(uri):
            predicate_cache[uri] = False
            return False
        ask = f"ASK WHERE {{ ?s <{uri}> ?o }}"
        try:
            res = sparql.query(ask)
            predicate_cache[uri] = bool(res.boolean)
            return predicate_cache[uri]
        except Exception as exc:
            logger.debug("Predicate existence check failed for %s: %s", uri, exc)
            predicate_cache[uri] = False
            return False

    def validate_predicates(preds: list[str], phase: str, cat: str, sparql_text: str) -> bool:
        if not allowed_predicates or not preds:
            return True
        unknown = [p for p in preds if p not in allowed_predicates]
        if not unknown:
            return True
        newly_allowed = []
        still_unknown = []
        for p in unknown:
            if predicate_exists(p):
                allowed_predicates.add(p)
                newly_allowed.append(p)
            else:
                still_unknown.append(p)
        if newly_allowed:
            logger.debug("Predicate validated via ASK (%s:%s): %s", phase, cat, newly_allowed)
        if still_unknown:
            logger.warning(
                "Reverse query uses predicates not in graph (%s:%s): unknown=%s all=%s sparql=%s",
                phase,
                cat,
                still_unknown,
                preds,
                sparql_text,
            )
            return False
        return True

    def validate_types(types: list[str], phase: str, cat: str, sparql_text: str) -> bool:
        if not allowed_types or not types:
            return True
        unknown = [t for t in types if t not in allowed_types]
        if not unknown:
            return True
        logger.warning(
            "Reverse query uses types not in schema (phase=%s category=%s): unknown=%s sparql=%s",
            phase,
            cat,
            unknown,
            sparql_text,
        )
        return False

    def binding_bank_file(cat: str, template: str, reverse_sparql: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", template).strip("_").lower()[:48] or "template"
        digest = hashlib.sha1(f"{cat}\n{template}\n{reverse_sparql}".encode("utf-8")).hexdigest()[:12]
        return binding_bank_dir / f"{cat}_{slug}_{digest}.jsonl"

    def collect_binding_iris(rows: list[dict[str, str]], slots: list[str]) -> list[str]:
        iris = []
        for row in rows:
            for slot in slots:
                value = row.get(slot)
                if isinstance(value, str) and is_iri(value):
                    iris.append(value)
        return list(dict.fromkeys(iris))

    def write_binding_bank(path: Path, rows: list[dict[str, str]], metadata: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def render_priority_reverse_query(category: str, template: str, slots: list[str]) -> str | None:
        lower = template.lower()
        if category == "generic" and slots == ["company"]:
            predicate = None
            if "located" in lower:
                predicate = "dbo:location ?location"
            elif "industry" in lower:
                predicate = "dbo:industry ?industry"
            elif "key person" in lower:
                predicate = "dbo:keyPerson ?person"
            elif "founding year" in lower:
                predicate = "dbo:foundingYear ?foundingYear"
            elif "employees" in lower:
                predicate = "dbo:numberOfEmployees ?numberOfEmployees"
            if predicate:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    f"  ?company {predicate} .",
                    "}",
                    "LIMIT 25",
                ])
        if category == "counting":
            if "companies are in {industry}" in lower and slots == ["industry"]:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?industry",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:industry ?industry .",
                    "}",
                    "LIMIT 25",
                ])
            if "companies are located in {location}" in lower and slots == ["location"]:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?location",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:location ?location .",
                    "}",
                    "LIMIT 25",
                ])
            if "companies have key person {person}" in lower and slots == ["person"]:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?person",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:keyPerson ?person .",
                    "}",
                    "LIMIT 25",
                ])
            if "key people are associated with {company}" in lower and slots == ["company"]:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:keyPerson ?person .",
                    "}",
                    "LIMIT 25",
                ])
            if "companies were founded in {year}" in lower and slots == ["year"]:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?year",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:foundingYear ?year .",
                    "}",
                    "LIMIT 25",
                ])

        if category == "comparative" and slots == ["company1", "company2"]:
            if "same industry" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company1 ?company2",
                    "WHERE {",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:industry ?industry .",
                    "  ?company2 dbo:industry ?industry .",
                    "  FILTER(?company1 != ?company2)",
                    "}",
                    "LIMIT 25",
                ])
            if "same location" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company1 ?company2",
                    "WHERE {",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:location ?location .",
                    "  ?company2 dbo:location ?location .",
                    "  FILTER(?company1 != ?company2)",
                    "}",
                    "LIMIT 25",
                ])
            if "same year" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company1 ?company2",
                    "WHERE {",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:foundingYear ?year .",
                    "  ?company2 dbo:foundingYear ?year .",
                    "  FILTER(?company1 != ?company2)",
                    "}",
                    "LIMIT 25",
                ])
            if "key person" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company1 ?company2",
                    "WHERE {",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:keyPerson ?person .",
                    "  ?company2 dbo:keyPerson ?person .",
                    "  FILTER(?company1 != ?company2)",
                    "}",
                    "LIMIT 25",
                ])
            if "employee count" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company1 ?company2",
                    "WHERE {",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:numberOfEmployees ?count .",
                    "  ?company2 dbo:numberOfEmployees ?count .",
                    "  FILTER(?company1 != ?company2)",
                    "}",
                    "LIMIT 25",
                ])

        if category == "difference" and slots == ["company1", "company2"]:
            if "different industries" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company1 ?company2",
                    "WHERE {",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:industry ?industry1 .",
                    "  ?company2 dbo:industry ?industry2 .",
                    "  FILTER(?company1 != ?company2 && ?industry1 != ?industry2)",
                    "}",
                    "LIMIT 25",
                ])
            if "different locations" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company1 ?company2",
                    "WHERE {",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:location ?location1 .",
                    "  ?company2 dbo:location ?location2 .",
                    "  FILTER(?company1 != ?company2 && ?location1 != ?location2)",
                    "}",
                    "LIMIT 25",
                ])
            if "different years" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company1 ?company2",
                    "WHERE {",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:foundingYear ?year1 .",
                    "  ?company2 dbo:foundingYear ?year2 .",
                    "  FILTER(?company1 != ?company2 && ?year1 != ?year2)",
                    "}",
                    "LIMIT 25",
                ])

        if category in {"multi-hop", "intersection"}:
            if set(slots) == {"industry", "location"}:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?industry ?location",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:industry ?industry .",
                    "  ?company dbo:location ?location .",
                    "}",
                    "LIMIT 25",
                ])
            if set(slots) == {"industry", "person"}:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?industry ?person",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:industry ?industry .",
                    "  ?company dbo:keyPerson ?person .",
                    "}",
                    "LIMIT 25",
                ])
            if set(slots) == {"person", "location"}:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?person ?location",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:keyPerson ?person .",
                    "  ?company dbo:location ?location .",
                    "}",
                    "LIMIT 25",
                ])

        if category == "superlative":
            if slots == ["industry"] and "{industry}" in lower:
                metric = "dbo:keyPerson ?person" if "key people" in lower else (
                    "dbo:foundingYear ?foundingYear" if "founded" in lower else "dbo:numberOfEmployees ?numberOfEmployees"
                )
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?industry",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:industry ?industry .",
                    f"  ?company {metric} .",
                    "}",
                    "LIMIT 25",
                ])
            if slots == ["location"] and "{location}" in lower:
                metric = "dbo:keyPerson ?person" if "key people" in lower else (
                    "dbo:foundingYear ?foundingYear" if "founded" in lower else "dbo:numberOfEmployees ?numberOfEmployees"
                )
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?location",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:location ?location .",
                    f"  ?company {metric} .",
                    "}",
                    "LIMIT 25",
                ])
        if category == "ordinal":
            rank_values = '"second" "third" "fourth" "fifth"'
            if slots == ["rank"]:
                metric = "dbo:foundingYear ?foundingYear" if "founded" in lower else "dbo:numberOfEmployees ?numberOfEmployees"
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?rank",
                    "WHERE {",
                    f"  VALUES ?rank {{ {rank_values} }}",
                    "  ?company rdf:type dbo:Company .",
                    f"  ?company {metric} .",
                    "}",
                    "LIMIT 25",
                ])
            if set(slots) == {"industry", "rank"} and "{industry}" in lower:
                metric = "dbo:foundingYear ?foundingYear" if "founded" in lower else "dbo:numberOfEmployees ?numberOfEmployees"
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?industry ?rank",
                    "WHERE {",
                    f"  VALUES ?rank {{ {rank_values} }}",
                    "  {",
                    "    SELECT ?industry",
                    "    WHERE {",
                    "      ?company rdf:type dbo:Company .",
                    "      ?company dbo:industry ?industry .",
                    f"      ?company {metric} .",
                    "    }",
                    "    GROUP BY ?industry",
                    "    HAVING (COUNT(DISTINCT ?company) >= 5)",
                    "  }",
                    "}",
                    "LIMIT 25",
                ])
            if set(slots) == {"location", "rank"} and "{location}" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?location ?rank",
                    "WHERE {",
                    f"  VALUES ?rank {{ {rank_values} }}",
                    "  {",
                    "    SELECT ?location",
                    "    WHERE {",
                    "      ?company rdf:type dbo:Company .",
                    "      ?company dbo:location ?location .",
                    "      ?company dbo:numberOfEmployees ?numberOfEmployees .",
                    "    }",
                    "    GROUP BY ?location",
                    "    HAVING (COUNT(DISTINCT ?company) >= 5)",
                    "  }",
                    "}",
                    "LIMIT 25",
                ])
        if category == "yesno":
            if set(slots) == {"company", "location"} and "located" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company ?location",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:location ?location .",
                    "}",
                    "LIMIT 25",
                ])
            if set(slots) == {"company", "industry"} and "industry" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company ?industry",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:industry ?industry .",
                    "}",
                    "LIMIT 25",
                ])
            if set(slots) == {"person", "company"} and "key person" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?person ?company",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:keyPerson ?person .",
                    "}",
                    "LIMIT 25",
                ])
        return None

    def build_binding_cache(
        reverse_sparql: str,
        slots: list[str],
        phase: str,
        cat: str,
        template: str,
        extend: bool = False,
    ) -> dict[str, object]:
        cache = reverse_row_cache.get(reverse_sparql)
        rows = list(cache.get("rows", [])) if cache else []
        used = cache.get("used", set()) if cache else set()
        offsets_done = set(cache.get("offsets_done", [])) if cache else set()
        offset_stride = max(binding_bank_offset_stride, binding_bank_query_limit, 1)
        start_offset = (
            int(hashlib.sha1(reverse_sparql.encode("utf-8")).hexdigest()[:8], 16)
            % offset_stride
        )
        next_offset = int(cache.get("next_offset", start_offset + offset_stride)) if cache else start_offset + offset_stride
        exhausted = bool(cache.get("exhausted", False)) if cache else False
        bank_path = Path(cache.get("bank_path")) if cache and cache.get("bank_path") else binding_bank_file(cat, template, reverse_sparql)

        if cache and not extend:
            return cache
        if exhausted and extend and 0 in offsets_done:
            return cache or {"rows": rows, "used": used, "offsets_done": offsets_done, "next_offset": next_offset, "exhausted": True, "bank_path": str(bank_path)}

        max_queries = binding_bank_extend_queries if extend else binding_bank_max_queries
        target_rows = len(rows) + binding_bank_query_limit if extend else binding_bank_target_rows
        queries_run = 0
        seen_keys = {row_key(row, slots) for row in rows}

        def pick_binding_offset() -> int:
            nonlocal next_offset
            if 0 not in offsets_done:
                return 0
            if start_offset not in offsets_done:
                return start_offset
            while next_offset in offsets_done:
                next_offset += offset_stride
            chosen = next_offset
            next_offset += offset_stride
            return chosen

        while len(rows) < target_rows and queries_run < max_queries and not exhausted:
            current_offset = pick_binding_offset()
            query = bounded_reverse_query(reverse_sparql, binding_bank_query_limit, current_offset)
            offsets_done.add(current_offset)
            queries_run += 1
            ok, _, fetched_rows, err = pipeline.execute_rows(query)
            if not ok:
                logger.warning(
                    "Binding-bank query failed (%s:%s): %s | template=%s",
                    phase,
                    cat,
                    err,
                    template,
                )
                break
            added = 0
            for row in fetched_rows:
                key = row_key(row, slots)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                rows.append(row)
                added += 1
            logger.debug(
                "Binding-bank query (%s:%s) offset=%d fetched=%d added=%d total=%d template=%s",
                phase,
                cat,
                current_offset,
                len(fetched_rows),
                added,
                len(rows),
                template,
            )
            if current_offset == 0 and len(fetched_rows) < binding_bank_query_limit:
                exhausted = True

        if rows:
            random.shuffle(rows)
            pipeline.prime_entity_metadata(collect_binding_iris(rows, slots), batch_size=binding_bank_batch_size)

        metadata = {
            "category": cat,
            "template": template,
            "slots": slots,
            "reverse_sparql": reverse_sparql,
            "rows": len(rows),
            "offsets_done": sorted(offsets_done),
            "next_offset": next_offset,
            "exhausted": exhausted,
        }
        write_binding_bank(bank_path, rows, metadata)
        cache = {
            "rows": rows,
            "used": used,
            "offsets_done": offsets_done,
            "next_offset": next_offset,
            "exhausted": exhausted,
            "bank_path": str(bank_path),
        }
        reverse_row_cache[reverse_sparql] = cache
        log_row_sample(rows, phase, cat, logger=logger)
        logger.info(
            "Binding bank ready (%s:%s): rows=%d path=%s",
            phase,
            cat,
            len(rows),
            bank_path,
        )
        return cache

    def sparql_iri(value: str) -> str:
        value = urllib.parse.unquote(str(value))
        return value if value.startswith("<") else f"<{value}>"

    def sparql_string(value: str) -> str:
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def rank_offset(value: str | None) -> int | None:
        if value is None:
            return None
        text = str(value).strip().strip('"').lower()
        mapping = {
            "first": 0,
            "1": 0,
            "second": 1,
            "2": 1,
            "third": 2,
            "3": 2,
            "fourth": 3,
            "4": 3,
            "fifth": 4,
            "5": 4,
        }
        return mapping.get(text)

    def scoped_company_patterns(slot: str | None, value: str | None) -> list[str]:
        lines = ["  ?company rdf:type dbo:Company ."]
        if slot == "industry" and value:
            lines.insert(0, f"  VALUES ?industry {{ {sparql_iri(value)} }}")
            lines.append("  ?company dbo:industry ?industry .")
        elif slot == "location" and value:
            lines.insert(0, f"  VALUES ?location {{ {sparql_iri(value)} }}")
            lines.append("  ?company dbo:location ?location .")
        return lines

    def render_priority_sparql(category: str, template: str, row: dict[str, str] | None) -> str | None:
        lower = template.lower()
        row = row or {}
        lines: list[str]

        if category == "generic" and row.get("company"):
            if "located" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?location",
                    "WHERE {",
                    f"  VALUES ?company {{ {sparql_iri(row['company'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:location ?location .",
                    "}",
                    "LIMIT 5",
                ])
            if "industry" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?industry",
                    "WHERE {",
                    f"  VALUES ?company {{ {sparql_iri(row['company'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:industry ?industry .",
                    "}",
                    "LIMIT 5",
                ])
            if "key person" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?person",
                    "WHERE {",
                    f"  VALUES ?company {{ {sparql_iri(row['company'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:keyPerson ?person .",
                    "}",
                    "LIMIT 5",
                ])
            if "founding year" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?foundingYear",
                    "WHERE {",
                    f"  VALUES ?company {{ {sparql_iri(row['company'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:foundingYear ?foundingYear .",
                    "}",
                    "LIMIT 5",
                ])
            if "employees" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?numberOfEmployees",
                    "WHERE {",
                    f"  VALUES ?company {{ {sparql_iri(row['company'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:numberOfEmployees ?numberOfEmployees .",
                    "}",
                    "LIMIT 5",
                ])

        if category == "counting":
            if "companies are in {industry}" in lower and row.get("industry"):
                lines = scoped_company_patterns("industry", row["industry"])
                return prefixes + "\n" + "\n".join([
                    "SELECT (COUNT(DISTINCT ?company) AS ?count)",
                    "WHERE {",
                    *lines,
                    "}",
                    "LIMIT 5",
                ])
            if "companies are located in {location}" in lower and row.get("location"):
                lines = scoped_company_patterns("location", row["location"])
                return prefixes + "\n" + "\n".join([
                    "SELECT (COUNT(DISTINCT ?company) AS ?count)",
                    "WHERE {",
                    *lines,
                    "}",
                    "LIMIT 5",
                ])
            if "companies have key person {person}" in lower and row.get("person"):
                return prefixes + "\n" + "\n".join([
                    "SELECT (COUNT(DISTINCT ?company) AS ?count)",
                    "WHERE {",
                    f"  VALUES ?person {{ {sparql_iri(row['person'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:keyPerson ?person .",
                    "}",
                    "LIMIT 5",
                ])
            if "key people are associated with {company}" in lower and row.get("company"):
                return prefixes + "\n" + "\n".join([
                    "SELECT (COUNT(DISTINCT ?person) AS ?count)",
                    "WHERE {",
                    f"  VALUES ?company {{ {sparql_iri(row['company'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:keyPerson ?person .",
                    "}",
                    "LIMIT 5",
                ])
            if "companies were founded in {year}" in lower and row.get("year"):
                return prefixes + "\n" + "\n".join([
                    "SELECT (COUNT(DISTINCT ?company) AS ?count)",
                    "WHERE {",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:foundingYear ?year .",
                    f"  FILTER(STR(?year) = {sparql_string(row['year'])})",
                    "}",
                    "LIMIT 5",
                ])
            return None

        if category == "comparative" and row.get("company1") and row.get("company2"):
            if "same industry" in lower:
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company1 {{ {sparql_iri(row['company1'])} }}",
                    f"  VALUES ?company2 {{ {sparql_iri(row['company2'])} }}",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:industry ?industry .",
                    "  ?company2 dbo:industry ?industry .",
                    "  FILTER(?company1 != ?company2)",
                    "}",
                ])
            if "same location" in lower:
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company1 {{ {sparql_iri(row['company1'])} }}",
                    f"  VALUES ?company2 {{ {sparql_iri(row['company2'])} }}",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:location ?location .",
                    "  ?company2 dbo:location ?location .",
                    "  FILTER(?company1 != ?company2)",
                    "}",
                ])
            if "same founding year" in lower or "founded in the same year" in lower:
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company1 {{ {sparql_iri(row['company1'])} }}",
                    f"  VALUES ?company2 {{ {sparql_iri(row['company2'])} }}",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:foundingYear ?year .",
                    "  ?company2 dbo:foundingYear ?year .",
                    "  FILTER(?company1 != ?company2)",
                    "}",
                ])
            if "same key person" in lower or "share a key person" in lower:
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company1 {{ {sparql_iri(row['company1'])} }}",
                    f"  VALUES ?company2 {{ {sparql_iri(row['company2'])} }}",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:keyPerson ?person .",
                    "  ?company2 dbo:keyPerson ?person .",
                    "  FILTER(?company1 != ?company2)",
                    "}",
                ])
            if "same employee count" in lower or "same number of employees" in lower:
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company1 {{ {sparql_iri(row['company1'])} }}",
                    f"  VALUES ?company2 {{ {sparql_iri(row['company2'])} }}",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:numberOfEmployees ?count .",
                    "  ?company2 dbo:numberOfEmployees ?count .",
                    "  FILTER(?company1 != ?company2)",
                    "}",
                ])
            return None

        if category == "difference" and row.get("company1") and row.get("company2"):
            if "different industries" in lower:
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company1 {{ {sparql_iri(row['company1'])} }}",
                    f"  VALUES ?company2 {{ {sparql_iri(row['company2'])} }}",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:industry ?industry1 .",
                    "  ?company2 dbo:industry ?industry2 .",
                    "  FILTER(?company1 != ?company2 && ?industry1 != ?industry2)",
                    "}",
                ])
            if "different locations" in lower:
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company1 {{ {sparql_iri(row['company1'])} }}",
                    f"  VALUES ?company2 {{ {sparql_iri(row['company2'])} }}",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:location ?location1 .",
                    "  ?company2 dbo:location ?location2 .",
                    "  FILTER(?company1 != ?company2 && ?location1 != ?location2)",
                    "}",
                ])
            if "different founding years" in lower or "founded in different years" in lower:
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company1 {{ {sparql_iri(row['company1'])} }}",
                    f"  VALUES ?company2 {{ {sparql_iri(row['company2'])} }}",
                    "  ?company1 rdf:type dbo:Company .",
                    "  ?company2 rdf:type dbo:Company .",
                    "  ?company1 dbo:foundingYear ?year1 .",
                    "  ?company2 dbo:foundingYear ?year2 .",
                    "  FILTER(?company1 != ?company2 && ?year1 != ?year2)",
                    "}",
                ])
            return None

        if category in {"multi-hop", "intersection"}:
            if row.get("industry") and row.get("location"):
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company",
                    "WHERE {",
                    f"  VALUES ?industry {{ {sparql_iri(row['industry'])} }}",
                    f"  VALUES ?location {{ {sparql_iri(row['location'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:industry ?industry .",
                    "  ?company dbo:location ?location .",
                    "}",
                    "LIMIT 5",
                ])
            if row.get("industry") and row.get("person"):
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company",
                    "WHERE {",
                    f"  VALUES ?industry {{ {sparql_iri(row['industry'])} }}",
                    f"  VALUES ?person {{ {sparql_iri(row['person'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:industry ?industry .",
                    "  ?company dbo:keyPerson ?person .",
                    "}",
                    "LIMIT 5",
                ])
            if row.get("person") and row.get("location"):
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company",
                    "WHERE {",
                    f"  VALUES ?person {{ {sparql_iri(row['person'])} }}",
                    f"  VALUES ?location {{ {sparql_iri(row['location'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:keyPerson ?person .",
                    "  ?company dbo:location ?location .",
                    "}",
                    "LIMIT 5",
                ])
            return None

        if category == "superlative":
            scope_slot = None
            scope_value = None
            if "{industry}" in lower and row.get("industry"):
                scope_slot = "industry"
                scope_value = row["industry"]
            elif "{location}" in lower and row.get("location"):
                scope_slot = "location"
                scope_value = row["location"]

            company_lines = scoped_company_patterns(scope_slot, scope_value)
            if "employee" in lower:
                order = "ASC" if any(word in lower for word in ("fewest", "least", "lowest")) else "DESC"
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company ?numberOfEmployees",
                    "WHERE {",
                    *company_lines,
                    "  ?company dbo:numberOfEmployees ?numberOfEmployees .",
                    "}",
                    f"ORDER BY {order}(?numberOfEmployees) ?company",
                    "LIMIT 1",
                ])
            if "founded" in lower:
                order = "ASC" if "earliest" in lower else "DESC"
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company ?foundingYear",
                    "WHERE {",
                    *company_lines,
                    "  ?company dbo:foundingYear ?foundingYear .",
                    "}",
                    f"ORDER BY {order}(?foundingYear) ?company",
                    "LIMIT 1",
                ])
            if "key people" in lower:
                return prefixes + "\n" + "\n".join([
                    "SELECT ?company (COUNT(DISTINCT ?person) AS ?count)",
                    "WHERE {",
                    *company_lines,
                    "  ?company dbo:keyPerson ?person .",
                    "}",
                    "GROUP BY ?company",
                    "ORDER BY DESC(?count) ?company",
                    "LIMIT 1",
                ])

        if category == "ordinal" and row.get("rank"):
            offset = rank_offset(row.get("rank"))
            if offset is None:
                return None
            scope_slot = None
            scope_value = None
            if "{industry}" in lower and row.get("industry"):
                scope_slot = "industry"
                scope_value = row["industry"]
            elif "{location}" in lower and row.get("location"):
                scope_slot = "location"
                scope_value = row["location"]
            company_lines = scoped_company_patterns(scope_slot, scope_value)
            if "employee" in lower:
                order = "ASC" if any(word in lower for word in ("fewest", "least", "lowest")) else "DESC"
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company ?numberOfEmployees",
                    "WHERE {",
                    *company_lines,
                    "  ?company dbo:numberOfEmployees ?numberOfEmployees .",
                    "}",
                    f"ORDER BY {order}(?numberOfEmployees) ?company",
                    f"LIMIT 1 OFFSET {offset}",
                ])
            if "founded" in lower:
                order = "DESC" if "most recently" in lower else "ASC"
                return prefixes + "\n" + "\n".join([
                    "SELECT DISTINCT ?company ?foundingYear",
                    "WHERE {",
                    *company_lines,
                    "  ?company dbo:foundingYear ?foundingYear .",
                    "}",
                    f"ORDER BY {order}(?foundingYear) ?company",
                    f"LIMIT 1 OFFSET {offset}",
                ])

        if category == "yesno":
            if row.get("company") and row.get("location") and ("located" in lower or "based" in lower):
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company {{ {sparql_iri(row['company'])} }}",
                    f"  VALUES ?location {{ {sparql_iri(row['location'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:location ?location .",
                    "}",
                ])
            if row.get("company") and row.get("industry") and "industry" in lower:
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company {{ {sparql_iri(row['company'])} }}",
                    f"  VALUES ?industry {{ {sparql_iri(row['industry'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:industry ?industry .",
                    "}",
                ])
            if row.get("company") and row.get("person") and "key person" in lower:
                return prefixes + "\n" + "\n".join([
                    "ASK WHERE {",
                    f"  VALUES ?company {{ {sparql_iri(row['company'])} }}",
                    f"  VALUES ?person {{ {sparql_iri(row['person'])} }}",
                    "  ?company rdf:type dbo:Company .",
                    "  ?company dbo:keyPerson ?person .",
                    "}",
                ])
        return None

    def run_known_sparql(
        question: str,
        category: str,
        template: str | None,
        row: dict[str, str] | None,
        examples: list[dict[str, str]] | None,
    ) -> GenerationRecord | None:
        if not template:
            return None
        sparql_text = render_priority_sparql(category, template, row)
        if not sparql_text:
            return None
        parse_ok, parse_err = parse_valid_sparql_detail(sparql_text)
        ast_nodes = 0
        ast_depth = 0
        if parse_ok:
            try:
                ast_nodes, ast_depth, _ = ast_stats(sparql_text)
            except Exception:
                ast_nodes, ast_depth = 0, 0
        exec_ok, exec_ms, answers, error = pipeline.execute(sparql_text) if parse_ok else (False, 0.0, [], None)
        error_type = None
        if not parse_ok:
            error = f"parse_error: {parse_err}"
            error_type = "parse_error"
        elif not exec_ok:
            error_type = "endpoint_error"
        elif not answers:
            error_type = "empty_result"
        elif category == "counting" and answers[0] in {"0", "0.0"}:
            error = "grounded_count_returned_zero"
            error_type = "semantic_empty_count"
            exec_ok = False
        elif not validate_answer_type(category, answers, sparql_text):
            error = "answer_type_mismatch"
            error_type = "answer_type_mismatch"
        if error_type:
            logger.debug(
                "Known-template SPARQL rejected | category=%s question=%s error_type=%s",
                category,
                question,
                error_type,
            )
        return GenerationRecord(
            category=category,
            question=question,
            sparql=sparql_text,
            answers=answers,
            exec_success=exec_ok,
            parse_valid=parse_ok,
            error=error,
            error_type=error_type,
            repair_attempts=0,
            llm_latency_ms=0.0,
            question_latency_ms=0.0,
            sparql_exec_ms=exec_ms,
            answer_count=len(answers),
            result_hash=result_set_hash(answers) if answers else None,
            prompt_chars=0,
            prompt_tokens_est=0,
            retrieved_examples=examples or [],
            ast_node_count=ast_nodes,
            ast_max_depth=ast_depth,
        )

    # Phase 1: template generation + reverse querying (seeds for retrieval)
    logger.info("Phase 1: template generation + reverse querying")
    phase1_records = []
    def new_proposal_bucket() -> dict[str, int | bool]:
        return {
            "target": 0,
            "max_outer_attempts": 0,
            "outer_attempts": 0,
            "template_items_seen": 0,
            "reverse_query_failures": 0,
            "reverse_query_parse_failures": 0,
            "reverse_query_form_rejections": 0,
            "predicate_type_rejections": 0,
            "empty_binding_caches": 0,
            "no_valid_binding_rows": 0,
            "pre_llm_duplicates": 0,
            "candidate_attempts": 0,
            "validation_failures": 0,
            "duplicate_drops": 0,
            "accepted": 0,
            "exhausted": False,
        }

    proposal_stats = {
        "phase1": defaultdict(new_proposal_bucket),
        "phase2": defaultdict(new_proposal_bucket),
        "phase3": defaultdict(new_proposal_bucket),
    }
    for cat_idx, cat in enumerate(categories, start=1):
        logger.info("Phase 1 category: %s (%d/%d) | overall=%.1f%%", cat, cat_idx, len(categories), overall_pct())
        target_phase1 = phase1_templates_per_category * phase1_seeds_per_template
        stats = proposal_stats["phase1"][cat]
        stats["target"] = target_phase1
        accepted = 0
        try:
            templates = pipeline.generate_templates(
                cat,
                n=max(phase1_templates_per_category, template_candidates),
                avoid_templates=avoid_templates.get(cat, [])[-10:],
            )
        except ValueError as exc:
            logger.warning("Phase 1 template generation skipped for category=%s: %s", cat, exc)
            continue
        templates = merge_priority_templates(cat, dedupe_templates(templates))
        if not templates:
            logger.warning("No templates generated in Phase 1 for category=%s", cat)
            continue
        phase1_templates_by_cat[cat] = templates
        for item in templates:
            if accepted >= target_phase1:
                break
            template = item.get("template", "")
            slots = item.get("slots", [])
            if not template:
                continue
            stats["template_items_seen"] += 1
            avoid_templates[cat].append(template)
            if slots:
                cache_key = (template, tuple(slots))
                reverse_sparql = template_reverse_cache.get(cache_key)
                if reverse_sparql is None:
                    reverse_sparql = render_priority_reverse_query(cat, template, slots)
                    if reverse_sparql is None:
                        try:
                            reverse_sparql = pipeline.reverse_query(template, slots)
                        except Exception as exc:
                            logger.warning("Reverse query failed in Phase 1 for category=%s: %s", cat, exc)
                            stats["reverse_query_failures"] += 1
                            continue
                    template_reverse_cache[cache_key] = reverse_sparql
                ok_parse, parse_err = parse_valid_sparql_detail(reverse_sparql)
                if not ok_parse:
                    stats["reverse_query_parse_failures"] += 1
                    logger.warning(
                        "Reverse query parse failed (phase1:%s): %s | sparql=%s",
                        cat,
                        parse_err,
                        reverse_sparql,
                    )
                if not is_simple_reverse_query(reverse_sparql, cat):
                    logger.warning("Reverse query not simple (phase1:%s): sparql=%s", cat, reverse_sparql)
                    stats["reverse_query_form_rejections"] += 1
                    continue
                preds = extract_predicates(reverse_sparql, prefix_map)
                if not validate_predicates(preds, "phase1", cat, reverse_sparql):
                    stats["predicate_type_rejections"] += 1
                    continue
                types = extract_types(reverse_sparql, prefix_map)
                if not validate_types(types, "phase1", cat, reverse_sparql):
                    stats["predicate_type_rejections"] += 1
                    continue
                if not select_vars_cover_slots(reverse_sparql, slots):
                    logger.warning("Reverse query missing slot vars (phase1:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                    stats["reverse_query_form_rejections"] += 1
                    continue
                if not body_contains_slots(reverse_sparql, slots):
                    logger.warning("Reverse query missing slot vars in body (phase1:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                    stats["reverse_query_form_rejections"] += 1
                    continue
                cache = build_binding_cache(reverse_sparql, slots, "phase1", cat, template)
                rows = cache["rows"]
                if not rows:
                    logger.debug("Reverse query cache empty (phase1:%s): sparql=%s", cat, reverse_sparql)
                    stats["empty_binding_caches"] += 1
                    continue
                for _ in range(phase1_seeds_per_template):
                    row = select_valid_row(
                        rows,
                        slots,
                        pipeline,
                        max_row_checks,
                        logger=logger,
                        used_keys=cache["used"],
                    )
                    if not row:
                        stats["no_valid_binding_rows"] += 1
                        break
                    entity_hints = {}
                    filled = template
                    for slot in slots:
                        val = row.get(slot)
                        if not val:
                            continue
                        label = pipeline.label_for_uri(val)
                        type_uri = slot_type_hints.get(re.sub(r"\d+$", "", slot)) or pipeline.entity_type_for_uri(val)
                        type_label = pipeline.label_for_type(type_uri) if type_uri else None
                        if type_label:
                            filled = filled.replace("{" + slot + "}", f"({type_label}: {label})")
                            entity_hints[slot] = f"{val} | type={type_uri}"
                        else:
                            filled = filled.replace("{" + slot + "}", label)
                            entity_hints[slot] = val
                    logger.debug("Filled question (phase1:%s cat=%d/%d): %s", cat, cat_idx, len(categories), filled)
                    stats["candidate_attempts"] += 1
                    rec = run_known_sparql(filled, cat, template, row, examples=None)
                    if rec is None:
                        rec = pipeline.run_single(filled, cat, entity_hints=entity_hints, repair_attempts=repair_attempts)
                    if (not rec.exec_success) or rec.error_type:
                        stats["validation_failures"] += 1
                        append_record_jsonl(str(log_path), rec)
                        continue
                    phase1_records.append(rec)
                    overall_done += 1
                    accepted += 1
                    append_record_jsonl(str(log_path), rec)
                    if accepted >= target_phase1:
                        break
            else:
                for _ in range(min(phase1_seeds_per_template, 1)):
                    logger.debug("Filled question (phase1:%s cat=%d/%d): %s", cat, cat_idx, len(categories), template)
                    stats["candidate_attempts"] += 1
                    rec = run_known_sparql(template, cat, template, row=None, examples=None)
                    if rec is None:
                        rec = pipeline.run_single(template, cat, repair_attempts=repair_attempts)
                    if (not rec.exec_success) or rec.error_type:
                        stats["validation_failures"] += 1
                        append_record_jsonl(str(log_path), rec)
                        continue
                    phase1_records.append(rec)
                    overall_done += 1
                    accepted += 1
                    append_record_jsonl(str(log_path), rec)
                    if accepted >= target_phase1:
                        break
        stats["accepted"] = accepted
        stats["exhausted"] = accepted < target_phase1
        logger.info("Phase 1 category complete: %s accepted=%d/%d", cat, accepted, target_phase1)

    logger.info("Phase 1 complete: %d records", len(phase1_records))

    # Build global FAISS index from Phase 1 seeds
    logger.info("Building FAISS index from Phase 1 seeds...")
    seed_examples = [
        {"question": r.question, "sparql": r.sparql, "category": r.category}
        for r in phase1_records
        if r.exec_success or r.parse_valid
    ]
    if not seed_examples:
        logger.warning("Phase 1 produced no usable seeds; proceeding without global retrieval.")
        global_store = None
    else:
        global_store = build_faiss_index(llm, seed_examples)
    logger.info("Global FAISS index ready: %d seed examples", len(seed_examples))

    # Phase 2: category-wise seed generation (uses global retrieval)
    logger.info("Phase 2: category-wise seed generation")
    for cache in reverse_row_cache.values():
        cache["used"] = set()
    phase2_records = []
    category_stores = {cat: None for cat in categories}
    for cat_idx, cat in enumerate(categories, start=1):
        logger.info("Phase 2 category: %s (%d/%d) | overall=%.1f%%", cat, cat_idx, len(categories), overall_pct())
        accepted = 0
        attempts = 0
        max_attempts = phase2_seeds_per_category * max_attempts_factor
        stats = proposal_stats["phase2"][cat]
        stats["target"] = phase2_seeds_per_category
        stats["max_outer_attempts"] = max_attempts
        stall_counts: dict[str, int] = {}
        stall_limit = max(3, template_candidates * 3)
        dead_templates: set[str] = set()
        refreshed_templates: set[str] = set()
        phase2_templates_by_cat[cat] = []
        while accepted < phase2_seeds_per_category and attempts < max_attempts:
            attempts += 1
            stats["outer_attempts"] = attempts
            t0 = time.time()
            try:
                templates = pipeline.generate_templates(
                    cat,
                    n=max(1, template_candidates),
                    avoid_templates=avoid_templates.get(cat, [])[-10:],
                )
            except ValueError as exc:
                logger.warning("Phase 2 template generation failed for category=%s: %s", cat, exc)
                continue
            templates = merge_priority_templates(cat, dedupe_templates(templates))
            if not templates:
                logger.warning("No templates generated in Phase 2 for category=%s", cat)
                continue
            q_latency = (time.time() - t0) * 1000
            filled = None
            entity_hints = {}
            selected_template = None
            selected_slots = None
            selected_row = None
            for tmpl in templates:
                template = tmpl.get("template", "")
                slots = tmpl.get("slots", [])
                if not template:
                    continue
                stats["template_items_seen"] += 1
                if template in dead_templates:
                    continue
                avoid_templates[cat].append(template)
                if slots:
                    cache_key = (template, tuple(slots))
                    reverse_sparql = template_reverse_cache.get(cache_key)
                    if reverse_sparql is None:
                        reverse_sparql = render_priority_reverse_query(cat, template, slots)
                        if reverse_sparql is None:
                            try:
                                reverse_sparql = pipeline.reverse_query(template, slots)
                            except Exception as exc:
                                logger.warning("Reverse query failed in Phase 2 for category=%s: %s", cat, exc)
                                stats["reverse_query_failures"] += 1
                                continue
                        template_reverse_cache[cache_key] = reverse_sparql
                    ok_parse, parse_err = parse_valid_sparql_detail(reverse_sparql)
                    if not ok_parse:
                        stats["reverse_query_parse_failures"] += 1
                        logger.warning(
                            "Reverse query parse failed (phase2:%s): %s | sparql=%s",
                            cat,
                            parse_err,
                            reverse_sparql,
                        )
                    if not is_simple_reverse_query(reverse_sparql, cat):
                        logger.warning("Reverse query not simple (phase2:%s): sparql=%s", cat, reverse_sparql)
                        stats["reverse_query_form_rejections"] += 1
                        continue
                    preds = extract_predicates(reverse_sparql, prefix_map)
                    if not validate_predicates(preds, "phase2", cat, reverse_sparql):
                        stats["predicate_type_rejections"] += 1
                        continue
                    types = extract_types(reverse_sparql, prefix_map)
                    if not validate_types(types, "phase2", cat, reverse_sparql):
                        stats["predicate_type_rejections"] += 1
                        continue
                    if not select_vars_cover_slots(reverse_sparql, slots):
                        logger.warning("Reverse query missing slot vars (phase2:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                        stats["reverse_query_form_rejections"] += 1
                        continue
                    if not body_contains_slots(reverse_sparql, slots):
                        logger.warning("Reverse query missing slot vars in body (phase2:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                        stats["reverse_query_form_rejections"] += 1
                        continue
                    cache = build_binding_cache(reverse_sparql, slots, "phase2", cat, template)
                    rows = cache["rows"]
                    if not rows:
                        logger.debug("Reverse query cache empty (phase2:%s): sparql=%s", cat, reverse_sparql)
                        stats["empty_binding_caches"] += 1
                        continue
                    row = select_valid_row(
                        rows,
                        slots,
                        pipeline,
                        max_row_checks,
                        logger=logger,
                        used_keys=cache["used"],
                        entity_diversity=entity_diversity,
                        category=cat,
                        target_per_cat=phase2_seeds_per_category,
                    )
                    if not row:
                        if template not in refreshed_templates:
                            cache = build_binding_cache(reverse_sparql, slots, "phase2", cat, template, extend=True)
                            refreshed_templates.add(template)
                            row = select_valid_row(
                                cache["rows"],
                                slots,
                                pipeline,
                                max_row_checks,
                                logger=logger,
                                used_keys=cache["used"],
                                entity_diversity=entity_diversity,
                                category=cat,
                                target_per_cat=phase2_seeds_per_category,
                            )
                        if not row:
                            stats["no_valid_binding_rows"] += 1
                            key = template
                            stall_counts[key] = stall_counts.get(key, 0) + 1
                            if stall_counts[key] >= stall_limit:
                                logger.warning(
                                    "Phase 2 stall limit reached for category=%s template=%s (no valid rows)",
                                    cat,
                                    template,
                                )
                                dead_templates.add(template)
                                continue
                            continue
                    entity_hints = {}
                    filled = template
                    for slot in slots:
                        val = row.get(slot)
                        if not val:
                            continue
                        if is_iri(val):
                            label = pipeline.label_for_uri(val)
                            type_uri = slot_type_hints.get(re.sub(r"\d+$", "", slot)) or pipeline.entity_type_for_uri(val)
                            type_label = pipeline.label_for_type(type_uri) if type_uri else None
                            if type_label:
                                filled = filled.replace("{" + slot + "}", f"({type_label}: {label})")
                                entity_hints[slot] = f"{val} | type={type_uri}"
                            else:
                                filled = filled.replace("{" + slot + "}", label)
                                entity_hints[slot] = val
                        else:
                            filled = filled.replace("{" + slot + "}", str(val))
                else:
                    filled = template
                    entity_hints = {}
                if filled:
                    selected_template = template
                    selected_slots = slots
                    selected_row = row if slots else None
                    break
            if not filled:
                stats["no_valid_binding_rows"] += 1
                continue

            logger.debug("Filled question (phase2:%s): %s", cat, filled)
            phase_key = f"phase2:{cat}"
            if filled in seen_filled[phase_key]:
                logger.debug("Filled question duplicate pre-LLM (phase2:%s): %s", cat, filled)
                stats["pre_llm_duplicates"] += 1
                continue
            seen_filled[phase_key].add(filled)
            examples = retrieve_examples(llm, global_store, filled, k=retrieval_top_k) if global_store else []
            examples = filter_retrieval_leakage(examples, filled)
            rec = run_known_sparql(filled, cat, selected_template, selected_row, examples)
            if rec is None:
                rec = pipeline.run_single(
                    filled,
                    cat,
                    examples=examples,
                    entity_hints=entity_hints,
                    repair_attempts=repair_attempts,
                )
            rec.question_latency_ms = q_latency
            stats["candidate_attempts"] += 1
            append_record_jsonl(str(log_path), rec)

            status = "OK" if rec.exec_success and not rec.error_type else "FAIL"
            logger.info(
                "[phase2:%s] %d/%d %s | parse=%s exec_ms=%.1f llm_ms=%.1f q_ms=%.1f answers=%d error=%s overall=%.1f%% cat=%d/%d",
                cat,
                accepted + 1,
                phase2_seeds_per_category,
                status,
                rec.parse_valid,
                rec.sparql_exec_ms,
                rec.llm_latency_ms,
                rec.question_latency_ms,
                rec.answer_count,
                rec.error_type,
                overall_pct(),
                cat_idx,
                len(categories),
            )

            if (not rec.exec_success) or rec.error_type:
                stats["validation_failures"] += 1
                continue
            if is_duplicate(llm, category_stores[cat], rec.question, dup_sim_threshold):
                logger.debug("Duplicate detected in phase2 for category=%s", cat)
                stats["duplicate_drops"] += 1
                continue

            phase2_records.append(rec)
            accepted += 1
            stats["accepted"] = accepted
            overall_done += 1
            # Record entity usage for diversity tracking
            if entity_hints:
                entity_uris = [v.split("|")[0].strip() for v in entity_hints.values() if v]
                entity_diversity.record(cat, entity_uris)
            category_stores[cat] = add_example_to_store(llm, category_stores[cat], rec.question, rec.sparql, rec.category)
            if selected_template and selected_template not in phase2_template_seen[cat]:
                phase2_template_seen[cat].add(selected_template)
                phase2_templates_by_cat[cat].append(
                    {"template": selected_template, "slots": selected_slots or []}
                )
        stats["accepted"] = accepted
        stats["exhausted"] = accepted < phase2_seeds_per_category

    for cat in categories:
        if category_stores[cat] is None or not category_stores[cat].metadata:
            if global_store is not None:
                logger.warning(
                    "Phase 2 produced no usable seeds for category=%s; falling back to global retrieval.",
                    cat,
                )
                category_stores[cat] = global_store
            else:
                logger.warning(
                    "Phase 2 produced no usable seeds for category=%s and no global retrieval; skipping category.",
                    cat,
                )

    # Phase 3: full dataset generation (category-specific retrieval)
    logger.info("Phase 3: full dataset generation")
    for cache in reverse_row_cache.values():
        cache["used"] = set()
    entity_diversity = EntityDiversityTracker(max_entity_pct=cfg.get("max_entity_pct", 0.15))
    phase3_records = []
    phase3_dedupe_stores = {cat: None for cat in categories}
    for cat_idx, cat in enumerate(categories, start=1):
        if category_stores.get(cat) is None:
            logger.warning("Skipping Phase 3 for category=%s (no retrieval store).", cat)
            continue
        logger.info("Phase 3 category: %s (%d/%d) | overall=%.1f%%", cat, cat_idx, len(categories), overall_pct())
        accepted = 0
        attempts = 0
        max_attempts = target_per_category * max_attempts_factor
        stats = proposal_stats["phase3"][cat]
        stats["target"] = target_per_category
        stats["max_outer_attempts"] = max_attempts
        stall_counts: dict[str, int] = {}
        stall_limit = max(3, template_candidates * 3)
        dead_templates: set[str] = set()
        refreshed_templates: set[str] = set()
        while accepted < target_per_category and attempts < max_attempts:
            attempts += 1
            stats["outer_attempts"] = attempts
            t0 = time.time()
            templates = phase2_templates_by_cat.get(cat) or []
            if not templates:
                try:
                    templates = pipeline.generate_templates(
                        cat,
                        n=max(1, template_candidates),
                        avoid_templates=avoid_templates.get(cat, [])[-10:],
                    )
                except ValueError as exc:
                    logger.warning("Phase 3 template generation failed for category=%s: %s", cat, exc)
                    continue
            templates = merge_priority_templates(cat, dedupe_templates(templates))
            templates = structural_dedupe_templates(templates)
            if not templates:
                logger.warning("No templates generated in Phase 3 for category=%s", cat)
                continue
            q_latency = (time.time() - t0) * 1000
            filled = None
            entity_hints = {}
            selected_template = None
            selected_row = None
            for tmpl in templates:
                template = tmpl.get("template", "")
                slots = tmpl.get("slots", [])
                if not template:
                    continue
                stats["template_items_seen"] += 1
                if template in dead_templates:
                    continue
                avoid_templates[cat].append(template)
                if slots:
                    cache_key = (template, tuple(slots))
                    reverse_sparql = template_reverse_cache.get(cache_key)
                    if reverse_sparql is None:
                        reverse_sparql = render_priority_reverse_query(cat, template, slots)
                        if reverse_sparql is None:
                            try:
                                reverse_sparql = pipeline.reverse_query(template, slots)
                            except Exception as exc:
                                logger.warning("Reverse query failed in Phase 3 for category=%s: %s", cat, exc)
                                stats["reverse_query_failures"] += 1
                                continue
                        template_reverse_cache[cache_key] = reverse_sparql
                    ok_parse, parse_err = parse_valid_sparql_detail(reverse_sparql)
                    if not ok_parse:
                        stats["reverse_query_parse_failures"] += 1
                        logger.warning(
                            "Reverse query parse failed (phase3:%s): %s | sparql=%s",
                            cat,
                            parse_err,
                            reverse_sparql,
                        )
                    if not is_simple_reverse_query(reverse_sparql, cat):
                        logger.warning("Reverse query not simple (phase3:%s): sparql=%s", cat, reverse_sparql)
                        stats["reverse_query_form_rejections"] += 1
                        continue
                    preds = extract_predicates(reverse_sparql, prefix_map)
                    if not validate_predicates(preds, "phase3", cat, reverse_sparql):
                        stats["predicate_type_rejections"] += 1
                        continue
                    types = extract_types(reverse_sparql, prefix_map)
                    if not validate_types(types, "phase3", cat, reverse_sparql):
                        stats["predicate_type_rejections"] += 1
                        continue
                    if not select_vars_cover_slots(reverse_sparql, slots):
                        logger.warning("Reverse query missing slot vars (phase3:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                        stats["reverse_query_form_rejections"] += 1
                        continue
                    if not body_contains_slots(reverse_sparql, slots):
                        logger.warning("Reverse query missing slot vars in body (phase3:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                        stats["reverse_query_form_rejections"] += 1
                        continue
                    cache = build_binding_cache(reverse_sparql, slots, "phase3", cat, template)
                    rows = cache["rows"]
                    if not rows:
                        logger.debug("Reverse query cache empty (phase3:%s): sparql=%s", cat, reverse_sparql)
                        stats["empty_binding_caches"] += 1
                        continue
                    row = select_valid_row(
                        rows,
                        slots,
                        pipeline,
                        max_row_checks,
                        logger=logger,
                        used_keys=cache["used"],
                        entity_diversity=entity_diversity,
                        category=cat,
                        target_per_cat=target_per_category,
                    )
                    if not row:
                        if template not in refreshed_templates:
                            cache = build_binding_cache(reverse_sparql, slots, "phase3", cat, template, extend=True)
                            refreshed_templates.add(template)
                            row = select_valid_row(
                                cache["rows"],
                                slots,
                                pipeline,
                                max_row_checks,
                                logger=logger,
                                used_keys=cache["used"],
                                entity_diversity=entity_diversity,
                                category=cat,
                                target_per_cat=target_per_category,
                            )
                        if not row:
                            stats["no_valid_binding_rows"] += 1
                            key = template
                            stall_counts[key] = stall_counts.get(key, 0) + 1
                            if stall_counts[key] >= stall_limit:
                                logger.warning(
                                    "Phase 3 stall limit reached for category=%s template=%s (no valid rows)",
                                    cat,
                                    template,
                                )
                                dead_templates.add(template)
                                continue
                            continue
                    entity_hints = {}
                    filled = template
                    for slot in slots:
                        val = row.get(slot)
                        if not val:
                            continue
                        if is_iri(val):
                            label = pipeline.label_for_uri(val)
                            type_uri = slot_type_hints.get(re.sub(r"\d+$", "", slot)) or pipeline.entity_type_for_uri(val)
                            type_label = pipeline.label_for_type(type_uri) if type_uri else None
                            if type_label:
                                filled = filled.replace("{" + slot + "}", f"({type_label}: {label})")
                                entity_hints[slot] = f"{val} | type={type_uri}"
                            else:
                                filled = filled.replace("{" + slot + "}", label)
                                entity_hints[slot] = val
                        else:
                            filled = filled.replace("{" + slot + "}", str(val))
                else:
                    filled = template
                    entity_hints = {}
                if filled:
                    selected_template = template
                    selected_row = row if slots else None
                    break
            if not filled:
                stats["no_valid_binding_rows"] += 1
                continue

            logger.debug("Filled question (phase3:%s): %s", cat, filled)
            phase_key = f"phase3:{cat}"
            if filled in seen_filled[phase_key]:
                logger.debug("Filled question duplicate pre-LLM (phase3:%s): %s", cat, filled)
                stats["pre_llm_duplicates"] += 1
                continue
            seen_filled[phase_key].add(filled)
            examples = retrieve_examples(llm, category_stores[cat], filled, k=retrieval_top_k)
            examples = filter_retrieval_leakage(examples, filled)
            rec = run_known_sparql(filled, cat, selected_template, selected_row, examples)
            if rec is None:
                rec = pipeline.run_single(
                    filled,
                    cat,
                    examples=examples,
                    entity_hints=entity_hints,
                    repair_attempts=repair_attempts,
                )
            rec.question_latency_ms = q_latency
            stats["candidate_attempts"] += 1
            append_record_jsonl(str(log_path), rec)

            status = "OK" if rec.exec_success and not rec.error_type else "FAIL"
            logger.info(
                "[phase3:%s] %d/%d %s | parse=%s exec_ms=%.1f llm_ms=%.1f q_ms=%.1f answers=%d error=%s overall=%.1f%% cat=%d/%d",
                cat,
                accepted + 1,
                target_per_category,
                status,
                rec.parse_valid,
                rec.sparql_exec_ms,
                rec.llm_latency_ms,
                rec.question_latency_ms,
                rec.answer_count,
                rec.error_type,
                overall_pct(),
                cat_idx,
                len(categories),
            )

            if (not rec.exec_success) or rec.error_type:
                stats["validation_failures"] += 1
                continue
            if accepted > 0 and is_duplicate(llm, phase3_dedupe_stores[cat], rec.question, dup_sim_threshold):
                logger.debug("Duplicate detected in phase3 for category=%s", cat)
                stats["duplicate_drops"] += 1
                continue

            phase3_records.append(rec)
            accepted += 1
            stats["accepted"] = accepted
            overall_done += 1
            # Record entity usage for diversity tracking
            if entity_hints:
                entity_uris = [v.split("|")[0].strip() for v in entity_hints.values() if v]
                entity_diversity.record(cat, entity_uris)
            phase3_dedupe_stores[cat] = add_example_to_store(
                llm,
                phase3_dedupe_stores[cat],
                rec.question,
                rec.sparql,
                rec.category,
            )
            category_stores[cat] = add_example_to_store(llm, category_stores[cat], rec.question, rec.sparql, rec.category)
        stats["accepted"] = accepted
        stats["exhausted"] = accepted < target_per_category

    # Save dataset outputs
    jsonl_phase1 = data_dir / "benchmark_phase1.jsonl"
    jsonl_phase2 = data_dir / "benchmark_phase2.jsonl"
    jsonl_phase3 = data_dir / "benchmark_phase3.jsonl"
    jsonl_full = data_dir / "benchmark_full.jsonl"
    csv_phase1 = data_dir / "benchmark_phase1.csv"
    csv_phase3 = data_dir / "benchmark_phase3.csv"
    csv_full = data_dir / "benchmark_full.csv"

    logger.info("Writing dataset outputs...")
    with jsonl_phase1.open("w", encoding="utf-8") as f:
        for r in phase1_records:
            f.write(json.dumps(r.to_dict()) + "\n")

    with jsonl_phase2.open("w", encoding="utf-8") as f:
        for r in phase2_records:
            f.write(json.dumps(r.to_dict()) + "\n")

    with jsonl_phase3.open("w", encoding="utf-8") as f:
        for r in phase3_records:
            f.write(json.dumps(r.to_dict()) + "\n")

    with jsonl_full.open("w", encoding="utf-8") as f:
        for r in phase1_records + phase2_records + phase3_records:
            f.write(json.dumps(r.to_dict()) + "\n")

    with csv_phase1.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "category", "question", "sparql", "answer_count", "exec_success", "parse_valid"],
        )
        writer.writeheader()
        for idx, r in enumerate(phase1_records, start=1):
            writer.writerow(
                {
                    "id": idx,
                    "category": r.category,
                    "question": r.question,
                    "sparql": r.sparql,
                    "answer_count": r.answer_count,
                    "exec_success": r.exec_success,
                    "parse_valid": r.parse_valid,
                }
            )

    with csv_phase3.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "category", "question", "sparql", "answer_count", "exec_success", "parse_valid"],
        )
        writer.writeheader()
        for idx, r in enumerate(phase3_records, start=1):
            writer.writerow(
                {
                    "id": idx,
                    "category": r.category,
                    "question": r.question,
                    "sparql": r.sparql,
                    "answer_count": r.answer_count,
                    "exec_success": r.exec_success,
                    "parse_valid": r.parse_valid,
                }
            )

    with csv_full.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "phase",
                "category",
                "question",
                "sparql",
                "answer_count",
                "exec_success",
                "parse_valid",
            ],
        )
        writer.writeheader()
        idx = 1
        for r in phase1_records:
            writer.writerow(
                {
                    "id": idx,
                    "phase": "phase1",
                    "category": r.category,
                    "question": r.question,
                    "sparql": r.sparql,
                    "answer_count": r.answer_count,
                    "exec_success": r.exec_success,
                    "parse_valid": r.parse_valid,
                }
            )
            idx += 1
        for r in phase2_records:
            writer.writerow(
                {
                    "id": idx,
                    "phase": "phase2",
                    "category": r.category,
                    "question": r.question,
                    "sparql": r.sparql,
                    "answer_count": r.answer_count,
                    "exec_success": r.exec_success,
                    "parse_valid": r.parse_valid,
                }
            )
            idx += 1
        for r in phase3_records:
            writer.writerow(
                {
                    "id": idx,
                    "phase": "phase3",
                    "category": r.category,
                    "question": r.question,
                    "sparql": r.sparql,
                    "answer_count": r.answer_count,
                    "exec_success": r.exec_success,
                    "parse_valid": r.parse_valid,
                }
            )
            idx += 1

    # Auto-generate figures for this run
    logger.info("Generating figures...")
    subprocess.run(
        [sys.executable, "scripts/generate_figures_from_logs.py", "--run-id", run_id],
        check=False,
    )
    logger.info("Generating strategy analysis...")
    subprocess.run(
        [sys.executable, "scripts/generate_strategy_analysis.py", "--run-id", run_id],
        check=False,
    )

    logger.info("Full pipeline run complete. Run ID: %s", run_id)

    # Summary
    def summarize(records, label):
        if not records:
            logger.info("Summary %s: no records", label)
            return {
                "total": 0,
                "exec_ok": 0,
                "parse_ok": 0,
                "empty": 0,
                "parse_err": 0,
                "endpoint_err": 0,
                "repairs": 0,
                "avg_llm_ms": 0.0,
                "avg_exec_ms": 0.0,
                "avg_q_ms": 0.0,
                "avg_answers": 0.0,
                "by_category": {},
            }
        total = len(records)
        exec_ok = sum(1 for r in records if r.exec_success)
        parse_ok = sum(1 for r in records if r.parse_valid)
        empty = sum(1 for r in records if r.error_type == "empty_result")
        parse_err = sum(1 for r in records if r.error_type == "parse_error")
        endpoint_err = sum(1 for r in records if r.error_type == "endpoint_error")
        repairs = sum(1 for r in records if r.repair_attempts > 0)
        llm_avg = sum(r.llm_latency_ms for r in records) / total
        exec_avg = sum(r.sparql_exec_ms for r in records) / total
        q_avg = sum(r.question_latency_ms for r in records) / total
        ans_avg = sum(r.answer_count for r in records) / total
        by_cat = {}
        for r in records:
            by_cat.setdefault(r.category, []).append(r)
        logger.info(
            "Summary %s: total=%d exec_ok=%d (%.1f%%) parse_ok=%d (%.1f%%) empty=%d parse_err=%d endpoint_err=%d repairs=%d",
            label,
            total,
            exec_ok,
            exec_ok / total * 100,
            parse_ok,
            parse_ok / total * 100,
            empty,
            parse_err,
            endpoint_err,
            repairs,
        )
        logger.info(
            "Summary %s: avg_llm_ms=%.1f avg_exec_ms=%.1f avg_q_ms=%.1f avg_answers=%.2f",
            label,
            llm_avg,
            exec_avg,
            q_avg,
            ans_avg,
        )
        for cat, rows in sorted(by_cat.items()):
            ok = sum(1 for r in rows if r.exec_success)
            logger.info("Summary %s category=%s count=%d exec_ok=%d (%.1f%%)", label, cat, len(rows), ok, ok / len(rows) * 100)
        return {
            "total": total,
            "exec_ok": exec_ok,
            "parse_ok": parse_ok,
            "empty": empty,
            "parse_err": parse_err,
            "endpoint_err": endpoint_err,
            "repairs": repairs,
            "avg_llm_ms": llm_avg,
            "avg_exec_ms": exec_avg,
            "avg_q_ms": q_avg,
            "avg_answers": ans_avg,
            "by_category": {
                cat: {
                    "count": len(rows),
                    "exec_ok": sum(1 for r in rows if r.exec_success),
                }
                for cat, rows in by_cat.items()
            },
        }

    summary = {
        "run_id": run_id,
        "phase1": summarize(phase1_records, "Phase1"),
        "phase2": summarize(phase2_records, "Phase2"),
        "phase3": summarize(phase3_records, "Phase3"),
        "proposal_stats": {
            phase_name: {
                category: dict(bucket)
                for category, bucket in sorted(phase_buckets.items())
            }
            for phase_name, phase_buckets in proposal_stats.items()
        },
    }

    elapsed = time.time() - run_start
    logger.info("Run artifacts: logs=%s data=%s figures=%s", log_path, data_dir, run_dir / "figures")
    logger.info("Total runtime: %.1f seconds", elapsed)
    summary_path = run_dir / "run_summary.json"
    summary["runtime_sec"] = elapsed
    summary["artifacts"] = {
        "logs": str(log_path),
        "data_dir": str(data_dir),
        "figures_dir": str(run_dir / "figures"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
