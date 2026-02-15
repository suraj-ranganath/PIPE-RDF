import csv
import json
import random
import re
import time
import logging
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
from pipekg.llm import LLMClient, LLMConfig
from pipekg.sparql_client import SparqlClient
import argparse
import subprocess
from datetime import datetime
from pipekg.pipeline_ollama import (
    OllamaPipeline,
    build_faiss_index,
    retrieve_examples,
    append_record_jsonl,
)
from pipekg.evaluation import parse_valid_sparql_detail
from pipekg.figures_extra import bar_by_category
from pipekg.logging_utils import result_set_hash
from pipekg.utils import tokenize, jaccard
from pipekg.schema_summary import build_schema_summary, build_schema_whitelist
from pipekg.logger import get_logger


def build_llm(settings):
    if settings.llm_provider == "ollama":
        return LLMClient(
            LLMConfig(
                provider="ollama",
                api_key="",
                model=settings.ollama_chat_model,
                embed_model=settings.ollama_embed_model,
                base_url=settings.ollama_base_url,
            )
        )
    return LLMClient(
        LLMConfig(
            provider="openai",
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            embed_model=settings.openai_embed_model,
            base_url="",
        )
    )


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


def randomize_reverse_query(sparql: str, limit: int) -> str:
    # Strip inline LIMIT/ORDER BY to avoid ORDER BY after LIMIT (GraphDB parse error).
    text = sparql
    text = re.sub(r"(?is)ORDER\\s+BY\\s+[^\\n]+", "", text)
    text = re.sub(r"(?is)LIMIT\\s+\\d+\\b", "", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        upper = ln.strip().upper()
        if upper.startswith(("ORDER BY", "LIMIT")):
            continue
        cleaned.append(ln)
    cleaned.append("ORDER BY RAND()")
    cleaned.append(f"LIMIT {limit}")
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


def is_simple_reverse_query(sparql: str) -> bool:
    upper = sparql.upper()
    banned = ["ORDER BY", "GROUP BY", "HAVING", "OPTIONAL", "UNION", "SUBSELECT", "COUNT(", "AVG(", "MIN(", "MAX(", "SUM("]
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
    if cfg.get("models", {}).get("chat"):
        settings.ollama_chat_model = cfg["models"]["chat"]
    if cfg.get("models", {}).get("embed"):
        settings.ollama_embed_model = cfg["models"]["embed"]
    if cfg.get("sparql_endpoint_url"):
        settings.sparql_endpoint_url = cfg["sparql_endpoint_url"]
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
    logger.info("Ollama base: %s", settings.ollama_base_url)
    logger.info("Models: chat=%s embed=%s", settings.ollama_chat_model, settings.ollama_embed_model)
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

    sparql_params = {}
    if not sparql_infer:
        sparql_params["infer"] = "false"
        logger.info("SPARQL inference disabled (infer=false)")
    logger.info("SPARQL endpoint (base): %s", settings.sparql_endpoint_url)
    logger.info("SPARQL timeouts: query=%ss schema=%ss", sparql_timeout_sec, schema_timeout_sec)
    if generated_query_limit:
        logger.info("Generated query LIMIT cap: %s", generated_query_limit)
    logger.info("Reverse query LIMIT cap: %s", reverse_query_limit)

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
    allowed_predicates_cfg = cfg.get("allowed_predicates") or []
    allowed_types_cfg = cfg.get("allowed_types") or []
    whitelist = {"predicates": [], "types": []}
    if allowed_predicates_cfg or allowed_types_cfg:
        allowed_predicates = {expand_qname(p, prefix_map) for p in allowed_predicates_cfg}
        allowed_types = [expand_qname(t, prefix_map) for t in allowed_types_cfg]
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

    # Phase 1: template generation + reverse querying (seeds for retrieval)
    logger.info("Phase 1: template generation + reverse querying")
    phase1_records = []
    for cat_idx, cat in enumerate(categories, start=1):
        logger.info("Phase 1 category: %s (%d/%d) | overall=%.1f%%", cat, cat_idx, len(categories), overall_pct())
        target_phase1 = phase1_templates_per_category * phase1_seeds_per_template
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
        templates = dedupe_templates(templates)
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
            avoid_templates[cat].append(template)
            if slots:
                cache_key = (template, tuple(slots))
                reverse_sparql = template_reverse_cache.get(cache_key)
                if reverse_sparql is None:
                    try:
                        reverse_sparql = pipeline.reverse_query(template, slots)
                    except Exception as exc:
                        logger.warning("Reverse query failed in Phase 1 for category=%s: %s", cat, exc)
                        continue
                    template_reverse_cache[cache_key] = reverse_sparql
                ok_parse, parse_err = parse_valid_sparql_detail(reverse_sparql)
                if not ok_parse:
                    logger.warning(
                        "Reverse query parse failed (phase1:%s): %s | sparql=%s",
                        cat,
                        parse_err,
                        reverse_sparql,
                    )
                if not is_simple_reverse_query(reverse_sparql):
                    logger.warning("Reverse query not simple (phase1:%s): sparql=%s", cat, reverse_sparql)
                    continue
                preds = extract_predicates(reverse_sparql, prefix_map)
                if not validate_predicates(preds, "phase1", cat, reverse_sparql):
                    continue
                types = extract_types(reverse_sparql, prefix_map)
                if not validate_types(types, "phase1", cat, reverse_sparql):
                    continue
                if not select_vars_cover_slots(reverse_sparql, slots):
                    logger.warning("Reverse query missing slot vars (phase1:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                    continue
                if not body_contains_slots(reverse_sparql, slots):
                    logger.warning("Reverse query missing slot vars in body (phase1:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                    continue
                cache = reverse_row_cache.get(reverse_sparql)
                if cache is None:
                    ok, _, rows, err = pipeline.execute_rows(reverse_sparql)
                    if not ok:
                        logger.warning("Reverse query exec failed (phase1:%s): %s | sparql=%s", cat, err, reverse_sparql)
                        reverse_row_cache[reverse_sparql] = {"rows": [], "used": set()}
                        continue
                    if not rows:
                        logger.debug("Reverse query returned 0 rows (phase1:%s): sparql=%s", cat, reverse_sparql)
                        reverse_row_cache[reverse_sparql] = {"rows": [], "used": set()}
                        continue
                    random.shuffle(rows)
                    cache = {"rows": rows, "used": set()}
                    reverse_row_cache[reverse_sparql] = cache
                    log_row_sample(rows, "phase1", cat, logger=logger)
                rows = cache["rows"]
                if not rows:
                    logger.debug("Reverse query cache empty (phase1:%s): sparql=%s", cat, reverse_sparql)
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
                        break
                    entity_hints = {}
                    filled = template
                    for slot in slots:
                        val = row.get(slot)
                        if not val:
                            continue
                        label = pipeline.label_for_uri(val)
                        type_uri = pipeline.entity_type_for_uri(val)
                        type_label = pipeline.label_for_type(type_uri) if type_uri else None
                        if type_label:
                            filled = filled.replace("{" + slot + "}", f"({type_label}: {label})")
                            entity_hints[slot] = f"{val} | type={type_uri}"
                        else:
                            filled = filled.replace("{" + slot + "}", label)
                            entity_hints[slot] = val
                    logger.debug("Filled question (phase1:%s cat=%d/%d): %s", cat, cat_idx, len(categories), filled)
                    rec = pipeline.run_single(filled, cat, entity_hints=entity_hints, repair_attempts=repair_attempts)
                    phase1_records.append(rec)
                    overall_done += 1
                    accepted += 1
                    append_record_jsonl(str(log_path), rec)
                    if accepted >= target_phase1:
                        break
            else:
                for _ in range(min(phase1_seeds_per_template, 1)):
                    logger.debug("Filled question (phase1:%s cat=%d/%d): %s", cat, cat_idx, len(categories), template)
                    rec = pipeline.run_single(template, cat, repair_attempts=repair_attempts)
                    phase1_records.append(rec)
                    overall_done += 1
                    accepted += 1
                    append_record_jsonl(str(log_path), rec)
                    if accepted >= target_phase1:
                        break
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
        stall_counts: dict[str, int] = {}
        stall_limit = max(3, template_candidates * 3)
        dead_templates: set[str] = set()
        refreshed_templates: set[str] = set()
        phase2_templates_by_cat[cat] = []
        while accepted < phase2_seeds_per_category and attempts < max_attempts:
            attempts += 1
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
            templates = dedupe_templates(templates)
            if not templates:
                logger.warning("No templates generated in Phase 2 for category=%s", cat)
                continue
            q_latency = (time.time() - t0) * 1000
            filled = None
            entity_hints = {}
            selected_template = None
            selected_slots = None
            for tmpl in templates:
                template = tmpl.get("template", "")
                slots = tmpl.get("slots", [])
                if not template:
                    continue
                if template in dead_templates:
                    continue
                avoid_templates[cat].append(template)
                if slots:
                    cache_key = (template, tuple(slots))
                    reverse_sparql = template_reverse_cache.get(cache_key)
                    if reverse_sparql is None:
                        try:
                            reverse_sparql = pipeline.reverse_query(template, slots)
                        except Exception as exc:
                            logger.warning("Reverse query failed in Phase 2 for category=%s: %s", cat, exc)
                            continue
                        template_reverse_cache[cache_key] = reverse_sparql
                    ok_parse, parse_err = parse_valid_sparql_detail(reverse_sparql)
                    if not ok_parse:
                        logger.warning(
                            "Reverse query parse failed (phase2:%s): %s | sparql=%s",
                            cat,
                            parse_err,
                            reverse_sparql,
                        )
                    if not is_simple_reverse_query(reverse_sparql):
                        logger.warning("Reverse query not simple (phase2:%s): sparql=%s", cat, reverse_sparql)
                        continue
                    preds = extract_predicates(reverse_sparql, prefix_map)
                    if not validate_predicates(preds, "phase2", cat, reverse_sparql):
                        continue
                    types = extract_types(reverse_sparql, prefix_map)
                    if not validate_types(types, "phase2", cat, reverse_sparql):
                        continue
                    if not select_vars_cover_slots(reverse_sparql, slots):
                        logger.warning("Reverse query missing slot vars (phase2:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                        continue
                    if not body_contains_slots(reverse_sparql, slots):
                        logger.warning("Reverse query missing slot vars in body (phase2:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                        continue
                    cache = reverse_row_cache.get(reverse_sparql)
                    if cache is None:
                        ok, _, rows, err = pipeline.execute_rows(reverse_sparql)
                        if not ok:
                            logger.warning("Reverse query exec failed (phase2:%s): %s | sparql=%s", cat, err, reverse_sparql)
                            reverse_row_cache[reverse_sparql] = {"rows": [], "used": set()}
                            continue
                        if not rows:
                            logger.debug("Reverse query returned 0 rows (phase2:%s): sparql=%s", cat, reverse_sparql)
                            reverse_row_cache[reverse_sparql] = {"rows": [], "used": set()}
                            continue
                        random.shuffle(rows)
                        cache = {"rows": rows, "used": set()}
                        reverse_row_cache[reverse_sparql] = cache
                        log_row_sample(rows, "phase2", cat, logger=logger)
                    rows = cache["rows"]
                    if not rows:
                        logger.debug("Reverse query cache empty (phase2:%s): sparql=%s", cat, reverse_sparql)
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
                            rand_sparql = randomize_reverse_query(
                                reverse_sparql,
                                reverse_query_limit,
                            )
                            ok, _, rrows, _ = pipeline.execute_rows(rand_sparql)
                            if ok and rrows:
                                cache["rows"] = rrows
                                cache["used"] = set()
                                refreshed_templates.add(template)
                                row = select_valid_row(
                                    rrows,
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
                            type_uri = pipeline.entity_type_for_uri(val)
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
                    break
            if not filled:
                continue

            logger.debug("Filled question (phase2:%s): %s", cat, filled)
            phase_key = f"phase2:{cat}"
            if filled in seen_filled[phase_key]:
                logger.debug("Filled question duplicate pre-LLM (phase2:%s): %s", cat, filled)
                continue
            seen_filled[phase_key].add(filled)
            examples = retrieve_examples(llm, global_store, filled, k=retrieval_top_k) if global_store else []
            examples = filter_retrieval_leakage(examples, filled)
            rec = pipeline.run_single(
                filled,
                cat,
                examples=examples,
                entity_hints=entity_hints,
                repair_attempts=repair_attempts,
            )
            rec.question_latency_ms = q_latency
            append_record_jsonl(str(log_path), rec)

            status = "OK" if rec.exec_success else "FAIL"
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

            if not rec.exec_success:
                continue
            if is_duplicate(llm, category_stores[cat], rec.question, dup_sim_threshold):
                logger.debug("Duplicate detected in phase2 for category=%s", cat)
                continue

            phase2_records.append(rec)
            accepted += 1
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
        stall_counts: dict[str, int] = {}
        stall_limit = max(3, template_candidates * 3)
        dead_templates: set[str] = set()
        refreshed_templates: set[str] = set()
        while accepted < target_per_category and attempts < max_attempts:
            attempts += 1
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
            templates = dedupe_templates(templates)
            templates = structural_dedupe_templates(templates)
            if not templates:
                logger.warning("No templates generated in Phase 3 for category=%s", cat)
                continue
            q_latency = (time.time() - t0) * 1000
            filled = None
            entity_hints = {}
            for tmpl in templates:
                template = tmpl.get("template", "")
                slots = tmpl.get("slots", [])
                if not template:
                    continue
                if template in dead_templates:
                    continue
                avoid_templates[cat].append(template)
                if slots:
                    cache_key = (template, tuple(slots))
                    reverse_sparql = template_reverse_cache.get(cache_key)
                    if reverse_sparql is None:
                        try:
                            reverse_sparql = pipeline.reverse_query(template, slots)
                        except Exception as exc:
                            logger.warning("Reverse query failed in Phase 3 for category=%s: %s", cat, exc)
                            continue
                        template_reverse_cache[cache_key] = reverse_sparql
                    ok_parse, parse_err = parse_valid_sparql_detail(reverse_sparql)
                    if not ok_parse:
                        logger.warning(
                            "Reverse query parse failed (phase3:%s): %s | sparql=%s",
                            cat,
                            parse_err,
                            reverse_sparql,
                        )
                    if not is_simple_reverse_query(reverse_sparql):
                        logger.warning("Reverse query not simple (phase3:%s): sparql=%s", cat, reverse_sparql)
                        continue
                    preds = extract_predicates(reverse_sparql, prefix_map)
                    if not validate_predicates(preds, "phase3", cat, reverse_sparql):
                        continue
                    types = extract_types(reverse_sparql, prefix_map)
                    if not validate_types(types, "phase3", cat, reverse_sparql):
                        continue
                    if not select_vars_cover_slots(reverse_sparql, slots):
                        logger.warning("Reverse query missing slot vars (phase3:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                        continue
                    if not body_contains_slots(reverse_sparql, slots):
                        logger.warning("Reverse query missing slot vars in body (phase3:%s): slots=%s sparql=%s", cat, slots, reverse_sparql)
                        continue
                    cache = reverse_row_cache.get(reverse_sparql)
                    if cache is None:
                        ok, _, rows, err = pipeline.execute_rows(reverse_sparql)
                        if not ok:
                            logger.warning("Reverse query exec failed (phase3:%s): %s | sparql=%s", cat, err, reverse_sparql)
                            reverse_row_cache[reverse_sparql] = {"rows": [], "used": set()}
                            continue
                        if not rows:
                            logger.debug("Reverse query returned 0 rows (phase3:%s): sparql=%s", cat, reverse_sparql)
                            reverse_row_cache[reverse_sparql] = {"rows": [], "used": set()}
                            continue
                        random.shuffle(rows)
                        cache = {"rows": rows, "used": set()}
                        reverse_row_cache[reverse_sparql] = cache
                        log_row_sample(rows, "phase3", cat, logger=logger)
                    rows = cache["rows"]
                    if not rows:
                        logger.debug("Reverse query cache empty (phase3:%s): sparql=%s", cat, reverse_sparql)
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
                            rand_sparql = randomize_reverse_query(
                                reverse_sparql,
                                reverse_query_limit,
                            )
                            ok, _, rrows, _ = pipeline.execute_rows(rand_sparql)
                            if ok and rrows:
                                cache["rows"] = rrows
                                cache["used"] = set()
                                refreshed_templates.add(template)
                                row = select_valid_row(
                                    rrows,
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
                            type_uri = pipeline.entity_type_for_uri(val)
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
                    break
            if not filled:
                continue

            logger.debug("Filled question (phase3:%s): %s", cat, filled)
            phase_key = f"phase3:{cat}"
            if filled in seen_filled[phase_key]:
                logger.debug("Filled question duplicate pre-LLM (phase3:%s): %s", cat, filled)
                continue
            seen_filled[phase_key].add(filled)
            examples = retrieve_examples(llm, category_stores[cat], filled, k=retrieval_top_k)
            examples = filter_retrieval_leakage(examples, filled)
            rec = pipeline.run_single(
                filled,
                cat,
                examples=examples,
                entity_hints=entity_hints,
                repair_attempts=repair_attempts,
            )
            rec.question_latency_ms = q_latency
            append_record_jsonl(str(log_path), rec)

            status = "OK" if rec.exec_success else "FAIL"
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

            if not rec.exec_success:
                continue
            if accepted > 0 and is_duplicate(llm, phase3_dedupe_stores[cat], rec.question, dup_sim_threshold):
                logger.debug("Duplicate detected in phase3 for category=%s", cat)
                continue

            phase3_records.append(rec)
            accepted += 1
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
        ["python", "scripts/generate_figures_from_logs.py", "--run-id", run_id],
        check=False,
    )
    logger.info("Generating strategy analysis...")
    subprocess.run(
        ["python", "scripts/generate_strategy_analysis.py", "--run-id", run_id],
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
