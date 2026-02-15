from __future__ import annotations

import json
import time
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .llm import LLMClient
from .prompts import (
    SYSTEM_NL_TEMPLATE,
    SYSTEM_SPARQL_GEN,
    SYSTEM_REPAIR,
    SYSTEM_PARAPHRASE,
    TEMPLATE_REQUEST,
    SPARQL_REQUEST,
    REPAIR_REQUEST,
    PARAPHRASE_REQUEST,
    REVERSE_QUERY_REQUEST,
)
from .sparql_client import SparqlClient
from .vector_store import FaissStore
from .logging_utils import estimate_tokens, result_set_hash
from .evaluation import parse_valid_sparql, parse_valid_sparql_detail, validate_answer_type
from .ast_utils import ast_stats


@dataclass
class GenerationRecord:
    category: str
    question: str
    sparql: str
    answers: List[str]
    exec_success: bool
    parse_valid: bool
    error: Optional[str]
    error_type: Optional[str]
    repair_attempts: int
    llm_latency_ms: float
    question_latency_ms: float
    sparql_exec_ms: float
    answer_count: int
    result_hash: Optional[str]
    prompt_chars: int
    prompt_tokens_est: int
    retrieved_examples: List[Dict[str, str]]
    ast_node_count: int
    ast_max_depth: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "category": self.category,
            "question": self.question,
            "sparql": self.sparql,
            "answers": self.answers,
            "exec_success": self.exec_success,
            "parse_valid": self.parse_valid,
            "error": self.error,
            "error_type": self.error_type,
            "repair_attempts": self.repair_attempts,
            "llm_latency_ms": self.llm_latency_ms,
            "question_latency_ms": self.question_latency_ms,
            "sparql_exec_ms": self.sparql_exec_ms,
            "answer_count": self.answer_count,
            "result_hash": self.result_hash,
            "prompt_chars": self.prompt_chars,
            "prompt_tokens_est": self.prompt_tokens_est,
            "retrieved_examples": self.retrieved_examples,
            "ast_node_count": self.ast_node_count,
            "ast_max_depth": self.ast_max_depth,
        }


def save_records_jsonl(path: str, records: List[GenerationRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")


def append_record_jsonl(path: str, record: GenerationRecord) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


class OllamaPipeline:
    def __init__(
        self,
        llm: LLMClient,
        sparql: SparqlClient,
        prefixes: str,
        schema_summary: str,
        slot_type_hints: Optional[Dict[str, str]] = None,
        logger=None,
        reverse_query_timeout_sec: int = 120,
        generated_query_limit: Optional[int] = None,
    ) -> None:
        self.llm = llm
        self.sparql = sparql
        self.prefixes = prefixes.strip()
        self.schema_summary = schema_summary.strip()
        self.slot_type_hints = slot_type_hints or {}
        self.logger = logger
        self.reverse_query_timeout_sec = reverse_query_timeout_sec
        self.generated_query_limit = generated_query_limit
        self._label_cache: Dict[str, str] = {}
        self._type_cache: Dict[str, Optional[str]] = {}
        self._type_label_cache: Dict[str, str] = {}

    def build_prompt(
        self,
        question: str,
        category: str,
        examples: Optional[List[Dict[str, str]]] = None,
        entity_hints: Optional[Dict[str, str]] = None,
    ) -> str:
        def esc(text: str) -> str:
            return text.replace("{", "{{").replace("}", "}}")

        example_block = ""
        if examples:
            rows = []
            for ex in examples:
                rows.append(f"Q: {ex.get('question','')}\nSPARQL: {ex.get('sparql','')}\n")
            example_block = "\nRetrieved examples:\n" + "\n".join(rows)

        hint_block = ""
        if entity_hints:
            lines = [f"{k}: {v}" for k, v in entity_hints.items()]
            hint_block = "\nEntity hints (use these IRIs when relevant):\n" + "\n".join(lines)
            hint_block += "\nSlot variables (use these variable names for hinted entities): " + ", ".join(entity_hints.keys())

        schema_text = esc(self.schema_summary + example_block + hint_block)
        question_text = esc(question)
        return SPARQL_REQUEST.format(
            prefixes=self.prefixes,
            schema=schema_text,
            question=question_text,
        )

    def generate_templates(
        self,
        category: str,
        n: int,
        avoid_templates: Optional[List[str]] = None,
    ) -> List[Dict[str, object]]:
        avoid_text = "; ".join(avoid_templates or []) if avoid_templates else "None"
        hints_map = {
            "generic": [
                "Where is {company} located?",
                "Who is a key person at {company}?",
                "What industry is {company} in?",
            ],
            "counting": [
                "How many companies are located in {location}?",
                "How many companies are in {industry}?",
                "How many key people are associated with {company}?",
            ],
            "comparative": [
                "Are {company1} and {company2} located in the same location?",
                "Are {company1} and {company2} in the same industry?",
            ],
            "superlative": [
                "Which company has the most employees?",
                "Which company was founded earliest?",
                "Which company has the highest number of employees?",
                "Which company has the most key people?",
                "Which company has the fewest employees?",
                "Which company has the lowest number of employees?",
                "Which company was founded most recently?",
                "Which company was founded latest?",
                "Which company has the most locations?",
                "Which company has the fewest locations?",
                "Which company in {industry} has the most employees?",
                "Which company in {location} has the most employees?",
                "Which company in {industry} was founded earliest?",
                "Which company in {industry} was founded most recently?",
                "Which company in {location} was founded earliest?",
                "Which company in {location} was founded most recently?",
                "Which company in {industry} has the most key people?",
                "Which company in {location} has the most key people?",
            ],
            "ordinal": [
                "What is the founding year of {company}?",
                "Which year was {company} founded?",
                "How many employees does {company} have?",
                "What is the number of employees of {company}?",
                "What is the employee count for {company}?",
                "What year was {company} established?",
                "In what year was {company} founded?",
                "What is {company}'s founding year?",
                "How many employees work at {company}?",
                "How many people does {company} employ?",
            ],
            "multi-hop": [
                "Which company is in {industry} and located in {location}?",
                "Which company has key person {person} and is located in {location}?",
                "Which company in {industry} has key person {person}?",
                "Which company located in {location} operates in {industry}?",
                "Which company has key person {person} and operates in {industry}?",
                "Which company in {industry} is located in {location}?",
                "Which company with key person {person} is located in {location}?",
                "Which company with key person {person} operates in {industry}?",
                "Which company in {location} operates in {industry}?",
                "Which company located in {location} is in {industry}?",
                "Which company in {industry} is located in {location}?",
            ],
            "intersection": [
                "Which company is in {industry} and has key person {person}?",
                "Which company is located in {location} and is in {industry}?",
                "Which company has key person {person} and is located in {location}?",
                "Which company operates in {industry} and is located in {location}?",
                "Which company in {industry} has key person {person}?",
                "Which company in {location} operates in {industry}?",
                "Which company located in {location} is in {industry}?",
                "Which company located in {location} has key person {person}?",
            ],
            "difference": [
                "Do {company1} and {company2} have different locations?",
                "Are {company1} and {company2} in different industries?",
            ],
            "yesno": [
                "Is {company} located in {location}?",
                "Does {company} have a key person {person}?",
                "Is {company} in the {industry} industry?",
                "Does {company} operate in {industry}?",
                "Is {company} based in {location}?",
                "Does {company} have any key person?",
                "Is {person} a key person at {company}?",
            ],
        }
        fallback_map = {
            "generic": [
                {"template": "Where is {company} located?", "slots": ["company"]},
                {"template": "What industry is {company} in?", "slots": ["company"]},
                {"template": "Who is a key person at {company}?", "slots": ["company"]},
            ],
            "counting": [
                {"template": "How many companies are in {industry}?", "slots": ["industry"]},
                {"template": "How many companies are located in {location}?", "slots": ["location"]},
            ],
            "comparative": [
                {"template": "Are {company1} and {company2} in the same industry?", "slots": ["company1", "company2"]},
                {"template": "Are {company1} and {company2} in the same location?", "slots": ["company1", "company2"]},
            ],
            "superlative": [
                {"template": "Which company has the most employees?", "slots": []},
                {"template": "Which company was founded earliest?", "slots": []},
                {"template": "Which company in {industry} has the most employees?", "slots": ["industry"]},
                {"template": "Which company in {location} has the most employees?", "slots": ["location"]},
                {"template": "Which company in {industry} was founded earliest?", "slots": ["industry"]},
                {"template": "Which company in {location} was founded earliest?", "slots": ["location"]},
                {"template": "Which company has the highest number of employees?", "slots": []},
                {"template": "Which company has the fewest employees?", "slots": []},
                {"template": "Which company has the most key people?", "slots": []},
                {"template": "Which company was founded most recently?", "slots": []},
                {"template": "Which company was founded latest?", "slots": []},
                {"template": "Which company has the most locations?", "slots": []},
                {"template": "Which company has the fewest locations?", "slots": []},
                {"template": "Which company in {industry} was founded most recently?", "slots": ["industry"]},
                {"template": "Which company in {location} was founded most recently?", "slots": ["location"]},
                {"template": "Which company in {industry} has the most key people?", "slots": ["industry"]},
                {"template": "Which company in {location} has the most key people?", "slots": ["location"]},
            ],
            "ordinal": [
                {"template": "What is the founding year of {company}?", "slots": ["company"]},
                {"template": "Which year was {company} founded?", "slots": ["company"]},
                {"template": "How many employees does {company} have?", "slots": ["company"]},
                {"template": "What is the number of employees of {company}?", "slots": ["company"]},
                {"template": "What is the employee count for {company}?", "slots": ["company"]},
                {"template": "What year was {company} established?", "slots": ["company"]},
                {"template": "In what year was {company} founded?", "slots": ["company"]},
                {"template": "What is {company}'s founding year?", "slots": ["company"]},
                {"template": "How many employees work at {company}?", "slots": ["company"]},
                {"template": "How many people does {company} employ?", "slots": ["company"]},
            ],
            "multi-hop": [
                {"template": "Which company has key person {person} and is located in {location}?", "slots": ["person", "location"]},
                {"template": "Which company is in {industry} and located in {location}?", "slots": ["industry", "location"]},
                {"template": "Which company in {industry} has key person {person}?", "slots": ["industry", "person"]},
                {"template": "Which company located in {location} operates in {industry}?", "slots": ["location", "industry"]},
                {"template": "Which company has key person {person} and operates in {industry}?", "slots": ["person", "industry"]},
                {"template": "Which company with key person {person} is located in {location}?", "slots": ["person", "location"]},
                {"template": "Which company with key person {person} operates in {industry}?", "slots": ["person", "industry"]},
                {"template": "Which company in {location} operates in {industry}?", "slots": ["location", "industry"]},
                {"template": "Which company located in {location} is in {industry}?", "slots": ["location", "industry"]},
                {"template": "Which company in {industry} is located in {location}?", "slots": ["industry", "location"]},
            ],
            "intersection": [
                {"template": "Which company is in {industry} and has key person {person}?", "slots": ["industry", "person"]},
                {"template": "Which company is located in {location} and is in {industry}?", "slots": ["location", "industry"]},
                {"template": "Which company has key person {person} and is located in {location}?", "slots": ["person", "location"]},
                {"template": "Which company operates in {industry} and is located in {location}?", "slots": ["industry", "location"]},
                {"template": "Which company in {industry} has key person {person}?", "slots": ["industry", "person"]},
                {"template": "Which company in {location} operates in {industry}?", "slots": ["location", "industry"]},
                {"template": "Which company located in {location} is in {industry}?", "slots": ["location", "industry"]},
                {"template": "Which company located in {location} has key person {person}?", "slots": ["location", "person"]},
            ],
            "difference": [
                {"template": "Do {company1} and {company2} have different locations?", "slots": ["company1", "company2"]},
                {"template": "Are {company1} and {company2} in different industries?", "slots": ["company1", "company2"]},
            ],
            "yesno": [
                {"template": "Is {company} located in {location}?", "slots": ["company", "location"]},
                {"template": "Does {company} have a key person {person}?", "slots": ["company", "person"]},
                {"template": "Was {company} founded in {year}?", "slots": ["company", "year"]},
                {"template": "Is {company} in the {industry} industry?", "slots": ["company", "industry"]},
                {"template": "Does {company} operate in {industry}?", "slots": ["company", "industry"]},
                {"template": "Is {company} based in {location}?", "slots": ["company", "location"]},
                {"template": "Does {company} have any key person?", "slots": ["company"]},
                {"template": "Is {person} a key person at {company}?", "slots": ["person", "company"]},
            ],
        }
        schema_lower = self.schema_summary.lower()
        literal_slots = set()
        if "foundingyear" in schema_lower or "founding year" in schema_lower:
            literal_slots.update({"year", "foundingyear"})
        if "numberofemployees" in schema_lower or "number of employees" in schema_lower:
            literal_slots.update({"number", "numberofemployees", "employees"})

        allowed_by_category = {
            "generic": {"company", "location", "industry", "person", "year", "number"},
            "counting": {"company", "location", "industry", "person", "year", "number"},
            "comparative": {
                "company1",
                "company2",
                "person1",
                "person2",
                "location1",
                "location2",
                "industry1",
                "industry2",
                "year1",
                "year2",
                "number1",
                "number2",
            },
            "superlative": {"industry", "location", "company", "person", "year", "number"},
            "ordinal": {"company", "location", "industry", "person", "year", "number"},
            "multi-hop": {"company", "location", "industry", "person", "year", "number"},
            "intersection": {"company", "location", "industry", "person", "year", "number"},
            "difference": {
                "company1",
                "company2",
                "person1",
                "person2",
                "location1",
                "location2",
                "industry1",
                "industry2",
                "year1",
                "year2",
                "number1",
                "number2",
            },
            "yesno": {"company", "location", "industry", "person", "year", "number"},
        }
        allowed_slots = set(allowed_by_category.get(category, {"company", "location", "industry", "person"}))
        allowed_slots.update(literal_slots)
        raw_hints = "; ".join(hints_map.get(category, [])) or "None"
        hints_text = raw_hints.replace("{", "{{").replace("}", "}}")
        prompt = TEMPLATE_REQUEST.format(
            schema=self.schema_summary,
            category=category,
            n=n,
            avoid=avoid_text,
            hints=hints_text,
        )
        if self.logger:
            self.logger.debug("Generating templates for category=%s", category)
        schema = {
            "type": "array",
            "minItems": n,
            "maxItems": n,
            "items": {
                "type": "object",
                "properties": {
                    "template": {"type": "string"},
                    "slots": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["template", "slots"],
            },
        }
        response = self.llm.chat(
            system=SYSTEM_NL_TEMPLATE,
            user=prompt,
            temperature=0.4,
            json_schema=schema,
            max_tokens=512,
        )
        if self.logger:
            self.logger.debug("Template response raw: %s", response)
        try:
            data = json.loads(response)
        except Exception as exc:
            raise ValueError(f"Template JSON parse failed for category={category}: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError(f"Template JSON is not a list for category={category}")
        def normalize_template(item: Dict[str, object]) -> Dict[str, object]:
            template = item.get("template", "")
            if not template:
                return item
            while "{{" in template or "}}" in template:
                template = template.replace("{{", "{").replace("}}", "}")
            slots = item.get("slots", [])
            placeholders = re.findall(r"\{([A-Za-z0-9_]+)\}", template)
            if not placeholders:
                item["slots"] = []
                return item
            if len(placeholders) > 2:
                return item
            counts = {p: placeholders.count(p) for p in set(placeholders)}
            idx_map: Dict[str, int] = {}
            new_slots: List[str] = []

            def repl(match):
                name = match.group(1)
                idx = idx_map.get(name, 0) + 1
                idx_map[name] = idx
                new_name = f"{name}{idx}" if counts.get(name, 0) > 1 else name
                if new_name not in new_slots:
                    new_slots.append(new_name)
                return "{" + new_name + "}"

            new_template = re.sub(r"\{([A-Za-z0-9_]+)\}", repl, template)
            item["template"] = new_template
            if new_slots:
                item["slots"] = new_slots
            elif slots:
                item["slots"] = slots
            return item

        filtered: List[Dict[str, object]] = []
        banned_phrases = (
            "type of",
            "rdf rank",
            "most popular",
            "most famous",
        )
        banned_slot_tokens = ()
        allowed_slot_tokens = set(literal_slots)
        class_noun_slots = ("person", "location", "company", "organization", "organisation", "feature", "place", "agent")
        superlatives = (
            "most",
            "least",
            "tallest",
            "oldest",
            "youngest",
            "largest",
            "smallest",
            "highest",
            "lowest",
            "earliest",
            "latest",
            "first",
            "last",
            "newest",
            "recent",
            "longest",
            "shortest",
            "biggest",
            "greatest",
            "top",
            "bottom",
            "best",
            "worst",
            "maximum",
            "minimum",
            "max",
            "min",
            "fewest",
            "highest-ranked",
            "lowest-ranked",
            "top-ranked",
            "bottom-ranked",
        )
        rejected = []
        for item in data:
            item = normalize_template(item)
            if not isinstance(item, dict):
                raise ValueError(f"Template entry is not an object for category={category}: {item}")
            if "template" not in item or "slots" not in item:
                raise ValueError(f"Template entry missing keys for category={category}: {item}")
            if not isinstance(item["template"], str):
                raise ValueError(f"Template entry template is not a string for category={category}: {item}")
            if not isinstance(item["slots"], list) or not all(isinstance(s, str) for s in item["slots"]):
                raise ValueError(f"Template entry slots invalid for category={category}: {item}")
            placeholders = re.findall(r"\{([A-Za-z0-9_]+)\}", item["template"])
            if placeholders and set(placeholders) != set(item["slots"]):
                item["slots"] = placeholders
            if len(item["slots"]) > 2:
                rejected.append(("too_many_slots", item))
                continue
            template_lower = item["template"].lower()
            slot_names = [s.lower() for s in item["slots"]]
            if category == "superlative" and not any(word in template_lower for word in superlatives):
                rejected.append(("superlative_missing_keyword", item))
                continue
            if any(word in template_lower for word in superlatives) and item["slots"]:
                if category != "superlative":
                    rejected.append(("superlative_with_slots", item))
                    continue
            if any(p in template_lower for p in banned_phrases):
                rejected.append(("banned_phrase", item))
                continue
            if len(set(slot_names)) != len(slot_names):
                rejected.append(("duplicate_slots", item))
                continue
            template_text = item["template"]
            template_lower = template_text.lower()
            for slot in slot_names:
                slot_base = re.sub(r"\\d+$", "", slot)
                if slot_base in class_noun_slots:
                    bad_patterns = [
                        rf"\bhow many\s+\{{{re.escape(slot)}\}}",
                        rf"\bwhich\s+\{{{re.escape(slot)}\}}",
                        rf"\bwhat\s+\{{{re.escape(slot)}\}}",
                        rf"\bwho\s+\{{{re.escape(slot)}\}}",
                        rf"\b(the|a|an)\s+\{{{re.escape(slot)}\}}",
                    ]
                    for sup in superlatives:
                        bad_patterns.append(rf"\b{sup}\b\s+\w*\s*\{{{re.escape(slot)}\}}")
                    if any(re.search(pat, template_lower) for pat in bad_patterns):
                        break
                    if f"the {{{slot}}}" in template_lower or f"a {{{slot}}}" in template_lower or f"an {{{slot}}}" in template_lower:
                        break
                    pattern = rf"(tallest|oldest|youngest|largest|smallest|highest|lowest|most|least)\s+\\w*\\s*\\{{{re.escape(slot)}\\}}"
                    if re.search(pattern, template_lower):
                        break
            else:
                for slot in slot_names:
                    slot_base = re.sub(r"\\d+$", "", slot)
                    if slot_base in {"company", "employee", "agent", "feature", "metric"} and slot_base not in schema_lower:
                        break
                else:
                    filtered.append(item)
                continue
            rejected.append(("class_noun_placeholder", item))
        if self.logger and rejected:
            for reason, item in rejected:
                self.logger.debug("Template rejected (%s): %s", reason, item.get("template"))
        if filtered:
            filtered = [
                item
                for item in filtered
                if not item["slots"] or all(slot in allowed_slots for slot in item["slots"])
            ]
            if category in {"comparative", "difference"}:
                allowed_pairs = [
                    {"company1", "company2"},
                    {"person1", "person2"},
                    {"location1", "location2"},
                    {"industry1", "industry2"},
                    {"year1", "year2"},
                    {"number1", "number2"},
                ]
                filtered = [item for item in filtered if set(item["slots"]) in allowed_pairs]
            if category == "ordinal":
                filtered = [item for item in filtered if set(item["slots"]).issubset(allowed_slots)]
        if not filtered:
            fallback = fallback_map.get(category, [])
            if avoid_templates:
                fallback = [item for item in fallback if item.get("template") not in avoid_templates]
            if fallback:
                if self.logger:
                    self.logger.warning(
                        "Using fallback templates for category=%s (LLM templates rejected).",
                        category,
                    )
                filtered = [
                    item
                    for item in fallback
                    if not item["slots"] or all(slot in allowed_slots for slot in item["slots"])
                ]
                if category in {"comparative", "difference"}:
                    filtered = [item for item in filtered if set(item["slots"]) == {"company1", "company2"}]
                if category == "ordinal":
                    filtered = [item for item in filtered if set(item["slots"]).issubset(allowed_slots)]
            if not filtered:
                raise ValueError(f"No templates within slot limit (<=2) for category={category}")
        return filtered

    def reverse_query(self, template: str, slots: List[str]) -> str:
        hint_lines = []
        if self.slot_type_hints:
            for slot in slots:
                base = re.sub(r"\\d+$", "", slot)
                t = self.slot_type_hints.get(base)
                if t:
                    hint_lines.append(f"{slot}: {t}")
        hint_text = "\n".join(hint_lines) if hint_lines else "None"
        prompt = REVERSE_QUERY_REQUEST.format(
            prefixes=self.prefixes,
            schema=self.schema_summary,
            template=template,
            slots=", ".join(slots),
            slot_type_hints=hint_text,
        )
        if self.logger:
            self.logger.debug("Reverse querying template=%s", template)
        response = self.llm.chat(
            system=SYSTEM_SPARQL_GEN,
            user=prompt,
            temperature=0.2,
            timeout_sec=self.reverse_query_timeout_sec,
            max_tokens=512,
        )
        if self.logger:
            self.logger.debug("Reverse query LLM raw: %s", response)
        sparql = self.extract_sparql(response)
        sanitized = self.sanitize_reverse_sparql(sparql, template, slots)
        if self.logger and sanitized != sparql:
            self.logger.debug("Reverse query sanitized: %s", sanitized)
        sparql = self.ensure_select_vars(sanitized, slots)
        if "PREFIX" not in sparql.upper():
            sparql = self.prefixes + "\n" + sparql
        if "LIMIT" in sparql.upper():
            sparql = re.sub(r"(?i)LIMIT\s+\d+", "LIMIT 25", sparql)
        else:
            sparql = sparql.strip() + "\nLIMIT 25"
        return sparql

    def fill_template(self, template: str, bindings: Dict[str, str]) -> str:
        text = template
        for slot, value in bindings.items():
            text = text.replace("{" + slot + "}", value)
        return text

    def entity_type_for_uri(self, uri: str) -> Optional[str]:
        if uri in self._type_cache:
            return self._type_cache[uri]
        bad_types = {
            "http://www.w3.org/2002/07/owl#Class",
            "http://www.w3.org/2002/07/owl#Thing",
            "http://www.w3.org/2000/01/rdf-schema#Class",
        }
        q = f"SELECT ?type WHERE {{ <{uri}> a ?type }} LIMIT 5"
        ok, _, rows, _ = self.execute_rows(q)
        if ok and rows:
            for row in rows:
                t = row.get("type")
                if t and t not in bad_types:
                    self._type_cache[uri] = t
                    return t
        self._type_cache[uri] = None
        return None

    def label_for_type(self, type_uri: str) -> str:
        if type_uri in self._type_label_cache:
            return self._type_label_cache[type_uri]
        label_props = [
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://www.ldbcouncil.org/spb#prefLabel",
            "http://purl.org/dc/terms/title",
        ]
        for prop in label_props:
            q = f"SELECT ?label WHERE {{ <{type_uri}> <{prop}> ?label }} LIMIT 1"
            ok, _, rows, _ = self.execute_rows(q)
            if ok and rows:
                label = rows[0].get("label", type_uri)
                self._type_label_cache[type_uri] = label
                return label
        if type_uri:
            local = type_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            if local.lower() == "feature":
                self._type_label_cache[type_uri] = "Location"
                return "Location"
            if local.lower() == "company":
                self._type_label_cache[type_uri] = "Company"
                return "Company"
            if local.lower() == "person":
                self._type_label_cache[type_uri] = "Person"
                return "Person"
        fallback = type_uri.rsplit("/", 1)[-1]
        self._type_label_cache[type_uri] = fallback
        return fallback

    def extract_sparql(self, text: str) -> str:
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
            else:
                text = parts[-1]
        text = text.replace("sparql", "").strip()
        text = self.strip_comment_lines(text)
        if text.lower().startswith("select") or text.lower().startswith("ask"):
            return self.truncate_sparql(text)
        for token in ("SELECT", "ASK"):
            idx = text.upper().find(token)
            if idx != -1:
                return self.truncate_sparql(text[idx:])
        return text

    def truncate_sparql(self, text: str) -> str:
        upper = text.upper()
        if "LIMIT" in upper:
            lim_idx = upper.find("LIMIT")
            line_end = text.find("\n", lim_idx)
            if line_end != -1:
                return text[:line_end].strip()
            return text.strip()
        if "}" in text:
            last = text.rfind("}")
            return text[: last + 1].strip()
        return text.strip()

    def strip_comment_lines(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("--"):
                continue
            # Remove inline comments
            if "#" in line:
                line = line.split("#", 1)[0]
            if "--" in line:
                line = line.split("--", 1)[0]
            if line.strip():
                lines.append(line.rstrip())
        return "\n".join(lines)

    def wrap_bare_iris(self, text: str) -> str:
        pattern = re.compile(r'(?<![<"\'])(https?://[^\s>\)"]+)(?!>)')
        return pattern.sub(lambda m: f"<{m.group(1)}>", text)

    def extract_first_query(self, text: str) -> str:
        lines = text.splitlines()
        prefixes = []
        rest = []
        for line in lines:
            if line.strip().upper().startswith("PREFIX"):
                prefixes.append(line)
            else:
                rest.append(line)
        body = "\n".join(rest).strip()
        match = re.search(r"(?i)\b(SELECT|ASK)\b", body)
        if not match:
            return text.strip()
        body = body[match.start():]
        body_full = body
        brace_start = body.find("{")
        if brace_start != -1:
            depth = 0
            end_idx = None
            for idx, ch in enumerate(body[brace_start:], start=brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = idx
                        break
            if end_idx is not None:
                tail = body_full[end_idx + 1 :]
                tail_lines = []
                for line in tail.splitlines():
                    stripped = line.strip()
                    if stripped.upper().startswith(("ORDER BY", "LIMIT", "OFFSET")):
                        tail_lines.append(stripped)
                body = body_full[: end_idx + 1]
                if tail_lines:
                    body = body + "\n" + "\n".join(tail_lines)
        merged = "\n".join(prefixes + [body]).strip()
        return merged

    def normalize_generated_select(self, sparql: str) -> str:
        match = re.search(r"(?is)\bSELECT\b(.*?)\bWHERE\b", sparql)
        if not match:
            return sparql
        select_clause = match.group(1)
        select_line = ""
        for line in sparql.splitlines():
            if line.strip().upper().startswith("SELECT"):
                select_line = line
                break
        simple_count = re.search(r"(?is)SELECT\\s+COUNT\\s*\\(\\s*([^\\)]+)\\s*\\)", sparql)
        if simple_count and "AS" not in select_clause.upper() and "AS" not in select_line.upper():
            var = simple_count.group(1).strip()
            new_select = f"SELECT (COUNT({var}) AS ?count)\nWHERE"
            sparql = re.sub(r"(?is)SELECT\\s+COUNT\\s*\\(\\s*[^\\)]+\\s*\\)\\s*WHERE", new_select, sparql)
            return sparql
        if "COUNT(" in select_clause.upper():
            count_match = re.search(r"COUNT\s*\(\s*([^\)]+)\s*\)", select_clause, re.IGNORECASE)
            if count_match:
                var = count_match.group(1).strip()
                new_select = f"SELECT (COUNT({var}) AS ?count)\nWHERE"
                sparql = sparql[: match.start()] + new_select + sparql[match.end():]
                return sparql
        if select_line and "COUNT(" in select_line.upper() and "AS" not in select_line.upper():
            line_match = re.search(r"COUNT\\s*\\(\\s*([^\\)]+)\\s*\\)", select_line, re.IGNORECASE)
            if line_match:
                var = line_match.group(1).strip()
                replacement = f"SELECT (COUNT({var}) AS ?count)"
                sparql = sparql.replace(select_line, replacement, 1)
                return sparql
        where_block = sparql[match.end():]
        count_in_where = re.search(r"COUNT\s*\(\s*(\?\w+|\*)\s*\)", where_block, re.IGNORECASE)
        if count_in_where and "COUNT" not in select_clause.upper():
            var = count_in_where.group(1)
            if var == "*":
                var = "?x"
            new_select = f"SELECT (COUNT({var}) AS ?count)\nWHERE"
            sparql = sparql[: match.start()] + new_select + sparql[match.end():]
            sparql = re.sub(r"^.*COUNT\s*\(.*\).*$\n?", "", sparql, flags=re.IGNORECASE | re.MULTILINE)
            return sparql
        bad_tokens = ("CASE", "DATE_SUB", "INTERVAL", "STRDT", "SUM(", "AVG(", "MIN(", "MAX(")
        if any(tok in select_clause.upper() for tok in bad_tokens):
            vars_found = re.findall(r"\?(\w+)", where_block)
            unique = []
            for name in vars_found:
                var = f"?{name}"
                if var not in unique:
                    unique.append(var)
            if unique:
                new_select = "SELECT " + " ".join(unique[:2]) + "\nWHERE"
                sparql = sparql[: match.start()] + new_select + sparql[match.end():]
        return sparql

    def sanitize_generated_sparql(self, sparql: str) -> str:
        sparql = self.strip_comment_lines(sparql)
        sparql = self.extract_first_query(sparql)
        while "{{" in sparql or "}}" in sparql:
            sparql = sparql.replace("{{", "{").replace("}}", "}")
        sparql = re.sub(r"(?is)SELECT\s*\(\s*(\?\w+)\s*\)", r"SELECT \1", sparql)
        sparql = re.sub(
            r"(?is)SELECT\s+COUNT\s*\(\s*([^\)]+)\s*\)",
            r"SELECT (COUNT(\1) AS ?count)",
            sparql,
        )
        sparql = self.wrap_bare_iris(sparql)
        sparql = re.sub(r'\"(https?://[^\"\\s]+)\"', r"<\\1>", sparql)
        sparql = self.normalize_generated_select(sparql)
        cleaned_lines = []
        order_by_lines = []
        tautology_re = re.compile(r"FILTER\s*\(\s*\?(\w+)\s*(!=|=)\s*\?\1\s*\)", re.IGNORECASE)
        for line in sparql.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if "GROUP BY" in upper or "HAVING" in upper:
                continue
            if "ORDER BY" in upper:
                normalized = line
                m = re.search(r"ORDER\s+BY\s+\?(\w+)\s+(ASC|DESC)\b", stripped, re.IGNORECASE)
                if m:
                    direction = m.group(2).upper()
                    normalized = f"ORDER BY {direction}(?{m.group(1)})"
                order_by_lines.append(normalized)
                continue
            if stripped.upper().startswith("BIND"):
                continue
            if stripped.upper() == "UNION":
                continue
            if tautology_re.search(stripped):
                continue
            if stripped.upper().startswith("FILTER") and "NULL" in stripped.upper():
                continue
            if stripped.upper().startswith(("FILTER", "BIND", "VALUES")) and stripped.endswith("."):
                line = line.rstrip().rstrip(".").rstrip()
            cleaned_lines.append(line)
        if order_by_lines:
            seen = set()
            deduped = []
            for line in order_by_lines:
                key = line.strip().upper()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(line)
            order_by_lines = deduped
            limit_idx = None
            for idx, line in enumerate(cleaned_lines):
                if line.strip().upper().startswith("LIMIT"):
                    limit_idx = idx
                    break
            if limit_idx is None:
                cleaned_lines.extend(order_by_lines)
            else:
                cleaned_lines[limit_idx:limit_idx] = order_by_lines
        sparql = "\n".join(cleaned_lines)
        open_braces = sparql.count("{")
        close_braces = sparql.count("}")
        if close_braces < open_braces:
            sparql = sparql.rstrip() + "\n" + ("}" * (open_braces - close_braces))
        if self.generated_query_limit:
            if "LIMIT" not in sparql.upper():
                sparql = sparql.strip() + f"\nLIMIT {self.generated_query_limit}"
        return sparql.strip()

    def apply_entity_hints(self, sparql: str, entity_hints: Dict[str, str]) -> str:
        if not entity_hints:
            return sparql
        iri_map: Dict[str, str] = {}
        for slot, hint in entity_hints.items():
            if not hint:
                continue
            iri = hint.split("|", 1)[0].strip()
            if iri:
                iri_map[slot] = iri
        if not iri_map:
            return sparql
        lines = []
        for line in sparql.splitlines():
            stripped = line.strip().rstrip(".").strip()
            if not stripped:
                lines.append(line)
                continue
            parts = stripped.split()
            if len(parts) == 2 and parts[0].startswith("?") and parts[1].startswith("<"):
                var = parts[0][1:]
                if var in iri_map:
                    continue
            lines.append(line)
        sparql = "\n".join(lines)
        insert_lines = []
        for slot, iri in iri_map.items():
            if not iri:
                continue
            if not iri.startswith("<"):
                iri = f"<{iri}>"
            insert_lines.append(f"  VALUES ?{slot} {{ {iri} }}")
        if insert_lines and "WHERE" in sparql.upper():
            sparql = re.sub(
                r"(?is)(WHERE\\s*\\{)",
                lambda m: m.group(1) + "\n" + "\n".join(insert_lines),
                sparql,
                count=1,
            )
        return sparql

    def sanitize_reverse_sparql(self, sparql: str, template: str = "", slots: Optional[List[str]] = None) -> str:
        sparql = self.strip_comment_lines(sparql)
        while "{{" in sparql or "}}" in sparql:
            sparql = sparql.replace("{{", "{").replace("}}", "}")
        sparql = re.sub(r"\bLOWER\s*\(", "LCASE(", sparql, flags=re.IGNORECASE)
        sparql = re.sub(r"\bUPPER\s*\(", "UCASE(", sparql, flags=re.IGNORECASE)
        template_has_quotes = '"' in template or "'" in template
        template_has_number = bool(re.search(r"\\d", template))
        slot_vars = set(slots or [])
        allowed_type_vars = set()
        if slot_vars and self.slot_type_hints:
            for slot in slot_vars:
                base = re.sub(r"\\d+$", "", slot)
                if base in self.slot_type_hints:
                    allowed_type_vars.add(slot)
        lines = []
        for line in sparql.splitlines():
            upper = line.upper()
            if "ORDER BY" in upper or "GROUP BY" in upper or "HAVING" in upper:
                continue
            if upper.lstrip().startswith(("FILTER", "BIND")):
                continue
            if not template_has_quotes and "FILTER" in upper and ("\"" in line or "'" in line):
                continue
            if not template_has_number and "FILTER" in upper and re.search(r"\\d", line):
                continue
            if not template_has_quotes and re.search(r"['\\\"]", line) and line.strip().startswith("?"):
                continue
            lines.append(line)
        sparql = "\n".join(lines)

        fn_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\\s*\\(.*\\)\\s*$")
        cleaned = []
        for line in sparql.splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned.append(line)
                continue
            if stripped.upper().startswith(("PREFIX", "SELECT", "WHERE", "ASK", "LIMIT", "FILTER", "BIND")):
                cleaned.append(line)
                continue
            if fn_pattern.match(stripped):
                continue
            if stripped.startswith("?") and "=" in stripped:
                continue
            type_match = re.match(r"^(\\?[A-Za-z_][A-Za-z0-9_]*)\\s+(a|rdf:type)\\s+(.+)$", stripped)
            if type_match and slot_vars:
                subj_var = type_match.group(1).lstrip("?")
                if subj_var not in allowed_type_vars:
                    continue
            if stripped.startswith("a ") and stripped.split()[1].startswith(("http://", "https://")):
                parts = stripped.split()
                parts[1] = f"<{parts[1]}>"
                cleaned.append(" ".join(parts))
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                for idx in (1, 2):
                    if parts[idx].startswith(("http://", "https://")):
                        parts[idx] = f"<{parts[idx]}>"
                cleaned.append(" ".join(parts))
                continue
            cleaned.append(line)
        sparql = "\n".join(cleaned)

        cleaned_lines = []
        for line in sparql.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith(("FILTER", "BIND", "VALUES")) and stripped.endswith("."):
                line = line.rstrip().rstrip(".").rstrip()
            cleaned_lines.append(line)
        sparql = "\n".join(cleaned_lines)

        sparql = re.sub(r"LIMIT\\s+\\d+\\b", "LIMIT 85", sparql, flags=re.IGNORECASE)
        if "LIMIT" not in sparql.upper():
            sparql = sparql.strip() + "\nLIMIT 85"
        return sparql

    def ensure_select_vars(self, sparql: str, slots: List[str]) -> str:
        if not slots:
            return sparql
        match = re.search(r"(?is)SELECT\s+(DISTINCT|REDUCED)?\s*(.*?)\s+WHERE", sparql)
        if not match:
            return sparql
        select_body = match.group(2)
        if "*" in select_body:
            return sparql
        existing = set(re.findall(r"\?\w+", select_body))
        missing = [f"?{s}" for s in slots if f"?{s}" not in existing]
        if not missing:
            return sparql
        new_select = select_body + " " + " ".join(missing)
        start, end = match.span(2)
        return sparql[:start] + new_select + sparql[end:]

    def enforce_category_patterns(self, sparql: str, category: str) -> str:
        """Apply category-specific SPARQL best practices."""
        upper = sparql.upper()
        # Yes/No: convert SELECT to ASK if not already ASK
        if category == "yesno" and "ASK" not in upper and "SELECT" in upper:
            # Extract WHERE block and convert to ASK
            where_match = re.search(r"(?is)(WHERE\s*\{.*\})", sparql)
            if where_match:
                prefix_lines = []
                for line in sparql.splitlines():
                    if line.strip().upper().startswith("PREFIX"):
                        prefix_lines.append(line)
                ask_body = where_match.group(1)
                # Remove LIMIT/ORDER BY from ASK
                ask_body = re.sub(r"(?i)\s*ORDER\s+BY\s+[^\n]+", "", ask_body)
                ask_body = re.sub(r"(?i)\s*LIMIT\s+\d+", "", ask_body)
                sparql = "\n".join(prefix_lines) + "\nASK " + ask_body

        # Counting: ensure COUNT(DISTINCT ...)
        if category == "counting" and "COUNT(" in upper:
            sparql = re.sub(
                r"COUNT\s*\(\s*(?!DISTINCT)(\?\w+)",
                r"COUNT(DISTINCT \1",
                sparql,
                flags=re.IGNORECASE,
            )

        # Set-returning categories: ensure SELECT DISTINCT
        if category in ("intersection", "difference", "multi-hop"):
            if re.search(r"(?i)^(.*?)SELECT\s+(?!DISTINCT)", sparql):
                sparql = re.sub(
                    r"(?i)(SELECT)\s+(?!DISTINCT)",
                    r"\1 DISTINCT ",
                    sparql,
                    count=1,
                )

        # Superlative/ordinal: ensure LIMIT 1 if ORDER BY present
        if category in ("superlative", "ordinal") and "ORDER BY" in upper:
            if "LIMIT" not in upper:
                sparql = sparql.rstrip() + "\nLIMIT 1"
            else:
                sparql = re.sub(r"(?i)LIMIT\s+\d+", "LIMIT 1", sparql)

        return sparql

    def generate_sparql(
        self,
        question: str,
        category: str,
        examples: Optional[List[Dict[str, str]]] = None,
        entity_hints: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, float, int, int]:
        prompt = self.build_prompt(question, category, examples, entity_hints)
        prompt_chars = len(prompt)
        prompt_tokens = estimate_tokens(prompt)
        start = time.time()
        response = self.llm.chat(system=SYSTEM_SPARQL_GEN, user=prompt, temperature=0.2, max_tokens=512)
        if self.logger:
            self.logger.debug("SPARQL gen LLM raw: %s", response)
        latency = (time.time() - start) * 1000
        sparql = self.extract_sparql(response)
        sanitized = self.sanitize_generated_sparql(sparql)
        if self.logger and sanitized != sparql:
            self.logger.debug("SPARQL gen sanitized: %s", sanitized)
        sparql = sanitized
        sparql = self.enforce_category_patterns(sparql, category)
        if entity_hints:
            sparql = self.apply_entity_hints(sparql, entity_hints)
        if "PREFIX" not in sparql.upper():
            sparql = self.prefixes + "\n" + sparql
        return sparql, latency, prompt_chars, prompt_tokens

    def execute(self, sparql: str) -> Tuple[bool, float, List[str], Optional[str]]:
        try:
            if self.logger:
                self.logger.debug("SPARQL exec query: %s", self.truncate_query(sparql))
            start = time.time()
            res = self.sparql.query(sparql)
            elapsed = (time.time() - start) * 1000
            if res.boolean is not None:
                answers = [str(res.boolean)]
            else:
                answers = []
                for row in res.rows:
                    answers.extend([str(v) for v in row.values()])
            if self.logger:
                self.logger.debug("SPARQL exec ok | answers=%d ms=%.1f", len(answers), elapsed)
            return True, elapsed, answers, None
        except Exception as exc:
            if self.logger:
                self.logger.debug("SPARQL exec error: %s | query=%s", exc, self.truncate_query(sparql))
            return False, 0.0, [], str(exc)

    def execute_rows(self, sparql: str) -> Tuple[bool, float, List[Dict[str, str]], Optional[str]]:
        try:
            if self.logger:
                self.logger.debug("SPARQL rows query: %s", self.truncate_query(sparql))
            start = time.time()
            res = self.sparql.query(sparql)
            elapsed = (time.time() - start) * 1000
            if self.logger:
                self.logger.debug("SPARQL rows ok | rows=%d ms=%.1f", len(res.rows), elapsed)
            return True, elapsed, res.rows, None
        except Exception as exc:
            if self.logger:
                self.logger.debug("SPARQL rows error: %s | query=%s", exc, self.truncate_query(sparql))
            return False, 0.0, [], str(exc)

    def truncate_query(self, sparql: str, limit: int = 500) -> str:
        text = sparql.replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def label_for_uri(self, uri: str) -> str:
        if uri in self._label_cache:
            return self._label_cache[uri]
        label_props = [
            "http://xmlns.com/foaf/0.1/name",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://www.ldbcouncil.org/spb#prefLabel",
            "http://purl.org/dc/terms/title",
        ]
        for prop in label_props:
            q = f"SELECT ?label WHERE {{ <{uri}> <{prop}> ?label }} LIMIT 1"
            ok, _, rows, _ = self.execute_rows(q)
            if ok and rows:
                label = rows[0].get("label", uri)
                self._label_cache[uri] = label
                return label
        fallback = uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        try:
            from urllib.parse import unquote

            fallback = unquote(fallback)
        except Exception:
            pass
        fallback = fallback.replace("_", " ").strip()
        self._label_cache[uri] = fallback or uri
        return self._label_cache[uri]

    def repair(self, question: str, sparql: str, error: str) -> str:
        prompt = REPAIR_REQUEST.format(schema=self.schema_summary, question=question, sparql=sparql, error=error)
        response = self.llm.chat(system=SYSTEM_REPAIR, user=prompt, temperature=0.2, max_tokens=512)
        repaired = self.extract_sparql(response)
        repaired = self.sanitize_generated_sparql(repaired)
        if "PREFIX" not in repaired.upper():
            repaired = self.prefixes + "\n" + repaired
        return repaired

    def paraphrase(self, question: str) -> List[str]:
        prompt = PARAPHRASE_REQUEST.format(question=question)
        schema = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"}}
        response = self.llm.chat(
            system=SYSTEM_PARAPHRASE,
            user=prompt,
            temperature=0.5,
            json_schema=schema,
            max_tokens=256,
        )
        if self.logger:
            self.logger.debug("Paraphrase response raw: %s", response)
        try:
            data = json.loads(response)
        except Exception as exc:
            raise ValueError(f"Paraphrase JSON parse failed: {exc}") from exc
        if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
            raise ValueError("Paraphrase response must be a JSON list of strings")
        return data

    def run_single(
        self,
        question: str,
        category: str,
        examples: Optional[List[Dict[str, str]]] = None,
        entity_hints: Optional[Dict[str, str]] = None,
        repair_attempts: int = 1,
    ) -> GenerationRecord:
        if self.logger:
            self.logger.debug("Run single: category=%s question=%s", category, question)
        sparql, llm_latency, prompt_chars, prompt_tokens = self.generate_sparql(
            question, category, examples, entity_hints
        )
        parse_ok, parse_err = parse_valid_sparql_detail(sparql)
        ast_nodes = 0
        ast_depth = 0
        if parse_ok:
            try:
                ast_nodes, ast_depth, _ = ast_stats(sparql)
            except Exception:
                ast_nodes, ast_depth = 0, 0
        exec_ok, exec_ms, answers, error = self.execute(sparql) if parse_ok else (False, 0.0, [], None)
        error_type = None
        if not parse_ok:
            error = f"parse_error: {parse_err}"
            error_type = "parse_error"
            if self.logger:
                self.logger.warning(
                    "SPARQL parse failed | category=%s question=%s error=%s sparql=%s",
                    category,
                    question,
                    parse_err,
                    sparql,
                )
        attempts = 0

        if not exec_ok and repair_attempts > 0:
            if not error_type:
                error_type = "endpoint_error" if error not in (None, "parse_error") else "parse_error"
            while attempts < repair_attempts:
                attempts += 1
                repaired = self.repair(question, sparql, error or "")
                parse_ok, parse_err = parse_valid_sparql_detail(repaired)
                exec_ok, exec_ms, answers, error = self.execute(repaired) if parse_ok else (False, 0.0, [], "parse_error")
                if exec_ok and answers:
                    sparql = repaired
                    if parse_ok:
                        try:
                            ast_nodes, ast_depth, _ = ast_stats(sparql)
                        except Exception:
                            ast_nodes, ast_depth = 0, 0
                    break

        if exec_ok and answers:
            error_type = None
            error = None

        if exec_ok and not answers:
            error_type = "empty_result"

        # Validate answer types match category expectations
        if exec_ok and answers and not validate_answer_type(category, answers, sparql):
            if self.logger:
                self.logger.warning(
                    "Answer type mismatch | category=%s question=%s answers=%s",
                    category, question, answers[:3],
                )
            error_type = "answer_type_mismatch"

        return GenerationRecord(
            category=category,
            question=question,
            sparql=sparql,
            answers=answers,
            exec_success=exec_ok,
            parse_valid=parse_ok,
            error=error,
            error_type=error_type,
            repair_attempts=attempts,
            llm_latency_ms=llm_latency,
            question_latency_ms=0.0,
            sparql_exec_ms=exec_ms,
            answer_count=len(answers),
            result_hash=result_set_hash(answers) if answers else None,
            prompt_chars=prompt_chars,
            prompt_tokens_est=prompt_tokens,
            retrieved_examples=examples or [],
            ast_node_count=ast_nodes,
            ast_max_depth=ast_depth,
        )


def build_faiss_index(llm: LLMClient, examples: List[Dict[str, str]]) -> FaissStore:
    texts = [ex["question"] for ex in examples]
    embeddings = np.array(llm.embed_texts(texts), dtype="float32")
    return FaissStore.build(embeddings, examples)


def retrieve_examples(llm: LLMClient, store: FaissStore, query: str, k: int = 3) -> List[Dict[str, str]]:
    emb = np.array(llm.embed_texts([query]), dtype="float32")
    hits = store.search_with_scores(emb, k=k)[0]
    return hits
