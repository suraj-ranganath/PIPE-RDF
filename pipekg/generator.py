from dataclasses import dataclass
from typing import Dict, List, Optional
import random

from rdflib import URIRef

from .kg import SyntheticKG
from .templates import QuestionTemplate
from .retrieval import Retriever, Example
from .utils import tokenize, cosine_sim


@dataclass
class GenerationResult:
    question: str
    sparql: str
    answers: List[str]
    category: str
    template_name: str
    phase: str
    retrieved_examples: List[Example]


class BenchmarkGenerator:
    def __init__(self, kg: SyntheticKG, templates: List[QuestionTemplate]) -> None:
        self.kg = kg
        self.templates = templates
        self.retriever = Retriever()
        self._bindings_cache: Dict[str, List[Dict[str, URIRef]]] = {}

    def _bindings_for_template(self, template: QuestionTemplate) -> List[Dict[str, URIRef]]:
        if template.name in self._bindings_cache:
            return self._bindings_cache[template.name]
        if not template.slot_vars:
            bindings = [{}]
            self._bindings_cache[template.name] = bindings
            return bindings
        try:
            bindings = self.kg.run_select(template.slot_query)
            self._bindings_cache[template.name] = bindings
            return bindings
        except Exception:
            self._bindings_cache[template.name] = []
            return []

    def _fill_question(self, template: QuestionTemplate, binding: Dict[str, URIRef]) -> str:
        question = template.question_template
        for var in template.slot_vars:
            uri = binding.get(var)
            if uri is None:
                continue
            label = self.kg.label_of(uri)
            question = question.replace("{" + var + "}", label)
        return question

    def _fill_sparql(self, template: QuestionTemplate, binding: Dict[str, URIRef]) -> str:
        sparql = template.sparql_template
        for var in template.slot_vars:
            uri = binding.get(var)
            if uri is None:
                continue
            sparql = sparql.replace("{" + var + "}", f"<{uri}>")
        return sparql

    def _execute(self, sparql: str) -> List[str]:
        try:
            return self.kg.run_query_answers(sparql)
        except Exception:
            return []

    def generate_one(self, template: QuestionTemplate, phase: str, retrieval_query: Optional[str] = None) -> Optional[GenerationResult]:
        bindings = self._bindings_for_template(template)
        if not bindings:
            return None
        binding = random.choice(bindings)
        question = self._fill_question(template, binding)
        sparql = self._fill_sparql(template, binding)
        answers = self._execute(sparql)
        if not answers and template.category != "yesno":
            return None

        retrieved = []
        if retrieval_query is not None:
            retrieved = self.retriever.top_k(retrieval_query, k=3, category=template.category)

        return GenerationResult(
            question=question,
            sparql=sparql,
            answers=answers,
            category=template.category,
            template_name=template.name,
            phase=phase,
            retrieved_examples=retrieved,
        )

    def rephrase(self, question: str) -> List[str]:
        variants = []
        swaps = {
            "Which": "What",
            "What": "Which",
            "How many": "Number of",
            "directed": "helmed",
            "produced": "made",
            "films": "movies",
            "actor": "performer",
            "actors": "performers",
            "genre": "category",
        }
        for src, tgt in swaps.items():
            if src in question:
                variants.append(question.replace(src, tgt))
        if question.startswith("Which"):
            variants.append("List " + question[6:].rstrip("?") + ".")
        return list(dict.fromkeys(variants))

    def is_duplicate(self, question: str, existing: List[str], threshold: float = 0.95) -> bool:
        q_tokens = tokenize(question)
        for ex in existing:
            sim = cosine_sim(q_tokens, tokenize(ex))
            if sim >= threshold:
                return True
        return False

    def generate_phase(self, templates: List[QuestionTemplate],
                       per_template: int,
                       phase: str,
                       use_retrieval: bool = False,
                       include_rephrase: bool = False) -> List[GenerationResult]:
        results: List[GenerationResult] = []
        questions_seen: List[str] = []
        for template in templates:
            attempts = 0
            added = 0
            while added < per_template and attempts < per_template * 5:
                attempts += 1
                retrieval_query = template.question_template if use_retrieval else None
                result = self.generate_one(template, phase=phase, retrieval_query=retrieval_query)
                if result is None:
                    continue
                if self.is_duplicate(result.question, questions_seen):
                    continue
                results.append(result)
                questions_seen.append(result.question)
                added += 1

                if include_rephrase:
                    for variant in self.rephrase(result.question):
                        if self.is_duplicate(variant, questions_seen):
                            continue
                        results.append(
                            GenerationResult(
                                question=variant,
                                sparql=result.sparql,
                                answers=result.answers,
                                category=result.category,
                                template_name=result.template_name,
                                phase=phase,
                                retrieved_examples=result.retrieved_examples,
                            )
                        )
                        questions_seen.append(variant)
        return results

    def index_examples(self, results: List[GenerationResult]) -> None:
        for r in results:
            self.retriever.add(Example(question=r.question, sparql=r.sparql, category=r.category))

    def generate_category_wise(self, per_category: int, phase: str, use_retrieval: bool = True, include_rephrase: bool = False) -> List[GenerationResult]:
        results: List[GenerationResult] = []
        questions_seen: List[str] = []
        categories = sorted(set(t.category for t in self.templates))
        for cat in categories:
            cat_templates = [t for t in self.templates if t.category == cat]
            added = 0
            attempts = 0
            while added < per_category and attempts < per_category * 6:
                attempts += 1
                template = random.choice(cat_templates)
                retrieval_query = template.question_template if use_retrieval else None
                result = self.generate_one(template, phase=phase, retrieval_query=retrieval_query)
                if result is None:
                    continue
                if self.is_duplicate(result.question, questions_seen):
                    continue
                results.append(result)
                questions_seen.append(result.question)
                added += 1
                if include_rephrase and added < per_category:
                    for variant in self.rephrase(result.question):
                        if added >= per_category:
                            break
                        if self.is_duplicate(variant, questions_seen):
                            continue
                        results.append(
                            GenerationResult(
                                question=variant,
                                sparql=result.sparql,
                                answers=result.answers,
                                category=result.category,
                                template_name=result.template_name,
                                phase=phase,
                                retrieved_examples=result.retrieved_examples,
                            )
                        )
                        questions_seen.append(variant)
                        added += 1
        return results
