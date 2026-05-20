import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from rdflib import Graph
from rdflib.plugins.sparql.parser import parseQuery

from .utils import tokenize
from .ast_utils import ast_label_f1


@dataclass
class MetricResult:
    sp_f1: float
    triple_f1: float
    exec_accuracy: float
    parse_valid_rate: float
    repaired_rate: float
    answer_f1: float
    answer_precision: float
    answer_recall: float
    predicate_f1: float
    sketch_similarity: float
    ast_label_f1: float


def normalize_vars(query: str) -> str:
    vars_found = {}
    var_id = 0

    def repl(match):
        nonlocal var_id
        var = match.group(0)
        if var not in vars_found:
            vars_found[var] = f"?v{var_id}"
            var_id += 1
        return vars_found[var]

    return re.sub(r"\?[A-Za-z_][A-Za-z0-9_]*", repl, query)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    pred_set = pred_tokens
    gold_set = gold_tokens
    if not pred_set or not gold_set:
        return 0.0
    pred_counts = {t: pred_set.count(t) for t in set(pred_set)}
    gold_counts = {t: gold_set.count(t) for t in set(gold_set)}
    common = set(pred_counts) & set(gold_counts)
    match = sum(min(pred_counts[t], gold_counts[t]) for t in common)
    precision = match / len(pred_set)
    recall = match / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_triples(query: str) -> List[str]:
    triples = []
    if "{" not in query or "}" not in query:
        return triples
    body = query.split("{", 1)[1].rsplit("}", 1)[0]
    for segment in body.split("."):
        seg = segment.strip()
        if not seg:
            continue
        if seg.upper().startswith("FILTER") or seg.upper().startswith("OPTIONAL"):
            continue
        tokens = seg.split()
        if len(tokens) >= 3:
            triples.append(" ".join(tokens[:3]))
    return triples


def triple_f1(pred: str, gold: str) -> float:
    pred_triples = extract_triples(pred)
    gold_triples = extract_triples(gold)
    if not pred_triples or not gold_triples:
        return 0.0
    pred_set = set(pred_triples)
    gold_set = set(gold_triples)
    match = len(pred_set & gold_set)
    precision = match / len(pred_set)
    recall = match / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_valid(graph: Graph, query: str) -> bool:
    try:
        graph.query(query)
        return True
    except Exception:
        return False


def parse_valid_sparql_detail(query: str) -> Tuple[bool, str | None]:
    try:
        parseQuery(query)
        return True, None
    except Exception as exc:
        return False, str(exc)


def parse_valid_sparql(query: str) -> bool:
    ok, _ = parse_valid_sparql_detail(query)
    return ok


def canonicalize_query(query: str) -> str:
    # Lightweight canonicalization: normalize variables, strip extra whitespace, sort triples
    norm = normalize_vars(query)
    triples = extract_triples(norm)
    triples_sorted = sorted(triples)
    body = " . ".join(triples_sorted)
    return f"{{ {body} }}"


def predicate_coverage(pred: str, gold: str) -> float:
    def preds(q: str) -> set[str]:
        items = set()
        for triple in extract_triples(q):
            parts = triple.split()
            if len(parts) >= 2:
                items.add(parts[1])
        return items

    pred_set = preds(pred)
    gold_set = preds(gold)
    if not pred_set or not gold_set:
        return 0.0
    match = len(pred_set & gold_set)
    precision = match / len(pred_set)
    recall = match / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sketch_similarity(pred: str, gold: str) -> float:
    def sketch(q: str) -> List[str]:
        q = normalize_vars(q)
        tokens = tokenize(q)
        # Remove IRIs and literals to focus on structure
        return [t for t in tokens if not t.startswith("http") and not t.startswith("mv") and not t.startswith("remv") and not t.startswith("dcterms")]

    return token_f1(" ".join(sketch(pred)), " ".join(sketch(gold)))


def execute_answers(graph: Graph, query: str) -> List[str]:
    try:
        results = graph.query(query)
    except Exception:
        return []
    answers = []
    for row in results:
        if hasattr(row, "asdict"):
            for val in row.asdict().values():
                answers.append(str(val))
        else:
            answers.append(str(row))
    return answers


def answer_set_scores(pred_answers: List[str], gold_answers: List[str]) -> Tuple[float, float, float]:
    pred_set = set(pred_answers)
    gold_set = set(gold_answers)
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    match = len(pred_set & gold_set)
    precision = match / len(pred_set)
    recall = match / len(gold_set)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def exec_accuracy_at_k(graph: Graph, gold_query: str, predictions: List[str]) -> float:
    gold_ans = execute_answers(graph, gold_query)
    if not gold_ans:
        return 0.0
    for pred in predictions:
        pred_ans = execute_answers(graph, pred)
        if pred_ans == gold_ans:
            return 1.0
    return 0.0


def success_at_k(graph: Graph, gold_query: str, predictions: List[str], k: int) -> int:
    return int(exec_accuracy_at_k(graph, gold_query, predictions[:k]) > 0)


def simulate_prediction(gold_query: str, corruption_rate: float = 0.3) -> str:
    if random.random() > corruption_rate:
        return gold_query
    # Corrupt by swapping a predicate token if possible
    tokens = gold_query.split()
    pred_indices = [i for i, tok in enumerate(tokens) if ":" in tok and not tok.startswith("PREFIX")]
    if pred_indices:
        idx = random.choice(pred_indices)
        tokens[idx] = tokens[idx].replace("has_", "has_") + "_bad"
    return " ".join(tokens)


def repair_query(query: str) -> str:
    # Simple heuristic: remove trailing '_bad'
    return query.replace("_bad", "")


def evaluate_predictions(graph: Graph, gold_queries: List[str], corruption_rate: float = 0.3) -> MetricResult:
    sp_f1_scores = []
    triple_scores = []
    exec_hits = 0
    parse_hits = 0
    repaired_hits = 0
    answer_f1_scores = []
    answer_prec_scores = []
    answer_rec_scores = []
    predicate_scores = []
    sketch_scores = []
    ast_scores = []

    for gold in gold_queries:
        pred = simulate_prediction(gold, corruption_rate=corruption_rate)
        pred_norm = normalize_vars(pred)
        gold_norm = normalize_vars(gold)
        sp_f1_scores.append(token_f1(pred_norm, gold_norm))
        triple_scores.append(triple_f1(pred, gold))
        predicate_scores.append(predicate_coverage(pred, gold))
        sketch_scores.append(sketch_similarity(pred, gold))
        ast_scores.append(ast_label_f1(pred, gold))

        if parse_valid(graph, pred):
            parse_hits += 1
        gold_ans = execute_answers(graph, gold)
        pred_ans = execute_answers(graph, pred)
        if gold_ans and pred_ans and gold_ans == pred_ans:
            exec_hits += 1
        else:
            repaired = repair_query(pred)
            if repaired != pred and execute_answers(graph, repaired) == gold_ans:
                repaired_hits += 1

        precision, recall, f1 = answer_set_scores(pred_ans, gold_ans)
        answer_prec_scores.append(precision)
        answer_rec_scores.append(recall)
        answer_f1_scores.append(f1)

    total = len(gold_queries) or 1
    return MetricResult(
        sp_f1=sum(sp_f1_scores) / total,
        triple_f1=sum(triple_scores) / total,
        exec_accuracy=exec_hits / total,
        parse_valid_rate=parse_hits / total,
        repaired_rate=repaired_hits / total,
        answer_f1=sum(answer_f1_scores) / total,
        answer_precision=sum(answer_prec_scores) / total,
        answer_recall=sum(answer_rec_scores) / total,
        predicate_f1=sum(predicate_scores) / total,
        sketch_similarity=sum(sketch_scores) / total,
        ast_label_f1=sum(ast_scores) / total,
    )


def validate_answer_type(category: str, answers: List[str], query: str) -> bool:
    """Validate that answers match expected type for the category.

    Returns True if answers pass type validation, False otherwise.
    """
    if not answers:
        return True  # Empty results are handled separately

    upper = query.upper()

    # Boolean ASK forms are only semantically valid for boolean/comparison categories.
    if "ASK" in upper and "SELECT" not in upper and category not in {"yesno", "comparative", "difference"}:
        return False

    if category == "ordinal":
        return "ORDER BY" in upper and "OFFSET" in upper and "LIMIT 1" in upper

    if category == "superlative":
        return "ORDER BY" in upper and "LIMIT 1" in upper

    if category == "counting" and "COUNT" not in upper:
        return False

    if category in {"yesno", "comparative", "difference"} and ("ASK" not in upper or "SELECT" in upper):
        return False

    if category in {"generic", "multi-hop", "intersection"} and ("ASK" in upper and "SELECT" not in upper):
        return False

    # Counting queries should return numeric answers
    if category == "counting":
        for ans in answers:
            try:
                int(ans)
            except (ValueError, TypeError):
                # Allow float counts too
                try:
                    float(ans)
                except (ValueError, TypeError):
                    return False
        return True

    # Yes/No queries should return boolean
    if category == "yesno":
        valid_bools = {"true", "false", "yes", "no", "1", "0"}
        for ans in answers:
            if str(ans).strip().lower() not in valid_bools:
                return False
        return True

    # ASK queries should return boolean
    if "ASK" in upper and "SELECT" not in upper:
        valid_bools = {"true", "false"}
        for ans in answers:
            if str(ans).strip().lower() not in valid_bools:
                return False
        return True

    return True
