from __future__ import annotations

from typing import List, Tuple
from rdflib.plugins.sparql.parser import parseQuery
from pyparsing.results import ParseResults


def _walk(node, depth: int, labels: List[str]) -> int:
    max_depth = depth
    if isinstance(node, ParseResults):
        name = node.getName()
        if name:
            labels.append(name)
        for item in node:
            max_depth = max(max_depth, _walk(item, depth + 1, labels))
    elif isinstance(node, (list, tuple)):
        for item in node:
            max_depth = max(max_depth, _walk(item, depth + 1, labels))
    return max_depth


def ast_stats(query: str) -> Tuple[int, int, List[str]]:
    """Return (node_count, max_depth, labels) for a SPARQL query AST."""
    parsed = parseQuery(query)
    labels: List[str] = []
    max_depth = _walk(parsed, 1, labels)
    node_count = len(labels)
    return node_count, max_depth, labels


def ast_label_f1(pred_query: str, gold_query: str) -> float:
    try:
        _, _, pred_labels = ast_stats(pred_query)
        _, _, gold_labels = ast_stats(gold_query)
    except Exception:
        return 0.0

    pred_set = set(pred_labels)
    gold_set = set(gold_labels)
    if not pred_set or not gold_set:
        return 0.0
    match = len(pred_set & gold_set)
    precision = match / len(pred_set)
    recall = match / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
