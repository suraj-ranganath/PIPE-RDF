from dataclasses import dataclass, field
from typing import Dict, List

from .utils import tokenize, cosine_sim


@dataclass
class Example:
    question: str
    sparql: str
    category: str


@dataclass
class Retriever:
    examples: List[Example] = field(default_factory=list)

    def add(self, example: Example) -> None:
        self.examples.append(example)

    def add_many(self, examples: List[Example]) -> None:
        self.examples.extend(examples)

    def top_k(self, query: str, k: int = 3, category: str | None = None) -> List[Example]:
        query_tokens = tokenize(query)
        scored = []
        for ex in self.examples:
            if category and ex.category != category:
                continue
            score = cosine_sim(query_tokens, tokenize(ex.question))
            scored.append((score, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:k]]
