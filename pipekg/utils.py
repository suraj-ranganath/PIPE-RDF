import random
import re
from typing import Iterable, List


def set_seed(seed: int) -> None:
    random.seed(seed)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def cosine_sim(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
    from collections import Counter

    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    common = set(a_counts) & set(b_counts)
    num = sum(a_counts[t] * b_counts[t] for t in common)
    denom_a = sum(v * v for v in a_counts.values()) ** 0.5
    denom_b = sum(v * v for v in b_counts.values()) ** 0.5
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return num / (denom_a * denom_b)


def jaccard(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)
