from __future__ import annotations

import hashlib
from typing import List


def estimate_tokens(text: str) -> int:
    # Rough estimate (avg 4 chars/token)
    return max(1, int(len(text) / 4))


def result_set_hash(values: List[str]) -> str:
    joined = "\n".join(sorted(values))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
