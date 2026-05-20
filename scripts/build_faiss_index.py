import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipekg.settings import get_settings
from pipekg.runtime import build_llm
from pipekg.vector_store import FaissStore


def main() -> None:
    settings = get_settings()
    input_path = Path("artifacts/data/benchmark_phase2.jsonl")
    if not input_path.exists():
        raise SystemExit("Expected Phase 2 seed file at artifacts/data/benchmark_phase2.jsonl")

    records = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]

    client = build_llm(settings)

    # Build per-category FAISS index
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)

    for category, items in by_cat.items():
        texts = [item["question"] for item in items]
        embeddings = np.array(client.embed_texts(texts), dtype="float32")
        metadata = [{"question": item["question"], "sparql": item["sparql"], "category": category} for item in items]
        store = FaissStore.build(embeddings, metadata)
        index_path = Path("artifacts/indexes") / f"{category}.faiss"
        meta_path = Path("artifacts/indexes") / f"{category}.json"
        store.save(index_path, meta_path)
        print(f"saved {category} index with {len(items)} items")


if __name__ == "__main__":
    main()
