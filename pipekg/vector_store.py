from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json

import faiss
import numpy as np


@dataclass
class FaissStore:
    dim: int
    index: faiss.Index
    metadata: List[Dict[str, str]]

    @classmethod
    def build(cls, embeddings: np.ndarray, metadata: List[Dict[str, str]]) -> "FaissStore":
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return cls(dim=dim, index=index, metadata=metadata)

    def search(self, embeddings: np.ndarray, k: int = 3) -> List[List[Dict[str, str]]]:
        faiss.normalize_L2(embeddings)
        scores, indices = self.index.search(embeddings, k)
        results: List[List[Dict[str, str]]] = []
        for row in indices:
            hits = []
            for idx in row:
                if idx < 0 or idx >= len(self.metadata):
                    continue
                hits.append(self.metadata[idx])
            results.append(hits)
        return results

    def search_with_scores(self, embeddings: np.ndarray, k: int = 3) -> List[List[Dict[str, str]]]:
        faiss.normalize_L2(embeddings)
        scores, indices = self.index.search(embeddings, k)
        results: List[List[Dict[str, str]]] = []
        for row_idx, row in enumerate(indices):
            hits = []
            for col_idx, idx in enumerate(row):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                item = dict(self.metadata[idx])
                item["score"] = float(scores[row_idx][col_idx])
                hits.append(item)
            results.append(hits)
        return results

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, str]]) -> None:
        if embeddings.size == 0:
            return
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def save(self, index_path: Path, meta_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(json.dumps(self.metadata, indent=2))

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "FaissStore":
        index = faiss.read_index(str(index_path))
        metadata = json.loads(meta_path.read_text())
        dim = index.d
        return cls(dim=dim, index=index, metadata=metadata)
