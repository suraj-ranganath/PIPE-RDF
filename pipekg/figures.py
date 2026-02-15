from pathlib import Path
from typing import Dict, List
import json

import matplotlib.pyplot as plt


def save_category_distribution(stats: Dict[str, int], path: Path) -> None:
    categories = list(stats.keys())
    counts = [stats[c] for c in categories]

    plt.figure(figsize=(10, 4))
    plt.bar(categories, counts, color="#4C78A8")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Questions")
    plt.title("Benchmark Category Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_metric_bars(metrics: Dict[str, float], path: Path) -> None:
    labels = list(metrics.keys())
    values = [metrics[k] for k in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#72B7B2")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics (Simulated)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_pipeline_dot(path: Path) -> None:
    dot = r"""
    digraph PIPEKG {
      rankdir=LR;
      node [shape=box, style=rounded];

      subgraph cluster_phase1 {
        label="Phase 1: Seed Generation";
        color="#B3CDE3";
        p1a [label="Ontology + Graph\nDescription"];
        p1b [label="Template NL Questions"];
        p1c [label="Reverse Querying\n(SPARQL)"];
        p1d [label="Execute + Repair"];
        p1e [label="Human Validation"];
        p1f [label="Seed Vector DB"];
        p1a -> p1b -> p1c -> p1d -> p1e -> p1f;
      }

      subgraph cluster_phase2 {
        label="Phase 2: Category-Wise Seeds";
        color="#CCEBC5";
        p2a [label="Category Prompting"];
        p2b [label="RAG (Seed DB)"];
        p2c [label="NL/SPARQL Generation"];
        p2d [label="Rephrase + Dedup"];
        p2e [label="Category Vector DBs"];
        p2a -> p2b -> p2c -> p2d -> p2e;
      }

      subgraph cluster_phase3 {
        label="Phase 3: Full Dataset";
        color="#FBB4AE";
        p3a [label="Category DB RAG"];
        p3b [label="NL/SPARQL Generation"];
        p3c [label="Execute + Repair"];
        p3d [label="Human Validation"];
        p3e [label="Benchmark Dataset"];
        p3a -> p3b -> p3c -> p3d -> p3e;
      }

      p1f -> p2b;
      p2e -> p3a;
    }
    """
    path.write_text(dot)


def save_histogram(values: List[int], path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=12, color="#F58518", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
