import csv
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from pipekg.config import (
    SEED,
    DATA_DIR,
    FIG_DIR,
    LOG_DIR,
    NUM_FILMS,
    NUM_DIRECTORS,
    NUM_ACTORS,
    NUM_PRODUCERS,
    NUM_GENRES,
    NUM_COUNTRIES,
    PHASE1_SEEDS_PER_TEMPLATE,
    PHASE2_SEEDS_PER_CATEGORY,
    PHASE3_SAMPLES_PER_CATEGORY,
    CORRUPTION_RATE,
)
from pipekg.schema import Schema
from pipekg.kg import SyntheticKG
from pipekg.templates import build_templates
from pipekg.generator import BenchmarkGenerator
from pipekg.evaluation import evaluate_predictions
from pipekg.figures import save_category_distribution, save_metric_bars, save_pipeline_dot, save_histogram
from pipekg.utils import set_seed


def main() -> None:
    set_seed(SEED)
    random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    schema = Schema.from_ttl("old_work/linkedmdb_structure.ttl")
    kg = SyntheticKG.build(
        schema,
        {
            "films": NUM_FILMS,
            "directors": NUM_DIRECTORS,
            "actors": NUM_ACTORS,
            "producers": NUM_PRODUCERS,
            "genres": NUM_GENRES,
            "countries": NUM_COUNTRIES,
        },
    )

    templates = build_templates()
    generator = BenchmarkGenerator(kg, templates)

    # Phase 1: seed generation
    phase1 = generator.generate_phase(
        templates,
        per_template=PHASE1_SEEDS_PER_TEMPLATE,
        phase="phase1",
        use_retrieval=False,
        include_rephrase=False,
    )
    generator.index_examples(phase1)

    # Phase 2: category-wise seeds
    phase2 = generator.generate_category_wise(
        per_category=PHASE2_SEEDS_PER_CATEGORY,
        phase="phase2",
        use_retrieval=True,
    )
    generator.index_examples(phase2)

    # Phase 3: full dataset
    phase3 = generator.generate_category_wise(
        per_category=PHASE3_SAMPLES_PER_CATEGORY,
        phase="phase3",
        use_retrieval=True,
        include_rephrase=True,
    )

    all_results = phase1 + phase2 + phase3
    phase3_results = phase3

    # Save datasets
    jsonl_path = DATA_DIR / "benchmark.jsonl"
    csv_path = DATA_DIR / "benchmark.csv"
    phase3_jsonl = DATA_DIR / "benchmark_phase3.jsonl"
    phase3_csv = DATA_DIR / "benchmark_phase3.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, r in enumerate(all_results, start=1):
            record = {
                "id": idx,
                "phase": r.phase,
                "category": r.category,
                "template": r.template_name,
                "question": r.question,
                "sparql": r.sparql,
                "answers": r.answers,
                "retrieved": [ex.question for ex in r.retrieved_examples],
            }
            f.write(json.dumps(record) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "phase", "category", "template", "question", "sparql", "answers"],
        )
        writer.writeheader()
        for idx, r in enumerate(all_results, start=1):
            writer.writerow(
                {
                    "id": idx,
                    "phase": r.phase,
                    "category": r.category,
                    "template": r.template_name,
                    "question": r.question,
                    "sparql": r.sparql,
                    "answers": "; ".join(r.answers),
                }
            )

    with phase3_jsonl.open("w", encoding="utf-8") as f:
        for idx, r in enumerate(phase3_results, start=1):
            record = {
                "id": idx,
                "phase": r.phase,
                "category": r.category,
                "template": r.template_name,
                "question": r.question,
                "sparql": r.sparql,
                "answers": r.answers,
                "retrieved": [ex.question for ex in r.retrieved_examples],
            }
            f.write(json.dumps(record) + "\n")

    with phase3_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "phase", "category", "template", "question", "sparql", "answers"],
        )
        writer.writeheader()
        for idx, r in enumerate(phase3_results, start=1):
            writer.writerow(
                {
                    "id": idx,
                    "phase": r.phase,
                    "category": r.category,
                    "template": r.template_name,
                    "question": r.question,
                    "sparql": r.sparql,
                    "answers": "; ".join(r.answers),
                }
            )

    # Stats
    category_counts = {}
    for r in all_results:
        category_counts[r.category] = category_counts.get(r.category, 0) + 1

    phase3_counts = {}
    for r in phase3_results:
        phase3_counts[r.category] = phase3_counts.get(r.category, 0) + 1

    # Balanced sampling for phase3
    min_count = min(phase3_counts.values()) if phase3_counts else 0
    balanced_phase3 = []
    per_cat = {c: 0 for c in phase3_counts}
    random.shuffle(phase3_results)
    for r in phase3_results:
        if per_cat[r.category] >= min_count:
            continue
        balanced_phase3.append(r)
        per_cat[r.category] += 1

    balanced_counts = {c: min_count for c in phase3_counts}
    (DATA_DIR / "phase3_balanced_counts.json").write_text(json.dumps(balanced_counts, indent=2))
    balanced_jsonl = DATA_DIR / "benchmark_phase3_balanced.jsonl"
    balanced_csv = DATA_DIR / "benchmark_phase3_balanced.csv"

    with balanced_jsonl.open("w", encoding="utf-8") as f:
        for idx, r in enumerate(balanced_phase3, start=1):
            record = {
                "id": idx,
                "phase": r.phase,
                "category": r.category,
                "template": r.template_name,
                "question": r.question,
                "sparql": r.sparql,
                "answers": r.answers,
                "retrieved": [ex.question for ex in r.retrieved_examples],
            }
            f.write(json.dumps(record) + "\n")

    with balanced_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "phase", "category", "template", "question", "sparql", "answers"],
        )
        writer.writeheader()
        for idx, r in enumerate(balanced_phase3, start=1):
            writer.writerow(
                {
                    "id": idx,
                    "phase": r.phase,
                    "category": r.category,
                    "template": r.template_name,
                    "question": r.question,
                    "sparql": r.sparql,
                    "answers": "; ".join(r.answers),
                }
            )

    # Evaluation (simulated predictions)
    gold_queries = [r.sparql for r in balanced_phase3] if balanced_phase3 else [r.sparql for r in phase3_results]
    metrics = evaluate_predictions(kg.graph, gold_queries, corruption_rate=CORRUPTION_RATE)
    metrics_dict = {
        "sp_f1": metrics.sp_f1,
        "triple_f1": metrics.triple_f1,
        "exec_accuracy": metrics.exec_accuracy,
        "parse_valid_rate": metrics.parse_valid_rate,
        "repaired_rate": metrics.repaired_rate,
        "answer_f1": metrics.answer_f1,
        "answer_precision": metrics.answer_precision,
        "answer_recall": metrics.answer_recall,
        "predicate_f1": metrics.predicate_f1,
        "sketch_similarity": metrics.sketch_similarity,
        "ast_label_f1": metrics.ast_label_f1,
    }
    (DATA_DIR / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))
    (DATA_DIR / "category_counts.json").write_text(json.dumps(category_counts, indent=2))
    (DATA_DIR / "phase3_counts.json").write_text(json.dumps(phase3_counts, indent=2))

    # Figures
    save_category_distribution(category_counts, FIG_DIR / "category_distribution.png")
    save_category_distribution(phase3_counts, FIG_DIR / "category_distribution_phase3.png")
    save_category_distribution(balanced_counts, FIG_DIR / "category_distribution_phase3_balanced.png")

    # Histograms from balanced phase3
    if balanced_phase3:
        question_lengths = [len(r.question.split()) for r in balanced_phase3]
        answer_counts = [len(r.answers) for r in balanced_phase3]
        save_histogram(question_lengths, FIG_DIR / "question_length_hist.png", "Question Length Distribution", "Tokens per question")
        save_histogram(answer_counts, FIG_DIR / "answer_count_hist.png", "Answer Count Distribution", "Answers per question")
    save_metric_bars(metrics_dict, FIG_DIR / "evaluation_metrics.png")

    dot_path = FIG_DIR / "pipeline_flow.dot"
    save_pipeline_dot(dot_path)
    png_path = FIG_DIR / "pipeline_flow.png"
    try:
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
