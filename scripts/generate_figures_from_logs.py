import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipekg.figures_extra import bar_by_category, hist, pie, radar_by_category, scatter


TRIPLE_PATTERN_RE = re.compile(r"\.[ \t]*(?:\n|$)")


def _estimate_triple_patterns(sparql: str) -> int:
    if not sparql:
        return 0
    text = "\n".join(
        line for line in sparql.splitlines() if not line.strip().upper().startswith(("PREFIX ", "BASE "))
    )
    # Fallback-friendly proxy: count terminating dots in WHERE blocks.
    return len(TRIPLE_PATTERN_RE.findall(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="", help="Run ID under artifacts/runs/<run-id>")
    args = parser.parse_args()

    if args.run_id:
        log_path = Path("artifacts/runs") / args.run_id / "pipeline_records.jsonl"
        fig_dir = Path("artifacts/runs") / args.run_id / "figures"
    else:
        log_path = Path("artifacts/logs/pipeline_records.jsonl")
        fig_dir = Path("artifacts/figures")
    if not log_path.exists():
        print("Log file not found. Run full pipeline first.")
        return

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not records:
        print("No records found in log file.")
        return

    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    # Metrics by category
    exec_rate = {}
    parse_rate = {}
    repair_rate = {}
    avg_llm_latency = {}
    avg_exec_latency = {}
    avg_ast_nodes = {}
    avg_ast_depth = {}
    non_empty_rate = {}
    avg_triple_patterns = {}

    for cat, rows in by_cat.items():
        exec_rate[cat] = sum(1 for r in rows if r.get("exec_success")) / len(rows)
        parse_rate[cat] = sum(1 for r in rows if r.get("parse_valid")) / len(rows)
        repair_rate[cat] = sum(1 for r in rows if r.get("repair_attempts", 0) > 0) / len(rows)
        avg_llm_latency[cat] = sum(r.get("llm_latency_ms", 0) for r in rows) / len(rows)
        avg_exec_latency[cat] = sum(r.get("sparql_exec_ms", 0) for r in rows) / len(rows)
        avg_ast_nodes[cat] = sum(r.get("ast_node_count", 0) for r in rows) / len(rows)
        avg_ast_depth[cat] = sum(r.get("ast_max_depth", 0) for r in rows) / len(rows)
        non_empty_rate[cat] = sum(1 for r in rows if (r.get("answer_count", 0) or 0) > 0) / len(rows)
        avg_triple_patterns[cat] = sum(_estimate_triple_patterns(r.get("sparql", "")) for r in rows) / len(rows)

    bar_by_category(exec_rate, "Execution Success Rate by Category", "Exec Success", fig_dir / "exec_success_by_category.png")
    bar_by_category(parse_rate, "Parse Valid Rate by Category", "Parse Valid", fig_dir / "parse_valid_by_category.png")
    bar_by_category(repair_rate, "Repair Attempt Rate by Category", "Repair Rate", fig_dir / "repair_rate_by_category.png")
    bar_by_category(avg_llm_latency, "Avg LLM Latency by Category", "Latency (ms)", fig_dir / "llm_latency_by_category.png")
    bar_by_category(avg_exec_latency, "Avg SPARQL Exec Latency by Category", "Latency (ms)", fig_dir / "exec_latency_by_category.png")
    bar_by_category(avg_ast_nodes, "Avg AST Node Count by Category", "AST nodes", fig_dir / "ast_nodes_by_category.png")
    bar_by_category(avg_ast_depth, "Avg AST Depth by Category", "AST depth", fig_dir / "ast_depth_by_category.png")

    complexity_signal = avg_ast_nodes if any(v > 0 for v in avg_ast_nodes.values()) else avg_triple_patterns
    max_ast_nodes = max(complexity_signal.values()) if complexity_signal else 1.0
    complexity_norm = {
        cat: (complexity_signal.get(cat, 0.0) / max_ast_nodes if max_ast_nodes > 0 else 0.0)
        for cat in by_cat.keys()
    }
    radar_by_category(
        categories=list(by_cat.keys()),
        exec_rate=exec_rate,
        non_empty_rate=non_empty_rate,
        complexity_norm=complexity_norm,
        path=fig_dir / "category_radar.png",
    )

    # Histograms
    hist([r.get("llm_latency_ms", 0) for r in records], "LLM Latency Distribution", "LLM latency (ms)", fig_dir / "llm_latency_hist.png")
    hist([r.get("question_latency_ms", 0) for r in records], "Question Generation Latency", "Question gen latency (ms)", fig_dir / "question_latency_hist.png")
    hist([r.get("sparql_exec_ms", 0) for r in records], "SPARQL Exec Latency Distribution", "Exec latency (ms)", fig_dir / "exec_latency_hist.png")
    hist([r.get("answer_count", 0) for r in records], "Answer Count Distribution", "Answers per query", fig_dir / "answer_count_hist.png")
    hist([r.get("prompt_tokens_est", 0) for r in records], "Prompt Length Distribution", "Estimated tokens", fig_dir / "prompt_length_hist.png")
    hist([r.get("ast_node_count", 0) for r in records], "AST Node Count Distribution", "AST nodes", fig_dir / "ast_node_count_hist.png")
    hist([r.get("ast_max_depth", 0) for r in records], "AST Depth Distribution", "AST depth", fig_dir / "ast_depth_hist.png")

    # Retrieval score distribution (if available)
    scores = []
    for r in records:
        for ex in r.get("retrieved_examples", []):
            if "score" in ex:
                scores.append(ex["score"])
    if scores:
        hist(scores, "Retrieval Similarity Scores", "FAISS score", fig_dir / "retrieval_scores_hist.png")

    # Error type distribution
    error_counts = defaultdict(int)
    for r in records:
        err = r.get("error_type") or "none"
        error_counts[err] += 1
    pie(error_counts, "Error Type Distribution", fig_dir / "error_type_pie.png")

    # Scatter plots
    scatter(
        [r.get("prompt_tokens_est", 0) for r in records],
        [r.get("llm_latency_ms", 0) for r in records],
        "LLM Latency vs Prompt Length",
        "Prompt tokens (est)",
        "LLM latency (ms)",
        fig_dir / "llm_latency_vs_prompt.png",
    )
    scatter(
        [r.get("question_latency_ms", 0) for r in records],
        [r.get("llm_latency_ms", 0) for r in records],
        "Question Gen Latency vs SPARQL Gen Latency",
        "Question gen latency (ms)",
        "SPARQL gen latency (ms)",
        fig_dir / "question_vs_sparql_latency.png",
    )
    scatter(
        [r.get("ast_node_count", 0) for r in records],
        [r.get("sparql_exec_ms", 0) for r in records],
        "Exec Latency vs AST Node Count",
        "AST node count",
        "Exec latency (ms)",
        fig_dir / "exec_latency_vs_ast_nodes.png",
    )
    scatter(
        [r.get("ast_max_depth", 0) for r in records],
        [1 if r.get("exec_success") else 0 for r in records],
        "AST Depth vs Exec Success",
        "AST depth",
        "Exec success (0/1)",
        fig_dir / "ast_depth_vs_exec_success.png",
    )
    scatter(
        [r.get("answer_count", 0) for r in records],
        [r.get("sparql_exec_ms", 0) for r in records],
        "Exec Latency vs Answer Count",
        "Answer count",
        "Exec latency (ms)",
        fig_dir / "exec_latency_vs_answer_count.png",
    )

    # Retrieval score vs exec success (if available)
    if scores:
        retrieval_scores = []
        success_vals = []
        for r in records:
            exs = r.get("retrieved_examples", [])
            if exs and "score" in exs[0]:
                retrieval_scores.append(exs[0]["score"])
                success_vals.append(1 if r.get("exec_success") else 0)
        if retrieval_scores:
            scatter(
                retrieval_scores,
                success_vals,
                "Retrieval Score vs Exec Success",
                "Top retrieval score",
                "Exec success (0/1)",
                fig_dir / "retrieval_score_vs_exec.png",
            )

    print("Figures generated in artifacts/figures/")


if __name__ == "__main__":
    main()
