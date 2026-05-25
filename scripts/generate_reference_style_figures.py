from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def _rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str = "#F6F8FB",
    edge: str = "#334155",
    fontsize: int = 12,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#0F172A",
    lw: float = 1.8,
    curve: float = 0.0,
) -> None:
    conn = f"arc3,rad={curve}" if curve else "arc3"
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        color=color,
        connectionstyle=conn,
    )
    ax.add_patch(arr)


def _routed_arrow(
    ax,
    points: list[tuple[float, float]],
    color: str = "#0F172A",
    lw: float = 1.8,
) -> None:
    if len(points) < 2:
        return
    for i in range(len(points) - 2):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    _arrow(ax, x1, y1, x2, y2, color=color, lw=lw)


def generate_pipeline_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Three-phase banded layout.
    ax.add_patch(Rectangle((0.02, 0.60), 0.96, 0.32, facecolor="#EAF1FB", edgecolor="#B6CCE8"))
    ax.add_patch(Rectangle((0.02, 0.35), 0.96, 0.22, facecolor="#ECF7EE", edgecolor="#B5DAB7"))
    ax.add_patch(Rectangle((0.02, 0.09), 0.96, 0.22, facecolor="#FFF5E8", edgecolor="#E8CFA7"))

    ax.text(
        0.5,
        0.965,
        "PIPE-RDF Workload Construction Pipeline",
        fontsize=20,
        weight="bold",
        ha="center",
        va="top",
    )
    ax.text(0.03, 0.88, "Phase 1: Seed Generation", fontsize=13, weight="bold")
    ax.text(0.03, 0.55, "Phase 2: Category-wise Seeding", fontsize=13, weight="bold")
    ax.text(0.03, 0.29, "Phase 3: Full Dataset Generation", fontsize=13, weight="bold")

    # Phase 1.
    _rounded_box(ax, 0.05, 0.69, 0.19, 0.14, "Target RDF\nGraph +\nSchema Summary", face="#FFFFFF")
    _rounded_box(ax, 0.28, 0.69, 0.17, 0.14, "Template\nGeneration")
    _rounded_box(ax, 0.49, 0.69, 0.18, 0.14, "Reverse Querying\n+ Validation")
    _rounded_box(ax, 0.71, 0.69, 0.23, 0.14, "Verified Seed Bank\n(NL, SPARQL, answers)")
    _arrow(ax, 0.24, 0.76, 0.28, 0.76)
    _arrow(ax, 0.45, 0.76, 0.49, 0.76)
    _arrow(ax, 0.67, 0.76, 0.71, 0.76)

    # Phase 2.
    _rounded_box(ax, 0.08, 0.38, 0.24, 0.13, "Category Template Planning\n(9 categories)")
    _rounded_box(ax, 0.38, 0.38, 0.24, 0.13, "Category-aware RAG Retrieval\n(top-k verified seeds)")
    _rounded_box(ax, 0.68, 0.38, 0.24, 0.13, "Category Seed Generation\n+ Validation/Repair")
    _arrow(ax, 0.32, 0.445, 0.38, 0.445)
    _arrow(ax, 0.62, 0.445, 0.68, 0.445)

    # Phase 3.
    _rounded_box(ax, 0.08, 0.11, 0.24, 0.13, "Category-specific\nRetrieval Banks")
    _rounded_box(ax, 0.38, 0.11, 0.24, 0.13, "Batch Generation\n(Questions + SPARQL)")
    _rounded_box(ax, 0.67, 0.11, 0.20, 0.13, "Execution Validation\n+ Metrics Logging")
    _rounded_box(ax, 0.90, 0.11, 0.07, 0.13, "Balanced\nBenchmark", face="#FFFFFF", fontsize=11)
    _arrow(ax, 0.32, 0.175, 0.38, 0.175)
    _arrow(ax, 0.62, 0.175, 0.67, 0.175)
    _arrow(ax, 0.87, 0.175, 0.90, 0.175)

    # Inter-phase transfer arrows routed outside active blocks.
    _routed_arrow(
        ax,
        [
            (0.94, 0.76),
            (0.97, 0.76),
            (0.97, 0.60),
            (0.50, 0.60),
            (0.50, 0.51),
        ],
    )
    _routed_arrow(
        ax,
        [
            (0.80, 0.38),
            (0.80, 0.34),
            (0.20, 0.34),
            (0.20, 0.24),
        ],
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close(fig)


def _entity_table(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    rows: list[str],
    face: str,
    edge: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.plot([x, x + w], [y + h * 0.74, y + h * 0.74], color=edge, linewidth=1.2)
    ax.text(x + 0.02, y + h * 0.84, title, fontsize=11.5, weight="bold", va="center")
    for idx, row in enumerate(rows):
        ry = y + h * (0.62 - idx * 0.16)
        ax.text(x + 0.02, ry, row, fontsize=10.2, va="center")


def generate_schema_figure(path: Path) -> None:
    fig, ax_graph = plt.subplots(1, 1, figsize=(11, 6.5))
    ax_graph.set_xlim(0, 1)
    ax_graph.set_ylim(0, 1)
    ax_graph.axis("off")
    ax_graph.set_title("Schema C: Company-Location Mini-Slice", fontsize=15, weight="bold", pad=12)

    _entity_table(
        ax_graph,
        0.31,
        0.44,
        0.40,
        0.30,
        "dbo:Company",
        [
            "PK: company_uri",
            "dbo:foundingYear (xsd:gYear)",
            "dbo:numberOfEmployees (xsd:integer)",
            "rdfs:label",
        ],
        face="#E8F0FB",
        edge="#5D7FAF",
    )
    _entity_table(
        ax_graph,
        0.05,
        0.18,
        0.25,
        0.22,
        "gn:Feature",
        ["PK: location_uri", "rdfs:label", "spb:prefLabel"],
        face="#EEF8ED",
        edge="#6A9D69",
    )
    _entity_table(
        ax_graph,
        0.70,
        0.18,
        0.25,
        0.22,
        "foaf:Person",
        ["PK: person_uri", "foaf:name", "rdfs:label"],
        face="#FFF2DE",
        edge="#C08A2C",
    )
    _entity_table(
        ax_graph,
        0.36,
        0.03,
        0.28,
        0.18,
        "dbo:Industry",
        ["PK: industry_uri", "rdfs:label"],
        face="#F2EAF7",
        edge="#8D6AA6",
    )

    _arrow(ax_graph, 0.31, 0.59, 0.18, 0.40)
    _arrow(ax_graph, 0.71, 0.59, 0.82, 0.40)
    _arrow(ax_graph, 0.51, 0.44, 0.51, 0.21)
    ax_graph.text(0.17, 0.41, "dbo:location\n[0..n]", fontsize=10, ha="center")
    ax_graph.text(0.83, 0.41, "dbo:keyPerson\n[0..n]", fontsize=10, ha="center")
    ax_graph.text(0.54, 0.27, "dbo:industry\n[1..n]", fontsize=10, ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close(fig)


def main() -> None:
    out_dir = Path("paper_acl2026_industry/figures")
    generate_pipeline_figure(out_dir / "pipeline_architecture.png")
    generate_schema_figure(out_dir / "schema_ontology.png")
    print("Generated reference-style figures in", out_dir)


if __name__ == "__main__":
    main()
