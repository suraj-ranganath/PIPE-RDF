from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from pipekg.config import CATEGORIES
from pipekg.plot_style import PALETTE, apply_publication_style

apply_publication_style()


def _save(fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def _ordered_categories(values: Dict[str, float]) -> List[str]:
    ordered = [c for c in CATEGORIES if c in values]
    extras = sorted([k for k in values if k not in set(ordered)])
    return ordered + extras


def _pretty_category(cat: str) -> str:
    mapping = {
        "yesno": "Yes/No",
        "multi-hop": "Multi-hop",
    }
    return mapping.get(cat, cat.replace("_", " ").title())


def bar_by_category(values: Dict[str, float], title: str, ylabel: str, path: Path) -> None:
    categories = _ordered_categories(values)
    scores = [values[c] for c in categories]
    labels = [_pretty_category(c) for c in categories]

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    bars = ax.bar(labels, scores, color=PALETTE[0], alpha=0.9, edgecolor="#1E293B", linewidth=0.6)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    if scores and max(scores) <= 1.05:
        ax.set_ylim(0, 1.05)
        for b, v in zip(bars, scores):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.015,
                f"{v*100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#0F172A",
            )
    else:
        y_max = max(scores) if scores else 1.0
        ax.set_ylim(0, y_max * 1.12 if y_max > 0 else 1.0)
        for b, v in zip(bars, scores):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + (0.02 * y_max if y_max > 0 else 0.02),
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#0F172A",
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, weight="semibold")
    _save(path)


def radar_by_category(
    categories: List[str],
    exec_rate: Dict[str, float],
    non_empty_rate: Dict[str, float],
    complexity_norm: Dict[str, float],
    path: Path,
) -> None:
    if not categories:
        return
    ordered = [c for c in CATEGORIES if c in categories] + sorted([c for c in categories if c not in CATEGORIES])
    labels = [_pretty_category(c) for c in ordered]
    n = len(ordered)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    s1 = [exec_rate.get(c, 0.0) for c in ordered]
    s2 = [non_empty_rate.get(c, 0.0) for c in ordered]
    s3 = [complexity_norm.get(c, 0.0) for c in ordered]
    series = [s1 + s1[:1], s2 + s2[:1], s3 + s3[:1]]
    names = ["Execution Success Rate", "Non-Empty Result Rate", "Structural Complexity (normalized)"]
    colors = [PALETTE[0], PALETTE[2], PALETTE[1]]

    fig, ax = plt.subplots(figsize=(8.2, 6.2), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_rgrids([0.25, 0.5, 0.75, 1.0], labels=["25%", "50%", "75%", "100%"], angle=0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.grid(color="#CBD5E1", linewidth=0.7)
    ax.spines["polar"].set_color("#94A3B8")

    for vals, name, color in zip(series, names, colors):
        ax.plot(angles, vals, color=color, linewidth=2.4, marker="o", markersize=4.8, label=name)
        ax.fill(angles, vals, color=color, alpha=0.10)

    ax.set_title("Category-wise Benchmark Metrics", pad=18, fontsize=16, weight="semibold")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.40, 0.02),
        frameon=False,
        borderaxespad=0.0,
    )
    _save(path)


def hist(values: List[float], title: str, xlabel: str, path: Path, bins: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.hist(values, bins=bins, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title(title, pad=8, weight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    _save(path)


def scatter(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.scatter(x, y, alpha=0.72, color=PALETTE[2], edgecolor="#0F172A", linewidth=0.25, s=22)
    ax.grid(axis="both")
    ax.set_axisbelow(True)
    ax.set_title(title, pad=8, weight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(path)


def pie(values: Dict[str, int], title: str, path: Path) -> None:
    labels = sorted(values.keys(), key=lambda k: values[k], reverse=True)
    sizes = [values[k] for k in labels]
    total = float(sum(sizes)) if sizes else 1.0

    def _autopct(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= 2.0 else ""

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=[l.replace("_", " ") for l in labels],
        autopct=_autopct,
        startangle=90,
        counterclock=False,
        colors=PALETTE[: len(labels)],
        wedgeprops={"linewidth": 0.9, "edgecolor": "white", "width": 0.42},
        pctdistance=0.72,
        labeldistance=1.08,
        textprops={"fontsize": 9.5, "color": "#0F172A"},
    )
    for t in autotexts:
        t.set_fontsize(9.5)
        t.set_weight("semibold")
    ax.text(0, 0, f"n={int(total)}", ha="center", va="center", fontsize=10, color="#334155")
    ax.set_title(title, pad=10, weight="semibold")
    ax.axis("equal")
    _save(path)
