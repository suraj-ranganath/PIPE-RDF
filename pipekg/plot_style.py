from __future__ import annotations

import matplotlib as mpl
from cycler import cycler


# Okabe-Ito inspired, colorblind-friendly palette.
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "font.family": "serif",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.9,
            "grid.color": "#CBD5E1",
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "grid.alpha": 0.55,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=PALETTE),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

