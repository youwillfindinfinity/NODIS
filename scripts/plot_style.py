"""
Shared matplotlib style for NODIS publication figures.

Nature Methods / Nature Communications compatible styling:
- Okabe-Ito colorblind-safe palette
- Arial/Helvetica fonts, 7-8 pt
- Column widths: 88 mm (single), 180 mm (double)
- Vector-safe font embedding (pdf.fonttype 42)
"""

import os
import warnings
import matplotlib

# ---------------------------------------------------------------------------
# Nature column widths (inches)
# ---------------------------------------------------------------------------
SINGLE_W = 3.46   # 88 mm
FULL_W   = 7.08   # 180 mm
MAX_H    = 9.45   # 240 mm

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette (2008)
# ---------------------------------------------------------------------------
OKABE = {
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "orange":         "#E69F00",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
    "yellow":         "#F0E442",
    "black":          "#000000",
}

# ---------------------------------------------------------------------------
# Method colours & labels  (shared across all figures)
# ---------------------------------------------------------------------------
METHOD_PALETTE = {
    "desparsified":        OKABE["sky_blue"],
    "glasso":              OKABE["bluish_green"],
    "gglasso":             OKABE["orange"],
    "ssglasso":            OKABE["blue"],
    "piglasso":            OKABE["vermillion"],
    "piglasso_corr":       OKABE["vermillion"],
    "piglasso_oracle_n02": OKABE["vermillion"],
    "genie3":              OKABE["reddish_purple"],
}

METHOD_LABELS = {
    "desparsified":        "Desparsified",
    "glasso":              "GLasso",
    "gglasso":             "GGLasso",
    "ssglasso":            "SSGLasso",
    "piglasso":            "PIGLasso",
    "piglasso_corr":       "PIGLasso",
    "piglasso_oracle_n02": "PIGLasso",
    "genie3":              "GENIE3",
}

# Methods rendered with thicker lines / heavier markers (prior-informed)
PIG_METHODS = {"piglasso", "piglasso_corr", "piglasso_oracle_n02"}

# ---------------------------------------------------------------------------
# Topology colours & markers  (shared across all figures)
# ---------------------------------------------------------------------------
TOPO_PALETTE = {
    "cluster":    OKABE["sky_blue"],
    "hub":        OKABE["orange"],
    "random":     OKABE["bluish_green"],
    "scale-free": OKABE["reddish_purple"],
}
TOPO_MARKERS = {
    "cluster":    "o",
    "hub":        "s",
    "random":     "^",
    "scale-free": "D",
}
TOPO_LABELS = {
    "cluster":    "Cluster",
    "hub":        "Hub",
    "random":     "Random",
    "scale-free": "Scale-free",
}

# ---------------------------------------------------------------------------
# rcParams  — Nature-journal specifications
# ---------------------------------------------------------------------------
RCPARAMS = {
    # Font
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":             7,
    "axes.labelsize":        8,
    "axes.titlesize":        8,
    "xtick.labelsize":       7,
    "ytick.labelsize":       7,
    "legend.fontsize":       7,
    "legend.title_fontsize": 7,
    # Axes
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.6,
    "axes.labelpad":         3,
    # Ticks
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    # Lines
    "lines.linewidth":       1.0,
    "lines.markersize":      4,
    # Figure
    "figure.dpi":            600,
    "savefig.dpi":           600,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    # PDF / PS — embeds fonts for Illustrator/InDesign
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
    # Legend
    "legend.frameon":        False,
    "legend.handlelength":   1.5,
    "legend.handletextpad":  0.5,
    "legend.labelspacing":   0.3,
    "legend.columnspacing":  1.0,
    # Grid
    "axes.grid":             False,
}


def apply() -> None:
    """Apply shared rcParams and suppress deprecation noise."""
    matplotlib.rcParams.update(RCPARAMS)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def panel_label(ax, letter, x=-0.14, y=1.08, size=9):
    """Bold panel label (A, B, C …) in Nature style."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=size, fontweight="bold", va="top", ha="left")


def save(fig, path: str, dpi: int = 300) -> None:
    """Save figure as PDF + PNG pair."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    print(f"Saved → {path}")
    if path.endswith(".pdf"):
        png = path.replace(".pdf", ".png")
        fig.savefig(png, dpi=dpi)
        print(f"Saved → {png}")
    plt.close(fig)
