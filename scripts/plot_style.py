"""
Shared matplotlib style for NODIS publication figures.
Import and call apply() before building any figure.
"""

import warnings
import matplotlib

RCPARAMS = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          12,
    "axes.labelsize":     13,
    "axes.titlesize":     14,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    12,
    "figure.dpi":         300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
}


def apply() -> None:
    """Apply shared rcParams and suppress matplotlib/pandas deprecation noise."""
    matplotlib.rcParams.update(RCPARAMS)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
