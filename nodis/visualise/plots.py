"""
Visualisation utilities for GGM inference results and benchmarks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(
    matrix: np.ndarray,
    title: str = "",
    ax: plt.Axes | None = None,
    cmap: str = "RdBu_r",
    centre: float = 0.0,
) -> plt.Axes:
    """
    Plot a square matrix (precision, z-score, or adjacency) as a heatmap.

    Parameters
    ----------
    matrix : (p, p) ndarray
    title  : plot title
    ax     : existing Axes or None (creates new figure)
    cmap   : matplotlib colourmap
    centre : colour scale centre value

    Returns
    -------
    ax : the Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, center=centre, cmap=cmap, ax=ax,
                xticklabels=False, yticklabels=False, square=True)
    ax.set_title(title)
    return ax


def plot_aupr_curves(
    results: list[dict],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot AUPR bar chart from a list of benchmark result dicts.

    Parameters
    ----------
    results : list of dicts with keys 'method' and 'aupr'
    ax      : existing Axes or None

    Returns
    -------
    ax : the Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    methods = [r["method"] for r in results]
    auprs = [r.get("aupr", float("nan")) for r in results]

    ax.bar(methods, auprs, color=sns.color_palette("Set2", len(methods)))
    ax.set_ylabel("AUPR")
    ax.set_ylim(0, 1)
    ax.set_title("Area Under Precision-Recall Curve by Method")
    plt.xticks(rotation=30, ha="right")
    return ax
