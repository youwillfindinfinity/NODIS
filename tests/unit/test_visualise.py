"""
Unit tests for nodis/visualise/plots.py.
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import numpy as np
import matplotlib.pyplot as plt
import pytest

from nodis.visualise.plots import plot_heatmap, plot_aupr_curves


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------

def test_plot_heatmap_returns_axes():
    matrix = np.random.default_rng(0).standard_normal((5, 5))
    ax = plot_heatmap(matrix)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_heatmap_with_existing_ax():
    _, existing_ax = plt.subplots()
    matrix = np.random.default_rng(1).standard_normal((5, 5))
    returned_ax = plot_heatmap(matrix, ax=existing_ax)
    assert returned_ax is existing_ax
    plt.close("all")


def test_plot_heatmap_title():
    matrix = np.eye(4)
    ax = plot_heatmap(matrix, title="My Title")
    assert ax.get_title() == "My Title"
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_aupr_curves
# ---------------------------------------------------------------------------

def test_plot_aupr_curves_returns_axes():
    results = [
        {"method": "desparsified", "aupr": 0.75},
        {"method": "glasso", "aupr": 0.60},
    ]
    ax = plot_aupr_curves(results)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_aupr_curves_single():
    results = [{"method": "desparsified", "aupr": 0.80}]
    ax = plot_aupr_curves(results)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_aupr_curves_nan_aupr():
    """aupr=nan in results should not raise."""
    results = [
        {"method": "desparsified", "aupr": float("nan")},
        {"method": "glasso", "aupr": 0.55},
    ]
    ax = plot_aupr_curves(results)
    assert isinstance(ax, plt.Axes)
    plt.close("all")
