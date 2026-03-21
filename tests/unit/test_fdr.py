"""
Unit tests for FDR control module.

Tests verify correct API, output properties, and method handling.
"""

import numpy as np
import pytest
from nodis.inference.fdr import fdr_control


def _make_p_matrix(p: int, seed: int = 0) -> np.ndarray:
    """Generate a symmetric (p, p) p-value matrix with diagonal = 1."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, (p, p))
    sym = (raw + raw.T) / 2
    np.fill_diagonal(sym, 1.0)
    return sym


def test_output_shape():
    P = _make_p_matrix(20)
    adj = fdr_control(P, alpha=0.05)
    assert adj.shape == (20, 20)


def test_output_binary():
    P = _make_p_matrix(15)
    adj = fdr_control(P, alpha=0.05)
    assert set(np.unique(adj)).issubset({0, 1})


def test_output_symmetric():
    P = _make_p_matrix(20)
    adj = fdr_control(P, alpha=0.05)
    np.testing.assert_array_equal(adj, adj.T)


def test_diagonal_zero():
    P = _make_p_matrix(20)
    adj = fdr_control(P, alpha=0.05)
    np.testing.assert_array_equal(np.diag(adj), np.zeros(20, dtype=int))


def test_bh_controls_fdr():
    """Under the global null (uniform p-values), FDR must be controlled at alpha."""
    rng = np.random.default_rng(123)
    n_trials = 200
    p = 30
    alpha = 0.05
    fdp_list = []

    for _ in range(n_trials):
        pvals = rng.uniform(0, 1, (p, p))
        pvals = (pvals + pvals.T) / 2
        np.fill_diagonal(pvals, 1.0)
        adj = fdr_control(pvals, alpha=alpha, method="BH")
        uidx = np.triu_indices(p, k=1)
        n_rejected = adj[uidx].sum()
        if n_rejected > 0:
            fdp_list.append(n_rejected / n_rejected)   # all are FP under global null
        else:
            fdp_list.append(0.0)

    # Under the global null, FDR = FDP ≤ alpha always for BH
    assert all(fdp <= alpha + 1e-9 for fdp in fdp_list), (
        "BH FDR control violated under global null."
    )


def test_by_method_accepted():
    P = _make_p_matrix(10)
    adj = fdr_control(P, alpha=0.05, method="BY")
    assert adj.shape == (10, 10)


def test_invalid_method_raises():
    P = _make_p_matrix(10)
    with pytest.raises(ValueError, match="method must be"):
        fdr_control(P, method="benjamini")


def test_all_significant_when_all_pvalues_zero():
    p = 10
    P = np.zeros((p, p))
    np.fill_diagonal(P, 1.0)
    adj = fdr_control(P, alpha=0.05)
    uidx = np.triu_indices(p, k=1)
    assert adj[uidx].sum() == len(uidx[0]), "Expected all edges when p-values = 0"


def test_none_significant_when_all_pvalues_one():
    p = 10
    P = np.ones((p, p))
    adj = fdr_control(P, alpha=0.05)
    assert adj.sum() == 0, "Expected no edges when all p-values = 1"
