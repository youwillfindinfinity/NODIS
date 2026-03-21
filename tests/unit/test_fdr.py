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
    """Monte Carlo FDR estimate under global null must be <= alpha.

    BH guarantees E[FDP] <= alpha, NOT FDP <= alpha per trial.
    Under the global null all rejections are false positives, so:
      FDP_trial = 1.0 if any rejection else 0.0
    The test checks the Monte Carlo mean over 200 trials.
    """
    rng = np.random.default_rng(0)
    alpha = 0.05
    p = 30
    n_trials = 200
    fdp_list = []

    for _ in range(n_trials):
        # Global null: uniform p-values on [0, 1]
        pvals = rng.uniform(size=(p, p))
        pvals = (pvals + pvals.T) / 2
        np.fill_diagonal(pvals, 1.0)
        adj = fdr_control(pvals, alpha=alpha, method="BH")
        uidx = np.triu_indices(p, k=1)
        n_rejected = int(adj[uidx].sum())
        # Under global null: every rejection is a FP → FDP = 1 if any, else 0
        fdp_list.append(1.0 if n_rejected > 0 else 0.0)

    mc_fdr = float(np.mean(fdp_list))
    # Allow generous tolerance: E[FDP] <= alpha with high probability over 200 trials
    assert mc_fdr <= alpha + 0.02, (
        f"Monte Carlo FDR estimate {mc_fdr:.4f} exceeds alpha={alpha} + 0.02. "
        "BH FDR control appears broken."
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
