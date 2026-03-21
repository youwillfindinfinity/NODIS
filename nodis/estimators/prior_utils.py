"""
prior_utils.py — Prior matrix builders for PIGLasso.

Two builders:
  build_corr_prior       — data-derived, no ground-truth leak (use in benchmarks)
  build_noisy_oracle_prior — simulation-study only (sensitivity sweep)
"""
from __future__ import annotations

import numpy as np


def build_corr_prior(X: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    Data-derived prior: |Pearson correlation|^gamma.

    Edges with strong marginal correlation get higher prior belief.
    gamma=2 (default) suppresses weak correlations cleanly.
    No ground-truth leak — derived purely from X.

    Parameters
    ----------
    X     : (n, p) data matrix
    gamma : exponent — higher = sparser prior (more conservative)

    Returns
    -------
    P : (p, p) symmetric prior matrix with zeros on diagonal, values in [0, 1]
    """
    C = np.abs(np.corrcoef(X.T)) ** gamma
    np.fill_diagonal(C, 0.0)
    return np.clip(C, 0.0, 1.0)


def build_noisy_oracle_prior(
    adj_true: np.ndarray,
    noise: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulation-study prior: true adjacency with a fraction of entries flipped.

    noise=0.0 → perfect oracle prior
    noise=0.5 → effectively random (no information)

    Use only when ground truth is available (synthetic benchmarks).
    Never use on real data.

    Parameters
    ----------
    adj_true : (p, p) binary true adjacency (symmetric, zero diagonal)
    noise    : fraction of off-diagonal entries to flip (0.0–0.5)
    seed     : RNG seed for reproducibility

    Returns
    -------
    P : (p, p) symmetric prior in {0, 1} with zero diagonal
    """
    rng = np.random.default_rng(seed)
    prior = adj_true.astype(float).copy()
    np.fill_diagonal(prior, 0.0)

    # work on upper triangle only, mirror after
    p = prior.shape[0]
    rows, cols = np.triu_indices(p, k=1)
    n_edges = len(rows)
    n_flip = max(0, int(round(noise * n_edges)))

    flip_idx = rng.choice(n_edges, size=n_flip, replace=False)
    for fi in flip_idx:
        i, j = rows[fi], cols[fi]
        prior[i, j] = 1.0 - prior[i, j]
        prior[j, i] = prior[i, j]

    np.fill_diagonal(prior, 0.0)
    return np.clip(prior, 0.0, 1.0)
