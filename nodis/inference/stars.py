"""
StARS (Stability Approach to Regularisation Selection) for GGMs.

Provides a non-parametric graph selector for regimes where asymptotic
guarantees of the de-sparsified estimator may not hold (n/p < 5).

Reference
---------
Liu H, Roeder K, Wasserman L (2010). Stability approach to regularisation
    selection (StARS) for high dimensional graphical models.
    NeurIPS 23: 1432–1440.
"""

import copy

import numpy as np


def _instability(freq: np.ndarray, p: int) -> float:
    """Mean edge instability D = mean_{i<j} 4 * F_ij * (1 - F_ij)."""
    uidx = np.triu_indices(p, k=1)
    f = freq[uidx]
    return float(np.mean(4.0 * f * (1.0 - f)))


def stars_select(
    X: np.ndarray,
    estimator,
    beta: float = 0.05,
    n_subsamples: int = 20,
    subsample_frac: float = 0.75,
    threshold_grid: np.ndarray | None = None,
    n_jobs: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Select a sparse graph via StARS stability criterion.

    Varies a threshold over ``threshold_grid``, fits ``estimator`` on
    ``n_subsamples`` random subsamples at each threshold, and returns the
    adjacency corresponding to the largest threshold (sparsest graph) whose
    mean edge instability D ≤ beta.

    This is the recommended alternative to parametric FDR control when n/p < 5
    and asymptotic normality of the de-sparsified z-scores cannot be assumed.

    Parameters
    ----------
    X : (n, p) ndarray
        Expression matrix; rows = samples, columns = genes / variables.
    estimator :
        Any object with ``.fit(X: ndarray)`` returning itself and
        ``.get_adjacency(threshold: float)`` returning a binary (p, p)
        integer ndarray.  ``DesparifiedGGM`` and ``SklearnGLasso`` both
        satisfy this interface.
    beta : float, default 0.05
        Instability threshold.  The selected graph has D ≤ beta, where
        D = mean_{i<j} 4 · F_ij · (1 − F_ij) and F_ij is the fraction of
        subsamples in which edge (i, j) was included.
    n_subsamples : int, default 20
        Number of subsampled replicates per threshold.
    subsample_frac : float, default 0.75
        Fraction of n used per subsample: sub_size = floor(subsample_frac * n).
    threshold_grid : ndarray or None
        Candidate threshold values passed to ``estimator.get_adjacency(θ)``.
        For ``DesparifiedGGM`` these are FDR alpha levels; for ``SklearnGLasso``
        these are edge-weight magnitude cutoffs.
        If None, defaults to 30 values linearly spaced in [0.01, 0.5].
    n_jobs : int, default 1
        Number of parallel workers (via joblib) for the subsample fits.
        ``-1`` uses all available CPUs.  Set to 1 to disable parallelism.
    seed : int, default 0
        Random seed for subsample indices.

    Returns
    -------
    adj : (p, p) integer ndarray
        Binary symmetric adjacency at the selected threshold.
        Diagonal is zero (no self-loops).
    selected_threshold : float
        The threshold value that was selected.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1); got {beta}.")
    if not (0.0 < subsample_frac < 1.0):
        raise ValueError(f"subsample_frac must be in (0, 1); got {subsample_frac}.")
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0; use 1 for sequential or -1 for all CPUs.")

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}.")

    n, p = X.shape
    sub_size = int(np.floor(subsample_frac * n))

    if threshold_grid is None:
        threshold_grid = np.linspace(0.01, 0.5, 30)
    threshold_grid = np.asarray(threshold_grid, dtype=float)
    if threshold_grid.size == 0:
        raise ValueError("threshold_grid must contain at least one value.")

    rng = np.random.default_rng(seed)
    # Pre-draw all subsample indices for reproducibility
    indices = [rng.choice(n, size=sub_size, replace=False)
               for _ in range(n_subsamples)]

    def _fit_subsample(idx: np.ndarray, theta: float) -> np.ndarray:
        est = copy.deepcopy(estimator)
        est.fit(X[idx])
        return est.get_adjacency(theta)

    # Sort grid descending: largest threshold = sparsest graph = first candidate
    grid_desc = np.sort(threshold_grid)[::-1]

    selected_adj = None
    selected_theta = float(grid_desc[-1])   # fallback: densest graph

    if n_jobs == 1:
        for theta in grid_desc:
            freq = np.zeros((p, p), dtype=float)
            for idx in indices:
                freq += _fit_subsample(idx, theta)
            freq /= n_subsamples
            D = _instability(freq, p)
            if D <= beta:
                est_full = copy.deepcopy(estimator)
                est_full.fit(X)
                selected_adj = est_full.get_adjacency(theta)
                selected_theta = float(theta)
                break
    else:
        from joblib import Parallel, delayed

        for theta in grid_desc:
            adjs = Parallel(n_jobs=n_jobs)(
                delayed(_fit_subsample)(idx, theta) for idx in indices
            )
            freq = np.mean(adjs, axis=0)
            D = _instability(freq, p)
            if D <= beta:
                est_full = copy.deepcopy(estimator)
                est_full.fit(X)
                selected_adj = est_full.get_adjacency(theta)
                selected_theta = float(theta)
                break

    if selected_adj is None:
        # No threshold gave D <= beta: return densest (smallest threshold in grid)
        est_full = copy.deepcopy(estimator)
        est_full.fit(X)
        selected_adj = est_full.get_adjacency(float(grid_desc[-1]))

    return selected_adj, selected_theta
