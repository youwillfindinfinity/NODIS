"""
Asymptotic confidence intervals for de-sparsified GGM precision matrix entries.

Under the asymptotic normal distribution:

    √n · ω̂_ij / σ̂_ij  →  N(0, 1)   (H₀: ω_ij = 0)

a (1 − α) confidence interval for ω_ij is:

    ω̂_ij  ±  z_{1−α/2} · σ̂_ij / √n

where σ̂²_ij = τ̂²_i · τ̂²_j is the asymptotic variance stored in the
``variance`` field of ``GGMInferenceResult``.
"""

import numpy as np
from scipy.stats import norm


def asymptotic_ci(
    precision: np.ndarray,
    variance: np.ndarray,
    n: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute asymptotic (1 − alpha) confidence intervals.

    Parameters
    ----------
    precision : (p, p) ndarray — de-biased precision matrix estimate
    variance  : (p, p) ndarray — asymptotic variances τ̂²_i · τ̂²_j
    n         : int — number of samples
    alpha     : float, default 0.05 — nominal error rate

    Returns
    -------
    lower, upper : ndarrays of shape (p, p)
        Element-wise lower and upper bounds.  Diagonal entries are
        [0, 0] (no self-loop).
    """
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    se = np.sqrt(variance / n)
    lower = precision - z_crit * se
    upper = precision + z_crit * se
    # diagonal is not meaningful — zero out
    np.fill_diagonal(lower, 0.0)
    np.fill_diagonal(upper, 0.0)
    return lower, upper


def _fit_split_for_ensemble(idx: np.ndarray, X: np.ndarray, lambda_scale: float) -> np.ndarray:
    """Fit DesparifiedGGM on a subsample; return precision matrix."""
    import warnings
    # Deferred import to avoid circular dependency
    from nodis.estimators.desparsified import DesparifiedGGM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        est = DesparifiedGGM(lambda_scale=lambda_scale)
        est.fit(X[idx])
    return est.result_.precision


def ensemble_ci(
    X: np.ndarray,
    n_splits: int = 25,
    subsample_frac: float = 0.5,
    alpha: float = 0.05,
    lambda_scale: float = 1.0,
    seed: int = 0,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensemble confidence intervals via split-and-average de-sparsified Lasso.

    Fits ``DesparifiedGGM`` on ``n_splits`` random subsamples of X, then
    aggregates the per-split de-biased precision estimates.  The ensemble
    average reduces variance of the de-biasing correction and produces CIs
    with better finite-sample coverage than the single-fit asymptotic variant,
    particularly at n/p < 10.

    Parameters
    ----------
    X : (n, p) ndarray
        Expression matrix; rows = samples, columns = genes / variables.
    n_splits : int, default 25
        Number of random subsamples.
    subsample_frac : float, default 0.5
        Fraction of n used per subsample: sub_size = floor(subsample_frac * n).
    alpha : float, default 0.05
        Nominal error rate for the CIs.
    lambda_scale : float, default 1.0
        Passed to ``DesparifiedGGM(lambda_scale=lambda_scale)``.
    seed : int, default 0
        Random seed for subsample indices.
    n_jobs : int, default 1
        Parallel workers via joblib.  ``-1`` = all CPUs.

    Returns
    -------
    omega_ensemble : (p, p) ndarray
        Ensemble-averaged de-biased precision matrix.  Symmetric; diagonal = 0.
    lower : (p, p) ndarray
        Lower CI bounds.  Diagonal = 0.
    upper : (p, p) ndarray
        Upper CI bounds.  Diagonal = 0.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}.")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2; got {n_splits}.")
    if not (0.0 < subsample_frac < 1.0):
        raise ValueError(f"subsample_frac must be in (0, 1); got {subsample_frac}.")

    n, p = X.shape
    sub_size = int(np.floor(subsample_frac * n))
    rng = np.random.default_rng(seed)
    indices = [rng.choice(n, size=sub_size, replace=False)
               for _ in range(n_splits)]

    if n_jobs == 1:
        omegas = np.stack([_fit_split_for_ensemble(idx, X, lambda_scale) for idx in indices], axis=0)
    else:
        from joblib import Parallel, delayed
        omegas = np.stack(
            Parallel(n_jobs=n_jobs)(
                delayed(_fit_split_for_ensemble)(idx, X, lambda_scale) for idx in indices
            ),
            axis=0,
        )   # shape (n_splits, p, p)

    omega_ensemble = np.mean(omegas, axis=0)
    se_ensemble = np.std(omegas, axis=0, ddof=1) / np.sqrt(n_splits)

    z_crit = norm.ppf(1.0 - alpha / 2.0)
    lower = omega_ensemble - z_crit * se_ensemble
    upper = omega_ensemble + z_crit * se_ensemble

    # Symmetrise (individual fits are symmetric; mean/std preserve symmetry,
    # but enforce it numerically to guard against floating-point asymmetry)
    omega_ensemble = (omega_ensemble + omega_ensemble.T) / 2.0
    lower = (lower + lower.T) / 2.0
    upper = (upper + upper.T) / 2.0

    np.fill_diagonal(omega_ensemble, 0.0)
    np.fill_diagonal(lower, 0.0)
    np.fill_diagonal(upper, 0.0)

    return omega_ensemble, lower, upper
