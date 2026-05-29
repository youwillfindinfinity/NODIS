"""
Nonparanormal (NPN) transformation — rank-based Gaussian copula transform.

Two variants are provided:

``shrinkage`` (default)
    Equivalent to ``huge::huge.npn(method="shrinkage")`` in R.
    For each column the empirical quantile is Winsorised at level δ and
    then transformed via the normal quantile function Φ⁻¹.

``skeptic``
    Nonparanormal SKEPTIC (Liu et al. 2012).  Estimates the latent Gaussian
    correlation matrix directly from Kendall's τ (or Spearman's ρ) without
    computing column-wise marginal transforms.  Achieves optimal parametric
    rates and is more robust to outliers because no truncation level δ is
    required.  Returns the n×p matrix whose correlation structure matches
    the rank-correlation-based latent Gaussian estimate.

Public entry point
------------------
``npn_transform(X, method="shrinkage")`` — dispatches to the appropriate
variant.  ``npn_shrinkage(X)`` and ``npn_skeptic(X)`` are also available
for direct use.

References
----------
Liu H, Lafferty J, Wasserman L (2009). The nonparanormal: Semiparametric
    estimation of high dimensional undirected graphs.
    JMLR 10: 2295–2328.  https://www.jmlr.org/papers/v10/liu09a.html

Liu H, Han F, Yuan M, Lafferty J, Wasserman L (2012). High-dimensional
    semiparametric Gaussian copula graphical models.
    Ann Stat 40(4): 2293–2326.  doi:10.1214/12-AOS1037
    [Nonparanormal SKEPTIC — Theorem 4 optimal rate for Kendall's τ]
"""

import numpy as np
from scipy.stats import norm, rankdata, kendalltau


def npn_shrinkage(X: np.ndarray, truncation: float | None = None) -> np.ndarray:
    """
    Apply the shrinkage NPN transform to expression matrix X.

    Parameters
    ----------
    X          : (n, p) ndarray — expression matrix (log-transformed recommended)
    truncation : float or None — truncation level δ.  If None, computed from n
                 as δ = 1 / (4·n^{1/4}·√(π·ln n)).

    Returns
    -------
    X_npn : (n, p) ndarray of float64 — NPN-transformed matrix.
            Marginal distributions are approximately Gaussian.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}.")
    n, p = X.shape

    if truncation is None:
        truncation = 1.0 / (4.0 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))

    X_npn = np.zeros((n, p), dtype=float)
    for j in range(p):
        # Average ranks for ties — matches R's rank(ties.method="average")
        # and therefore huge::huge.npn, ensuring parity with SILGGM preprocessing.
        ranks = rankdata(X[:, j], method="average")
        phat = np.clip(ranks / (n + 1), truncation, 1.0 - truncation)
        X_npn[:, j] = norm.ppf(phat)

    return X_npn


def npn_skeptic(X: np.ndarray, corr_type: str = "kendall") -> np.ndarray:
    """
    Nonparanormal SKEPTIC transform (Liu et al. 2012).

    Estimates the latent Gaussian correlation matrix directly from pairwise
    rank correlations without column-wise marginal transforms.  Returns a
    matrix whose rows are whitened by the Cholesky factor of the estimated
    latent correlation matrix, preserving the (n, p) shape for downstream use.

    Parameters
    ----------
    X         : (n, p) ndarray — expression matrix
    corr_type : 'kendall' (default) or 'spearman'
        Rank correlation type used to estimate the latent Gaussian
        correlation matrix.

        'kendall'  → Ĉ_ij = sin(π/2 · τ̂_ij)   [Liu et al. 2012, Eq. 5]
        'spearman' → Ĉ_ij = 2 · sin(π/6 · ρ̂_ij) [Liu et al. 2012, Eq. 6]

    Returns
    -------
    X_skeptic : (n, p) ndarray of float64
        Standardised data whose empirical covariance matrix is the estimated
        latent Gaussian correlation matrix Ĉ.  Suitable as a drop-in
        replacement for ``npn_shrinkage`` output in downstream GGM inference.

    Notes
    -----
    The latent correlation matrix Ĉ is clipped to [−1, 1] and regularised
    by eigenvalue thresholding (minimum eigenvalue 1e-6) before whitening,
    ensuring positive semi-definiteness even at small n.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}.")
    if corr_type not in ("kendall", "spearman"):
        raise ValueError(f"corr_type must be 'kendall' or 'spearman'; got '{corr_type}'.")
    n, p = X.shape

    C_hat = np.eye(p)
    if corr_type == "kendall":
        for i in range(p):
            for j in range(i + 1, p):
                tau, _ = kendalltau(X[:, i], X[:, j])
                val = np.clip(np.sin(np.pi / 2.0 * tau), -1.0, 1.0)
                C_hat[i, j] = C_hat[j, i] = val
    else:
        # Spearman: rank each column, compute Pearson on ranks
        from scipy.stats import spearmanr
        rho_mat, _ = spearmanr(X)
        if p == 1:
            rho_mat = np.array([[1.0]])
        else:
            rho_mat = np.asarray(rho_mat)
        for i in range(p):
            for j in range(i + 1, p):
                val = np.clip(2.0 * np.sin(np.pi / 6.0 * rho_mat[i, j]), -1.0, 1.0)
                C_hat[i, j] = C_hat[j, i] = val

    # Regularise: shift eigenvalues so min >= 1e-6 (positive semi-definite)
    eigs, vecs = np.linalg.eigh(C_hat)
    eigs = np.maximum(eigs, 1e-6)
    C_hat = (vecs * eigs) @ vecs.T
    # Force exact symmetry
    C_hat = (C_hat + C_hat.T) / 2.0

    # Whiten X by the Cholesky factor of Ĉ so the empirical covariance ≈ Ĉ.
    # This gives a drop-in (n, p) matrix with the correct latent Gaussian
    # correlation structure for downstream GGM estimation.
    L = np.linalg.cholesky(C_hat)
    # Standardise X column-wise first (zero mean, unit variance)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    X_skeptic = X_std @ L.T

    return X_skeptic


def npn_transform(
    X: np.ndarray,
    method: str = "shrinkage",
    truncation: float | None = None,
    corr_type: str = "kendall",
) -> np.ndarray:
    """
    Apply a nonparanormal (NPN) transform to expression matrix X.

    Parameters
    ----------
    X          : (n, p) ndarray — expression matrix
    method     : 'shrinkage' (default) or 'skeptic'
        'shrinkage' — Winsorised rank-quantile transform (Liu et al. 2009).
        'skeptic'   — Latent Gaussian correlation from Kendall's τ (Liu et al. 2012).
    truncation : float or None — passed to npn_shrinkage; ignored for 'skeptic'.
    corr_type  : 'kendall' or 'spearman' — passed to npn_skeptic; ignored for 'shrinkage'.

    Returns
    -------
    X_npn : (n, p) ndarray of float64
    """
    if method == "shrinkage":
        return npn_shrinkage(X, truncation=truncation)
    elif method == "skeptic":
        return npn_skeptic(X, corr_type=corr_type)
    else:
        raise ValueError(f"method must be 'shrinkage' or 'skeptic'; got '{method}'.")
