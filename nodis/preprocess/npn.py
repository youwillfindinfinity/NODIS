"""
Nonparanormal (NPN) transformation — rank-based Gaussian copula transform.

Implements the shrinkage variant of the NPN transform equivalent to
``huge::huge.npn(method="shrinkage")`` in R, for pre-processing
non-Gaussian expression data before GGM inference.

Algorithm
---------
For each column j:
  1. Compute 1-based ranks r_1, …, r_n.
  2. Compute scaled, truncated empirical quantiles:
         p̂_k = clamp(r_k / (n + 1),  δ,  1 − δ)
     where δ = 1 / (4·n^{1/4}·√(π·ln n))  [Liu et al. 2009, Theorem 2].
  3. Apply the normal quantile transform: X̃_kj = Φ^{−1}(p̂_k).

Reference
---------
Liu H, Lafferty J, Wasserman L (2009). The nonparanormal: Semiparametric
    estimation of high dimensional undirected graphs.
    JMLR 10: 2295–2328.  https://www.jmlr.org/papers/v10/liu09a.html
"""

import numpy as np
from scipy.stats import norm, rankdata


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
