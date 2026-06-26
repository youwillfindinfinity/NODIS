"""
Reference implementation of the B_NW_SL estimator (Ren et al. 2015).

This module is deliberately unoptimised — it follows the paper formula
line-by-line using vanilla sklearn Lasso — so it can serve as an independent
correctness reference for parity tests against DesparifiedGGM.

Do NOT import this in production code.  Use nodis.estimators.desparsified.
"""

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


def _scaled_lambda(n: int, p: int) -> float:
    """Scaled Lasso tuning parameter (Sun & Zhang 2012; matches SILGGM B_NW_SL)."""
    if p / np.sqrt(n) <= 1:
        return np.sqrt(2 * np.log(p) / n)
    return np.sqrt(2 * np.log(p / np.sqrt(n)) / n)


def fit_reference_bnwsl(
    X: np.ndarray,
    lambda_scale: float = 1.0,
    standardise: bool = True,
) -> dict:
    """
    Fit the symmetrised B_NW_SL estimator from Ren et al. (2015).

    Equation references follow Ren et al. Ann Stat 43(3): 991–1026.

    Parameters
    ----------
    X : (n, p) array — input data matrix
    lambda_scale : float — multiplicative scaling of the tuning parameter
    standardise : bool — centre and scale columns before fitting

    Returns
    -------
    dict with keys:
        z_scores  : (p, p) symmetric z-score matrix, diagonal = 0
        p_values  : (p, p) two-sided p-values, diagonal = 1
        precision : (p, p) de-biased precision matrix estimate
        tau2      : (p,) nodewise residual variances
    """
    if standardise:
        X = StandardScaler().fit_transform(X)

    n, p = X.shape
    lam = lambda_scale * _scaled_lambda(n, p)

    # --- Step 1: nodewise Lasso regressions (Eq. 2.1–2.2) ---
    Beta = np.zeros((p, p))   # Beta[i, j] = β̂_ij (coef of X_j in reg. of X_i)
    tau2 = np.zeros(p)        # τ̂²_i = ||ẑ_i||² / n

    for i in range(p):
        mask = np.ones(p, dtype=bool)
        mask[i] = False
        X_sub = X[:, mask]
        y = X[:, i]

        est = Lasso(alpha=lam, fit_intercept=False, max_iter=10_000, tol=1e-6)
        est.fit(X_sub, y)

        resid = y - X_sub @ est.coef_
        tau2[i] = np.dot(resid, resid) / n
        Beta[i, mask] = est.coef_

    # --- Step 2: symmetrised de-biased precision (Ren et al. Eq. 2.5) ---
    Omega = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            # β̂_ij / τ̂²_i + β̂_ji / τ̂²_j, averaged
            omega_ij = -0.5 * (Beta[i, j] / tau2[i] + Beta[j, i] / tau2[j])
            Omega[i, j] = omega_ij
            Omega[j, i] = omega_ij

    # --- Step 3: asymptotic z-scores (Ren et al. Eq. 2.6) ---
    tau = np.sqrt(tau2)
    Z = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            sigma_ij = tau[i] * tau[j]
            z = np.sqrt(n) * Omega[i, j] / sigma_ij if sigma_ij > 0 else 0.0
            Z[i, j] = z
            Z[j, i] = z

    # --- Step 4: two-sided p-values ---
    P = np.ones((p, p))
    uidx = np.triu_indices(p, k=1)
    P[uidx] = 2 * (1 - norm.cdf(np.abs(Z[uidx])))
    P[(uidx[1], uidx[0])] = P[uidx]

    return {"z_scores": Z, "p_values": P, "precision": Omega, "tau2": tau2}
