"""
Native Python implementation of the de-sparsified nodewise Lasso estimator
for Gaussian Graphical Models.

Implements the symmetrised B_NW_SL (Bias-corrected Nodewise regression with
Scaled Lasso) estimator.  For each off-diagonal pair (i, j) the precision
matrix entry is estimated as the average of both nodewise regression directions:

    ω̂_ij = (−β̂_ij / τ̂²_i  −  β̂_ji / τ̂²_j) / 2

where β̂_ij is the Lasso coefficient of X_j in the nodewise regression of X_i
on X_{-i}, and τ̂²_i = ||ẑ_i||² / n is the nodewise residual variance.

Asymptotic null distribution (H₀: ω_ij = 0):

    Z_ij = √n · ω̂_ij / (τ̂_i · τ̂_j)  →  N(0, 1)

Tuning parameter (Scaled Lasso):

    λ = λ_scale · √( 2 log(p / √n) / n )

References
----------
van de Geer S, Bühlmann P, Ritov Y, Dezeure R (2014).
    On asymptotically optimal confidence regions and tests for
    high-dimensional models. Ann Stat 42(3): 1166–1202.
    doi:10.1214/14-AOS1221

Zhang C-H, Zhang SS (2014). Confidence intervals for low dimensional
    parameters in high dimensional linear models.
    J R Stat Soc B 76(1): 217–242. doi:10.1111/rssb.12026

Zhang R, Ren Z, Chen W (2018). SILGGM: An extensive R package for efficient
    statistical inference in large-scale gene networks.
    PLoS Comput Biol 14(8): e1006369. doi:10.1371/journal.pcbi.1006369
    [Reference implementation — used for parity validation in RQ1]

Shinkyu P, Sueishi N (2022). Inference with de-sparsified Lasso under a
    small tuning parameter. arXiv:2208.08679.
    [Relaxed lambda_method: λ ~ 1/√n, weaker sparsity requirement]

Bellec PC, Zhang C-H (2022). De-biasing the Lasso with degrees-of-freedom
    adjustment. Bernoulli 28(2): 713–743. doi:10.3150/21-BEJ1348
    [dof_correction: efficiency gain across all sparsity levels]

Note
----
Correctness of this implementation relative to SILGGM B_NW_SL is validated
empirically via the parity test in tests/integration/test_silggm_parity.py
(target: Pearson r > 0.99 on z-scores across all four graph topologies).
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Module-level helper — must be at module scope for joblib pickling
# ---------------------------------------------------------------------------

def _fit_node(i: int, X: np.ndarray, lam: float, p: int,
              max_iter: int, tol: float):
    """Fit one nodewise Lasso regression; called by joblib workers.

    Returns
    -------
    i    : node index
    tau2 : nodewise residual variance ||ẑ_i||² / n
    coef : (p,) coefficient vector (zero at position i)
    nnz  : number of non-zero coefficients (for DoF correction)
    """
    mask = np.ones(p, dtype=bool)
    mask[i] = False
    lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=max_iter, tol=tol)
    lasso.fit(X[:, mask], X[:, i])
    resid = X[:, i] - lasso.predict(X[:, mask])
    tau2 = float(np.dot(resid, resid) / len(resid))
    coef = np.zeros(p)
    coef[mask] = lasso.coef_
    nnz = int(np.sum(lasso.coef_ != 0))
    return i, tau2, coef, nnz


@dataclass
class GGMInferenceResult:
    """Container for de-sparsified GGM inference outputs.

    Attributes
    ----------
    z_scores  : (p, p) ndarray — asymptotic z-scores; diagonal = 0
    p_values  : (p, p) ndarray — two-sided p-values; diagonal = 1
    precision : (p, p) ndarray — de-biased precision matrix estimate; symmetric
    variance  : (p, p) ndarray — asymptotic variance τ̂²_i · τ̂²_j per entry
    adj_fdr   : (p, p) ndarray or None — FDR-controlled binary adjacency
    fdr_alpha : float or None — FDR level used for adj_fdr
    """

    z_scores: np.ndarray
    p_values: np.ndarray
    precision: np.ndarray
    variance: np.ndarray
    adj_fdr: Optional[np.ndarray] = None
    fdr_alpha: Optional[float] = None


class DesparifiedGGM:
    """
    De-sparsified nodewise Lasso estimator for GGM inference.

    Parameters
    ----------
    lambda_scale : float, default 1.0
        Multiplicative scaling of the Scaled Lasso tuning parameter.
        Full formula: λ = lambda_scale · √(2 log(p / √n) / n).
    lambda_method : str, default 'scaled'
        Tuning parameter formula.

        'scaled'  — van de Geer et al. (2014) / Zhang & Zhang (2014):
                    λ = lambda_scale · √(2 log(p / √n) / n).
                    Requires sparsity s₀ = o(n / log p).  Matches SILGGM B_NW_SL.

        'relaxed' — Shinkyu & Sueishi (2022, arXiv:2208.08679):
                    λ = lambda_scale / √n.
                    Weaker sparsity requirement: s₀ = o(√(n / log p)).
                    Reduces bias when p/n is small; may increase variance.
    dof_correction : bool, default False
        Apply the degrees-of-freedom correction to the de-biased precision
        entries (Bellec & Zhang 2022, doi:10.3150/21-BEJ1348).

        When True, the de-biasing formula is adjusted by the effective
        degrees of freedom of each nodewise Lasso fit:
            df_i = n - ||β̂_i||₀   (number of non-zero coefficients)
            ω̂_ij^{DoF} = ω̂_ij · n / mean(df_i, df_j)

        This correction improves efficiency when sparsity is uncertain or
        moderate (s₀ approaching n^{2/3}); it has no effect when all
        β̂_i = 0 (fully sparse Lasso solution, df_i = n).
    standardise : bool, default True
        Centre and scale each column of X to zero mean and unit variance
        before fitting.  Strongly recommended; set to False only when X
        has already been standardised.
    max_iter : int, default 10_000
        Maximum Lasso solver iterations per nodewise regression.
    tol : float, default 1e-6
        Convergence tolerance for the Lasso solver.
    n_jobs : int, default 1
        Number of parallel workers for the p nodewise Lasso regressions.
        ``-1`` uses all available CPU cores.  Uses joblib with the loky
        backend (process-based); each worker receives a copy of X.
        Set to 1 to disable parallelism (deterministic, no overhead).
    """

    def __init__(
        self,
        lambda_scale: float = 1.0,
        lambda_method: str = "scaled",
        dof_correction: bool = False,
        standardise: bool = True,
        max_iter: int = 10_000,
        tol: float = 1e-6,
        n_jobs: int = 1,
    ) -> None:
        if lambda_method not in ("scaled", "relaxed"):
            raise ValueError(
                f"lambda_method must be 'scaled' or 'relaxed'; got '{lambda_method}'."
            )
        self.lambda_scale = lambda_scale
        self.lambda_method = lambda_method
        self.dof_correction = dof_correction
        self.standardise = standardise
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lambda(self, n: int, p: int) -> float:
        """Tuning parameter λ for nodewise Lasso regressions.

        Two formulas are supported via ``self.lambda_method``:

        'scaled' (default)
            λ = lambda_scale · √(2 log(p / √n) / n)
            Matches SILGGM B_NW_SL (SILGGMCpp.cpp line 801).
            References: Zhang & Zhang (2014) JRSS-B Eq. 2.3;
                        Zhang et al. (2018) PLoS Comput Biol Eq. S1.

        'relaxed'
            λ = lambda_scale / √n
            Shinkyu & Sueishi (2022, arXiv:2208.08679).
            Reduces bias when p/n is small; relaxes sparsity requirement
            from s₀ = o(n / log p) to s₀ = o(√(n / log p)).
        """
        if self.lambda_method == "relaxed":
            return self.lambda_scale / np.sqrt(n)

        # 'scaled' path
        log_arg = p / np.sqrt(n)
        if log_arg > 1.0:
            return self.lambda_scale * np.sqrt(2.0 * np.log(log_arg) / n)
        # p < sqrt(n): scaled-Lasso formula undefined (log ≤ 0).
        # Fall back to the standard oracle Lasso lambda sqrt(2 log(p) / n).
        warnings.warn(
            f"p={p} < sqrt(n)={np.sqrt(n):.1f}: the scaled Lasso tuning formula "
            "sqrt(2·log(p/sqrt(n))/n) is undefined for p/sqrt(n) ≤ 1. "
            "Falling back to sqrt(2·log(p)/n) for the low-dimensional regime. "
            "Inference remains valid; parity with SILGGM applies only when p > sqrt(n).",
            UserWarning,
            stacklevel=3,
        )
        return self.lambda_scale * np.sqrt(2.0 * np.log(max(p, 2)) / n)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DesparifiedGGM":
        """
        Fit the de-sparsified GGM to an expression matrix.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Expression matrix; rows = samples, columns = genes / variables.

        Returns
        -------
        self
            Fitted estimator.  Results are stored in ``self.result_``.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}.")
        n, p = X.shape

        if n < 5 * p:
            warnings.warn(
                f"n={n} < 5·p={5 * p}: asymptotic normality of Z_ij may not hold "
                "at this n/p ratio (van de Geer et al. 2014, Theorem 2.1). "
                "Consider using nodis.inference.stars.stars_select() as a "
                "non-parametric alternative that does not rely on the Gaussian null.",
                UserWarning,
                stacklevel=2,
            )

        if self.standardise:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        else:
            X = X.copy()

        lam = self._get_lambda(n, p)

        # ----------------------------------------------------------------
        # Step 1 — p nodewise Lasso regressions
        # Beta[i, j]: coefficient of X_j in the regression of X_i on X_{-i}
        # Tau2[i]:    nodewise residual variance ||ẑ_i||² / n
        # ----------------------------------------------------------------
        Beta = np.zeros((p, p))
        Tau2 = np.zeros(p)
        Nnz = np.zeros(p, dtype=int)  # non-zero coef count per node (DoF correction)

        if self.n_jobs == 1:
            # Sequential path — no joblib overhead
            for i in range(p):
                _, tau2_i, coef_i, nnz_i = _fit_node(
                    i, X, lam, p, self.max_iter, self.tol
                )
                Tau2[i] = tau2_i
                Beta[i] = coef_i
                Nnz[i] = nnz_i
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_node)(i, X, lam, p, self.max_iter, self.tol)
                for i in range(p)
            )
            for i, tau2_i, coef_i, nnz_i in results:
                Tau2[i] = tau2_i
                Beta[i] = coef_i
                Nnz[i] = nnz_i

        # Guard against degenerate nodewise variance (near-perfect Lasso fit).
        # np.finfo(float).tiny (~5e-324) would cause -Beta[i,j]/tiny → ±inf in
        # Omega_hat and Z, producing spurious p-values of 0.  Use a relative
        # floor at 1 ppm of the mean positive Tau2 instead.
        tau2_pos = Tau2[Tau2 > 0]
        tau2_floor = (tau2_pos.mean() * 1e-6) if tau2_pos.size > 0 else 1e-8
        tau2_floor = max(tau2_floor, 1e-8)

        degenerate = np.where(Tau2 <= tau2_floor)[0]
        if degenerate.size > 0:
            warnings.warn(
                f"Nodes {degenerate.tolist()} have near-zero nodewise residual "
                f"variance (Tau2 ≤ {tau2_floor:.2e}). These nodes achieved a "
                "near-perfect Lasso fit. Precision entries and p-values for "
                "these nodes are unreliable. Consider reducing lambda_scale or "
                "checking for collinear variables.",
                UserWarning,
                stacklevel=2,
            )
        self.degenerate_nodes_ = degenerate
        Tau2 = np.where(Tau2 > tau2_floor, Tau2, tau2_floor)

        # ----------------------------------------------------------------
        # Step 2 — symmetrised de-biased precision matrix
        #
        # ω̂_ij = (−β̂_ij / τ̂²_i  −  β̂_ji / τ̂²_j) / 2
        #
        # This averages the two nodewise estimates and is symmetric by
        # construction.  Asymptotic variance under H₀: σ̂²_ij = τ̂²_i · τ̂²_j.
        #
        # Optional DoF correction (Bellec & Zhang 2022):
        #   df_i = n − ||β̂_i||₀   (effective degrees of freedom)
        #   ω̂_ij^{DoF} = ω̂_ij · n / mean(df_i, df_j)
        # Improves efficiency across all sparsity levels; no-op when Nnz=0.
        # ----------------------------------------------------------------
        Omega_hat = np.zeros((p, p))
        Var_hat = np.zeros((p, p))

        # Effective degrees of freedom per node
        Df = (n - Nnz).astype(float)
        Df = np.maximum(Df, 1.0)  # guard: df >= 1

        for i in range(p):
            for j in range(i + 1, p):
                omega_ij = (-Beta[i, j] / Tau2[i] - Beta[j, i] / Tau2[j]) / 2.0
                if self.dof_correction:
                    df_mean = (Df[i] + Df[j]) / 2.0
                    omega_ij = omega_ij * n / df_mean
                var_ij = Tau2[i] * Tau2[j]
                Omega_hat[i, j] = Omega_hat[j, i] = omega_ij
                Var_hat[i, j] = Var_hat[j, i] = var_ij

        # ----------------------------------------------------------------
        # Step 3 — z-scores and two-sided p-values
        # Z_ij = √n · ω̂_ij / σ̂_ij  →  N(0,1) under H₀
        # ----------------------------------------------------------------
        from scipy.stats import norm

        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(
                Var_hat > 0,
                np.sqrt(n) * Omega_hat / np.sqrt(Var_hat),
                0.0,
            )
        P = 2.0 * norm.sf(np.abs(Z))
        np.fill_diagonal(Z, 0.0)
        np.fill_diagonal(P, 1.0)

        self._n = n
        self._p = p
        self.result_ = GGMInferenceResult(
            z_scores=Z,
            p_values=P,
            precision=Omega_hat,
            variance=Var_hat,
        )
        return self

    def get_adjacency(
        self, alpha: float = 0.05, method: str = "BH"
    ) -> np.ndarray:
        """
        Apply FDR control to p-values and return the binary adjacency matrix.

        Parameters
        ----------
        alpha  : float, default 0.05 — target FDR level
        method : str, 'BH' or 'BY' — Benjamini–Hochberg or Benjamini–Yekutieli

        Returns
        -------
        adj : (p, p) integer ndarray — symmetric binary adjacency; no self-loops
        """
        from nodis.inference.fdr import fdr_control

        adj = fdr_control(self.result_.p_values, alpha=alpha, method=method)
        self.result_.adj_fdr = adj
        self.result_.fdr_alpha = alpha
        return adj

    def confidence_intervals(
        self,
        alpha: float = 0.05,
        method: str = 'asymptotic',
        X: np.ndarray | None = None,
        n_splits: int = 25,
    ) -> tuple:
        """
        (1 − alpha) confidence intervals for all precision matrix entries.

        Parameters
        ----------
        alpha    : float, default 0.05 — nominal error rate
        method   : 'asymptotic' or 'ensemble'
            'asymptotic' — single-fit CIs via the Gaussian z-score null
                           (correct asymptotically; may under-cover at n/p < 10).
            'ensemble'   — split-and-average CIs (better finite-sample coverage).
                           Requires ``X`` to be passed in.
        X        : (n, p) ndarray or None
            Raw expression matrix.  Required when ``method='ensemble'``.
        n_splits : int, default 25
            Number of subsamples for ``method='ensemble'``.

        Returns
        -------
        lower, upper : ndarrays of shape (p, p)    [asymptotic]
        omega_ensemble, lower, upper : ndarrays     [ensemble]
        """
        if method == 'asymptotic':
            from nodis.inference.confidence import asymptotic_ci
            return asymptotic_ci(
                self.result_.precision,
                self.result_.variance,
                self._n,
                alpha,
            )
        elif method == 'ensemble':
            if X is None:
                raise ValueError(
                    "X must be provided when method='ensemble'. "
                    "Pass the original expression matrix used to fit the estimator."
                )
            from nodis.inference.confidence import ensemble_ci
            return ensemble_ci(X, n_splits=n_splits, alpha=alpha,
                               lambda_scale=self.lambda_scale)
        else:
            raise ValueError(
                f"method must be 'asymptotic' or 'ensemble'; got '{method}'."
            )
