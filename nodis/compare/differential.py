"""
Differential network analysis: compare GGMs across two conditions.

DifferentialNetwork(X1, X2, method="desparsified_test").fit()

DifferentialResult fields
--------------------------
adj_cond1       : (p,p) adjacency, condition 1
adj_cond2       : (p,p) adjacency, condition 2
adj_shared      : edges present in both (AND)
adj_cond1_only  : edges unique to condition 1
adj_cond2_only  : edges unique to condition 2
adj_differential: edges with significantly different weights (method B only)
p_values_diff   : (p,p) differential test p-values (method B only)
adj_diff_fdr    : FDR-controlled differential edges (method B only)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DifferentialResult:
    adj_cond1: np.ndarray
    adj_cond2: np.ndarray
    adj_shared: np.ndarray
    adj_cond1_only: np.ndarray
    adj_cond2_only: np.ndarray
    adj_differential: np.ndarray | None = None
    p_values_diff: np.ndarray | None = None
    adj_diff_fdr: np.ndarray | None = None

    @property
    def n_shared(self) -> int:
        return int(np.triu(self.adj_shared, k=1).sum())

    @property
    def n_cond1_only(self) -> int:
        return int(np.triu(self.adj_cond1_only, k=1).sum())

    @property
    def n_cond2_only(self) -> int:
        return int(np.triu(self.adj_cond2_only, k=1).sum())


class DifferentialNetwork:
    """Differential GGM analysis across two conditions."""

    def __init__(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        method: str = "desparsified_test",
        alpha: float = 0.05,
        fdr: str = "BH",
        n_jobs: int = -1,
    ):
        """
        Parameters
        ----------
        X1 : (n1, p) array  — condition 1 (e.g., treated)
        X2 : (n2, p) array  — condition 2 (e.g., control)
        method : "desparsified_test" | "fused_glasso"
            desparsified_test: fit DesparifiedGGM on each group separately,
                then test for edge-weight differences using asymptotic Z-test.
                Requires n1/p ≥ 1.5 and n2/p ≥ 1.5.
            fused_glasso: use GGLasso ADMM_FGL; yields structural comparison
                but no edge-level p-values. Works at lower n/p.
        alpha : float  FDR level for differential edges
        fdr   : "BH" | "BY"
        n_jobs : int  parallel jobs for desparsified fitting
        """
        if method not in ("desparsified_test", "fused_glasso"):
            raise ValueError(
                f"method must be 'desparsified_test' or 'fused_glasso', got '{method}'"
            )
        self._X1 = np.asarray(X1, dtype=float)
        self._X2 = np.asarray(X2, dtype=float)
        self._method = method
        self._alpha = alpha
        self._fdr = fdr
        self._n_jobs = n_jobs

    def fit(self) -> DifferentialResult:
        if self._method == "desparsified_test":
            return self._fit_desparsified_test()
        else:
            return self._fit_fused_glasso()

    def _fit_desparsified_test(self) -> DifferentialResult:
        from nodis.estimators.desparsified import DesparifiedGGM
        from nodis.inference.fdr import fdr_control
        from scipy.stats import norm

        n1, p = self._X1.shape
        n2 = self._X2.shape[0]

        est1 = DesparifiedGGM(n_jobs=self._n_jobs).fit(self._X1)
        est2 = DesparifiedGGM(n_jobs=self._n_jobs).fit(self._X2)

        adj1 = est1.get_adjacency(alpha=self._alpha, method=self._fdr)
        adj2 = est2.get_adjacency(alpha=self._alpha, method=self._fdr)

        # Per-edge asymptotic variance of omega_hat:
        #   Var(omega_hat_ij) = variance[i,j] / n
        # where variance[i,j] = tau2_i * tau2_j  (from GGMInferenceResult)
        var1 = est1.result_.variance  # (p, p): tau2_i * tau2_j
        var2 = est2.result_.variance

        omega1 = est1.result_.precision
        omega2 = est2.result_.precision

        # Differential Z-test: Z_diff ~ N(0,1) under H0: omega1_ij == omega2_ij
        se2 = var1 / n1 + var2 / n2
        # Guard against near-zero SE (degenerate nodes produce zero variance)
        se2 = np.where(se2 > 0, se2, np.finfo(float).tiny)

        z_diff = (omega1 - omega2) / np.sqrt(se2)
        p_diff = 2.0 * (1.0 - norm.cdf(np.abs(z_diff)))

        # Symmetrise and zero diagonal
        np.fill_diagonal(p_diff, 1.0)
        np.fill_diagonal(z_diff, 0.0)

        adj_diff_fdr = fdr_control(p_diff, alpha=self._alpha, method=self._fdr)

        adj1 = np.asarray(adj1, dtype=int)
        adj2 = np.asarray(adj2, dtype=int)
        adj_shared = adj1 & adj2
        adj_cond1_only = adj1 & ~adj2
        adj_cond2_only = adj2 & ~adj1

        return DifferentialResult(
            adj_cond1=adj1,
            adj_cond2=adj2,
            adj_shared=adj_shared,
            adj_cond1_only=adj_cond1_only,
            adj_cond2_only=adj_cond2_only,
            adj_differential=adj_diff_fdr,
            p_values_diff=p_diff,
            adj_diff_fdr=adj_diff_fdr,
        )

    def _fit_fused_glasso(self) -> DifferentialResult:
        try:
            from gglasso.solver.admm_solver import ADMM_FGL
        except ImportError:
            raise ImportError(
                "gglasso required for fused_glasso method. pip install gglasso"
            )

        n1, p = self._X1.shape
        n2 = self._X2.shape[0]

        S1 = np.cov(self._X1, rowvar=False)
        S2 = np.cov(self._X2, rowvar=False)
        S = np.stack([S1, S2])

        lambda1 = 0.1
        lambda2 = 0.05
        sol, _, _ = ADMM_FGL(
            S, lambda1=lambda1, lambda2=lambda2, reg="FGL",
            n_samples=np.array([n1, n2]),
        )

        prec1 = sol["Theta"][0]
        prec2 = sol["Theta"][1]

        threshold = 1e-4
        adj1 = (np.abs(prec1) > threshold).astype(int)
        adj2 = (np.abs(prec2) > threshold).astype(int)
        np.fill_diagonal(adj1, 0)
        np.fill_diagonal(adj2, 0)

        adj_shared = adj1 & adj2
        adj_cond1_only = adj1 & ~adj2
        adj_cond2_only = adj2 & ~adj1

        return DifferentialResult(
            adj_cond1=adj1,
            adj_cond2=adj2,
            adj_shared=adj_shared,
            adj_cond1_only=adj_cond1_only,
            adj_cond2_only=adj_cond2_only,
        )
