"""
PIGLASSO stability-selection GGM estimator — with optional prior support.

Implements subsampling stability selection (Meinshausen & Bühlmann 2010)
using GGLasso ADMM_SGL as the base estimator.  For each of Q random
subsamples of size b = ceil(b_perc * n), ADMM_SGL is fitted at each of
n_lambda regularisation values.  The stability score for edge (i, j) is the
maximum selection frequency across the lambda grid.

When a prior matrix P is supplied to fit(), the effective per-edge penalty is:
    lambda_ij = lambda * (1 - prior_weight * P_ij)
giving edges with high prior belief a lower regularisation penalty.

This is a pure-Python re-implementation of the QJSweeper logic from the
original PIGLASSO pipeline (Roland Bumbuc, Amsterdam UMC).  rpy2 is not used.

API contract: fit(X, prior=None) / get_adjacency(threshold) / precision_ property.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
from sklearn.covariance import empirical_covariance


class PIGLassoEstimator:
    """
    PIGLASSO stability-selection GGM estimator.

    Parameters
    ----------
    Q            : number of subsamples
    b_perc       : subsample fraction (0 < b_perc < 1)
    n_lambda     : number of regularisation grid points
    lambda_lo    : lower regularisation bound
    lambda_hi    : upper regularisation bound
    pi_thr       : stability threshold for adjacency call.  Pass a float in
                   (0.5, 1.0) to use a fixed threshold, or the string
                   "adaptive" to apply the Meinshausen & Bühlmann (2010)
                   data-adaptive formula:
                       pi_thr = 0.5 + q̄ / (2 * sqrt(max_edges * v_target))
                   where q̄ is the mean number of edges selected per subsample
                   (averaged over the lambda grid and all Q subsamples) and
                   max_edges = p(p-1)/2.  The computed value is stored in
                   pi_thr_adaptive_ after fit() is called.
    v_target     : target PFER (expected number of false edges) used when
                   pi_thr="adaptive".  Default 1.0.  Increase to be more
                   permissive; decrease to be more conservative.
    prior_weight : alpha in mask = 1 - alpha * prior  (0 = no prior effect)
    n_jobs       : parallel joblib workers (1 = sequential)
    seed         : random seed
    """

    def __init__(
        self,
        Q: int = 50,
        b_perc: float = 0.65,
        n_lambda: int = 20,
        lambda_lo: float = 0.05,
        lambda_hi: float = 0.30,
        pi_thr: float | str = 0.5,
        v_target: float = 1.0,
        prior_weight: float = 0.5,
        n_jobs: int = 1,
        seed: int = 42,
    ) -> None:
        if isinstance(pi_thr, str) and pi_thr != "adaptive":
            raise ValueError(f"pi_thr must be a float in (0.5, 1.0) or 'adaptive', got {pi_thr!r}")
        if isinstance(pi_thr, float) and not (0.5 < pi_thr <= 1.0):
            raise ValueError(f"Fixed pi_thr must be in (0.5, 1.0], got {pi_thr}")
        if v_target <= 0:
            raise ValueError(f"v_target must be positive, got {v_target}")
        self.Q = Q
        self.b_perc = b_perc
        self.n_lambda = n_lambda
        self.lambda_lo = lambda_lo
        self.lambda_hi = lambda_hi
        self.pi_thr = pi_thr
        self.v_target = v_target
        self.prior_weight = prior_weight
        self.n_jobs = n_jobs
        self.seed = seed
        self._stability: np.ndarray | None = None
        self.pi_thr_adaptive_: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_subsamples(self, n: int, b: int) -> list[tuple[int, ...]]:
        rng = random.Random(self.seed)
        indices: set[tuple[int, ...]] = set()
        max_attempts = max(int(1e6), self.Q * 100)
        attempts = 0
        while len(indices) < self.Q and attempts < max_attempts:
            idx = tuple(sorted(rng.sample(range(n), b)))
            indices.add(idx)
            attempts += 1
        return list(indices)

    @staticmethod
    def _edge_counts_for_subsample(
        sub: np.ndarray,
        lambda_range: np.ndarray,
        lambda1_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit ADMM_SGL at each lambda for one subsample.

        When lambda1_mask is provided (prior-informed mode), the effective
        per-edge penalty is lambda * lambda1_mask[i,j].

        Returns
        -------
        counts  : (p, p, n_lams) int8 — 1 where |Theta_ij| > 1e-5
        success : (n_lams,) int8     — 1 where solver converged
        """
        from gglasso.solver.single_admm_solver import ADMM_SGL

        p = sub.shape[1]
        n_lams = len(lambda_range)
        counts  = np.zeros((p, p, n_lams), dtype=np.int8)
        success = np.zeros(n_lams, dtype=np.int8)

        S = empirical_covariance(sub)
        Omega_0 = np.eye(p)

        for li, lam in enumerate(lambda_range):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    sol = ADMM_SGL(
                        S, float(lam), Omega_0,
                        lambda1_mask=lambda1_mask,
                        max_iter=500,
                        tol=1e-5,
                        rtol=1e-4,
                        verbose=False,
                    )
                Theta = sol[1]          # precision matrix estimate
                edge_mask = (np.abs(Theta) > 1e-5).astype(np.int8)
                np.fill_diagonal(edge_mask, 0)
                counts[:, :, li] = edge_mask
                success[li] = 1
                Omega_0 = sol[0]        # warm-start next lambda
            except Exception:
                pass

        return counts, success

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, prior: np.ndarray | None = None) -> "PIGLassoEstimator":
        """
        Fit the stability-selection estimator.

        Parameters
        ----------
        X     : (n, p) data matrix (samples × features)
        prior : (p, p) symmetric prior matrix with values in [0, 1] and
                zero diagonal.  Edges with prior_ij > 0 receive reduced
                regularisation: effective_lambda_ij = lambda * (1 - prior_weight * prior_ij).
                If None, runs standard stability selection without prior.
        """
        n, p = X.shape
        b = max(2, int(np.ceil(self.b_perc * n)))
        if b >= n:
            raise ValueError(
                f"b_perc={self.b_perc} yields b={b} >= n={n}. "
                "Reduce b_perc so that at least one sample is held out."
            )

        # Build per-edge penalty mask
        if prior is not None:
            prior = np.clip(prior, 0.0, 1.0)
            np.fill_diagonal(prior, 0.0)
            prior = (prior + prior.T) / 2.0          # enforce symmetry
            lambda1_mask = 1.0 - self.prior_weight * prior
            lambda1_mask = np.clip(lambda1_mask, 0.1, 1.0)   # floor at 10% lambda
        else:
            lambda1_mask = None

        lambda_range = np.linspace(self.lambda_lo, self.lambda_hi, self.n_lambda)
        subsamples   = self._draw_subsamples(n, b)

        edge_counts   = np.zeros((p, p, self.n_lambda), dtype=np.float32)
        success_counts = np.zeros(self.n_lambda, dtype=np.float32)

        if self.n_jobs == 1:
            for idx in subsamples:
                sub = X[np.array(idx), :]
                c, s = self._edge_counts_for_subsample(sub, lambda_range, lambda1_mask)
                edge_counts   += c
                success_counts += s
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(PIGLassoEstimator._edge_counts_for_subsample)(
                    X[np.array(idx), :], lambda_range, lambda1_mask
                )
                for idx in subsamples
            )
            for c, s in results:
                edge_counts   += c
                success_counts += s

        denom     = np.maximum(success_counts, 1.0)
        freq      = edge_counts / denom[np.newaxis, np.newaxis, :]
        stability = freq.max(axis=2).astype(np.float64)
        np.fill_diagonal(stability, 0.0)
        self._stability = stability

        # Adaptive pi_thr — Meinshausen & Bühlmann (2010) Corollary 1.
        # q̄ = mean number of upper-triangle edges selected per (subsample, lambda) pair.
        # pi_thr = 0.5 + q̄ / (2 * sqrt(max_edges * v_target))
        # Clipped to [0.5 + ε, 0.9] to stay in a sensible operating range.
        max_edges = p * (p - 1) / 2
        # freq[i,j,l] = selection frequency for edge (i,j) at lambda index l.
        # Mean over upper triangle and lambda gives q̄ per subsample slot.
        triu = np.triu_indices(p, k=1)
        q_bar = float(freq[triu[0], triu[1], :].mean()) * max_edges
        pi_adaptive = 0.5 + q_bar / (2.0 * np.sqrt(max_edges * self.v_target))
        pi_adaptive = float(np.clip(pi_adaptive, 0.501, 0.90))
        self.pi_thr_adaptive_ = pi_adaptive

        return self

    def get_adjacency(self, threshold: float | None = None) -> np.ndarray:
        """
        Return binary adjacency matrix.

        Parameters
        ----------
        threshold : float or None
            Override the instance pi_thr for this call.  If None, uses
            self.pi_thr; if self.pi_thr == "adaptive", uses the value
            computed by fit() and stored in pi_thr_adaptive_.
        """
        if self._stability is None:
            raise RuntimeError("Call fit() before get_adjacency().")

        if threshold is not None:
            thr = float(threshold)
        elif self.pi_thr == "adaptive":
            if self.pi_thr_adaptive_ is None:
                raise RuntimeError("Adaptive threshold not yet computed — call fit() first.")
            thr = self.pi_thr_adaptive_
        else:
            thr = float(self.pi_thr)

        adj = (self._stability >= thr).astype(int)
        np.fill_diagonal(adj, 0)
        return adj

    @property
    def precision_(self) -> np.ndarray:
        """Stability scores used as edge-ranking scores for AUPR/AUROC."""
        if self._stability is None:
            raise RuntimeError("Call fit() before accessing precision_.")
        return self._stability
