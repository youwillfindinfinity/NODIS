"""
PIGLASSO stability-selection GGM estimator.

Implements subsampling stability selection (Meinshausen & Bühlmann 2010)
using sklearn graphical_lasso as the base estimator.  For each of Q random
subsamples of size b = ceil(b_perc * n), graphical_lasso is fitted at each of
n_lambda regularisation values.  The stability score for edge (i, j) is the
maximum selection frequency across the lambda grid.

This is a pure-Python re-implementation of the QJSweeper logic from the
original PIGLASSO pipeline (Roland Bumbuc, Amsterdam UMC).  rpy2 is not used.

API contract: fit(X) / get_adjacency(threshold) / precision_ property.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
from sklearn.covariance import empirical_covariance


class PIGLassoEstimator:
    """
    PIGLASSO stability-selection GGM estimator.

    Draws Q random subsamples of size b = ceil(b_perc * n) from X and fits
    sklearn graphical_lasso at each of n_lambda regularisation values evenly
    spaced between lambda_lo and lambda_hi.  The stability score for each edge
    is the maximum selection frequency across the lambda grid.

    Parameters
    ----------
    Q         : number of subsamples
    b_perc    : subsample fraction (0 < b_perc < 1)
    n_lambda  : number of regularisation grid points
    lambda_lo : lower regularisation bound
    lambda_hi : upper regularisation bound
    pi_thr    : stability threshold for adjacency call (default 0.5)
    n_jobs    : parallel joblib workers for the subsampling loop (1 = sequential)
    seed      : random seed
    """

    def __init__(
        self,
        Q: int = 50,
        b_perc: float = 0.65,
        n_lambda: int = 20,
        lambda_lo: float = 0.05,
        lambda_hi: float = 0.30,
        pi_thr: float = 0.5,
        n_jobs: int = 1,
        seed: int = 42,
    ) -> None:
        self.Q = Q
        self.b_perc = b_perc
        self.n_lambda = n_lambda
        self.lambda_lo = lambda_lo
        self.lambda_hi = lambda_hi
        self.pi_thr = pi_thr
        self.n_jobs = n_jobs
        self.seed = seed
        self._stability: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_subsamples(self, n: int, b: int) -> list[tuple[int, ...]]:
        """Draw up to Q unique subsamples of size b from {0, …, n-1}."""
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit graphical_lasso at each lambda for one subsample.

        Returns
        -------
        counts  : (p, p, n_lams) int8 — 1 where |precision| > 1e-5
        success : (n_lams,) int8     — 1 where the solver converged
        """
        from sklearn.covariance import graphical_lasso as sk_glasso

        p = sub.shape[1]
        n_lams = len(lambda_range)
        counts = np.zeros((p, p, n_lams), dtype=np.int8)
        success = np.zeros(n_lams, dtype=np.int8)

        S = empirical_covariance(sub)

        for li, lam in enumerate(lambda_range):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    _, prec = sk_glasso(S, alpha=float(lam), max_iter=200, tol=1e-4)
                counts[:, :, li] = (np.abs(prec) > 1e-5).astype(np.int8)
                success[li] = 1
            except Exception:
                pass  # counts and success remain 0 for this lambda

        return counts, success

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "PIGLassoEstimator":
        n, p = X.shape
        b = max(2, int(np.ceil(self.b_perc * n)))
        if b >= n:
            raise ValueError(
                f"b_perc={self.b_perc} yields b={b} >= n={n}. "
                "Reduce b_perc so that at least one sample is held out."
            )

        lambda_range = np.linspace(self.lambda_lo, self.lambda_hi, self.n_lambda)
        subsamples = self._draw_subsamples(n, b)

        edge_counts = np.zeros((p, p, self.n_lambda), dtype=np.float32)
        success_counts = np.zeros(self.n_lambda, dtype=np.float32)

        if self.n_jobs == 1:
            for idx in subsamples:
                sub = X[np.array(idx), :]
                c, s = self._edge_counts_for_subsample(sub, lambda_range)
                edge_counts += c
                success_counts += s
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(PIGLassoEstimator._edge_counts_for_subsample)(
                    X[np.array(idx), :], lambda_range
                )
                for idx in subsamples
            )
            for c, s in results:
                edge_counts += c
                success_counts += s

        # Selection frequency per lambda — shape (p, p, n_lams)
        denom = np.maximum(success_counts, 1.0)
        freq = edge_counts / denom[np.newaxis, np.newaxis, :]

        # Stability score = max selection frequency over the lambda grid
        stability = freq.max(axis=2).astype(np.float64)
        np.fill_diagonal(stability, 0.0)
        self._stability = stability
        return self

    def get_adjacency(self, threshold: float | None = None) -> np.ndarray:
        """
        Binary adjacency where stability score >= threshold.

        Parameters
        ----------
        threshold : if None, uses self.pi_thr (default 0.5)
        """
        if self._stability is None:
            raise RuntimeError("Call fit() before get_adjacency().")
        thr = self.pi_thr if threshold is None else threshold
        adj = (self._stability >= thr).astype(int)
        np.fill_diagonal(adj, 0)
        return adj

    @property
    def precision_(self) -> np.ndarray:
        """Stability scores, used as edge-ranking scores for AUPR / AUROC."""
        if self._stability is None:
            raise RuntimeError("Call fit() before accessing precision_.")
        return self._stability
