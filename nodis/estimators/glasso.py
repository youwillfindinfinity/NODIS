"""
Baseline GGM estimators: sklearn GraphicalLassoCV and GGLasso.

These provide point estimates (no p-values or confidence intervals) and
serve as comparison baselines in the benchmark study (RQ2–RQ5).

API contract: all estimators expose ``fit(X)`` and ``get_adjacency(threshold)``.
"""

import numpy as np
from sklearn.covariance import GraphicalLassoCV


class SklearnGLasso:
    """
    sklearn GraphicalLassoCV wrapper.

    Selects the regularisation parameter via cross-validation, then returns
    the estimated sparse precision matrix.

    Parameters
    ----------
    cv         : int — number of CV folds (default 5)
    max_iter   : int — maximum EM iterations (default 200)
    n_jobs     : int — parallel CV jobs (default -1 = all CPUs)
    """

    def __init__(self, cv: int = 5, max_iter: int = 200, n_jobs: int = -1) -> None:
        self.cv = cv
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self._model: GraphicalLassoCV | None = None

    def fit(self, X: np.ndarray) -> "SklearnGLasso":
        self._model = GraphicalLassoCV(
            cv=self.cv, max_iter=self.max_iter, n_jobs=self.n_jobs
        )
        self._model.fit(X)
        return self

    def get_adjacency(self, threshold: float = 0.0) -> np.ndarray:
        """Return binary adjacency for |precision| > threshold."""
        if self._model is None:
            raise RuntimeError("Call fit() before get_adjacency().")
        prec = self._model.precision_
        adj = (np.abs(prec) > threshold).astype(int)
        np.fill_diagonal(adj, 0)
        return adj

    @property
    def precision_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before accessing precision_.")
        return self._model.precision_


class GGLassoEstimator:
    """
    GGLasso wrapper (pip install gglasso).

    GGLasso implements the Group Graphical Lasso, which reduces to
    standard GLasso in the single-dataset setting.

    Parameters
    ----------
    lambda1 : float — regularisation parameter (default 0.1)
    """

    def __init__(self, lambda1: float = 0.1) -> None:
        self.lambda1 = lambda1
        self._precision: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "GGLassoEstimator":
        try:
            from gglasso.problem import glasso_problem  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "gglasso is not installed.  Install with: pip install gglasso"
            ) from exc

        n, p = X.shape
        S = np.cov(X.T, bias=False)          # (p, p) sample covariance
        problem = glasso_problem(S, n, reg_params={"lambda1": self.lambda1})
        problem.solve()
        self._precision = problem.solution.precision_
        return self

    def get_adjacency(self, threshold: float = 0.0) -> np.ndarray:
        if self._precision is None:
            raise RuntimeError("Call fit() before get_adjacency().")
        adj = (np.abs(self._precision) > threshold).astype(int)
        np.fill_diagonal(adj, 0)
        return adj

    @property
    def precision_(self) -> np.ndarray:
        if self._precision is None:
            raise RuntimeError("Call fit() before accessing precision_.")
        return self._precision
