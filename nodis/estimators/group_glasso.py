"""
Multi-condition Graphical Lasso estimators.

Two penalty structures are supported (both via GGLasso's ADMM solver):

  GGL — Group Graphical Lasso (Yuan & Lin 2006 / Danaher et al. 2014)
    Shared *sparsity pattern* across K conditions.
    λ₂ penalises the ℓ₁ / ℓ₂ norm of the group of precision entries
    {Θᵏᵢⱼ}ₖ, encouraging the same edge set to appear in all conditions
    while allowing different magnitudes.
    Use case: multiple tissues / time points where you expect a common
    network backbone but condition-specific edge weights.

  FGL — Fused Graphical Lasso (Danaher et al. 2014)
    Shared *precision values* across K conditions.
    λ₂ penalises ‖Θᵏᵢⱼ − Θˡᵢⱼ‖₁ for all condition pairs (k, l),
    encouraging similar numerical values, not just a shared support.
    Use case: ordered conditions (time series, dose escalation) where
    small but consistent changes are expected.

Regularisation selection
------------------------
If ``lambda1`` and ``lambda2`` are not specified (both None), the estimator
runs GGLasso's eBIC grid search over ``lambda1_range`` × ``lambda2_range``
and selects the pair minimising the extended BIC.  The grid can be
configured via the ``lambda1_range`` / ``lambda2_range`` constructor args.

AnnData integration
-------------------
``MultiConditionGLasso.to_anndata(adata, condition_key)`` writes results
back into an AnnData object by splitting it on ``adata.obs[condition_key]``:

  adata.varp["nodis_mgl_prec_{cond}"]   — precision matrix, condition k
  adata.varp["nodis_mgl_adj_{cond}"]    — binary adjacency, condition k
  adata.varp["nodis_mgl_shared"]        — edges present in ALL conditions
  adata.uns["nodis_mgl"]                — run metadata

The module-level convenience function ``fit_multi_condition(adata, ...)``
wraps the full pipeline in a single call.

References
----------
Danaher P, Wang P, Witten DM (2014). The joint graphical lasso for inverse
    covariance estimation across multiple classes.
    J R Stat Soc B 76(2): 373–397. doi:10.1111/rssb.12033

Hsieh C-J, Dhillon IS, Ravikumar PK, Sustik MA (2014). QUIC: Quadratic
    approximation for sparse inverse covariance estimation.
    JMLR 15: 2911–2947.

Schaipp F et al. (2021). GGLasso: A Python package for General Graphical
    Lasso Computation. JOSS 6(60): 3536. doi:10.21105/joss.03536

Yuan M, Lin Y (2006). Model selection and estimation in the Gaussian
    graphical model. Biometrika 94(1): 19–35. doi:10.1093/biomet/asm018
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MultiConditionGLassoResult:
    """Result of a multi-condition (Group / Fused) Graphical Lasso fit.

    Attributes
    ----------
    precision_       : dict[str, (p, p) ndarray] — per-condition precision matrices
    adjacency_       : dict[str, (p, p) ndarray] — per-condition binary adjacencies
    shared_adjacency : (p, p) ndarray — edges present in ALL conditions (AND mask)
    unique_adjacency : dict[str, (p, p) ndarray] — edges unique to each condition
    reg              : str — 'GGL' or 'FGL'
    lambda1_         : float — fitted λ₁ (sparsity)
    lambda2_         : float — fitted λ₂ (group / fusion)
    condition_names  : list[str] — ordered condition labels
    ebic_selected    : bool — True if λ was chosen by eBIC model selection
    n_samples_       : dict[str, int] — per-condition sample sizes
    """
    precision_: dict
    adjacency_: dict
    shared_adjacency: np.ndarray
    unique_adjacency: dict
    reg: str
    lambda1_: float
    lambda2_: float
    condition_names: list
    ebic_selected: bool = False
    n_samples_: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_conditions(self) -> int:
        return len(self.condition_names)

    @property
    def n_shared_edges(self) -> int:
        return int(np.triu(self.shared_adjacency, k=1).sum())

    def n_edges(self, condition: str) -> int:
        """Number of edges in the given condition."""
        return int(np.triu(self.adjacency_[condition], k=1).sum())

    def summary(self) -> dict:
        """Return a summary dict of edge counts per condition."""
        return {
            "reg": self.reg,
            "lambda1": self.lambda1_,
            "lambda2": self.lambda2_,
            "ebic_selected": self.ebic_selected,
            "n_shared_edges": self.n_shared_edges,
            "edges_per_condition": {
                c: self.n_edges(c) for c in self.condition_names
            },
        }


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class MultiConditionGLasso:
    """
    Group / Fused Graphical Lasso for multi-condition GGM estimation.

    Wraps GGLasso's ADMM solver (pip install gglasso) under a NODIS-style
    API: ``fit(X_dict)`` → ``get_adjacency()`` → ``to_anndata(adata, key)``.

    Parameters
    ----------
    reg : str, default 'GGL'
        Penalty structure: 'GGL' (shared sparsity) or 'FGL' (shared values).
    lambda1 : float or None
        Sparsity penalty λ₁.  If None, eBIC model selection is run.
    lambda2 : float or None
        Group / fusion penalty λ₂.  If None, eBIC model selection is run.
    lambda1_range : array-like or None
        Grid for eBIC search over λ₁ (used when lambda1 is None).
        Default: 10 log-spaced values in [0.02, 0.5].
    lambda2_range : array-like or None
        Grid for eBIC search over λ₂ (used when lambda2 is None).
        Default: 5 log-spaced values in [0.01, 0.2].
    ebic_gamma : float, default 0.1
        eBIC regularisation hyperparameter (0 = BIC; larger → sparser).
    threshold : float, default 0.0
        Absolute precision value below which an edge is removed when
        computing the binary adjacency.  0.0 returns the GGLasso support
        (any non-zero off-diagonal entry is an edge).
    npn : bool, default False
        Apply NPN shrinkage transform per condition before fitting.
    standardise : bool, default True
        Centre and scale each column within each condition before computing
        the sample covariance matrix.
    max_iter : int, default 1_000
        Maximum ADMM iterations.
    tol : float, default 1e-5
        ADMM primal convergence tolerance.
    rtol : float, default 1e-4
        ADMM relative convergence tolerance.
    verbose : bool, default False
        Print GGLasso ADMM iteration info.
    """

    def __init__(
        self,
        reg: str = "GGL",
        lambda1: Optional[float] = None,
        lambda2: Optional[float] = None,
        lambda1_range: Optional[np.ndarray] = None,
        lambda2_range: Optional[np.ndarray] = None,
        ebic_gamma: float = 0.1,
        threshold: float = 0.0,
        npn: bool = False,
        standardise: bool = True,
        max_iter: int = 1_000,
        tol: float = 1e-5,
        rtol: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        if reg not in ("GGL", "FGL"):
            raise ValueError(f"reg must be 'GGL' or 'FGL'; got '{reg}'.")
        self.reg = reg
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # lambda1_range must be in descending order (sparse → dense) for GGLasso
        self.lambda1_range = (
            np.logspace(-0.3, -1.7, 10) if lambda1_range is None
            else np.sort(np.asarray(lambda1_range, dtype=float))[::-1]
        )
        self.lambda2_range = (
            np.logspace(-0.7, -2.0, 5) if lambda2_range is None
            else np.asarray(lambda2_range, dtype=float)
        )
        self.ebic_gamma = ebic_gamma
        self.threshold = threshold
        self.npn = npn
        self.standardise = standardise
        self.max_iter = max_iter
        self.tol = tol
        self.rtol = rtol
        self.verbose = verbose
        self.result_: Optional[MultiConditionGLassoResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_dict: dict) -> "MultiConditionGLasso":
        """
        Fit the multi-condition GGM.

        Parameters
        ----------
        X_dict : dict[str, (n_k, p) ndarray]
            Mapping from condition name to expression matrix.
            All conditions must have the same number of genes (p), but can
            differ in sample size (n_k).

        Returns
        -------
        self  — fitted estimator; results in ``self.result_``.
        """
        self._require_gglasso()

        if len(X_dict) < 2:
            raise ValueError(
                f"MultiConditionGLasso requires at least 2 conditions; "
                f"got {len(X_dict)}."
            )

        condition_names = list(X_dict.keys())
        Xs = [np.asarray(X_dict[c], dtype=float) for c in condition_names]

        # Validate shapes
        p = Xs[0].shape[1]
        for i, (c, X) in enumerate(zip(condition_names, Xs)):
            if X.ndim != 2:
                raise ValueError(
                    f"X_dict['{c}'] must be 2-D (n, p); got shape {X.shape}."
                )
            if X.shape[1] != p:
                raise ValueError(
                    f"All conditions must have the same number of genes p={p}; "
                    f"condition '{c}' has p={X.shape[1]}."
                )

        # Preprocessing
        if self.npn:
            from nodis.preprocess.npn import npn_shrinkage
            Xs = [npn_shrinkage(X) for X in Xs]

        if self.standardise:
            from sklearn.preprocessing import StandardScaler
            Xs = [StandardScaler().fit_transform(X) for X in Xs]

        # Sample covariances and sizes
        ns = np.array([X.shape[0] for X in Xs])
        S = np.stack([np.cov(X.T, bias=False) for X in Xs])  # (K, p, p)

        # Run GGLasso
        lam1, lam2, prec_stack = self._solve(S, ns, p)

        # Build result
        precision_dict = {
            c: prec_stack[k] for k, c in enumerate(condition_names)
        }
        adjacency_dict = {
            c: self._to_adj(prec_stack[k]) for c in condition_names
            for k, cc in enumerate(condition_names) if cc == c
        }

        # Shared adjacency: AND across all conditions
        shared = np.ones((p, p), dtype=int)
        for adj in adjacency_dict.values():
            shared = shared & adj
        np.fill_diagonal(shared, 0)

        # Per-condition unique edges (present in this condition, absent in all others)
        unique_dict = {}
        for c in condition_names:
            others_union = np.zeros((p, p), dtype=int)
            for c2, adj2 in adjacency_dict.items():
                if c2 != c:
                    others_union = others_union | adj2
            unique_dict[c] = adjacency_dict[c] & ~others_union
            np.fill_diagonal(unique_dict[c], 0)

        ebic_selected = self.lambda1 is None or self.lambda2 is None
        self.result_ = MultiConditionGLassoResult(
            precision_=precision_dict,
            adjacency_=adjacency_dict,
            shared_adjacency=shared,
            unique_adjacency=unique_dict,
            reg=self.reg,
            lambda1_=lam1,
            lambda2_=lam2,
            condition_names=condition_names,
            ebic_selected=ebic_selected,
            n_samples_={c: int(n) for c, n in zip(condition_names, ns)},
        )
        return self

    def get_adjacency(self, condition: Optional[str] = None) -> dict | np.ndarray:
        """
        Return binary adjacency matrices.

        Parameters
        ----------
        condition : str or None
            If given, return the (p, p) adjacency for that condition only.
            If None, return a dict of all per-condition adjacencies.

        Returns
        -------
        dict[str, (p, p) ndarray] or (p, p) ndarray
        """
        self._check_fitted()
        if condition is not None:
            if condition not in self.result_.adjacency_:
                raise KeyError(
                    f"Condition '{condition}' not found. "
                    f"Available: {self.result_.condition_names}"
                )
            return self.result_.adjacency_[condition]
        return dict(self.result_.adjacency_)

    def get_shared_adjacency(self) -> np.ndarray:
        """Return the (p, p) adjacency present in ALL conditions."""
        self._check_fitted()
        return self.result_.shared_adjacency

    def to_anndata(
        self,
        adata,
        condition_key: Optional[str] = None,
        key: str = "nodis_mgl",
    ) -> None:
        """Write results into an AnnData object.

        Writes to ``adata.varp`` (gene × gene slots) and ``adata.uns``.

        Parameters
        ----------
        adata         : AnnData — target object; modified in place.
        condition_key : str or None — ``adata.obs`` column used to split
                        conditions.  Required only for shape validation;
                        the fit was already done on ``X_dict`` in ``fit()``.
        key           : str — namespace prefix (default ``'nodis_mgl'``).
        """
        self._check_fitted()
        from scipy.sparse import csr_matrix

        r = self.result_
        p = next(iter(r.precision_.values())).shape[0]
        if hasattr(adata, "n_vars") and adata.n_vars != p:
            raise ValueError(
                f"Precision matrix size p={p} does not match "
                f"adata.n_vars={adata.n_vars}."
            )

        for cond in r.condition_names:
            safe = cond.replace(" ", "_")
            adata.varp[f"{key}_prec_{safe}"]  = csr_matrix(r.precision_[cond])
            adata.varp[f"{key}_adj_{safe}"]   = csr_matrix(r.adjacency_[cond])
            adata.varp[f"{key}_unique_{safe}"] = csr_matrix(r.unique_adjacency[cond])

        adata.varp[f"{key}_shared"] = csr_matrix(r.shared_adjacency)

        adata.uns[key] = {
            "reg": r.reg,
            "lambda1": r.lambda1_,
            "lambda2": r.lambda2_,
            "ebic_selected": r.ebic_selected,
            "conditions": r.condition_names,
            "n_samples": r.n_samples_,
            "n_shared_edges": r.n_shared_edges,
            "edges_per_condition": {
                c: r.n_edges(c) for c in r.condition_names
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_gglasso(self):
        try:
            import gglasso  # noqa: F401
        except ImportError:
            raise ImportError(
                "gglasso is required for MultiConditionGLasso. "
                "Install with: pip install gglasso\n"
                "(or: pip install nodis[gglasso])"
            )

    def _check_fitted(self):
        if self.result_ is None:
            raise RuntimeError("Call fit() before accessing results.")

    def _to_adj(self, prec: np.ndarray) -> np.ndarray:
        """Convert a precision matrix to a binary adjacency (no self-loops)."""
        adj = (np.abs(prec) > self.threshold).astype(int)
        np.fill_diagonal(adj, 0)
        return adj

    def _solve(
        self,
        S: np.ndarray,    # (K, p, p)
        ns: np.ndarray,   # (K,)
        p: int,
    ) -> tuple[float, float, np.ndarray]:
        """Run GGLasso and return (lambda1, lambda2, precision_stack (K,p,p))."""
        from gglasso.problem import glasso_problem

        if self.lambda1 is not None and self.lambda2 is not None:
            # Fixed hyperparameters — single solve
            prob = glasso_problem(
                S, ns,
                reg=self.reg,
                reg_params={"lambda1": self.lambda1, "lambda2": self.lambda2},
            )
            # solve(Omega_0, solver_params, tol, rtol, solver, verbose)
            prob.solve(
                solver_params={"max_iter": self.max_iter},
                tol=self.tol,
                rtol=self.rtol,
                verbose=self.verbose,
            )
            prec = prob.solution.precision_
            return float(self.lambda1), float(self.lambda2), prec

        # eBIC model selection over a 2-D grid
        prob = glasso_problem(S, ns, reg=self.reg)
        prob.set_modelselect_params({
            "lambda1_range": self.lambda1_range,
            "lambda2_range": self.lambda2_range,
        })
        prob.model_selection(
            method="eBIC",
            gamma=self.ebic_gamma,
            tol=self.tol,
            rtol=self.rtol,
        )
        prec = prob.solution.precision_
        selected = prob.reg_params
        lam1 = float(selected.get("lambda1", float(self.lambda1_range.mean())))
        lam2 = float(selected.get("lambda2", float(self.lambda2_range.mean())))
        return lam1, lam2, prec


# ---------------------------------------------------------------------------
# AnnData-native convenience function (tl-style)
# ---------------------------------------------------------------------------

def fit_multi_condition(
    adata,
    condition_key: str,
    reg: str = "GGL",
    lambda1: Optional[float] = None,
    lambda2: Optional[float] = None,
    lambda1_range: Optional[np.ndarray] = None,
    lambda2_range: Optional[np.ndarray] = None,
    ebic_gamma: float = 0.1,
    threshold: float = 0.0,
    layer: Optional[str] = None,
    genes: Optional[list] = None,
    use_hvg: bool = False,
    npn: bool = False,
    standardise: bool = True,
    key: str = "nodis_mgl",
    max_iter: int = 1_000,
    tol: float = 1e-5,
    rtol: float = 1e-4,
    verbose: bool = False,
) -> MultiConditionGLasso:
    """
    Fit a multi-condition GGM directly from an AnnData object.

    Splits ``adata`` by ``adata.obs[condition_key]``, fits a
    ``MultiConditionGLasso`` on each split, and writes results back into
    ``adata.varp`` and ``adata.uns``.

    Parameters
    ----------
    adata         : AnnData
    condition_key : str — ``adata.obs`` column (e.g. ``'cell_type'``,
                    ``'treatment'``, ``'time_point'``)
    reg           : 'GGL' or 'FGL'
    lambda1       : float or None — sparsity penalty (None → eBIC search)
    lambda2       : float or None — group/fusion penalty (None → eBIC search)
    lambda1_range : array-like or None — grid for eBIC over λ₁
    lambda2_range : array-like or None — grid for eBIC over λ₂
    ebic_gamma    : float — eBIC regularisation (default 0.1)
    threshold     : float — adjacency threshold on |precision| (default 0.0)
    layer         : str or None — AnnData layer to extract (None → ``.X``)
    genes         : list[str] or None — subset of genes; None uses all (or HVG)
    use_hvg       : bool — subset to highly variable genes
    npn           : bool — apply NPN shrinkage per condition
    standardise   : bool — standardise columns within each condition
    key           : str — namespace prefix for AnnData output slots
    max_iter      : int — ADMM max iterations
    tol, rtol     : float — ADMM convergence tolerances
    verbose       : bool — print ADMM progress

    Returns
    -------
    est : MultiConditionGLasso — fitted estimator (results also written to adata)

    Examples
    --------
    >>> from nodis.estimators.group_glasso import fit_multi_condition
    >>> est = fit_multi_condition(adata, condition_key='condition',
    ...                          reg='GGL', lambda1=0.1, lambda2=0.05)
    >>> adata.varp['nodis_mgl_shared']        # shared adjacency
    >>> adata.uns['nodis_mgl']['n_shared_edges']
    """
    from nodis.preprocess.anndata_compat import from_anndata

    if condition_key not in adata.obs.columns:
        raise KeyError(
            f"condition_key '{condition_key}' not found in adata.obs.columns. "
            f"Available: {list(adata.obs.columns)}"
        )

    conditions = adata.obs[condition_key].unique()
    if len(conditions) < 2:
        raise ValueError(
            f"condition_key '{condition_key}' has only {len(conditions)} unique "
            "value(s); need at least 2 for multi-condition analysis."
        )

    # Split adata and extract expression matrices
    X_dict = {}
    for cond in conditions:
        mask = (adata.obs[condition_key] == cond).values
        sub = adata[mask]
        X_dict[str(cond)] = from_anndata(
            sub,
            layer=layer,
            genes=genes,
            use_hvg=use_hvg,
            npn=False,           # NPN applied inside MultiConditionGLasso.fit()
        )

    # Warn if any condition has fewer samples than genes
    for cond, X in X_dict.items():
        n, p = X.shape
        if n < p:
            warnings.warn(
                f"Condition '{cond}': n={n} < p={p}. The sample covariance "
                "matrix is singular; GGLasso regularisation will compensate, "
                "but results may be unreliable. Consider using more samples or "
                "fewer genes (use_hvg=True).",
                UserWarning,
                stacklevel=2,
            )

    est = MultiConditionGLasso(
        reg=reg,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda1_range=lambda1_range,
        lambda2_range=lambda2_range,
        ebic_gamma=ebic_gamma,
        threshold=threshold,
        npn=npn,
        standardise=standardise,
        max_iter=max_iter,
        tol=tol,
        rtol=rtol,
        verbose=verbose,
    )
    est.fit(X_dict)
    est.to_anndata(adata, condition_key=condition_key, key=key)
    return est
