"""
Diffusion fidelity and perturbative knockout evaluation for GGM benchmarking.

Two complementary evaluations measure whether an inferred graph supports the
same downstream functional analyses as the ground-truth graph:

  1. Diffusion fidelity: does heat diffusion on the inferred graph reproduce
     node-level signal dynamics observed on the ground-truth graph?

  2. Knockout analysis: does the inferred graph identify the same critical genes
     (those whose perturbation most disrupts signal propagation)?

Mathematical framework
----------------------
Heat diffusion PDE:
    ∂S/∂t = −L S(t),   S(0) = δ
    Solution: S(t) = exp(−t·L) · δ

Implemented via eigendecomposition L = V Λ Vᵀ (computed once per graph):
    v = Vᵀ · δ
    S(t) = V · (exp(−t·λ) ⊙ v)
Complexity: O(p³ + T·p²) vs O(T·p³) for naïve per-timepoint expm.

References
----------
Kondor RI, Lafferty J (2002). Diffusion kernels on graphs and other discrete
input spaces. ICML 2002.
"""
import warnings

import numpy as np
from scipy.stats import spearmanr


def erdos_renyi_matching(adj: np.ndarray, seed: int) -> np.ndarray:
    """
    Return a random Erdős–Rényi graph with the same number of edges as ``adj``.

    Used as a null baseline for diffusion/knockout metrics: a method whose
    inferred graph performs no better than a random graph with identical
    density provides no functional information.

    Parameters
    ----------
    adj  : (p, p) binary symmetric adjacency (no self-loops)
    seed : RNG seed for reproducibility

    Returns
    -------
    (p, p) binary symmetric adjacency with the same edge count as ``adj``
    """
    p = adj.shape[0]
    n_edges = int(adj.sum()) // 2
    rng = np.random.default_rng(seed)

    # All possible upper-triangle positions
    rows, cols = np.triu_indices(p, k=1)
    chosen = rng.choice(len(rows), size=min(n_edges, len(rows)), replace=False)

    adj_rand = np.zeros((p, p), dtype=int)
    adj_rand[rows[chosen], cols[chosen]] = 1
    adj_rand[cols[chosen], rows[chosen]] = 1
    return adj_rand


def make_laplacian(adj: np.ndarray, normalised: bool = False) -> np.ndarray:
    """
    Compute the graph Laplacian from a binary (or weighted) adjacency matrix.

    Parameters
    ----------
    adj        : (p, p) adjacency matrix (symmetric, no self-loops)
    normalised : if True, returns the symmetric normalised Laplacian
                 L_norm = D^{-1/2} (D − A) D^{-1/2}

    Returns
    -------
    (p, p) Laplacian matrix (symmetric positive-semidefinite)

    Notes
    -----
    n_components_pred is always derived from the unnormalised Laplacian regardless
    of the `normalised` flag, because connected components correspond to zero
    eigenvalues of (D − A) only.
    """
    A = adj.astype(float)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    if not normalised:
        return L
    with np.errstate(divide='ignore', invalid='ignore'):
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ L @ D_inv_sqrt


def make_delta(
    adj_true: np.ndarray,
    L_true: np.ndarray,
    mode: str,
    seed: int,
) -> np.ndarray:
    """
    Construct an initial signal vector for diffusion experiments.

    All vectors are constructed from the **true** graph to ensure a consistent
    reference signal across estimators.

    Parameters
    ----------
    adj_true : (p, p) ground-truth binary adjacency
    L_true   : (p, p) unnormalised Laplacian of adj_true
    mode     : 'random' | 'hub' | 'fiedler'
               random  — unit-norm Gaussian, seeded by `seed`
               hub     — one-hot on the highest-degree node
               fiedler — Fiedler vector (second eigenvector of L_true);
                         falls back to random if the graph is disconnected
    seed     : RNG seed (used for 'random' mode and Fiedler fallback)

    Returns
    -------
    (p,) unit-norm vector
    """
    p = adj_true.shape[0]

    if mode == 'random':
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(p)
        return v / np.linalg.norm(v)

    if mode == 'hub':
        hub_idx = int(np.argmax(adj_true.sum(axis=1)))
        v = np.zeros(p)
        v[hub_idx] = 1.0
        return v

    if mode == 'fiedler':
        eigenvalues = np.linalg.eigvalsh(L_true)
        n_zero = int((eigenvalues < 1e-10).sum())
        if n_zero > 1:
            warnings.warn(
                f"True graph has {n_zero} connected components — Fiedler vector "
                "is undefined. Falling back to random delta.",
                stacklevel=2,
            )
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(p)
            return v / np.linalg.norm(v)
        # eigvalsh returns eigenvalues ascending; Fiedler vector is column index 1
        fiedler_vec = np.linalg.eigh(L_true)[1][:, 1]
        norm = np.linalg.norm(fiedler_vec)
        return fiedler_vec / norm if norm > 0 else fiedler_vec

    raise ValueError(f"Unknown delta mode: {mode!r}. Choose 'random', 'hub', or 'fiedler'.")


def diffuse(
    L: np.ndarray,
    delta: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute heat diffusion S(t) = exp(−t·L) · delta for each t in t_grid.

    Parameters
    ----------
    L      : (p, p) Laplacian matrix
    delta  : (p,) unit-norm initial signal vector
    t_grid : (T,) array of timepoints

    Returns
    -------
    (p, T) signal matrix — S[:, i] is the node-signal vector at time t_grid[i]
    """
    eigenvalues, V = np.linalg.eigh(L)     # L = V Λ Vᵀ
    v = V.T @ delta                          # project into eigenbasis: (p,)
    exp_decay = np.exp(-np.outer(eigenvalues, t_grid))  # (p, T)
    return V @ (exp_decay * v[:, np.newaxis])            # (p, T)


def evaluate_diffusion(
    adj_pred: np.ndarray,
    adj_true: np.ndarray,
    delta: np.ndarray,
    t_grid: np.ndarray,
    normalised: bool = False,
) -> dict:
    """
    Quantify how well diffusion on the inferred graph replicates diffusion
    on the ground-truth graph.

    Parameters
    ----------
    adj_pred   : (p, p) inferred binary adjacency
    adj_true   : (p, p) ground-truth binary adjacency
    delta      : (p,) unit-norm initial signal (same for both graphs)
    t_grid     : (T,) timepoints
    normalised : use normalised Laplacian for the diffusion kernel

    Returns
    -------
    dict
        diffusion_spearman_mean  mean Spearman ρ(S_true[:,t], S_pred[:,t]) over T
        diffusion_spearman_min   worst-case Spearman ρ over T
        diffusion_mae_mean       mean MAE per node averaged over T
        n_components_pred        connected components of adj_pred (unnormalised L)
    """
    L_true = make_laplacian(adj_true, normalised)
    L_pred = make_laplacian(adj_pred, normalised)

    S_true = diffuse(L_true, delta, t_grid)   # (p, T)
    S_pred = diffuse(L_pred, delta, t_grid)   # (p, T)

    T = t_grid.shape[0]
    spearman_vals = np.empty(T)
    mae_vals      = np.empty(T)

    for t in range(T):
        rho, _ = spearmanr(S_true[:, t], S_pred[:, t])
        spearman_vals[t] = float(rho) if np.isfinite(rho) else 0.0
        mae_vals[t]      = float(np.mean(np.abs(S_true[:, t] - S_pred[:, t])))

    # n_components always from unnormalised Laplacian
    if normalised:
        L_pred_unnorm = make_laplacian(adj_pred, normalised=False)
        eigs_unnorm   = np.linalg.eigvalsh(L_pred_unnorm)
    else:
        eigs_unnorm = np.linalg.eigvalsh(L_pred)
    n_components = int((eigs_unnorm < 1e-10).sum())

    return {
        "diffusion_spearman_mean": float(spearman_vals.mean()),
        "diffusion_spearman_min":  float(spearman_vals.min()),
        "diffusion_mae_mean":      float(mae_vals.mean()),
        "n_components_pred":       n_components,
    }


def evaluate_knockouts(
    adj_pred: np.ndarray,
    adj_true: np.ndarray,
    delta: np.ndarray,
    t_grid: np.ndarray,
    reduction: float = 0.3,
    topk: int = 10,
) -> dict:
    """
    Evaluate whether the inferred graph identifies the same critical genes
    as the ground-truth graph under perturbative edge-weight reduction.

    For gene i, all edges incident to i are scaled by `reduction`, producing
    a weighted Laplacian perturbation. The perturbation is applied identically
    to adj_true and adj_pred. Impact is measured as the maximum L2 norm of the
    signal difference across timepoints.

    Parameters
    ----------
    adj_pred  : (p, p) inferred binary adjacency
    adj_true  : (p, p) ground-truth binary adjacency
    delta     : (p,) unit-norm initial signal vector
    t_grid    : (T,) timepoints
    reduction : edge-weight multiplier for the perturbed gene (default 0.3;
                retains 30% of original edge weight)
    topk      : target top-K; clipped to k_eff = max(1, min(topk, p // 5))
                — prevents K approaching p. Examples: p=50 → k_eff=10,
                  p=78 → k_eff=15, p=164 → k_eff=32.

    Returns
    -------
    dict
        knockout_spearman      Spearman ρ of impact_true vs impact_pred (primary)
        knockout_top10_recall  |top10_true ∩ top10_pred| / 10
        knockout_topk_recall   |topK_true ∩ topK_pred| / k_eff
    """
    p = adj_pred.shape[0]
    k_eff = max(1, min(topk, p // 5))
    top10 = min(10, p)

    def _gene_impacts(adj: np.ndarray) -> np.ndarray:
        A_base = adj.astype(float)
        L_base = make_laplacian(A_base)
        S_base = diffuse(L_base, delta, t_grid)   # (p, T)
        impacts = np.zeros(p)
        for i in range(p):
            A_pert = A_base.copy()
            A_pert[i, :] *= reduction
            A_pert[:, i] *= reduction
            L_pert    = make_laplacian(A_pert)
            S_pert    = diffuse(L_pert, delta, t_grid)
            diff_norms = np.linalg.norm(S_pert - S_base, axis=0)  # (T,)
            impacts[i] = float(diff_norms.max())
        return impacts

    impact_true = _gene_impacts(adj_true)
    impact_pred = _gene_impacts(adj_pred)

    rho, _ = spearmanr(impact_true, impact_pred)
    knockout_spearman = float(rho) if np.isfinite(rho) else 0.0

    top10_true = set(np.argsort(impact_true)[-top10:])
    top10_pred = set(np.argsort(impact_pred)[-top10:])
    knockout_top10_recall = len(top10_true & top10_pred) / top10

    topk_true = set(np.argsort(impact_true)[-k_eff:])
    topk_pred = set(np.argsort(impact_pred)[-k_eff:])
    knockout_topk_recall = len(topk_true & topk_pred) / k_eff

    return {
        "knockout_spearman":     knockout_spearman,
        "knockout_top10_recall": float(knockout_top10_recall),
        "knockout_topk_recall":  float(knockout_topk_recall),
    }
