"""
Pure Python synthetic GGM data generator — no R dependency.

Generates data from a known sparse precision matrix for four graph topologies,
mirroring ``huge::huge.generator()`` behaviour.  Serves as the primary data
source for the synthetic benchmark (Phases 1 and 2 of the SLURM array jobs).

Topologies
----------
hub        — hub graph: hub nodes each connected to a cluster of spoke nodes
scale-free — Barabási–Albert preferential attachment (networkx)
cluster    — block-diagonal with dense intra-block and zero inter-block edges
random     — Erdős–Rényi: each edge included independently with probability prob

Reference
---------
Zhao T, Liu H (2012). The huge package for high-dimensional undirected graph
    estimation in R.  JMLR 13: 1059–1062.
    https://www.jmlr.org/papers/volume13/zhao12a/zhao12a.pdf

Zhang R, Ren Z, Chen W (2018). SILGGM. PLoS Comput Biol 14(8): e1006369.
    [Simulation study: edge density ~5%, n/p ∈ {2, 3, 5}]
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from scipy.linalg import block_diag


@dataclass
class GeneratedData:
    """Container for a synthetic GGM dataset.

    Attributes
    ----------
    X        : (n, p) ndarray — observed data sampled from N(0, Sigma)
    Theta    : (p, p) ndarray — true precision matrix (positive definite)
    Omega    : (p, p) integer ndarray — true binary adjacency (no self-loops)
    n        : int — number of samples
    p        : int — number of variables (genes)
    topology : str — graph topology label
    seed     : int — random seed used
    """

    X: np.ndarray
    Theta: np.ndarray
    Omega: np.ndarray
    n: int
    p: int
    topology: str
    seed: int


def _precision_from_adjacency(
    A: np.ndarray,
    edge_weight: float = 0.3,
    min_eig: float = 0.1,
) -> np.ndarray:
    """
    Construct a positive-definite precision matrix from a binary adjacency A.

    Off-diagonal entries receive weight ``edge_weight``.  Diagonal is shifted
    to ensure all eigenvalues exceed ``min_eig``.

    Parameters
    ----------
    A           : (p, p) symmetric binary adjacency, no self-loops
    edge_weight : magnitude of off-diagonal precision entries (default 0.3)
    min_eig     : minimum eigenvalue guarantee (default 0.1)
    """
    p = A.shape[0]
    Theta = A.astype(float) * edge_weight
    np.fill_diagonal(Theta, 0.0)
    eigs = np.linalg.eigvalsh(Theta)
    if eigs.min() <= min_eig:
        shift = abs(eigs.min()) + min_eig + 0.1
        Theta += np.eye(p) * shift
    return Theta


def generate(
    n: int,
    p: int,
    topology: str = "hub",
    prob: float = 0.05,
    seed: int = 42,
) -> GeneratedData:
    """
    Generate synthetic GGM data from a sparse precision matrix.

    Parameters
    ----------
    n        : number of samples
    p        : number of variables (genes)
    topology : 'hub' | 'scale-free' | 'cluster' | 'random'
    prob     : target edge density (~0.05 matches SILGGM simulation study)
    seed     : random seed for reproducibility

    Returns
    -------
    GeneratedData with fields X, Theta, Omega, n, p, topology, seed
    """
    rng = np.random.default_rng(seed)

    if topology == "random":
        # Erdős–Rényi: each upper-triangle entry included with prob
        upper = rng.random((p, p)) < prob
        A = np.triu(upper.astype(int), k=1)
        A = A + A.T

    elif topology == "hub":
        # Hub graph: n_hubs hub nodes, each connected to ~p/5 spokes
        A = np.zeros((p, p), dtype=int)
        n_hubs = max(1, p // 20)
        hub_spacing = p // n_hubs
        for h in range(n_hubs):
            hub = h * hub_spacing
            candidates = [k for k in range(p) if k != hub]
            n_spokes = max(1, p // 5)
            spokes = rng.choice(candidates, size=min(n_spokes, len(candidates)), replace=False)
            for s in spokes:
                A[hub, s] = A[s, hub] = 1

    elif topology == "scale-free":
        # Barabási–Albert: m edges per new node; m chosen to approximate prob
        m = max(1, int(round(prob * p)))
        G = nx.barabasi_albert_graph(p, m=m, seed=seed)
        A = nx.to_numpy_array(G, dtype=int)

    elif topology == "cluster":
        # Block-diagonal: dense intra-cluster (~40% density), zero inter-cluster
        cluster_size = max(3, p // 5)
        n_clusters = p // cluster_size
        blocks = []
        for _ in range(n_clusters):
            cs = cluster_size
            B = rng.random((cs, cs)) < 0.4
            B = np.triu(B.astype(int), k=1)
            B = B + B.T
            blocks.append(B)
        remainder = p - n_clusters * cluster_size
        if remainder > 0:
            blocks.append(np.zeros((remainder, remainder), dtype=int))
        A = block_diag(*blocks).astype(int)

    else:
        raise ValueError(
            f"Unknown topology '{topology}'. "
            "Choose from: 'hub', 'scale-free', 'cluster', 'random'."
        )

    np.fill_diagonal(A, 0)
    Theta = _precision_from_adjacency(A)
    Sigma = np.linalg.inv(Theta)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)

    return GeneratedData(X=X, Theta=Theta, Omega=A, n=n, p=p, topology=topology, seed=seed)
