"""
Network topology analysis for GGMs: community detection, hub identification,
and network backbone extraction.

Community detection
-------------------
Three algorithm families are supported with automatic backend selection:

  Leiden    — best-in-class modularity optimisation; requires ``leidenalg`` +
               ``igraph`` (``pip install leidenalg igraph``).  Falls back to
               Louvain when unavailable.
  Louvain   — Blondel et al. (2008); available via ``python-louvain``
               (``pip install python-louvain``) or via networkx ≥ 3.0.
  Greedy    — Clauset–Newman–Moore greedy modularity; always available via
               networkx (no optional deps).
  Spectral  — k-means on Laplacian eigenvectors; always available via scipy /
               sklearn.

Hub gene identification
-----------------------
Computes degree / betweenness / eigenvector centrality for each node and
tests significance against a permutation null (random edge rewiring that
preserves node degrees).

Backbone extraction
-------------------
Two methods:

  disparity — Serrano et al. (2009) disparity filter; retains edges that are
               statistically significant relative to the node's total strength.
  threshold — simple edge-weight threshold (falls back when precision matrix is
               not available or graph is unweighted).

AnnData integration
-------------------
``write_anndata_network`` writes topology results into the AnnData object:
  adata.uns["nodis_communities"]   — dict with labels, modularity, algorithm
  adata.uns["nodis_hubs"]          — DataFrame of hub gene scores
  adata.obsp["nodis_adjacency"]    — sparse binary adjacency (if provided)
  adata.obsp["nodis_backbone"]     — sparse backbone adjacency

References
----------
Blondel VD et al. (2008). Fast unfolding of communities in large networks.
    J Stat Mech 2008: P10008. doi:10.1088/1742-5468/2008/10/P10008

Traag VA, Waltman L, van Eck NJ (2019). From Louvain to Leiden: guaranteeing
    well-connected communities. Sci Rep 9: 5233. doi:10.1038/s41598-019-41695-z

Serrano MA, Boguñá M, Vespignani A (2009). Extracting the multiscale backbone
    of complex weighted networks. PNAS 106(16): 6483–6488.
    doi:10.1073/pnas.0808904106

Newman MEJ (2006). Finding community structure in networks using the
    eigenvectors of matrices. Phys Rev E 74: 036104.
    doi:10.1103/PhysRevE.74.036104
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CommunityResult:
    """Results of community detection on a GGM adjacency matrix.

    Attributes
    ----------
    labels       : (p,) int array — community index per node (0-based)
    modularity   : float — modularity Q of the partition
    n_communities: int — number of detected communities
    algorithm    : str — name of the algorithm used
    gene_names   : list[str] or None — gene labels (same order as labels)
    communities  : list[set] — each set contains node indices in that community
    """
    labels: np.ndarray
    modularity: float
    n_communities: int
    algorithm: str
    gene_names: Optional[list] = None
    communities: list = field(default_factory=list)

    def as_dataframe(self) -> pd.DataFrame:
        """Return community assignments as a DataFrame."""
        names = self.gene_names if self.gene_names is not None else [
            str(i) for i in range(len(self.labels))
        ]
        return pd.DataFrame({"gene": names, "community": self.labels})


@dataclass
class HubResult:
    """Results of hub gene identification.

    Attributes
    ----------
    scores       : pd.DataFrame — columns: gene, degree, betweenness,
                   eigenvector, hub_score, p_value, significant
    n_hubs       : int — number of significant hub genes at the given alpha
    metric       : str — centrality metric used for ranking
    alpha        : float — significance threshold used
    n_permutations: int — number of permutations used for the null distribution
    """
    scores: pd.DataFrame
    n_hubs: int
    metric: str
    alpha: float
    n_permutations: int


# ---------------------------------------------------------------------------
# Helper: adjacency matrix → networkx Graph
# ---------------------------------------------------------------------------

def to_networkx(
    adjacency: np.ndarray,
    weights: Optional[np.ndarray] = None,
    gene_names: Optional[list] = None,
) -> nx.Graph:
    """Convert a (p, p) adjacency matrix to a networkx Graph.

    Parameters
    ----------
    adjacency  : (p, p) binary or weighted ndarray — symmetric adjacency
    weights    : (p, p) ndarray or None — edge weights (e.g. |precision|);
                 when None, edges are unweighted
    gene_names : list of str or None — node labels; integer indices used if None

    Returns
    -------
    G : networkx.Graph
    """
    p = adjacency.shape[0]
    adj = np.asarray(adjacency)
    G = nx.Graph()

    node_ids = gene_names if gene_names is not None else list(range(p))
    G.add_nodes_from(node_ids)

    rows, cols = np.where(np.triu(adj, k=1) != 0)
    for r, c in zip(rows, cols):
        w = float(np.abs(weights[r, c])) if weights is not None else 1.0
        G.add_edge(node_ids[r], node_ids[c], weight=w)

    return G


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NetworkTopology:
    """Community detection, hub identification, and backbone extraction.

    Parameters
    ----------
    adjacency  : (p, p) binary ndarray — symmetric adjacency matrix
    gene_names : list of str or None — gene labels; integer indices if None
    weights    : (p, p) ndarray or None — edge weights for weighted analysis
                 (e.g. absolute precision entries |Omega_hat|)
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        gene_names: Optional[list] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        adj = np.asarray(adjacency, dtype=float)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(
                f"adjacency must be square 2-D; got shape {adj.shape}."
            )
        p = adj.shape[0]
        if gene_names is not None and len(gene_names) != p:
            raise ValueError(
                f"gene_names length {len(gene_names)} != adjacency size {p}."
            )

        # Force symmetry and zero diagonal
        adj = (adj + adj.T) / 2.0
        np.fill_diagonal(adj, 0.0)
        self._adj = adj
        self._p = p
        self._gene_names = gene_names
        self._weights = (
            np.asarray(weights, dtype=float) if weights is not None else None
        )
        self._G: Optional[nx.Graph] = None  # lazy-built

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _graph(self) -> nx.Graph:
        if self._G is None:
            self._G = to_networkx(
                self._adj, weights=self._weights, gene_names=self._gene_names
            )
        return self._G

    def _node_list(self) -> list:
        return (
            self._gene_names if self._gene_names is not None
            else list(range(self._p))
        )

    @staticmethod
    def _leiden_available() -> bool:
        try:
            import leidenalg  # noqa: F401
            import igraph      # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _louvain_available() -> bool:
        """Check for python-louvain (community package) or networkx Louvain."""
        try:
            import community  # python-louvain  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------

    def detect_communities(
        self,
        algorithm: str = "leiden",
        resolution: float = 1.0,
        seed: int = 42,
        n_clusters: int = 5,
    ) -> CommunityResult:
        """Detect communities in the inferred GGM.

        Parameters
        ----------
        algorithm  : str
            'leiden'   — Traag et al. 2019 (requires leidenalg + igraph)
            'louvain'  — Blondel et al. 2008 (requires python-louvain or
                         networkx ≥ 3.0)
            'greedy'   — Clauset-Newman-Moore; always available via networkx
            'spectral' — k-means on Laplacian eigenvectors; requires sklearn
        resolution : float, default 1.0
            Resolution parameter for Leiden/Louvain (higher → more, smaller
            communities).  Ignored for 'greedy' and 'spectral'.
        seed       : int, default 42
            Random seed for reproducibility (Leiden/Louvain/spectral).
        n_clusters : int, default 5
            Target number of clusters for 'spectral' (ignored otherwise).

        Returns
        -------
        CommunityResult
        """
        algo = algorithm.lower()
        if algo == "leiden":
            if self._leiden_available():
                return self._detect_leiden(resolution=resolution, seed=seed)
            warnings.warn(
                "leidenalg or igraph not installed; falling back to Louvain. "
                "Install with: pip install leidenalg igraph",
                ImportWarning,
                stacklevel=2,
            )
            algo = "louvain"

        if algo == "louvain":
            return self._detect_louvain(resolution=resolution, seed=seed)

        if algo == "greedy":
            return self._detect_greedy()

        if algo == "spectral":
            return self._detect_spectral(n_clusters=n_clusters, seed=seed)

        raise ValueError(
            f"algorithm must be 'leiden', 'louvain', 'greedy', or 'spectral'; "
            f"got '{algorithm}'."
        )

    def _detect_leiden(self, resolution: float, seed: int) -> CommunityResult:
        import igraph as ig
        import leidenalg

        G_nx = self._graph()
        nodes = list(G_nx.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}

        ig_edges = [
            (node_idx[u], node_idx[v])
            for u, v in G_nx.edges()
        ]
        weights = [
            G_nx[u][v].get("weight", 1.0) for u, v in G_nx.edges()
        ]
        ig_G = ig.Graph(n=len(nodes), edges=ig_edges, edge_attrs={"weight": weights})

        partition = leidenalg.find_partition(
            ig_G,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed,
        )
        labels = np.array(partition.membership)
        modularity = float(partition.modularity)
        communities = [set(c) for c in partition]

        return CommunityResult(
            labels=labels,
            modularity=modularity,
            n_communities=len(communities),
            algorithm="leiden",
            gene_names=self._gene_names,
            communities=communities,
        )

    def _detect_louvain(self, resolution: float, seed: int) -> CommunityResult:
        """Louvain via python-louvain (preferred) or networkx fallback."""
        G = self._graph()

        try:
            import community as comm_pkg  # python-louvain
            part = comm_pkg.best_partition(G, weight="weight",
                                           resolution=resolution, random_state=seed)
            labels = np.array([part[n] for n in G.nodes()])
            modularity = float(comm_pkg.modularity(part, G, weight="weight"))
            n_comm = int(labels.max()) + 1
            communities = [
                {i for i, l in enumerate(labels) if l == c}
                for c in range(n_comm)
            ]
            return CommunityResult(
                labels=labels,
                modularity=modularity,
                n_communities=n_comm,
                algorithm="louvain",
                gene_names=self._gene_names,
                communities=communities,
            )
        except ImportError:
            pass

        # networkx Louvain (nx >= 3.0)
        rng = np.random.default_rng(seed)
        py_seed = int(rng.integers(0, 2**31))
        comms = nx.community.louvain_communities(G, weight="weight",
                                                 resolution=resolution, seed=py_seed)
        node_list = self._node_list()
        node_idx = {n: i for i, n in enumerate(node_list)}
        labels = np.zeros(self._p, dtype=int)
        for c_idx, comm in enumerate(comms):
            for node in comm:
                labels[node_idx[node]] = c_idx
        modularity = float(nx.community.modularity(G, comms, weight="weight"))
        communities = [{node_idx[n] for n in c} for c in comms]
        return CommunityResult(
            labels=labels,
            modularity=modularity,
            n_communities=len(comms),
            algorithm="louvain_nx",
            gene_names=self._gene_names,
            communities=communities,
        )

    def _detect_greedy(self) -> CommunityResult:
        G = self._graph()
        comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        node_list = self._node_list()
        node_idx = {n: i for i, n in enumerate(node_list)}
        labels = np.zeros(self._p, dtype=int)
        for c_idx, comm in enumerate(comms):
            for node in comm:
                labels[node_idx[node]] = c_idx
        modularity = float(nx.community.modularity(G, comms, weight="weight"))
        communities = [{node_idx[n] for n in c} for c in comms]
        return CommunityResult(
            labels=labels,
            modularity=modularity,
            n_communities=len(comms),
            algorithm="greedy",
            gene_names=self._gene_names,
            communities=communities,
        )

    def _detect_spectral(self, n_clusters: int, seed: int) -> CommunityResult:
        from sklearn.cluster import SpectralClustering

        # Affinity = weighted adjacency (or binary if unweighted)
        A = (
            np.abs(self._weights) * (self._adj > 0)
            if self._weights is not None
            else self._adj
        )
        # Symmetrise and ensure non-negative
        A = np.maximum(A, 0.0)

        n_clust = min(n_clusters, self._p)
        sc = SpectralClustering(
            n_clusters=n_clust,
            affinity="precomputed",
            random_state=seed,
            assign_labels="kmeans",
        )
        labels = sc.fit_predict(A)
        G = self._graph()
        node_list = self._node_list()
        node_idx = {n: i for i, n in enumerate(node_list)}
        comms = [
            {node_idx[n] for n in G.nodes() if labels[node_idx[n]] == c}
            for c in range(n_clust)
        ]
        comms_nx = [
            {node_list[i] for i in range(self._p) if labels[i] == c}
            for c in range(n_clust)
        ]
        try:
            modularity = float(nx.community.modularity(G, comms_nx, weight="weight"))
        except Exception:
            modularity = float("nan")
        return CommunityResult(
            labels=labels,
            modularity=modularity,
            n_communities=n_clust,
            algorithm="spectral",
            gene_names=self._gene_names,
            communities=comms,
        )

    # ------------------------------------------------------------------
    # Hub gene identification
    # ------------------------------------------------------------------

    def hub_genes(
        self,
        top_n: int = 10,
        metric: str = "degree",
        n_permutations: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> HubResult:
        """Identify hub genes with permutation significance testing.

        Parameters
        ----------
        top_n         : int — number of top hubs to flag (in addition to
                        permutation-based significance)
        metric        : str
            'degree'      — degree centrality (fraction of possible connections)
            'betweenness' — betweenness centrality (fraction of shortest paths)
            'eigenvector' — eigenvector centrality (recursive neighbour influence)
            'strength'    — weighted degree (sum of incident edge weights);
                            falls back to degree if graph is unweighted
        n_permutations: int — number of random edge rewirings for null dist
        alpha         : float — significance threshold (empirical p-value)
        seed          : int — random seed for permutation

        Returns
        -------
        HubResult with a DataFrame containing all genes sorted by hub_score
        """
        G = self._graph()
        if G.number_of_edges() == 0:
            warnings.warn(
                "Graph has no edges; hub identification is not meaningful.",
                UserWarning, stacklevel=2,
            )
            empty = pd.DataFrame(
                {"gene": self._node_list(),
                 "degree": 0.0, "betweenness": 0.0,
                 "eigenvector": 0.0, "strength": 0.0,
                 "hub_score": 0.0, "p_value": 1.0,
                 "significant": False}
            )
            return HubResult(scores=empty, n_hubs=0, metric=metric,
                             alpha=alpha, n_permutations=n_permutations)

        node_list = list(G.nodes())
        scores_obs = self._centrality_scores(G, node_list)

        # Permutation null: randomly rewire edges, preserving degree sequence
        rng = np.random.default_rng(seed)
        null_scores = {m: [] for m in ("degree", "betweenness", "eigenvector", "strength")}
        for _ in range(n_permutations):
            G_null = self._rewire(G, rng)
            s = self._centrality_scores(G_null, node_list)
            for m in null_scores:
                null_scores[m].append(s[m])

        null_arr = {m: np.array(null_scores[m]) for m in null_scores}

        # Empirical p-values: fraction of permutations where null score >= observed
        metric_scores = scores_obs[metric]
        pvals = np.zeros(self._p)
        for i in range(self._p):
            pvals[i] = float(np.mean(null_arr[metric][:, i] >= metric_scores[i]))

        df = pd.DataFrame({
            "gene": node_list,
            "degree": scores_obs["degree"],
            "betweenness": scores_obs["betweenness"],
            "eigenvector": scores_obs["eigenvector"],
            "strength": scores_obs["strength"],
            "hub_score": metric_scores,
            "p_value": pvals,
            "significant": pvals < alpha,
        }).sort_values("hub_score", ascending=False).reset_index(drop=True)

        n_hubs = int(df["significant"].sum())
        return HubResult(
            scores=df,
            n_hubs=n_hubs,
            metric=metric,
            alpha=alpha,
            n_permutations=n_permutations,
        )

    @staticmethod
    def _centrality_scores(G: nx.Graph, node_list: list) -> dict:
        """Compute centrality metrics for all nodes, returned as (p,) arrays."""
        deg = nx.degree_centrality(G)
        try:
            bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
        except Exception:
            bet = {n: 0.0 for n in G.nodes()}
        try:
            eig = nx.eigenvector_centrality(G, weight="weight", max_iter=300, tol=1e-6)
        except Exception:
            eig = {n: 0.0 for n in G.nodes()}
        strength = dict(G.degree(weight="weight"))
        # Normalise strength to [0, 1]
        max_s = max(strength.values()) if strength else 1.0
        if max_s == 0:
            max_s = 1.0
        strength_norm = {n: v / max_s for n, v in strength.items()}

        return {
            "degree":      np.array([deg.get(n, 0.0) for n in node_list]),
            "betweenness": np.array([bet.get(n, 0.0) for n in node_list]),
            "eigenvector": np.array([eig.get(n, 0.0) for n in node_list]),
            "strength":    np.array([strength_norm.get(n, 0.0) for n in node_list]),
        }

    @staticmethod
    def _rewire(G: nx.Graph, rng: np.random.Generator) -> nx.Graph:
        """Random edge rewiring that preserves the degree sequence (Maslov-Sneppen).

        Performs up to Q × |E| double-edge swap attempts.  Dense graphs may
        not achieve all requested swaps; the function returns what it managed.
        """
        G_r = G.copy()
        n_edges = G_r.number_of_edges()
        if n_edges < 2:
            return G_r
        # Attempt nswap = max(n_edges, 10) swaps with generous max_tries.
        # Catch both NetworkXError and NetworkXAlgorithmError (dense graphs
        # exhaust swap attempts before reaching the target count).
        n_swaps = max(n_edges, 10)
        seed_val = int(rng.integers(0, 2**31))
        try:
            nx.double_edge_swap(G_r, nswap=n_swaps,
                                max_tries=n_swaps * 200,
                                seed=seed_val)
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            pass
        return G_r

    # ------------------------------------------------------------------
    # Network backbone extraction
    # ------------------------------------------------------------------

    def backbone(
        self,
        method: str = "disparity",
        alpha: float = 0.05,
        threshold: float = 0.1,
    ) -> np.ndarray:
        """Extract the network backbone.

        Parameters
        ----------
        method    : str
            'disparity' — Serrano et al. (2009) statistical disparity filter;
                          retains edge (i,j) if it is significant in EITHER
                          node i's or node j's local weight distribution.
                          Requires a weighted graph (``weights`` provided at
                          construction); falls back to 'threshold' if unweighted.
            'threshold' — keep edges with |weight| > ``threshold``; or if no
                          weights are provided, return the original adjacency.
        alpha     : float, default 0.05
            Significance level for the disparity filter.
        threshold : float, default 0.1
            Edge weight cut-off for the 'threshold' method (or disparity
            fallback).

        Returns
        -------
        backbone_adj : (p, p) binary ndarray — symmetric backbone adjacency
        """
        method = method.lower()
        if method not in ("disparity", "threshold"):
            raise ValueError(
                f"method must be 'disparity' or 'threshold'; got '{method}'."
            )

        if method == "disparity":
            if self._weights is None:
                warnings.warn(
                    "No weights provided; disparity filter requires edge weights. "
                    "Falling back to threshold method.",
                    UserWarning, stacklevel=2,
                )
                return self.backbone(method="threshold", threshold=threshold)
            return self._disparity_filter(alpha)

        # threshold method
        if self._weights is not None:
            W = np.abs(self._weights) * (self._adj > 0)
            backbone_adj = (W > threshold).astype(int)
        else:
            backbone_adj = self._adj.astype(int).copy()
        np.fill_diagonal(backbone_adj, 0)
        return backbone_adj

    def _disparity_filter(self, alpha: float) -> np.ndarray:
        """Serrano et al. (2009) disparity filter.

        For each node i with degree k_i and total strength s_i, the weight
        w_ij of edge (i,j) is significant if:

            (1 − w_ij / s_i)^{k_i − 1}  <  alpha

        Retain edge if significant from EITHER endpoint.
        """
        W = np.abs(self._weights) * (self._adj > 0)
        strength = W.sum(axis=1)  # s_i for each node i
        degree = (self._adj > 0).sum(axis=1)  # k_i

        p_val = np.ones((self._p, self._p))
        for i in range(self._p):
            si = strength[i]
            ki = int(degree[i])
            if si == 0 or ki <= 1:
                continue
            for j in np.where(self._adj[i] > 0)[0]:
                pij = (1.0 - W[i, j] / si) ** (ki - 1)
                p_val[i, j] = pij

        # Edge is significant if it passes from EITHER direction
        sig = (p_val < alpha) | (p_val.T < alpha)
        backbone_adj = (sig & (self._adj > 0)).astype(int)
        np.fill_diagonal(backbone_adj, 0)
        return backbone_adj

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return basic graph statistics."""
        G = self._graph()
        if G.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "density": 0.0,
                    "mean_degree": 0.0, "n_components": 0}
        comps = list(nx.connected_components(G))
        degrees = [d for _, d in G.degree()]
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
            "n_components": len(comps),
            "largest_component_fraction": (
                max(len(c) for c in comps) / G.number_of_nodes()
                if comps else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# AnnData integration
# ---------------------------------------------------------------------------

def write_anndata_network(
    adata,
    adjacency: np.ndarray,
    communities: Optional[CommunityResult] = None,
    hubs: Optional[HubResult] = None,
    backbone_adj: Optional[np.ndarray] = None,
    precision: Optional[np.ndarray] = None,
) -> None:
    """Write network topology results into an AnnData object.

    Slots used
    ----------
    adata.varp["nodis_adjacency"]  — sparse binary adjacency (p × p)
    adata.varp["nodis_backbone"]   — sparse backbone adjacency (p × p)
    adata.varp["nodis_precision"]  — sparse precision matrix (p × p)
    adata.var["nodis_community"]   — community label per gene (int)
    adata.var["nodis_hub_score"]   — hub centrality score per gene (float)
    adata.var["nodis_hub_pvalue"]  — hub p-value per gene (float)
    adata.var["nodis_hub"]         — True/False hub flag (bool)
    adata.uns["nodis_communities"] — dict with modularity, n_communities, algorithm
    adata.uns["nodis_hubs"]        — DataFrame of hub results

    ``varp`` (variable × variable) is the correct AnnData slot for gene-gene
    precision / adjacency matrices, analogous to ``obsp`` for cell-cell graphs.

    Parameters
    ----------
    adata       : anndata.AnnData
    adjacency   : (p, p) binary adjacency
    communities : CommunityResult or None
    hubs        : HubResult or None
    backbone_adj: (p, p) binary backbone adjacency or None
    precision   : (p, p) precision matrix or None
    """
    from scipy.sparse import csr_matrix

    p = adjacency.shape[0]
    if hasattr(adata, "n_vars") and adata.n_vars != p:
        raise ValueError(
            f"adjacency size {p} does not match adata.n_vars={adata.n_vars}."
        )

    adata.varp["nodis_adjacency"] = csr_matrix(adjacency)

    if backbone_adj is not None:
        adata.varp["nodis_backbone"] = csr_matrix(backbone_adj)

    if precision is not None:
        adata.varp["nodis_precision"] = csr_matrix(precision)

    if communities is not None:
        adata.var["nodis_community"] = communities.labels
        adata.uns["nodis_communities"] = {
            "modularity": communities.modularity,
            "n_communities": communities.n_communities,
            "algorithm": communities.algorithm,
        }

    if hubs is not None:
        # Map hub scores back to adata.var order (gene_names may differ from index)
        gene_to_score = dict(zip(hubs.scores["gene"], hubs.scores["hub_score"]))
        gene_to_pval  = dict(zip(hubs.scores["gene"], hubs.scores["p_value"]))
        gene_to_hub   = dict(zip(hubs.scores["gene"], hubs.scores["significant"]))

        var_names = list(adata.var_names)
        adata.var["nodis_hub_score"]  = [gene_to_score.get(g, float("nan"))
                                          for g in var_names]
        adata.var["nodis_hub_pvalue"] = [gene_to_pval.get(g, float("nan"))
                                          for g in var_names]
        adata.var["nodis_hub"]        = [bool(gene_to_hub.get(g, False))
                                          for g in var_names]
        adata.uns["nodis_hubs"] = hubs.scores
