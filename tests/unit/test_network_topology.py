"""
Unit tests for nodis.network.topology.

Covers:
  - to_networkx
  - NetworkTopology.detect_communities (greedy, spectral; louvain_nx)
  - NetworkTopology.hub_genes
  - NetworkTopology.backbone (disparity, threshold)
  - NetworkTopology.summary
  - write_anndata_network
  - CommunityResult.as_dataframe
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_adj():
    """5-node ring graph adjacency."""
    A = np.zeros((5, 5), dtype=int)
    for i in range(5):
        A[i, (i + 1) % 5] = 1
        A[(i + 1) % 5, i] = 1
    return A


@pytest.fixture
def small_weights(small_adj):
    """Simple weights: uniform 0.5 on all edges."""
    return small_adj.astype(float) * 0.5


@pytest.fixture
def cluster_adj():
    """15-node two-cluster adjacency (clique-like within, sparse across)."""
    p = 15
    A = np.zeros((p, p), dtype=int)
    # cluster 0: nodes 0–6 (dense)
    for i in range(7):
        for j in range(i + 1, 7):
            A[i, j] = A[j, i] = 1
    # cluster 1: nodes 7–14 (dense)
    for i in range(7, 15):
        for j in range(i + 1, 15):
            A[i, j] = A[j, i] = 1
    # one bridge
    A[3, 10] = A[10, 3] = 1
    return A


@pytest.fixture
def cluster_weights(cluster_adj):
    return cluster_adj.astype(float) * 0.3


@pytest.fixture
def gene_names():
    return [f"G{i}" for i in range(15)]


# ---------------------------------------------------------------------------
# to_networkx
# ---------------------------------------------------------------------------

def test_to_networkx_unweighted(small_adj):
    from nodis.network.topology import to_networkx
    G = to_networkx(small_adj)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 5  # ring: 5 edges


def test_to_networkx_weighted(small_adj, small_weights):
    from nodis.network.topology import to_networkx
    G = to_networkx(small_adj, weights=small_weights)
    for u, v, d in G.edges(data=True):
        assert abs(d["weight"] - 0.5) < 1e-9


def test_to_networkx_gene_names(small_adj):
    from nodis.network.topology import to_networkx
    names = ["A", "B", "C", "D", "E"]
    G = to_networkx(small_adj, gene_names=names)
    assert set(G.nodes()) == set(names)


# ---------------------------------------------------------------------------
# NetworkTopology construction
# ---------------------------------------------------------------------------

def test_constructor_validation():
    from nodis.network.topology import NetworkTopology
    with pytest.raises(ValueError, match="square"):
        NetworkTopology(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="gene_names length"):
        NetworkTopology(np.eye(3), gene_names=["a", "b"])


def test_summary_basic(cluster_adj):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj)
    s = nt.summary()
    assert s["nodes"] == 15
    assert s["edges"] > 0
    assert 0 <= s["density"] <= 1
    assert s["n_components"] >= 1


def test_summary_empty():
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(np.zeros((5, 5)))
    s = nt.summary()
    assert s["edges"] == 0
    assert s["density"] == 0.0


# ---------------------------------------------------------------------------
# Community detection — greedy (always available)
# ---------------------------------------------------------------------------

def test_detect_communities_greedy(cluster_adj, gene_names):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj, gene_names=gene_names)
    result = nt.detect_communities(algorithm="greedy")

    assert result.algorithm == "greedy"
    assert len(result.labels) == 15
    assert result.n_communities >= 2
    assert result.modularity >= 0.0


def test_community_result_dataframe(cluster_adj, gene_names):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj, gene_names=gene_names)
    result = nt.detect_communities(algorithm="greedy")
    df = result.as_dataframe()
    assert set(df.columns) == {"gene", "community"}
    assert len(df) == 15
    assert set(df["gene"]) == set(gene_names)


def test_detect_communities_greedy_no_names(small_adj):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(small_adj)
    result = nt.detect_communities(algorithm="greedy")
    assert len(result.labels) == 5


# ---------------------------------------------------------------------------
# Community detection — louvain_nx
# ---------------------------------------------------------------------------

def test_detect_communities_louvain(cluster_adj):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj)
    result = nt.detect_communities(algorithm="louvain", seed=0)
    assert len(result.labels) == 15
    assert result.n_communities >= 1


# ---------------------------------------------------------------------------
# Community detection — spectral
# ---------------------------------------------------------------------------

def test_detect_communities_spectral(cluster_adj, gene_names):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj, gene_names=gene_names)
    result = nt.detect_communities(algorithm="spectral", n_clusters=2, seed=0)
    assert result.algorithm == "spectral"
    assert result.n_communities == 2
    assert len(result.labels) == 15
    # The two dense clusters should mostly land in different groups
    labels0 = result.labels[:7]
    labels1 = result.labels[7:]
    # Most within each natural cluster should share a label
    assert (np.bincount(labels0).max() >= 5)
    assert (np.bincount(labels1).max() >= 5)


def test_detect_communities_spectral_clamp(small_adj):
    """n_clusters > p → clamp to p."""
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(small_adj)
    result = nt.detect_communities(algorithm="spectral", n_clusters=100, seed=0)
    assert result.n_communities == 5  # clamped to p=5


# ---------------------------------------------------------------------------
# Leiden fallback warning
# ---------------------------------------------------------------------------

def test_leiden_falls_back_to_louvain(cluster_adj, monkeypatch):
    """When leidenalg is unavailable, Leiden should fall back with a warning."""
    from nodis.network import topology as topo_mod
    monkeypatch.setattr(topo_mod.NetworkTopology, "_leiden_available",
                        staticmethod(lambda: False))
    nt = topo_mod.NetworkTopology(cluster_adj)
    with pytest.warns(ImportWarning, match="leidenalg"):
        result = nt.detect_communities(algorithm="leiden", seed=0)
    # Should still return a valid result via louvain fallback
    assert len(result.labels) == 15


# ---------------------------------------------------------------------------
# Hub gene identification
# ---------------------------------------------------------------------------

def test_hub_genes_basic(cluster_adj, gene_names):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj, gene_names=gene_names)
    result = nt.hub_genes(top_n=3, metric="degree",
                          n_permutations=50, alpha=0.05, seed=0)
    assert len(result.scores) == 15
    assert set(result.scores.columns) >= {
        "gene", "degree", "betweenness", "hub_score", "p_value", "significant"
    }
    # Genes inside the dense clusters should have higher degree than the bridge
    assert result.scores.iloc[0]["degree"] > 0


def test_hub_genes_empty_graph():
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(np.zeros((5, 5)))
    with pytest.warns(UserWarning, match="no edges"):
        result = nt.hub_genes(n_permutations=10)
    assert result.n_hubs == 0


def test_hub_genes_metrics(cluster_adj):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj)
    for metric in ("degree", "betweenness", "eigenvector", "strength"):
        r = nt.hub_genes(metric=metric, n_permutations=20, seed=1)
        assert "hub_score" in r.scores.columns
        assert r.metric == metric


def test_hub_genes_reproducible(cluster_adj):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj)
    r1 = nt.hub_genes(n_permutations=30, seed=7)
    r2 = nt.hub_genes(n_permutations=30, seed=7)
    np.testing.assert_array_equal(r1.scores["p_value"].values,
                                  r2.scores["p_value"].values)


# ---------------------------------------------------------------------------
# Backbone extraction
# ---------------------------------------------------------------------------

def test_backbone_threshold(cluster_adj, cluster_weights):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj, weights=cluster_weights)
    bb = nt.backbone(method="threshold", threshold=0.1)
    assert bb.shape == (15, 15)
    assert np.all(np.diag(bb) == 0)
    # All weighted edges above 0.1 should be retained (weight = 0.3 > 0.1)
    np.testing.assert_array_equal(bb, cluster_adj)


def test_backbone_threshold_no_weights(cluster_adj):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj)
    bb = nt.backbone(method="threshold")
    np.testing.assert_array_equal(bb, cluster_adj.astype(int))


def test_backbone_disparity(cluster_adj, cluster_weights):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj, weights=cluster_weights)
    bb = nt.backbone(method="disparity", alpha=0.5)
    assert bb.shape == (15, 15)
    assert np.all(np.diag(bb) == 0)
    # Backbone should be a subset of the original adjacency
    assert np.all(bb <= cluster_adj)


def test_backbone_disparity_no_weights_warns(cluster_adj):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(cluster_adj)
    with pytest.warns(UserWarning, match="No weights"):
        bb = nt.backbone(method="disparity", threshold=0.0)
    # Falls back to threshold method → returns full adjacency
    np.testing.assert_array_equal(bb, cluster_adj.astype(int))


def test_backbone_invalid_method(small_adj):
    from nodis.network.topology import NetworkTopology
    nt = NetworkTopology(small_adj)
    with pytest.raises(ValueError, match="method must be"):
        nt.backbone(method="unknown")


# ---------------------------------------------------------------------------
# write_anndata_network
# ---------------------------------------------------------------------------

def test_write_anndata_network():
    anndata = pytest.importorskip("anndata")
    from nodis.network.topology import NetworkTopology, write_anndata_network

    p = 15
    cluster_adj = np.zeros((p, p), dtype=int)
    for i in range(7):
        for j in range(i + 1, 7):
            cluster_adj[i, j] = cluster_adj[j, i] = 1
    gene_names = [f"G{i}" for i in range(p)]

    adata = anndata.AnnData(np.random.default_rng(0).normal(size=(10, p)))
    adata.var_names = gene_names

    nt = NetworkTopology(cluster_adj, gene_names=gene_names)
    comm = nt.detect_communities(algorithm="greedy")
    hubs = nt.hub_genes(n_permutations=20, seed=0)
    bb = nt.backbone(method="threshold")

    write_anndata_network(
        adata,
        adjacency=cluster_adj,
        communities=comm,
        hubs=hubs,
        backbone_adj=bb,
    )

    assert "nodis_adjacency" in adata.varp
    assert "nodis_backbone" in adata.varp
    assert "nodis_community" in adata.var.columns
    assert "nodis_hub_score" in adata.var.columns
    assert "nodis_hub_pvalue" in adata.var.columns
    assert "nodis_hub" in adata.var.columns
    assert "nodis_communities" in adata.uns
    assert "nodis_hubs" in adata.uns
    assert len(adata.var["nodis_community"]) == p


def test_write_anndata_network_size_mismatch():
    anndata = pytest.importorskip("anndata")
    from nodis.network.topology import write_anndata_network

    adata = anndata.AnnData(np.zeros((5, 10)))
    with pytest.raises(ValueError, match="adjacency size"):
        write_anndata_network(adata, np.zeros((5, 5)))
