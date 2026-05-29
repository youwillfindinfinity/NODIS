import numpy as np
import pandas as pd
import pytest
from nodis.enrich.extract import hub_genes, ranked_genes, community_gene_sets


def _make_adj(p: int = 8, seed: int = 0) -> np.ndarray:
    """Random symmetric binary adjacency with ~30% density."""
    rng = np.random.default_rng(seed)
    A = (rng.random((p, p)) < 0.3).astype(int)
    A = np.triu(A, k=1)
    A = A + A.T
    np.fill_diagonal(A, 0)
    return A


def _make_pval(p: int = 8, seed: int = 0) -> np.ndarray:
    """Symmetric p-value matrix with diagonal = 1."""
    rng = np.random.default_rng(seed)
    P = rng.random((p, p))
    P = (P + P.T) / 2
    np.fill_diagonal(P, 1.0)
    return P


def test_hub_genes_returns_list():
    adj = _make_adj()
    names = [f"G{i}" for i in range(8)]
    result = hub_genes(adj, names, centrality="degree", quantile=0.75)
    assert isinstance(result, list)
    assert all(isinstance(g, str) for g in result)
    assert len(result) > 0


def test_hub_genes_top_quantile():
    # Hub at G0: connected to all others
    adj = np.zeros((5, 5), dtype=int)
    adj[0, 1:] = 1
    adj[1:, 0] = 1
    names = [f"G{i}" for i in range(5)]
    hubs = hub_genes(adj, names, centrality="degree", quantile=0.9)
    assert "G0" in hubs


def test_hub_genes_betweenness():
    adj = _make_adj()
    names = [f"G{i}" for i in range(8)]
    result = hub_genes(adj, names, centrality="betweenness", quantile=0.5)
    assert isinstance(result, list)


def test_hub_genes_invalid_centrality():
    adj = _make_adj()
    names = [f"G{i}" for i in range(8)]
    with pytest.raises(ValueError, match="centrality must be one of"):
        hub_genes(adj, names, centrality="nonsense", quantile=0.9)


def test_ranked_genes_returns_series():
    adj = _make_adj()
    pvals = _make_pval()
    names = [f"G{i}" for i in range(8)]
    ranked = ranked_genes(adj, names, pvals, method="min_pvalue")
    assert isinstance(ranked, pd.Series)
    assert len(ranked) == 8
    assert ranked.index.tolist() == names


def test_ranked_genes_min_pvalue_score():
    # G0 connects to G1 with p=0.001 — should have low min_pvalue → high rank score
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = 1; adj[1, 0] = 1
    pvals = np.ones((4, 4))
    pvals[0, 1] = 0.001; pvals[1, 0] = 0.001
    names = ["A", "B", "C", "D"]
    ranked = ranked_genes(adj, names, pvals, method="min_pvalue")
    # A and B should have higher scores (lower min p-value → -log10(p) higher)
    assert ranked["A"] > ranked["C"]
    assert ranked["B"] > ranked["D"]


def test_ranked_genes_degree_method():
    adj = _make_adj()
    pvals = _make_pval()
    names = [f"G{i}" for i in range(8)]
    ranked = ranked_genes(adj, names, pvals, method="degree")
    assert isinstance(ranked, pd.Series)


def test_ranked_genes_invalid_method():
    adj = _make_adj()
    pvals = _make_pval()
    names = [f"G{i}" for i in range(8)]
    with pytest.raises(ValueError, match="method must be one of"):
        ranked_genes(adj, names, pvals, method="bogus")


def test_community_gene_sets_returns_dict():
    adj = _make_adj(p=10, seed=1)
    names = [f"G{i}" for i in range(10)]
    comms = community_gene_sets(adj, names, algorithm="greedy_modularity")
    assert isinstance(comms, dict)
    assert len(comms) >= 1
    # All genes must appear in exactly one community
    all_genes = [g for genes in comms.values() for g in genes]
    assert sorted(all_genes) == sorted(names)


def test_community_gene_sets_label_format():
    adj = _make_adj(p=10, seed=1)
    names = [f"G{i}" for i in range(10)]
    comms = community_gene_sets(adj, names)
    for key in comms.keys():
        assert key.startswith("community_"), f"unexpected key: {key}"


def test_community_gene_sets_invalid_algorithm():
    adj = _make_adj()
    names = [f"G{i}" for i in range(8)]
    with pytest.raises(ValueError, match="algorithm must be one of"):
        community_gene_sets(adj, names, algorithm="kmeans")


def test_community_gene_sets_disconnected():
    adj = np.zeros((4, 4), dtype=int)
    names = ["A", "B", "C", "D"]
    comms = community_gene_sets(adj, names)
    all_genes = [g for genes in comms.values() for g in genes]
    assert sorted(all_genes) == sorted(names)
    assert len(all_genes) == len(names)  # no duplicates


def test_hub_genes_empty_graph_returns_empty():
    adj = np.zeros((5, 5), dtype=int)
    names = [f"G{i}" for i in range(5)]
    assert hub_genes(adj, names, centrality="degree", quantile=0.9) == []


def test_hub_genes_shape_mismatch_raises():
    adj = _make_adj(p=8)
    with pytest.raises(ValueError, match="gene_names length"):
        hub_genes(adj, ["A", "B"], centrality="degree", quantile=0.5)


def test_ranked_genes_shape_mismatch_raises():
    adj = _make_adj(p=8)
    pvals = _make_pval(p=8)
    with pytest.raises(ValueError, match="gene_names length"):
        ranked_genes(adj, ["A", "B"], pvals, method="degree")


def test_community_gene_sets_label_propagation_coverage():
    adj = _make_adj(p=10, seed=2)
    names = [f"G{i}" for i in range(10)]
    comms = community_gene_sets(adj, names, algorithm="label_propagation", seed=42)
    all_genes = [g for genes in comms.values() for g in genes]
    assert sorted(all_genes) == sorted(names)
    assert len(all_genes) == len(names)
