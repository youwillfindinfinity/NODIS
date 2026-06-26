"""Unit tests for nodis.preprocess.pseudobulk.aggregate_pseudobulk."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("anndata")

from nodis.preprocess.pseudobulk import aggregate_pseudobulk


def make_test_adata(n_cells=200, n_genes=50, n_samples=4, n_celltypes=3):
    """Construct minimal AnnData for pseudobulk tests."""
    import anndata as ad

    rng = np.random.default_rng(42)
    X = rng.integers(1, 100, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(
        {
            "sample_id": [f"S{i % n_samples}" for i in range(n_cells)],
            "cell_type": [f"CT{i % n_celltypes}" for i in range(n_cells)],
        }
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_output_shape():
    adata = make_test_adata(n_cells=200, n_samples=4, n_celltypes=3, n_genes=50)
    pb = aggregate_pseudobulk(adata, groupby=["sample_id", "cell_type"])
    assert pb.n_vars == 50
    assert pb.n_obs <= 4 * 3  # at most 12 groups; some may be < min_cells


def test_sum_vs_mean():
    adata = make_test_adata()
    pb_sum = aggregate_pseudobulk(adata, groupby=["sample_id"], agg="sum")
    pb_mean = aggregate_pseudobulk(adata, groupby=["sample_id"], agg="mean")
    # sum should be larger than mean (positive expression values)
    assert pb_sum.X.mean() >= pb_mean.X.mean()


def test_min_cells_filter():
    adata = make_test_adata(n_cells=100)
    pb_strict = aggregate_pseudobulk(
        adata, groupby=["sample_id", "cell_type"], min_cells=50
    )
    pb_loose = aggregate_pseudobulk(
        adata, groupby=["sample_id", "cell_type"], min_cells=1
    )
    assert pb_strict.n_obs <= pb_loose.n_obs


def test_preserves_gene_names():
    adata = make_test_adata()
    pb = aggregate_pseudobulk(adata, groupby=["sample_id"])
    assert list(pb.var_names) == list(adata.var_names)


def test_obs_metadata_populated():
    adata = make_test_adata()
    pb = aggregate_pseudobulk(adata, groupby=["sample_id"])
    assert "sample_id" in pb.obs.columns
    assert "pseudobulk_n_cells" in pb.uns


def test_sparse_input():
    import scipy.sparse as sp

    adata = make_test_adata()
    adata.X = sp.csr_matrix(adata.X)
    pb = aggregate_pseudobulk(adata, groupby=["sample_id"])
    assert pb.n_obs > 0


def test_missing_column_raises():
    adata = make_test_adata()
    with pytest.raises(KeyError):
        aggregate_pseudobulk(adata, groupby=["nonexistent_col"])


def test_n_cells_recorded_per_group():
    adata = make_test_adata(n_cells=200, n_samples=4, n_celltypes=1)
    pb = aggregate_pseudobulk(adata, groupby=["sample_id"], min_cells=1)
    for group_key, count in pb.uns["pseudobulk_n_cells"].items():
        assert count > 0


def test_sum_values_correct():
    """sum per group should equal manual sum for a small example."""
    import anndata as ad

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    obs = pd.DataFrame({"sample_id": ["A", "A", "B", "B"]})
    var = pd.DataFrame(index=["g0", "g1"])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    pb = aggregate_pseudobulk(adata, groupby=["sample_id"], agg="sum", min_cells=1)
    # A: [1+3, 2+4] = [4, 6]; B: [5+7, 6+8] = [12, 14]
    assert pb.n_obs == 2
    totals = {
        row.sample_id: pb.X[idx] for idx, row in enumerate(pb.obs.itertuples())
    }
    np.testing.assert_array_equal(totals["A"], [4.0, 6.0])
    np.testing.assert_array_equal(totals["B"], [12.0, 14.0])


def test_invalid_agg_raises():
    adata = make_test_adata()
    with pytest.raises(ValueError, match="agg must be"):
        aggregate_pseudobulk(adata, groupby=["sample_id"], agg="median")


def test_empty_result_when_all_filtered():
    adata = make_test_adata(n_cells=10, n_samples=2, n_celltypes=2)
    pb = aggregate_pseudobulk(
        adata, groupby=["sample_id", "cell_type"], min_cells=100
    )
    assert pb.n_obs == 0
    assert pb.n_vars == adata.n_vars
