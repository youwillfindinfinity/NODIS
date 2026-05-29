"""
Unit tests for nodis/simulate/loaders.py.
"""
import pathlib

import numpy as np
import pandas as pd
import pytest

from nodis.simulate.loaders import load_dream5_insilico, load_sergio_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_official_expr(path: pathlib.Path) -> None:
    """Write a net1_expression_data.tsv in official transposed format.

    Format: first column is a string gene-name label; rows are genes; columns
    are experiment labels.  After set_index + .T → (n_experiments, n_genes).
    """
    # 3 genes (rows), 2 experiments (data columns) + 1 label column
    content = (
        "gene\texp1\texp2\n"
        "G1\t0.1\t0.4\n"
        "G2\t0.2\t0.5\n"
        "G3\t0.3\t0.6\n"
    )
    path.write_text(content)


def _write_avg_expr(path: pathlib.Path) -> None:
    """Write a net1_expression_data_avg.tsv in avg format (numeric first col)."""
    # Gene names are column headers; first column is numeric → no transposition
    content = (
        "G1\tG2\tG3\n"
        "0.1\t0.2\t0.3\n"
        "0.4\t0.5\t0.6\n"
    )
    path.write_text(content)


def _write_official_gold(path: pathlib.Path) -> None:
    """Write an official DREAM5 gold standard TSV."""
    content = "TF\ttarget\tlabel\nG1\tG2\t1\nG1\tG3\t0\n"
    path.write_text(content)


def _write_jbris_gold(path: pathlib.Path) -> None:
    """Write a JBris CSV gold standard."""
    content = "from,to,weight\nG1,G2,1.0\nG2,G3,0.8\n"
    path.write_text(content)


# ---------------------------------------------------------------------------
# DREAM5 tests
# ---------------------------------------------------------------------------

def test_dream5_official_format(tmp_path):
    """Official transposed format: expr has correct shape & columns; gold has TF/target/label."""
    _write_official_expr(tmp_path / "net1_expression_data.tsv")
    _write_official_gold(tmp_path / "DREAM5_NetworkInference_GoldStandard_Network1.tsv")

    expr, gold = load_dream5_insilico(data_dir=tmp_path, network=1)

    # After set_index + .T → (n_experiments=2, n_genes=3)
    assert expr.shape == (2, 3)
    assert list(expr.columns) == ["G1", "G2", "G3"]
    assert list(gold.columns) == ["TF", "target", "label"]


def test_dream5_avg_format(tmp_path):
    """Avg format: gene names already as column headers; no transposition needed."""
    _write_avg_expr(tmp_path / "net1_expression_data_avg.tsv")
    _write_official_gold(tmp_path / "DREAM5_NetworkInference_GoldStandard_Network1.tsv")

    expr, gold = load_dream5_insilico(data_dir=tmp_path, network=1)

    # Avg format: 2 rows, 3 columns (G1, G2, G3)
    assert expr.shape == (2, 3)
    assert "G1" in expr.columns


def test_dream5_jbris_gold(tmp_path):
    """JBris CSV gold: from/to/weight → TF/target, label=1 for all rows."""
    _write_official_expr(tmp_path / "net1_expression_data.tsv")
    _write_jbris_gold(tmp_path / "in_silico_gold_standard.csv")

    _, gold = load_dream5_insilico(data_dir=tmp_path, network=1)

    assert list(gold.columns) == ["TF", "target", "label"]
    assert (gold["label"] == 1).all()
    assert len(gold) == 2


def test_dream5_missing_expr_raises(tmp_path):
    """No expression file → FileNotFoundError."""
    _write_official_gold(tmp_path / "DREAM5_NetworkInference_GoldStandard_Network1.tsv")

    with pytest.raises(FileNotFoundError):
        load_dream5_insilico(data_dir=tmp_path, network=1)


def test_dream5_missing_gold_raises(tmp_path):
    """Expression present but no gold standard → FileNotFoundError."""
    _write_official_expr(tmp_path / "net1_expression_data.tsv")

    with pytest.raises(FileNotFoundError):
        load_dream5_insilico(data_dir=tmp_path, network=1)


# ---------------------------------------------------------------------------
# SERGIO tests
# ---------------------------------------------------------------------------

def _make_sergio_ds(tmp_path: pathlib.Path, n_genes: int = 4, n_cells: int = 5) -> pathlib.Path:
    """Create a minimal fake SERGIO DS1 directory."""
    ds_dir = tmp_path / "myDS1"
    ds_dir.mkdir()

    # Two cell-type expression files: genes × cells (with header + index)
    for i in range(2):
        df = pd.DataFrame(
            np.random.default_rng(i).standard_normal((n_genes, n_cells)),
            index=[f"gene{j}" for j in range(n_genes)],
            columns=[f"cell{k}" for k in range(n_cells)],
        )
        df.to_csv(ds_dir / f"simulated_noNoise_{i}.csv")

    # gt_GRN.csv: 1-indexed directed edges (source, target), no header
    grn = pd.DataFrame([[1, 2], [2, 3]])
    grn.to_csv(ds_dir / "gt_GRN.csv", header=False, index=False)

    return ds_dir


def test_sergio_loads_correctly(tmp_path):
    """Fake DS1: correct expr shape, adj shape, adjacency is symmetric, no self-loops."""
    n_genes, n_cells, n_cell_types = 4, 5, 2
    _make_sergio_ds(tmp_path, n_genes=n_genes, n_cells=n_cells)

    expr, adj = load_sergio_dataset(data_dir=tmp_path, dataset_id=1)

    assert expr.shape == (n_genes, n_cells * n_cell_types)
    assert adj.shape == (n_genes, n_genes)
    np.testing.assert_array_equal(adj, adj.T)
    assert np.diag(adj).sum() == 0


def test_sergio_missing_dataset_raises(tmp_path):
    """No DS1 directory → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_sergio_dataset(data_dir=tmp_path, dataset_id=1)


def test_sergio_missing_expr_files_raises(tmp_path):
    """DS1 dir exists but no simulated_noNoise_*.csv → FileNotFoundError."""
    ds_dir = tmp_path / "myDS1"
    ds_dir.mkdir()
    # Write only gt_GRN.csv, no expression files
    pd.DataFrame([[1, 2]]).to_csv(ds_dir / "gt_GRN.csv", header=False, index=False)

    with pytest.raises(FileNotFoundError):
        load_sergio_dataset(data_dir=tmp_path, dataset_id=1)
