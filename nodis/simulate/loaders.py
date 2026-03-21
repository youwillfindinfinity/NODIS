"""
Data loaders for public benchmark datasets.

Datasets
--------
DREAM5  — Network Inference Challenge (Marbach et al. 2012, Nat Methods)
SERGIO  — Single-cell RNA-seq simulator (Dibaeinia & Sinha 2020, Cell Syst)
GNW     — GeneNetWeaver synthetic networks (Schaffter et al. 2011, Bioinformatics)

Notes
-----
Data must be downloaded manually; these loaders only parse locally cached files.
See docs/installation.md for download instructions.
"""

import pathlib
import numpy as np
import pandas as pd


def load_dream5_insilico(
    data_dir: str | pathlib.Path = "data/dream5",
    network: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a DREAM5 in-silico network expression matrix and gold standard.

    Supports two expression file formats:
    - Official format (experiments × genes, with experiment index column)
    - Alternative/averaged format (experiments × genes, gene names as column headers,
      no index column): ``net{N}_expression_data_avg.tsv`` from gnw.sourceforge.net

    Supports two gold standard formats:
    - Official TSV: tab-separated with columns TF, target, label (0/1)
    - JBris CSV: comma-separated with columns from, to, weight (positives only)

    Parameters
    ----------
    data_dir : path to directory containing DREAM5 files
    network  : 1 (E. coli in silico) or 3 (E. coli in vivo)

    Returns
    -------
    expr  : DataFrame, shape (n_samples, n_genes), gene names as columns
    gold  : DataFrame with columns ['TF', 'target', 'label']
    """
    data_dir = pathlib.Path(data_dir)

    # Expression file — try official name then averaged alternative
    expr_candidates = [
        data_dir / f"net{network}_expression_data.tsv",
        data_dir / f"net{network}_expression_data_avg.tsv",
        data_dir / "extracted" / "DREAM5_NetworkInferenceChallenge_AlternativeDataFormats"
                 / f"net{network}" / f"net{network}_expression_data_avg.tsv",
    ]
    expr_file = next((f for f in expr_candidates if f.exists()), None)
    if expr_file is None:
        raise FileNotFoundError(
            f"DREAM5 expression file not found in {data_dir}.\n"
            "Download from https://gnw.sourceforge.net/dreamchallenge.html"
        )

    # Gold standard — try official name then JBris CSV
    gold_candidates = [
        data_dir / f"DREAM5_NetworkInference_GoldStandard_Network{network}.tsv",
        data_dir / "in_silico_gold_standard.csv",
    ]
    gold_file = next((f for f in gold_candidates if f.exists()), None)
    if gold_file is None:
        raise FileNotFoundError(
            f"DREAM5 gold standard not found in {data_dir}.\n"
            "Download from https://www.synapse.org/Synapse:syn2787209 "
            "or https://github.com/JBris/dream5_grn_data"
        )

    # Read expression — avg format has gene names as column headers (no index column)
    expr = pd.read_csv(expr_file, sep="\t")
    if expr.iloc[:, 0].dtype == object:
        # First column contains gene names (official format with experiment index)
        expr = expr.set_index(expr.columns[0])
        expr = expr.T  # → (n_experiments, n_genes)
    # expr columns are now gene names (G1, G2, ...)

    # Read gold standard and normalise to TF / target / label
    sep = "," if gold_file.suffix == ".csv" else "\t"
    gold_raw = pd.read_csv(gold_file, sep=sep)
    if "from" in gold_raw.columns:
        # JBris format: from/to/weight, positives only
        gold = gold_raw.rename(columns={"from": "TF", "to": "target"})[["TF", "target"]]
        gold["label"] = 1
    else:
        gold = gold_raw[["TF", "target", "label"]]

    return expr, gold


def load_sergio_dataset(
    data_dir: str | pathlib.Path = "data/sergio",
    dataset_id: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a pre-built SERGIO dataset (expression + GRN ground truth).

    Reads the CSV format from https://github.com/PayamDiba/SERGIO.
    Each dataset directory is named ``*DS{id}`` and contains:
      - ``gt_GRN.csv``            — 1-indexed directed edge list (source, target)
      - ``simulated_noNoise_{i}.csv`` — (n_genes+1) × (n_cells+1) expression matrix
        with a header row (cell indices) and index column (gene indices).

    All cell-type files are concatenated along the cell axis.

    Parameters
    ----------
    data_dir   : path to directory containing SERGIO dataset subdirectories
    dataset_id : integer ID (1–8) of the pre-built SERGIO dataset

    Returns
    -------
    expr : (n_genes, n_cells_total) ndarray — concatenated across cell types
    adj  : (n_genes, n_genes) binary adjacency (GRN, symmetrised)
    """
    data_dir = pathlib.Path(data_dir)
    matches = sorted(data_dir.glob(f"*DS{dataset_id}"))
    if not matches:
        raise FileNotFoundError(
            f"SERGIO dataset DS{dataset_id} not found in {data_dir}.\n"
            "Clone from https://github.com/PayamDiba/SERGIO and copy data_sets/ here."
        )
    ds_dir = matches[0]

    # Expression: concatenate all cell-type files (load first to get n_genes)
    expr_files = sorted(ds_dir.glob("simulated_noNoise_*.csv"))
    if not expr_files:
        raise FileNotFoundError(f"No simulated_noNoise_*.csv files in {ds_dir}")
    frames = []
    for f in expr_files:
        df = pd.read_csv(f, header=0, index_col=0)  # genes × cells
        frames.append(df.values)
    expr = np.hstack(frames).astype(float)   # (n_genes, n_cells_total)
    n_genes = expr.shape[0]

    # Ground-truth GRN: 1-indexed directed edges → symmetric binary adjacency
    # Use n_genes from expression matrix so adj is always (n_genes, n_genes);
    # genes that are leaves in the GRN (no edges) receive all-zero rows/cols.
    grn = pd.read_csv(ds_dir / "gt_GRN.csv", header=None).values.astype(int)
    adj = np.zeros((n_genes, n_genes), dtype=int)
    for src, tgt in grn:
        adj[src - 1, tgt - 1] = 1
        adj[tgt - 1, src - 1] = 1
    np.fill_diagonal(adj, 0)

    return expr, adj
