"""
Pseudobulk aggregation for single-cell AnnData objects.

aggregate_pseudobulk(adata, groupby, layer, agg, min_cells) → AnnData

Groups cells by `groupby` keys (typically ["sample_id", "cell_type"]),
sums or averages raw counts within each group, and returns a new AnnData
where each observation is one pseudobulk sample.

This converts scRNA-seq into a bulk-equivalent matrix suitable for GGM
inference: the resulting n_obs is the number of (sample × cell-type)
combinations, not the number of cells.

Parameters
----------
adata       : AnnData  per-cell expression (raw counts recommended)
groupby     : list[str]  columns in adata.obs to group by
layer       : str | None  layer to aggregate; None → .X
agg         : "sum" | "mean"  aggregation function
min_cells   : int  minimum cells per group; groups below this are dropped
              (default 10)

Returns
-------
AnnData  shape (n_groups, n_genes), with .obs containing group metadata
         and .uns["pseudobulk_n_cells"] recording cells per group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_pseudobulk(
    adata,
    groupby: list[str],
    layer: str | None = None,
    agg: str = "sum",
    min_cells: int = 10,
):
    """Aggregate single-cell counts to pseudobulk samples."""
    try:
        import anndata as ad
    except ImportError:
        raise ImportError(
            "anndata is required for pseudobulk aggregation. "
            "Install it with: pip install anndata"
        )
    import scipy.sparse

    if agg not in ("sum", "mean"):
        raise ValueError(f"agg must be 'sum' or 'mean', got '{agg}'")

    # Validate groupby columns
    for col in groupby:
        if col not in adata.obs.columns:
            raise KeyError(f"Column '{col}' not found in adata.obs")

    # Get expression matrix
    X_full = adata.layers[layer] if layer is not None else adata.X

    # Build a stable group key per cell
    obs = adata.obs
    group_series = obs[groupby].astype(str).agg("__".join, axis=1)
    unique_groups = group_series.unique()

    rows: list[np.ndarray] = []
    group_meta: list[dict] = []
    n_cells_dict: dict[str, int] = {}

    for group_key in unique_groups:
        mask = (group_series == group_key).values
        n_cells = int(mask.sum())
        if n_cells < min_cells:
            continue

        X_group = X_full[mask]
        if scipy.sparse.issparse(X_group):
            X_group = X_group.toarray()

        agg_expr = X_group.sum(axis=0) if agg == "sum" else X_group.mean(axis=0)
        rows.append(np.asarray(agg_expr).ravel())

        # Collect group-level metadata from the first matching obs row
        first_idx = obs.index[mask][0]
        meta: dict = {col: obs.loc[first_idx, col] for col in groupby}
        meta["pseudobulk_group"] = group_key
        group_meta.append(meta)
        n_cells_dict[group_key] = n_cells

    if not rows:
        pb = ad.AnnData(
            X=np.zeros((0, adata.n_vars), dtype=float),
            obs=pd.DataFrame(columns=groupby + ["pseudobulk_group"]),
            var=adata.var.copy(),
        )
        pb.uns["pseudobulk_n_cells"] = {}
        return pb

    X_agg = np.vstack(rows)
    obs_df = pd.DataFrame(group_meta).reset_index(drop=True)

    pb = ad.AnnData(
        X=X_agg,
        obs=obs_df,
        var=adata.var.copy(),
    )
    pb.uns["pseudobulk_n_cells"] = n_cells_dict
    return pb
