"""
AnnData input adapter for NODIS.

Extracts an expression matrix from an AnnData object and returns a plain
``np.ndarray`` ready for NODIS inference.  AnnData is a soft dependency —
the function uses duck-typed attribute access (``.X``, ``.layers``, ``.var``)
rather than importing the package, so ``import nodis`` succeeds even without
``anndata`` installed.

Typical usage
-------------
>>> import scanpy as sc
>>> adata = sc.read_h5ad("my_data.h5ad")
>>> sc.pp.normalize_total(adata)
>>> sc.pp.log1p(adata)
>>> sc.pp.highly_variable_genes(adata, n_top_genes=200)
>>> from nodis import from_anndata
>>> X = from_anndata(adata, layer="log1p_norm", use_hvg=True, npn=True)
>>> from nodis import DesparifiedGGM
>>> est = DesparifiedGGM().fit(X)
"""

import numpy as np


def from_anndata(
    adata,
    layer: str | None = None,
    genes: list[str] | None = None,
    use_hvg: bool = False,
    npn: bool = False,
    sparse_to_dense: bool = True,
) -> np.ndarray:
    """
    Extract an expression matrix from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Single-cell expression data container (from the ``anndata`` package).
    layer : str or None, default None
        Which layer to extract.  ``None`` → ``adata.X``;
        a string key → ``adata.layers[layer]``.
    genes : list of str or None, default None
        Subset of gene names (from ``adata.var_names``) to extract.
        If None, all genes in the selected layer are returned (after HVG
        filtering if ``use_hvg=True``).  When both ``genes`` and
        ``use_hvg=True`` are supplied, ``genes`` takes priority and HVG
        filtering is not applied.
    use_hvg : bool, default False
        If True, subset to ``adata.var['highly_variable']`` before returning.
        Silently ignored if ``adata.var`` does not contain a
        ``'highly_variable'`` column.
    npn : bool, default False
        If True, apply the NPN shrinkage transform (``npn_shrinkage``) to the
        extracted matrix before returning.  Recommended for scRNA-seq data.
    sparse_to_dense : bool, default True
        If True, convert scipy sparse matrices to dense ``np.ndarray``.
        Set to False only if you know downstream code handles sparse input.

    Returns
    -------
    X : (n_obs, n_genes) ndarray of float64
        Expression matrix ready for NODIS inference.

    Raises
    ------
    KeyError
        If ``layer`` does not exist in ``adata.layers``.
    ValueError
        If any gene name in ``genes`` is not found in ``adata.var_names``.
    """
    # Extract matrix
    # No anndata import is performed here — the function uses only duck-typed
    # attribute access (.X, .layers, .var), so no ImportError is raised if
    # anndata is absent. Passing a non-AnnData-like object will raise
    # AttributeError at the point of access.
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]   # raises KeyError if absent — intentional

    # Convert sparse to dense
    if sparse_to_dense and hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float64)

    # Gene subsetting by name (takes priority over use_hvg)
    if genes is not None:
        var_names = list(adata.var_names)
        missing = [g for g in genes if g not in var_names]
        if missing:
            raise ValueError(
                f"The following gene names were not found in adata.var_names: {missing}"
            )
        col_idx = [var_names.index(g) for g in genes]
        X = X[:, col_idx]
    elif use_hvg and 'highly_variable' in adata.var:
        hvg_mask = np.asarray(adata.var['highly_variable'], dtype=bool)
        X = X[:, hvg_mask]

    if npn:
        from nodis.preprocess.npn import npn_shrinkage
        X = npn_shrinkage(X)

    return X
