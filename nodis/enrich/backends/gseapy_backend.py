# nodis/enrich/backends/gseapy_backend.py
"""
GSEApy backend — pre-ranked GSEA and Enrichr ORA.

Covers all three biological levels via Enrichr library names from databases.py:
  RNA           → GO_Biological_Process_2023, KEGG_2021_Human, Reactome_2022
  post-transcriptional → miRTarBase_2017, TargetScan_microRNA_2017
  protein       → CORUM, PPI_Hub_Proteins, InterPro_Domains_2019
"""
from __future__ import annotations

import pandas as pd

try:
    import gseapy  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "gseapy is required for the gseapy backend.\n"
        "Install it with: pip install gseapy\n"
        "Or install all enrichment dependencies: pip install 'nodis[enrich]'"
    ) from exc

_PRERANK_OUTPUT_COLS = [
    "term_name", "ES", "NES", "p_value", "adjusted_p_value", "lead_genes",
]
_ORA_OUTPUT_COLS = [
    "gene_set", "term_name", "overlap", "p_value", "adjusted_p_value",
    "odds_ratio", "combined_score", "genes",
]


def run_prerank(
    ranked_series: pd.Series,
    gene_sets: list[str],
    threads: int = 1,
    permutation_num: int = 1000,
    min_size: int = 5,
    max_size: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Run pre-ranked GSEA using GSEApy.

    Parameters
    ----------
    ranked_series : pd.Series
        Index = gene names, values = ranking scores (higher = more significant).
    gene_sets : list of str
        Enrichr library names or paths to local GMT files.
    threads : int, default 1
        Number of parallel threads for permutation testing.
    permutation_num : int, default 1000
        Number of permutations for p-value estimation.
    min_size : int, default 5
        Minimum gene set size to test.
    max_size : int, default 500
        Maximum gene set size to test.
    seed : int, default 42
        Random seed for reproducible permutations.

    Returns
    -------
    pd.DataFrame
        Columns: term_name, ES, NES, p_value, adjusted_p_value, lead_genes.
        Empty DataFrame if ranked_series is empty.
    """
    if ranked_series.empty:
        return pd.DataFrame(columns=_PRERANK_OUTPUT_COLS)

    rnk = ranked_series.sort_values(ascending=False)

    res = gseapy.prerank(
        rnk=rnk,
        gene_sets=gene_sets,
        threads=threads,
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        seed=seed,
        outdir=None,
        verbose=False,
    )

    df = res.res2d if hasattr(res, "res2d") else res.results

    if df is None or df.empty:
        return pd.DataFrame(columns=_PRERANK_OUTPUT_COLS)

    rename: dict[str, str] = {}
    col_map = {
        "Term": "term_name",
        "NOM p-val": "p_value",
        "FGSEA p-val": "p_value",
        "FDR q-val": "adjusted_p_value",
        "Lead_genes": "lead_genes",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            rename[old] = new
    df = df.rename(columns=rename).copy()

    present = [c for c in _PRERANK_OUTPUT_COLS if c in df.columns]
    extra = [c for c in ["ES", "NES"] if c in df.columns and c not in present]
    return df[present + extra].reset_index(drop=True)


def run_ora(
    gene_list: list[str],
    gene_sets: list[str],
    background: list[str] | None = None,
    organism: str = "Human",
) -> pd.DataFrame:
    """Run Over-Representation Analysis via Enrichr (GSEApy wrapper).

    Parameters
    ----------
    gene_list : list of str
        Query gene identifiers (HGNC symbols).
    gene_sets : list of str
        Enrichr library names.
    background : list of str or None
        Background gene set. ``None`` → Enrichr default.
    organism : str, default ``"Human"``
        Organism name for Enrichr.

    Returns
    -------
    pd.DataFrame
        Columns: gene_set, term_name, overlap, p_value, adjusted_p_value,
        odds_ratio, combined_score, genes.
        Empty DataFrame if gene_list is empty.
    """
    if not gene_list:
        return pd.DataFrame(columns=_ORA_OUTPUT_COLS)

    kwargs: dict = {
        "gene_list": gene_list,
        "gene_sets": gene_sets,
        "organism": organism,
        "outdir": None,
        "verbose": False,
    }
    if background is not None:
        kwargs["background"] = background

    res = gseapy.enrichr(**kwargs)

    if res is None:
        return pd.DataFrame(columns=_ORA_OUTPUT_COLS)

    df = res.results if hasattr(res, "results") else res

    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame(columns=_ORA_OUTPUT_COLS)

    rename: dict[str, str] = {
        "Term": "term_name",
        "Gene_set": "gene_set",
        "Overlap": "overlap",
        "P-value": "p_value",
        "Adjusted P-value": "adjusted_p_value",
        "Odds Ratio": "odds_ratio",
        "Combined Score": "combined_score",
        "Genes": "genes",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    present = [c for c in _ORA_OUTPUT_COLS if c in df.columns]
    return df[present].reset_index(drop=True)
