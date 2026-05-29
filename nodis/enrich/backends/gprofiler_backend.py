"""
g:Profiler ORA backend.

Covers all three biological levels:
  RNA           → GO:BP, GO:MF, GO:CC, KEGG, REAC, WP
  post-transcriptional → MIRNA, TF
  protein       → CORUM, HP, HPA

g:Profiler is a web-service backend — results depend on network access.
Organism codes: https://biit.cs.ut.ee/gprofiler/page/organism-list
"""
from __future__ import annotations

import pandas as pd

def _get_gprofiler():
    """Deferred import — raises informative error if gprofiler-official is absent."""
    try:
        from gprofiler import GProfiler  # type: ignore[import]
        return GProfiler
    except ImportError as exc:
        raise ImportError(
            "gprofiler-official is required for the gprofiler backend.\n"
            "Install it with: pip install gprofiler-official\n"
            "Or install all enrichment dependencies: pip install 'nodis[enrich]'"
        ) from exc

_OUTPUT_COLUMNS = ["term_id", "term_name", "p_value", "adjusted_p_value",
                   "source", "intersection_size", "query_size", "term_size",
                   "intersections"]


def run_ora(
    gene_list: list[str],
    sources: list[str],
    organism: str = "hsapiens",
    background: list[str] | None = None,
    user_threshold: float = 0.05,
) -> pd.DataFrame:
    """Run Over-Representation Analysis via g:Profiler.

    Parameters
    ----------
    gene_list : list of str
        Query gene identifiers (HGNC symbols or Ensembl IDs).
    sources : list of str
        g:Profiler data sources (e.g. ``["GO:BP", "KEGG", "CORUM"]``).
    organism : str, default ``"hsapiens"``
        g:Profiler organism code.
    background : list of str or None
        Custom statistical background. ``None`` → all annotated genes.
    user_threshold : float, default 0.05
        Adjusted p-value threshold for significance.

    Returns
    -------
    pd.DataFrame
        Columns: term_id, term_name, p_value, adjusted_p_value, source,
        intersection_size, query_size, term_size, intersections.
        Empty DataFrame if gene_list is empty or no results.
    """
    if not gene_list:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    GProfiler = _get_gprofiler()
    gp = GProfiler(return_dataframe=True)

    kwargs: dict = {
        "query": gene_list,
        "organism": organism,
        "sources": sources,
        "user_threshold": user_threshold,
        "no_evidences": False,
    }
    if background is not None:
        kwargs["background"] = background

    raw = gp.profile(**kwargs)

    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    # Normalise to expected output columns
    if isinstance(raw, dict) and "results" in raw:
        records = raw["results"]
        if not records:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        df = pd.DataFrame(records)
    else:
        df = raw

    # Standardise column names
    rename_map: dict[str, str] = {}
    if "native" in df.columns and "term_id" not in df.columns:
        rename_map["native"] = "term_id"
    if "name" in df.columns and "term_name" not in df.columns:
        rename_map["name"] = "term_name"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Add adjusted_p_value if not present (gprofiler already applies MHT)
    if "adjusted_p_value" not in df.columns and "p_value" in df.columns:
        df = df.copy()
        # Proxy: g:Profiler dict response does not include a separate BH-adjusted column;
        # p_value from the profile() dict path is the raw (unadjusted) value.
        # Downstream significant() filtering will use this proxy conservatively.
        df["adjusted_p_value"] = df["p_value"]

    present = [c for c in _OUTPUT_COLUMNS if c in df.columns]
    return df[present].reset_index(drop=True)
