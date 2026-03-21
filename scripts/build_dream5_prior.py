"""
Build STRING v12 prior matrices for DREAM5 E. coli benchmarks.

Downloads STRING v12 protein links and info for E. coli (taxon 511145),
filters by experimental score >= threshold, then builds (p×p) prior
matrices for each DREAM5 gene subset (top-p by variance).

Output
------
results/dream5/prior_string_p{p}.npy          — (p,p) float32 binary prior
results/dream5/prior_string_p{p}_genes.txt    — gene names (one per line)

Usage
-----
python scripts/build_dream5_prior.py \\
    --data-dir  data/dream5/ \\
    --out-dir   results/dream5/ \\
    --threshold 400
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import subprocess

import numpy as np
import pandas as pd


TAXON = "511145"
_STRING_BASE = "https://stringdb-downloads.org/download"
STRING_LINKS_URL = (
    f"{_STRING_BASE}/protein.links.full.v12.0/{TAXON}.protein.links.full.v12.0.txt.gz"
)
STRING_INFO_URL = (
    f"{_STRING_BASE}/protein.info.v12.0/{TAXON}.protein.info.v12.0.txt.gz"
)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_if_missing(url: str, dest: pathlib.Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [cached] {dest.name}")
        return
    print(f"  Downloading {dest.name} ...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-L", "-A", "Mozilla/5.0", "--progress-bar", "-o", str(dest), url],
        check=False,
    )
    if result.returncode != 0 or dest.stat().st_size < 1000:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}")
    print(f"  Saved → {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# STRING parsers
# ---------------------------------------------------------------------------

def _load_string_info(path: pathlib.Path) -> dict[str, str]:
    """Return mapping string_protein_id → preferred_name (gene symbol)."""
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t")
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    return dict(zip(df["string_protein_id"], df["preferred_name"]))


def _load_string_edges(
    path: pathlib.Path,
    id_to_name: dict[str, str],
    threshold: int,
) -> set[tuple[str, str]]:
    """
    Return canonical (gene_a, gene_b) pairs with experimental score >=
    threshold.  Pair is stored as (min, max) for easy lookup.
    """
    edges: set[tuple[str, str]] = set()
    with gzip.open(path, "rt") as fh:
        raw_header = fh.readline().rstrip().split()
        header = [c.lstrip("#").strip() for c in raw_header]

        p1_col = header.index("protein1")
        p2_col = header.index("protein2")

        # STRING column is "experiments" in v12; fall back to "experimental"
        exp_col: int | None = None
        for candidate in ("experiments", "experimental"):
            if candidate in header:
                exp_col = header.index(candidate)
                break
        if exp_col is None:
            raise ValueError(
                f"No experimental-score column found in STRING links file.\n"
                f"Available columns: {header}"
            )

        for line in fh:
            parts = line.rstrip().split()
            if int(parts[exp_col]) < threshold:
                continue
            g1 = id_to_name.get(parts[p1_col])
            g2 = id_to_name.get(parts[p2_col])
            if g1 is None or g2 is None or g1 == g2:
                continue
            edges.add((min(g1, g2), max(g1, g2)))

    return edges


# ---------------------------------------------------------------------------
# Prior matrix builder
# ---------------------------------------------------------------------------

def _build_prior(genes: list[str], edges: set[tuple[str, str]]) -> np.ndarray:
    """Return symmetric binary (p,p) float32 prior matrix."""
    p = len(genes)
    idx = {g: i for i, g in enumerate(genes)}
    P = np.zeros((p, p), dtype=np.float32)
    for g1, g2 in edges:
        i = idx.get(g1)
        j = idx.get(g2)
        if i is not None and j is not None:
            P[i, j] = 1.0
            P[j, i] = 1.0
    return P


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build STRING v12 prior matrices for DREAM5."
    )
    parser.add_argument(
        "--data-dir", default="data/dream5/",
        help="Directory containing DREAM5 expression files.",
    )
    parser.add_argument(
        "--string-dir", default=None,
        help="Directory for caching STRING files (default: data/string/).",
    )
    parser.add_argument("--out-dir",   default="results/dream5/")
    parser.add_argument(
        "--network", type=int, default=1, choices=[1, 3],
        help="DREAM5 network (1 = E. coli in silico, 3 = E. coli in vivo).",
    )
    parser.add_argument(
        "--threshold", type=int, default=400,
        help="Minimum STRING experimental score (0–1000, default 400).",
    )
    parser.add_argument(
        "--p-values", nargs="+", type=int, default=[200, 500, 1000], metavar="P",
        help="Gene subset sizes to build priors for (default: 200 500 1000).",
    )
    args = parser.parse_args()

    data_dir   = pathlib.Path(args.data_dir)
    string_dir = pathlib.Path(args.string_dir) if args.string_dir else pathlib.Path("data/string/")
    out_dir    = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Download STRING files -------------------------------------------
    links_path = string_dir / f"{TAXON}.protein.links.full.v12.0.txt.gz"
    info_path  = string_dir / f"{TAXON}.protein.info.v12.0.txt.gz"

    print("Downloading STRING files:")
    _download_if_missing(STRING_INFO_URL,  info_path)
    _download_if_missing(STRING_LINKS_URL, links_path)

    # ---- Parse STRING data -----------------------------------------------
    print("\nParsing STRING protein info ...")
    id_to_name = _load_string_info(info_path)
    print(f"  {len(id_to_name):,} proteins mapped to gene symbols")

    print(f"\nParsing STRING edges (experimental >= {args.threshold}) ...")
    edges = _load_string_edges(links_path, id_to_name, args.threshold)
    print(f"  {len(edges):,} edges retained")

    # ---- Load DREAM5 expression file -------------------------------------
    from nodis.simulate.loaders import load_dream5_insilico

    print(f"\nLoading DREAM5 network {args.network} expression ...")
    expr_df, _ = load_dream5_insilico(data_dir, network=args.network)
    print(f"  {expr_df.shape[0]} samples × {expr_df.shape[1]} genes")

    gene_var = expr_df.var(axis=0)

    # ---- Build gene set for STRING coverage report -----------------------
    string_gene_set: set[str] = set()
    for g1, g2 in edges:
        string_gene_set.add(g1)
        string_gene_set.add(g2)

    # ---- Build prior for each p ------------------------------------------
    print()
    for p in sorted(args.p_values):
        top_genes = gene_var.nlargest(p).index.tolist()
        coverage  = sum(1 for g in top_genes if g in string_gene_set)

        P = _build_prior(top_genes, edges)
        n_edges_prior = int(P[np.triu_indices(p, k=1)].sum())
        density = n_edges_prior / (p * (p - 1) / 2)

        out_npy   = out_dir / f"prior_string_p{p}.npy"
        out_genes = out_dir / f"prior_string_p{p}_genes.txt"

        np.save(out_npy, P)
        out_genes.write_text("\n".join(top_genes) + "\n")

        print(
            f"  p={p}: {coverage}/{p} genes in STRING, "
            f"{n_edges_prior} prior edges, density={density:.4f}"
        )
        print(f"    → {out_npy}")
        print(f"    → {out_genes}")

    print("\nDone.")


if __name__ == "__main__":
    main()
