"""
Rebuild results/metrics_summary.csv from all per-method result directories.

Usage (from NODIS/ root):
    python scripts/rebuild_metrics_summary.py
"""
import pathlib
import pandas as pd

NODIS_DIR = pathlib.Path(__file__).resolve().parents[1]
results = NODIS_DIR / "results"
summary_path = results / "metrics_summary.csv"

sections = {
    "synthetic": (list(results.glob("raw/results_*.csv")) +
                  list(results.glob("glasso/synthetic/results_*.csv")) +
                  list(results.glob("gglasso/synthetic/results_*.csv")) +
                  list(results.glob("desparsified/synthetic/results_*.csv")) +
                  list(results.glob("piglasso/synthetic/results_*.csv"))),
    "dream5":    list(results.glob("dream5/dream5_*.csv")),
    "diffusion": list(results.glob("diffusion/diffusion_*.csv")),
    "sergio":    list(results.glob("sergio/sergio_*.csv")),
}

new_dfs = []
found_benchmarks = set()
for source, files in sections.items():
    if not files:
        print(f"  {source}: no files found — skipping")
        continue
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["benchmark"] = source
    new_dfs.append(df)
    found_benchmarks.add(source)
    print(f"  {source}: {len(files)} files, {len(df)} rows")

if not new_dfs:
    print("No result files found.")
else:
    new_data = pd.concat(new_dfs, ignore_index=True)

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        keep = existing[~existing["benchmark"].isin(found_benchmarks)]
        print(f"  preserving {len(keep)} existing rows from untouched benchmarks")
        summary = pd.concat([keep, new_data], ignore_index=True)
    else:
        summary = new_data

    summary.to_csv(summary_path, index=False)
    print(f"Saved {len(summary)} total rows → {summary_path}")