"""
Rebuild results/metrics_summary.csv from all per-method result directories.
Extracts seed_offset from filenames (_s01, _s02, etc); existing files without
a seed tag are treated as seed_offset=0.

Usage (from NODIS/ root):
    python scripts/rebuild_metrics_summary.py
"""
import pathlib
import re
import pandas as pd

NODIS_DIR = pathlib.Path(__file__).resolve().parents[1]
results = NODIS_DIR / "results"
summary_path = results / "metrics_summary.csv"

SEED_RE = re.compile(r"_s(\d{2})_rep")


def load_csv_with_seed(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "seed_offset" not in df.columns:
        m = SEED_RE.search(path.stem)
        df["seed_offset"] = int(m.group(1)) if m else 0
    return df


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
    df = pd.concat([load_csv_with_seed(f) for f in files], ignore_index=True)
    df["benchmark"] = source
    new_dfs.append(df)
    found_benchmarks.add(source)
    print(f"  {source}: {len(files)} files, {len(df)} rows")

if not new_dfs:
    print("No result files found.")
else:
    new_data = pd.concat(new_dfs, ignore_index=True)

    if summary_path.exists():
        existing = pd.read_csv(summary_path, low_memory=False)
        if "seed_offset" not in existing.columns:
            existing["seed_offset"] = 0

        # Only drop existing rows for (benchmark, method, config, seed_offset)
        # combinations present in new data — preserves rows for methods/configs
        # whose CSV files don't exist locally (e.g. ssglasso small3 on Snellius).
        if "method" in new_data.columns and "seed_offset" in new_data.columns:
            new_keys = set(zip(new_data["benchmark"], new_data["method"],
                               new_data["config"], new_data["seed_offset"]))
            mask_drop = existing.apply(
                lambda r: (r.get("benchmark"), r.get("method"),
                           r.get("config"), r.get("seed_offset", 0)) in new_keys,
                axis=1
            )
            keep = existing[~mask_drop]
        else:
            keep = existing[~existing["benchmark"].isin(found_benchmarks)]

        print(f"  preserving {len(keep)} existing rows not covered by new files")
        summary = pd.concat([keep, new_data], ignore_index=True)
    else:
        summary = new_data

    summary.to_csv(summary_path, index=False)
    print(f"Saved {len(summary)} total rows → {summary_path}")