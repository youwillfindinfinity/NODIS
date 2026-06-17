"""
Analyse DoF correction benchmark results from results/dof_sweep/.

Usage
-----
    python scripts/analyse_dof.py --results results/dof_sweep/

Prints a verdict table comparing AUPR, F1, MCC with and without DoF
correction across all topologies and configs, and FPR from null-graph runs.

Decision rule: if DoF correction improves mean AUPR by > 0.02 (2 pp) across
all configs → keep prominent in methods section; otherwise relegate to
supplementary note.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd


THRESHOLD_PP = 0.02   # 2 percentage-point improvement threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/dof_sweep/",
                        help="Directory containing result CSV files.")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results)
    csvs = list(results_dir.glob("results_*.csv"))
    if not csvs:
        print(f"No result CSVs found in {results_dir}")
        return

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)

    # ----------------------------------------------------------------
    # DoF correction comparison (structured topologies only)
    # ----------------------------------------------------------------
    dof_df = df[~df["null_graph"].fillna(False)]

    metrics = ["aupr", "f1", "mcc"]
    print("\n=== DoF correction: mean metrics (with vs without) ===\n")
    print(f"{'topology':<12} {'config':<12} {'metric':<8} {'no_dof':>8} {'dof':>8} {'delta':>8}")
    print("-" * 60)

    deltas = []
    for (topo, cfg), grp in dof_df.groupby(["topology", "config"]):
        no_dof = grp[~grp["dof_correction"].fillna(False)]
        with_dof = grp[grp["dof_correction"].fillna(False)]
        if no_dof.empty or with_dof.empty:
            continue
        for m in metrics:
            nd = no_dof[m].mean()
            wd = with_dof[m].mean()
            delta = wd - nd
            if m == "aupr":
                deltas.append(delta)
            print(f"{topo:<12} {cfg:<12} {m:<8} {nd:>8.4f} {wd:>8.4f} {delta:>+8.4f}")

    mean_aupr_delta = np.mean(deltas) if deltas else float("nan")
    print(f"\nMean AUPR delta (DoF on - off): {mean_aupr_delta:+.4f}")

    if mean_aupr_delta > THRESHOLD_PP:
        verdict = "KEEP PROMINENT — DoF correction improves AUPR by > 2 pp on average."
    elif mean_aupr_delta > 0:
        verdict = "BORDERLINE — small positive effect; consider keeping in methods with caveat."
    else:
        verdict = "RELEGATE TO SUPPLEMENTARY — DoF correction shows no consistent benefit."

    print(f"\nVerdict: {verdict}\n")

    # ----------------------------------------------------------------
    # Null-graph FPR calibration
    # ----------------------------------------------------------------
    null_df = df[df["null_graph"].fillna(False)]
    if not null_df.empty:
        print("=== Null-graph FPR calibration ===\n")
        print(f"{'config':<12} {'alpha':>6} {'n_runs':>7} {'mean_FPR':>10} {'expected':>10}")
        print("-" * 50)
        alpha = 0.05
        for cfg, grp in null_df.groupby("config"):
            n, p = grp[["n", "p"]].iloc[0]
            n_pairs = p * (p - 1) // 2
            fpr = (grp["fp"] / n_pairs).mean() if "fp" in grp.columns else float("nan")
            print(f"{cfg:<12} {alpha:>6.2f} {len(grp):>7} {fpr:>10.4f} {alpha:>10.2f}")
    else:
        print("No null-graph results found yet.")


if __name__ == "__main__":
    main()
