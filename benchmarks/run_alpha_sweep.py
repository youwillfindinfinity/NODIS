"""
run_alpha_sweep.py — parallel local runner for the prior-weight (alpha) sweep.

Sweeps alpha in {0.1, 0.2, ..., 0.9} using a perfect oracle prior (0% noise)
to isolate the effect of alpha from prior quality. Runs all 4 topologies and
20 reps per (alpha, topology) combination at config n513p164 (closest to the
actual GSE182616 dataset size).

Results are saved to results/alpha_sweep/a<NN>/ where NN = alpha * 10.
The SSGLasso baseline (no prior) is also run for direct comparison.

Usage:
    cd NODIS/
    python benchmarks/run_alpha_sweep.py
    python benchmarks/run_alpha_sweep.py --workers 8 --reps 20
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

TOPOLOGIES  = ["scale-free", "hub", "cluster", "random"]
CONFIG      = "n513p164"
ALPHA_VALS  = [round(a / 10, 1) for a in range(1, 10)]  # 0.1 … 0.9
DEFAULT_REPS = 20

PYEXE = str(Path(__file__).parent.parent.parent / ".venv" / "bin" / "python")
BENCH = str(Path(__file__).parent / "run_synthetic.py")
BASE_RESULTS = str(Path(__file__).parent.parent / "results" / "alpha_sweep")


def alpha_tag(alpha: float) -> str:
    return f"a{round(alpha * 10):02d}"


def out_dir(alpha: float) -> str:
    return str(Path(BASE_RESULTS) / alpha_tag(alpha))


def run_one(topology: str, alpha: float, rep: int, n_jobs: int) -> tuple:
    cmd = [
        PYEXE, BENCH,
        "--topology",    topology,
        "--config",      CONFIG,
        "--method",      "piglasso_oracle_n00",
        "--prior-weight", str(alpha),
        "--rep",         str(rep),
        "--n-jobs",      str(n_jobs),
        "--out",         out_dir(alpha),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    last = (result.stdout + result.stderr).strip().splitlines()[-1] if \
           (result.stdout + result.stderr).strip() else ""
    return topology, alpha, rep, result.returncode, last


def run_baseline(topology: str, rep: int, n_jobs: int) -> tuple:
    """SSGLasso (no prior) at the same config for the baseline reference."""
    cmd = [
        PYEXE, BENCH,
        "--topology", topology,
        "--config",   CONFIG,
        "--method",   "ssglasso",
        "--rep",      str(rep),
        "--n-jobs",   str(n_jobs),
        "--out",      str(Path(BASE_RESULTS) / "baseline"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    last = (result.stdout + result.stderr).strip().splitlines()[-1] if \
           (result.stdout + result.stderr).strip() else ""
    return topology, "baseline", rep, result.returncode, last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel worker processes")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="Repetitions per (alpha, topology) cell")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="n_jobs forwarded to PIGLassoEstimator")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip SSGLasso baseline runs")
    args = parser.parse_args()

    reps = list(range(args.reps))

    tasks = [(t, a, r, args.n_jobs)
             for t, a, r in product(TOPOLOGIES, ALPHA_VALS, reps)]

    if not args.skip_baseline:
        baseline_tasks = [(t, r, args.n_jobs)
                          for t, r in product(TOPOLOGIES, reps)]
    else:
        baseline_tasks = []

    total = len(tasks) + len(baseline_tasks)
    print(f"[ALPHA SWEEP] {len(ALPHA_VALS)} alpha values × "
          f"{len(TOPOLOGIES)} topologies × {args.reps} reps = {len(tasks)} runs")
    print(f"[ALPHA SWEEP] + {len(baseline_tasks)} SSGLasso baseline runs")
    print(f"[ALPHA SWEEP] Total: {total} jobs | workers: {args.workers}")

    done = errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *t): t for t in tasks}
        futures.update({pool.submit(run_baseline, *t): t for t in baseline_tasks})

        for fut in as_completed(futures):
            try:
                *key, rc, msg = fut.result()
                done += 1
                status = "OK" if rc == 0 else f"ERR({rc})"
                print(f"[{done}/{total}] {status} {key} — {msg}")
                if rc != 0:
                    errors += 1
            except Exception as exc:
                errors += 1
                done += 1
                print(f"[{done}/{total}] EXCEPTION {futures[fut]}: {exc}")

    print(f"\n[DONE] {done} runs completed, {errors} errors.")
    print(f"Results in: {BASE_RESULTS}")


if __name__ == "__main__":
    main()
