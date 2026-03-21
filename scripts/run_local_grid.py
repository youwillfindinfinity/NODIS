"""
run_local_grid.py — parallel local launcher for a single config.

Usage:
    python scripts/run_local_grid.py --config n1026p328 --workers 14 \
        --methods desparsified gglasso glasso
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

TOPOLOGIES = ["hub", "scale-free", "cluster", "random"]
REPS = list(range(50))

PYEXE = str(Path(__file__).parent.parent.parent / ".venv" / "bin" / "python")
BENCH = str(Path(__file__).parent.parent / "benchmarks" / "run_synthetic.py")
RESULTS = str(Path(__file__).parent.parent / "results")


def run_one(topology, config, method, rep):
    out_dir = f"{RESULTS}/{method}/synthetic/"
    cmd = [
        PYEXE, BENCH,
        "--topology", topology,
        "--config", config,
        "--method", method,
        "--rep", str(rep),
        "--out", out_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    line = (result.stdout + result.stderr).strip().splitlines()[-1] if (result.stdout + result.stderr).strip() else ""
    return topology, config, method, rep, result.returncode, line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="n1026p328")
    parser.add_argument("--methods", nargs="+",
                        default=["desparsified", "gglasso", "glasso"])
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    tasks = list(product(TOPOLOGIES, [args.config], args.methods, REPS))
    total = len(tasks)
    print(f"Launching {total} tasks across {args.workers} workers "
          f"(config={args.config}, methods={args.methods})")

    done = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(run_one, *t): t for t in tasks}
        for fut in as_completed(futs):
            topo, cfg, meth, rep, rc, msg = fut.result()
            done += 1
            tag = "✓" if rc == 0 else "✗"
            if rc != 0:
                errors += 1
            print(f"  [{done:>4}/{total}] {tag} {meth:>14}  {topo:<12}  rep{rep:03d}  {msg}",
                  flush=True)

    print(f"\nDone. {done - errors}/{total} succeeded, {errors} failed.")


if __name__ == "__main__":
    main()
