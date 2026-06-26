# NODIS Reproduction Guide

This document provides exact shell commands to reproduce the main benchmark
figures and metrics table from the NODIS manuscript.

## Requirements

```bash
pip install nodis[all]          # includes gglasso, numba, joblib, click
# or from source:
pip install -e ".[dev]"
```

Pinned dependency versions are in `requirements.txt` (CPU-only) or
`environment.yml` (conda, includes R optional dependencies).

## Figure 1 and MCC table (main benchmark)

Reproduces Figure 1 (benchmark_comparison.pdf) and Table 2 (MCC/AUPR rankings)
via 4,000 simulation runs across 4 topologies × 4 (n,p) configurations × 5 methods.

**Local run (moderate size, ~2–4 h on 8 cores):**

```bash
# Run the three moderate-dimension configurations (all methods)
for method in desparsified glasso gglasso ssglasso piglasso_oracle_n02; do
  for topology in hub scale-free cluster random; do
    for rep in $(seq 0 49); do
      python3 benchmarks/run_synthetic.py \
        --topology $topology --n 513 --p 164 \
        --method $method --rep $rep --seed-offset 0
    done
  done
done

# Collect results and rebuild metrics summary
python3 scripts/rebuild_metrics_summary.py

# Regenerate Figure 1
python3 scripts/plot_benchmark_comparison.py
```

**Snellius HPC (full grid, SLURM array, ~30 min wall time):**

```bash
sbatch jobs/synthetic_array.job
# After completion:
bash scripts/sync_results_back.sh
python3 scripts/rebuild_metrics_summary.py
python3 scripts/plot_benchmark_comparison.py
```

Random seeds are deterministic per `(seed_offset, rep, topology)` combination
as documented in `benchmarks/run_synthetic.py` (line 106):
```
seed = seed_offset * 100_000 + rep * 1000 + hash(topology) % 1000
```

## Supplementary calibration figure (Figure S_calib)

```bash
python3 scripts/calibration_figure.py --n-reps 100 --seed 42
# Output: paper/Fig/calibration.pdf
```

## Supplementary concordance figure (Figure S_concordance)

```bash
python3 scripts/concordance_analytical.py --n-reps 20 --seed 42
# Output: paper/Fig/concordance.pdf
```

## DREAM5 benchmark (Supplementary Table S1)

```bash
# Download DREAM5 data (requires Synapse account):
# syn3130840 -> data/dream5/
python3 benchmarks/run_dream5.py --network 1 --p 200 --method desparsified
python3 benchmarks/run_dream5.py --network 1 --p 500 --method desparsified
python3 benchmarks/run_dream5.py --network 1 --p 1000 --method desparsified
# Repeat for all methods; then:
python3 scripts/plot_fig5_dream5.py
```

## Full test suite

```bash
python3 -m pytest tests/ -v --tb=short
# Expected: 453 tests collected; 432 pass without optional Numba dependency
```

## Hardware reference

- Main benchmark figures: Snellius HPC (AMD Rome, 128 cores, 256 GB RAM)
- Timing table (Table S_timing): large configs on Snellius; small configs local
  (8-core Intel, 32 GB RAM, numpy 1.26, sklearn 1.4, Python 3.10)
- Calibration and concordance figures: local workstation

## Dependency versions

See `requirements.txt` for pinned CPU-only dependencies.
Key versions used for manuscript results:
- Python 3.10
- numpy 1.26
- scipy 1.13
- scikit-learn 1.4
- joblib 1.4
- gglasso 0.3
- networkx 3.3
