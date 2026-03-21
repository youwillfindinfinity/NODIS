# NODIS Benchmark Results
**Last updated:** 2026-03-20
**Status:** Synthetic complete (600 reps/method). DREAM5 mostly complete — piglasso p=1000 and gglasso p=1000 pending.

---

## TABLE 1 — Synthetic Grand Mean
12 configs × 50 reps = 600 per method. Configs: n100p50, n237p78, n513p164 × 4 topologies (hub, scale-free, cluster, random). Methods evaluated at FDR α=0.05. F1_opt = oracle-threshold F1 (upper bound).

| Method | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|
| glasso | **0.834** | 0.950 | **0.798** | 0.503 |
| desparsified | 0.792 | 0.926 | 0.778 | **0.537** |
| gglasso | 0.725 | 0.901 | 0.732 | 0.511 |
| piglasso | 0.685 | **0.961** | 0.730 | 0.368 |

---

## TABLE 2 — Synthetic Per Topology
Grand mean across all 3 n/p configs and 50 reps per (topology, method) cell.

| Topology | Method | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|---|
| hub | glasso | **0.936** | 0.991 | **0.895** | 0.571 |
| | desparsified | 0.866 | 0.975 | 0.840 | **0.671** |
| | gglasso | 0.790 | 0.955 | 0.812 | 0.518 |
| | piglasso | 0.641 | **0.988** | 0.733 | 0.335 |
| scale-free | glasso | **0.702** | 0.899 | **0.680** | 0.499 |
| | piglasso | 0.657 | **0.925** | 0.660 | **0.458** |
| | desparsified | 0.649 | 0.855 | 0.663 | 0.330 |
| | gglasso | 0.533 | 0.793 | 0.571 | 0.489 |
| cluster | glasso | **0.781** | **0.932** | 0.746 | 0.456 |
| | desparsified | 0.751 | 0.903 | **0.747** | **0.501** |
| | gglasso | 0.709 | 0.893 | 0.711 | 0.481 |
| | piglasso | 0.679 | 0.950 | 0.720 | 0.342 |
| random | glasso | **0.915** | 0.981 | **0.872** | 0.488 |
| | desparsified | 0.900 | 0.971 | 0.861 | **0.648** |
| | gglasso | 0.868 | 0.965 | 0.833 | 0.555 |
| | piglasso | 0.762 | **0.981** | 0.809 | 0.339 |

---

## TABLE 3 — PIGLASSO Scaling with n
Grand mean across all 4 topologies per config.

| Config | n | p | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|---|---|
| n100p50 | 100 | 50 | 0.517 | 0.942 | 0.618 | 0.179 |
| n237p78 | 237 | 78 | 0.726 | 0.974 | 0.777 | 0.332 |
| n513p164 | 513 | 164 | **0.810** | **0.968** | **0.796** | **0.594** |

PIGLASSO F1_opt at n=513 (0.796) nearly matches glasso grand mean (0.798), demonstrating convergence at large n.

---

## TABLE 4 — DREAM5 Net1
E. coli in silico GRN (Marbach et al. 2012). n=487 experiments. Top-p genes selected by variance. Directed gold standard symmetrised prior to evaluation (see Methods). p=1000 piglasso and gglasso pending.

| Method | p | AUPR | AUROC | F1_opt | MCC | Status |
|---|---|---|---|---|---|---|
| glasso | 200 | 0.154 | 0.700 | 0.259 | 0.142 | ✓ |
| glasso | 500 | 0.106 | 0.641 | 0.194 | 0.111 | ✓ |
| glasso | 1000 | 0.084 | 0.617 | 0.168 | 0.124 | ✓ |
| gglasso | 200 | 0.155 | 0.710 | 0.258 | 0.101 | ✓ |
| gglasso | 500 | 0.105 | 0.643 | 0.195 | 0.071 | ✓ |
| gglasso | 1000 | — | — | — | — | not submitted |
| desparsified | 200 | 0.148 | 0.714 | 0.251 | 0.164 | ✓ |
| desparsified | 500 | 0.098 | 0.638 | 0.183 | 0.146 | ✓ |
| desparsified | 1000 | 0.069 | 0.618 | 0.156 | 0.119 | ✓ |
| piglasso | 200 | 0.039 | 0.714 | 0.111 | 0.078 | ✓ |
| piglasso | 500 | 0.027 | 0.642 | 0.097 | 0.055 | ✓ |
| piglasso | 1000 | — | — | — | — | job 21008764 running |

---

## TABLE 5 — Computational Cost
Mean wall time in seconds per single rep on Snellius (rome partition).

| Method | n100p50 | n237p78 | n513p164 |
|---|---|---|---|
| desparsified | 0 | 1 | 3 |
| gglasso | 8 | 7 | 26 |
| glasso | 68 | 90 | 91 |
| piglasso | 155 | 503 | 1700 |

desparsified is 20–600× faster than glasso at equivalent p. PIGLASSO is ~19× slower than glasso at n513p164 due to Q=200 subsampled GLasso fits across 20 λ values.

---

## Notes

### Synthetic
- All methods run on NPN-transformed data (rank-based shrinkage transform).
- Edge density ~5%, matching SILGGM simulation study (Zhang et al. 2018).
- PIGLASSO run with `lambda_wp=0` (no biological prior active) — stability-selected GLasso only.
- PIGLASSO threshold calibration (knee detection) is the primary weakness at small n; F1_opt shows competitive oracle performance at n≥237.

### DREAM5
- PIGLASSO AUPR (0.039, 0.027) is 4–5× lower than all other methods — attributed to non-Gaussian ODE-generated data degrading subsample consistency, and absence of prior (`lambda_wp=0`).
- Desparsified is fastest by far (3s at p=1000 vs 1377s for glasso); PIGLASSO is the slowest (14535s at p=500).
- MCC overflow bug in `evaluate.py` fixed (cast to `np.float64` before sqrt) — affected p=1000 piglasso DREAM5 run only.

### Pending
- `gglasso` DREAM5 p=1000: not yet submitted.
- `piglasso` DREAM5 p=1000: job 21008764 running, ~18h wall time remaining.
- Diffusion & knockout benchmark: design spec complete (`docs/superpowers/specs/2026-03-20-diffusion-knockout-benchmark-design.md`), implementation plan not yet written.
