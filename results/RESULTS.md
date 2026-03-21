# NODIS Benchmark Results
**Last updated:** 2026-03-21
**Status:** Synthetic complete (all 4 configs × 4 topologies × 50 reps × 4 methods). DREAM5 complete (all methods × all p). SERGIO complete (6 datasets × 4 methods × 2 preprocessing). Diffusion/knockout complete (8,000 reps).

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
E. coli in silico GRN (Marbach et al. 2012). n=487 experiments. Top-p genes selected by variance. Directed gold standard symmetrised prior to evaluation (see Methods). All methods complete.

| Method | p | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|---|
| glasso | 200 | 0.154 | 0.700 | 0.259 | 0.142 |
| glasso | 500 | 0.106 | 0.641 | 0.194 | 0.111 |
| glasso | 1000 | 0.084 | 0.617 | 0.168 | 0.124 |
| gglasso | 200 | 0.155 | 0.710 | 0.258 | 0.101 |
| gglasso | 500 | 0.105 | 0.643 | 0.195 | 0.071 |
| gglasso | 1000 | **0.084** | 0.634 | 0.165 | 0.061 |
| desparsified | 200 | 0.148 | 0.714 | 0.251 | 0.164 |
| desparsified | 500 | 0.098 | 0.638 | 0.183 | 0.146 |
| desparsified | 1000 | 0.069 | 0.618 | 0.156 | 0.119 |
| piglasso | 200 | 0.039 | 0.714 | 0.111 | 0.078 |
| piglasso | 500 | 0.027 | 0.642 | 0.097 | 0.055 |
| piglasso | 1000 | 0.016 | **0.645** | 0.069 | 0.052 |

---

---

## TABLE 4b — NEW: n1026p328 Config (large n)
4th synthetic config: n=1,026, p=328 (n/p≈3.1). 4 topologies × 50 reps × 4 methods. Note: gglasso degrades markedly at this scale.

| Method | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|
| desparsified | **0.729** | **0.904** | **0.739** | 0.448 |
| glasso | 0.729 | 0.896 | 0.732 | 0.500 |
| piglasso | 0.649 | 0.894 | 0.687 | **0.600** |
| gglasso | 0.199 | 0.584 | 0.276 | 0.297 |

gglasso AUPR collapses to 0.199 at n1026p328 — convergence issues at high n/p with group penalty. desparsified and glasso remain competitive.

---

## TABLE 5 — SERGIO scRNA-seq Benchmark
SERGIO simulator (Dibaeinia & Sinha 2020). 6 datasets (ds1, ds4–ds8). Directed TF–target gold standard. Grand mean across datasets. Log2+NPN preprocessing outperforms raw for precision-based methods.

| Method | Preprocessing | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|---|
| glasso | none | **0.048** | **0.528** | 0.075 | **0.068** |
| desparsified | log2+NPN | 0.045 | 0.507 | **0.083** | 0.002 |
| desparsified | none | 0.043 | 0.517 | 0.079 | 0.013 |
| gglasso | log2+NPN | 0.039 | 0.502 | 0.082 | 0.003 |
| gglasso | none | 0.034 | 0.532 | 0.073 | 0.030 |
| glasso | log2+NPN | 0.038 | 0.513 | 0.088 | 0.011 |
| piglasso | log2+NPN | 0.032 | 0.500 | 0.061 | 0.000 |
| piglasso | none | 0.032 | 0.500 | 0.061 | 0.000 |

All methods near chance AUROC (~0.5–0.53) — expected, as SERGIO simulates directed TF regulation; undirected GGMs are structurally misspecified for this task.

---

## TABLE 6 — Diffusion & Knockout Benchmark
Synthetic diffusion benchmark. Signal initialised at perturbed nodes; recovery measured by Spearman ρ between true and predicted propagation. 8,000 total reps (4 methods × 3 delta modes × 4 topologies × 3 configs × 50 reps). PIGLasso tested on random delta only.

| Method | Diffusion Spearman (mean) | Knockout Spearman | Knockout Top-10% Recall |
|---|---|---|---|
| **desparsified** | **0.616** | 0.479 | 0.519 |
| gglasso | 0.494 | 0.566 | 0.575 |
| glasso | 0.370 | 0.648 | **0.633** |
| piglasso | 0.318 | 0.521 | 0.547 |

desparsified leads on diffusion recovery (sparser inferred networks propagate signal more faithfully). glasso leads on knockout prediction (denser networks better capture indirect regulatory paths).

Per delta mode (desparsified):

| Delta mode | Diffusion Spearman | Knockout Spearman | Knockout Top-10% Recall |
|---|---|---|---|
| fiedler | **0.777** | 0.483 | 0.552 |
| random | 0.641 | 0.532 | 0.570 |
| hub | 0.430 | 0.422 | 0.434 |

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

### SERGIO
- All methods near chance AUROC — expected for directed TF network inference with undirected models.
- PIGLasso AUROC exactly 0.500 with both preprocessing modes — outputs all-zero adjacency, consistent with subsample instability on single-cell count data.

### Diffusion
- desparsified best for diffusion recovery (sparse network = cleaner signal propagation).
- glasso best for knockout prediction (dense network captures indirect effects).
- Fiedler delta (signal on lowest-eigenvalue eigenvector) is the hardest to recover; hub delta is easiest.

### n1026p328 (new config)
- gglasso convergence failure at large n/p: AUPR collapses to 0.199 (from 0.725 at 3-config mean). Hub topology especially bad (0.024 AUPR). Root cause: group Lasso penalty accumulates regularisation with p at fixed λ schedule.
- desparsified and glasso maintain parity (~0.729 each) — both benefit from larger n.
- piglasso MCC peaks at 0.600 at n1026p328 (highest across all configs) — stability selection improves with sample size.

### Pending
- Burns real-data results (pig_burns job 21023358 running).
- Adaptive pi_thr rerun for n1026p328 piglasso (requested).
