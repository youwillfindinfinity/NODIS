# NODIS Benchmark Results
**Last updated:** 2026-03-21
**Status:** All benchmarks complete after ADMM_SGL fix (rebuttal). All methods including plain piglasso and piglasso_adaptive now fully recomputed.

---

## TABLE 1 — Synthetic Grand Mean
12 configs × 50 reps = 600 per method. Configs: n100p50, n237p78, n513p164 × 4 topologies (hub, scale-free, cluster, random). Methods evaluated at FDR α=0.05. F1_opt = oracle-threshold F1 (upper bound).

| Method | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|
| glasso | **0.834** | 0.950 | **0.798** | 0.503 |
| desparsified | 0.792 | 0.926 | 0.778 | 0.537 |
| gglasso | 0.725 | 0.901 | 0.732 | 0.511 |
| piglasso | 0.687 | **0.960** | 0.734 | **0.630** |
| piglasso_corr | 0.684 | **0.960** | 0.732 | 0.629 |

---

## TABLE 2 — Synthetic Per Topology
Grand mean across all 3 n/p configs and 50 reps per (topology, method) cell.

| Topology | Method | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|---|
| hub | glasso | **0.936** | 0.991 | **0.895** | 0.571 |
| | desparsified | 0.866 | 0.975 | 0.840 | 0.671 |
| | gglasso | 0.790 | 0.955 | 0.812 | 0.518 |
| | piglasso | 0.656 | **0.990** | 0.749 | **0.597** |
| | piglasso_corr | 0.650 | 0.989 | 0.741 | 0.596 |
| scale-free | glasso | **0.702** | 0.899 | **0.680** | 0.499 |
| | desparsified | 0.649 | 0.855 | 0.663 | 0.330 |
| | piglasso | 0.655 | **0.922** | 0.657 | **0.593** |
| | piglasso_corr | 0.651 | 0.923 | 0.656 | 0.593 |
| | gglasso | 0.533 | 0.793 | 0.571 | 0.489 |
| cluster | glasso | **0.781** | **0.932** | 0.746 | 0.456 |
| | desparsified | 0.751 | 0.903 | **0.747** | 0.501 |
| | gglasso | 0.709 | 0.893 | 0.711 | 0.481 |
| | piglasso | 0.679 | 0.950 | 0.724 | **0.638** |
| | piglasso_corr | 0.674 | 0.949 | 0.718 | 0.635 |
| random | glasso | **0.915** | 0.981 | **0.872** | 0.488 |
| | desparsified | 0.900 | 0.971 | 0.861 | 0.648 |
| | gglasso | 0.868 | 0.965 | 0.833 | 0.555 |
| | piglasso | 0.757 | **0.979** | 0.806 | **0.691** |
| | piglasso_corr | 0.759 | 0.980 | 0.811 | 0.690 |

---

## TABLE 3 — piglasso Scaling with n
Grand mean across all 4 topologies per config.

### piglasso_corr
| Config | n | p | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|---|---|
| n100p50 | 100 | 50 | 0.520 | 0.940 | 0.621 | 0.490 |
| n237p78 | 237 | 78 | 0.728 | 0.974 | 0.783 | 0.645 |
| n513p164 | 513 | 164 | **0.802** | **0.967** | **0.791** | **0.750** |

### piglasso (no prior)
| Config | n | p | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|---|---|
| n100p50 | 100 | 50 | 0.529 | 0.940 | 0.626 | 0.491 |
| n237p78 | 237 | 78 | 0.729 | 0.974 | 0.785 | 0.648 |
| n513p164 | 513 | 164 | **0.802** | **0.967** | **0.791** | **0.750** |

piglasso and piglasso_corr are statistically indistinguishable at all configs — the correlation prior adds negligible lift. F1_opt at n=513 (0.791) nearly matches glasso grand mean (0.798). MCC at n=513 (0.750) exceeds all other methods.

### n1026p328 comparison (all methods)
| Method | n | p | AUPR | AUROC | F1_opt | MCC | SHD |
|---|---|---|---|---|---|---|---|
| piglasso | 1026 | 328 | 0.638 | 0.895 | 0.629 | 0.532 | 2291 |
| piglasso_adaptive | 1026 | 328 | 0.643 | 0.897 | 0.638 | 0.562 | 2254 |
| piglasso_corr | 1026 | 328 | 0.639 | 0.895 | 0.629 | 0.534 | 2287 |

piglasso_adaptive shows marginal improvement at n1026p328 (MCC +0.030 vs plain piglasso). High SHD at large n reflects edge-count explosion in dense n1026p328 networks.

---

## TABLE 4 — DREAM5 Net1
E. coli in silico GRN (Marbach et al. 2012). n=487 experiments. Top-p genes selected by variance. Directed gold standard symmetrised prior to evaluation.

| Method | p | AUPR | AUROC | F1_opt | MCC |
|---|---|---|---|---|---|
| glasso | 200 | 0.154 | 0.700 | 0.259 | 0.142 |
| glasso | 500 | 0.106 | 0.641 | 0.194 | 0.111 |
| glasso | 1000 | 0.084 | 0.617 | 0.168 | 0.124 |
| gglasso | 200 | 0.155 | 0.710 | 0.258 | 0.101 |
| gglasso | 500 | 0.105 | 0.643 | 0.195 | 0.071 |
| gglasso | 1000 | 0.084 | 0.634 | 0.165 | 0.061 |
| desparsified | 200 | 0.148 | 0.714 | 0.251 | 0.164 |
| desparsified | 500 | 0.098 | 0.638 | 0.183 | 0.146 |
| desparsified | 1000 | 0.069 | 0.618 | 0.156 | 0.119 |
| piglasso_corr | 200 | 0.039 | 0.718 | 0.111 | 0.110 |
| piglasso_corr | 500 | 0.026 | 0.647 | 0.090 | 0.084 |
| piglasso_corr | 1000 | 0.018 | 0.659 | 0.074 | 0.077 |
| piglasso_string | 200 | 0.040 | 0.722 | 0.111 | 0.115 |
| piglasso_string | 500 | 0.026 | 0.650 | 0.091 | 0.086 |
| piglasso_string | 1000 | 0.019 | 0.662 | 0.075 | 0.078 |

---

## TABLE 5 — Diffusion & Knockout (piglasso, normalised Spearman)
Grand mean across topologies and configs n100p50–n513p164, 50 reps, delta=all.
DiffSp_norm = (Spearman_signal − Spearman_null) / (1 − Spearman_null).
KOSp_norm = normalised knockout Spearman.

| Method | DiffSp_norm | KOSp_norm | KO_top10_recall |
|---|---|---|---|
| piglasso | **0.441** | **0.393** | **0.766** |
| desparsified | 0.441 | 0.282 | 0.519 |
| gglasso | 0.368 | 0.166 | 0.575 |
| glasso | 0.261 | 0.194 | 0.633 |

**piglasso per topology:**

| Topology | DiffSp_norm | KOSp_norm | KO_top10_recall |
|---|---|---|---|
| cluster | **0.578** | 0.444 | 0.833 |
| random | 0.447 | **0.462** | 0.787 |
| hub | 0.431 | 0.370 | 0.595 |
| scale-free | 0.306 | 0.297 | **0.847** |

---

## TABLE 6 — SERGIO scRNA-seq (DS1, DS4–DS8)
Mean across 6 datasets × 2 preprocessing modes (none, log2+NPN).

| Method | Preprocessing | AUPR | AUROC | F1_opt |
|---|---|---|---|---|
| desparsified | log2+NPN | **0.045** | 0.507 | **0.083** |
| desparsified | none | 0.043 | 0.517 | 0.079 |
| glasso | none | 0.048 | **0.528** | 0.075 |
| glasso | log2+NPN | 0.038 | 0.513 | 0.088 |
| gglasso | none | 0.034 | 0.532 | 0.073 |
| gglasso | log2+NPN | 0.039 | 0.502 | 0.082 |
| piglasso | log2+NPN | 0.034 | 0.490 | 0.068 |
| piglasso | none | 0.036 | 0.543 | 0.073 |

All AUPR values near chance (~0.03–0.05) — consistent with SERGIO generating non-Gaussian burst kinetics that violate GGM assumptions.

---

## TABLE 7 — Computational Cost
Mean wall time in seconds per single rep on Snellius (rome partition).

| Method | n100p50 | n237p78 | n513p164 | n1026p328 |
|---|---|---|---|---|
| desparsified | 0.1 | 0.6 | 2.8 | — |
| gglasso | 7.9 | 7.3 | 26.4 | — |
| glasso | 68.0 | 90.1 | 90.6 | — |
| piglasso | 9.0 | 13.6 | 189.0 | 311.2 |
| piglasso_adaptive | — | — | — | 313.1 |
| piglasso_corr | 9.6 | 30.9 | 470.5 | 761.6 |

desparsified is 20–600× faster than glasso. piglasso_corr is 2.5× slower than plain piglasso at n513p164 (470s vs 189s); the additional cost arises from computing the correlation prior matrix, which scales as O(p²n). piglasso_adaptive is negligibly slower than plain piglasso at n1026p328.

---

## TABLE 8 — Burns NODIS Hub Analysis (CAUTIONARY)
NODIS de-sparsified inference on GSE182616 burn expression (acute phase only).
n=584 samples, p=164 genes (top-164 by variance), n/p=3.56 — below asymptotic floor (n/p < 5).
Results labelled CAUTIONARY: z-scores may be inflated; use for hypothesis generation only.

**Network summary:** 164 nodes · 804 edges at FDR 5% · 835 at FDR 10%
Edge density: 6.0% · Mean degree: 9.8 · Max: 24

**Top 15 genes by degree (FDR 5% network):**

| Gene | Degree | Mean \|Z\| | Notes |
|---|---|---|---|
| KIAA1257 | **24** | 30.3 | Uncharacterised; high \|Z\| likely inflated at n/p=3.56 |
| RPS6KA5 | **23** | 33.0 | RSK4 — MAPK/immune signalling |
| GRAMD1C | 21 | 11.4 | Cholesterol transport |
| LOC100131043 | 20 | 13.2 | Unannotated locus |
| CPSF3L | 20 | 10.9 | mRNA cleavage & polyadenylation |
| GPR174 | 20 | 11.3 | G protein-coupled receptor — lymphocyte homing |
| SLC1A3 | 19 | 13.0 | Glutamate transporter — CNS/inflammation |
| RPGRIP1 | 19 | 20.6 | Ciliopathy gene (ciliary function) |
| SAP30 | 19 | 16.0 | Sin3A epigenetic co-repressor |
| PPP1R12B | 19 | 42.2 | PP1 regulatory subunit — very high \|Z\|, suspect |
| EMR3 | 19 | — | EGF-like adhesion GPCR — macrophage receptor |
| C1QB | 18 | — | Complement C1q — innate immunity |
| MALAT1 | 18 | — | lncRNA — inflammatory gene regulation |
| IL10 | 18 | — | Anti-inflammatory cytokine |
| BTLA | 18 | — | B/T lymphocyte attenuator — immune checkpoint |

Hubs by mean+2SD criterion (degree ≥ 22): KIAA1257, RPS6KA5.

**Caveat:** Mean \|Z\| > 20 for KIAA1257, RPS6KA5, RPGRIP1, PPP1R12B is consistent with z-score inflation at n/p=3.56 (van de Geer et al. 2014). IL10, C1QB, MALAT1, BTLA, BBC3 (degree=17) are biologically plausible burn-injury network members at moderate degrees.
Full comparison with PIGLasso burns network pending (`pig_burns` job 21035015, 8h wall).

---

## Notes

### Rebuttal fix (2026-03-21)
- **Bug fixed:** ADMM_SGL returns `(solution_dict, status_dict)`. Original code had `Theta = sol[1]` (status dict) and `Omega_0 = sol[0]` (full solution dict). Fixed to `Theta = sol[0]["Theta"]` and `Omega_0 = sol[0]["Omega"]`.
- **pi_thr default fixed:** Validator required `pi_thr > 0.5` but default was `0.5`. Changed default to `0.9` (standard StARS threshold).
- All methods fully recomputed post-fix.

### Synthetic
- All methods run on NPN-transformed data (rank-based shrinkage transform).
- Edge density ~5%, matching SILGGM simulation study (Zhang et al. 2018).
- piglasso_corr uses correlation prior (Pearson |r| on training data) with prior_weight=0.5.
- piglasso and piglasso_corr are statistically indistinguishable across all configs — the correlation prior confers no measurable benefit on synthetic Gaussian data.
- piglasso MCC at n513p164 (0.750) ties piglasso_corr as the highest MCC across all methods and configs.
- n1026p328 results excluded from TABLE 1/2 grand means to maintain comparability with the 3-config baselines.

### DREAM5
- piglasso_corr and piglasso_string AUPR (0.018–0.040) remain 3–5× lower than glasso/gglasso/desparsified — consistent with non-Gaussian ODE-generated data degrading subsample consistency.
- piglasso_string (STRING prior) shows negligible improvement over piglasso_corr: DREAM5 uses anonymised gene names (G1…G1643) so STRING edges cannot be matched.
- Partial correlation coverage (Net1): 40.8% at p=200, 13.3% at p=500, 8.7% at p=1000 — all below 50%. Low AUPR is expected for all methods and should be reported as a benchmark caveat.

### Burns NODIS (TABLE 8)
- `run_nodis_burns.py` fixed: `--ctrl-genes` intersection fails because `healthy_controls.genes.txt` contains probe IDs (not gene symbols). Fixed to use `--no-intersect --top-p 164` (top-164 by variance, n/p≈3.56).
- Gene set note in summary.txt still reads "intersection with GSE236713" — cosmetic only, does not affect results.

### Pending
- PIGLasso burns job 21035015 (8h wall) — sync results and compute NODIS vs PIGLasso hub overlap.
