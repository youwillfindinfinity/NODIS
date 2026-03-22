# Burns Network — Knockout Impact Rankings
**Date:** 2026-03-22

## Data source
**Dataset:** GSE182616 (NCBI GEO) — blood transcriptomics from burn injury patients
**Platform:** Agilent-039494 SurePrint G3 Human GE v2 8×60K microarray (GPL17077)
**Samples:** n=584 acute-phase post-injury samples
**Gene set:** 164 genes — intersection with healthy control gene list from GSE236713 (same Agilent GPL17077 platform, used to eliminate cross-platform batch effects)
**Preprocessing:** probe→gene collapse (highest-variance probe per gene symbol), Winsorisation, z-score normalisation, NPN transform

## Network
**Network:** NODIS FDR 5% adjacency (`burns_nodis_adj_fdr05.csv`)
**Genes / Edges:** 164 nodes, 804 edges (4 isolates)

## Knockout method
**Method:** Perturbative diffusion — each gene's incident edge weights scaled by `reduction=0.3`; impact = max‖S_mod(t) − S_baseline(t)‖₂ over t ∈ [0, 3.0] (100 timepoints, linear grid)
**Initial signal:** Hub-mode — unit impulse at KIAA1257 (highest-degree node, degree=24)

> **Note:** KIAA1257 is both the signal source and the top-ranked knockout. Its impact score is ~20× higher than rank 2 by construction. Re-run with a uniform initial signal for an unbiased ranking of the remaining genes.

---

| Rank | Gene | Impact (max L2) |
|-----:|------|----------------:|
| 1 | KIAA1257 | 0.405633 |
| 2 | ERMN | 0.025951 |
| 3 | KLHL21 | 0.018954 |
| 4 | CASS4 | 0.018442 |
| 5 | HAL | 0.018294 |
| 6 | TTC18 | 0.018289 |
| 7 | KCNK7 | 0.017916 |
| 8 | CCDC19 | 0.017909 |
| 9 | ARAP3 | 0.017839 |
| 10 | TIGD3 | 0.017764 |
| 11 | CD86 | 0.017710 |
| 12 | GOT1 | 0.017667 |
| 13 | KIAA0087 | 0.017666 |
| 14 | LOC100507006 | 0.017609 |
| 15 | INHBB | 0.017579 |
| 16 | VNN3 | 0.017279 |
| 17 | SNORA21 | 0.017212 |
| 18 | KIAA0319L | 0.017008 |
| 19 | RAF1 | 0.016902 |
| 20 | RELL1 | 0.016626 |
| 21 | UBN1 | 0.016170 |
| 22 | MME | 0.016130 |
| 23 | RPGRIP1 | 0.015914 |
| 24 | EMR3 | 0.015590 |
| 25 | LRWD1 | 0.015556 |
| 26 | RUNX2 | 0.006677 |
| 27 | TGM3 | 0.005516 |
| 28 | RTN1 | 0.004898 |
| 29 | ISY1-RAB43 | 0.004830 |
| 30 | CCDC153 | 0.004801 |
| 31 | PLEKHG3 | 0.004799 |
| 32 | MIRLET7BHG | 0.004763 |
| 33 | LRRC6 | 0.004557 |
| 34 | PPP1R12B | 0.004519 |
| 35 | MID1IP1 | 0.004493 |
| 36 | GLI3 | 0.004356 |
| 37 | IRS2 | 0.004346 |
| 38 | ITPKB-IT1 | 0.004321 |
| 39 | MADCAM1 | 0.004229 |
| 40 | CHI3L1 | 0.004143 |
| 41 | SPATA2L | 0.003801 |
| 42 | C1QA | 0.003746 |
| 43 | PPP6R1 | 0.003698 |
| 44 | SPAG9 | 0.003646 |
| 45 | RPS6KA5 | 0.003564 |
| 46 | VPS18 | 0.003518 |
| 47 | IFRD1 | 0.003499 |
| 48 | ALDH1A1 | 0.003489 |
| 49 | BTNL3 | 0.003461 |
| 50 | FOSB | 0.003438 |
| 51 | KIAA1841 | 0.003436 |
| 52 | F11R | 0.003373 |
| 53 | CD22 | 0.003358 |
| 54 | SPARCL1 | 0.003352 |
| 55 | L2HGDH | 0.003351 |
| 56 | GRAMD1C | 0.003346 |
| 57 | CPSF3L | 0.003341 |
| 58 | CCDC147 | 0.003339 |
| 59 | BTG2 | 0.003335 |
| 60 | HECW2 | 0.003303 |
| 61 | EPHB1 | 0.003293 |
| 62 | ATG2A | 0.003270 |
| 63 | PRKDC | 0.003242 |
| 64 | SCYL1 | 0.003219 |
| 65 | BBC3 | 0.003195 |
| 66 | NFASC | 0.003131 |
| 67 | KIR3DL3 | 0.003118 |
| 68 | TUFT1 | 0.003076 |
| 69 | ZNF230 | 0.003032 |
| 70 | AREG | 0.003032 |
| 71 | CR2 | 0.003025 |
| 72 | PIK3IP1 | 0.003008 |
| 73 | HMBOX1 | 0.002985 |
| 74 | PLA2G7 | 0.002979 |
| 75 | VMO1 | 0.002947 |
| 76 | ALG2 | 0.002927 |
| 77 | SLC1A3 | 0.002894 |
| 78 | TMEM184B | 0.002889 |
| 79 | SAP30 | 0.002879 |
| 80 | TIGIT | 0.002867 |
| 81 | HS3ST1 | 0.002845 |
| 82 | NUP133 | 0.002841 |
| 83 | NSMAF | 0.002833 |
| 84 | GJB4 | 0.002828 |
| 85 | PIM2 | 0.002824 |
| 86 | C1QC | 0.002810 |
| 87 | MCM9 | 0.002806 |
| 88 | PRO0471 | 0.002797 |
| 89 | LRRC2 | 0.002791 |
| 90 | MYBPH | 0.002761 |
| 91 | P2RY12 | 0.002760 |
| 92 | THRB | 0.002748 |
| 93 | CYMP | 0.002744 |
| 94 | NRG1 | 0.002741 |
| 95 | FOXK2 | 0.002726 |
| 96 | EXOC3L4 | 0.002717 |
| 97 | KALRN | 0.002712 |
| 98 | IKZF3 | 0.002709 |
| 99 | NOV | 0.002685 |
| 100 | UPK3A | 0.002678 |
| 101 | SEMA6B | 0.002673 |
| 102 | LOC100131043 | 0.002671 |
| 103 | APOA5 | 0.002647 |
| 104 | CHSY1 | 0.002608 |
| 105 | GK5 | 0.002596 |
| 106 | SYAP1 | 0.002552 |
| 107 | HPGD | 0.002547 |
| 108 | CCDC66 | 0.002513 |
| 109 | ZNF222 | 0.002500 |
| 110 | COL8A2 | 0.002488 |
| 111 | DNAJB1 | 0.002468 |
| 112 | PRDM8 | 0.002437 |
| 113 | RFC3 | 0.002407 |
| 114 | ACVRL1 | 0.002400 |
| 115 | MYOM2 | 0.002396 |
| 116 | ADAMTS2 | 0.002388 |
| 117 | CCDC151 | 0.002382 |
| 118 | DNAH12 | 0.002346 |
| 119 | C1QB | 0.002345 |
| 120 | CCDC7 | 0.002340 |
| 121 | RGS1 | 0.002310 |
| 122 | SYN2 | 0.002299 |
| 123 | OXR1 | 0.002292 |
| 124 | STYXL1 | 0.002280 |
| 125 | TRIB1 | 0.002271 |
| 126 | P2RY1 | 0.002265 |
| 127 | IL10 | 0.002263 |
| 128 | S1PR5 | 0.002236 |
| 129 | BTLA | 0.002236 |
| 130 | VSIG4 | 0.002229 |
| 131 | FAM107A | 0.002228 |
| 132 | NEK10 | 0.002223 |
| 133 | JOSD1 | 0.002223 |
| 134 | GPR174 | 0.002188 |
| 135 | ZNF793 | 0.002188 |
| 136 | PRL | 0.002180 |
| 137 | MINPP1 | 0.002173 |
| 138 | BTBD18 | 0.002159 |
| 139 | FGF9 | 0.002157 |
| 140 | IKBKB | 0.002154 |
| 141 | ARFGAP3 | 0.002126 |
| 142 | RGS9 | 0.002117 |
| 143 | BTC | 0.002092 |
| 144 | DUSP2 | 0.002092 |
| 145 | CREM | 0.002085 |
| 146 | STMN2 | 0.002080 |
| 147 | DRD5 | 0.001980 |
| 148 | COL6A6 | 0.001959 |
| 149 | KLF8 | 0.001925 |
| 150 | KIR3DL1 | 0.001919 |
| 151 | MALAT1 | 0.001918 |
| 152 | SYT1 | 0.001846 |
| 153 | LOC100507616 | 0.001796 |
| 154 | KIR2DL2 | 0.001777 |
| 155 | KIR2DL5A | 0.001605 |
| 156 | KIR2DS4 | 0.001443 |
| 157 | SPHKAP | 0.001275 |
| 158 | MUC13 | 0.000700 |
| 159 | LOC100506403 | 0.000327 |
| 160 | XG | 0.000131 |
| 161 | KRTAP4-6 | 0.000000 |
| 162 | PYGO1 | 0.000000 |
| 163 | SAG | 0.000000 |
| 164 | LRRC31 | 0.000000 |
