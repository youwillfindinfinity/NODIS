"""
plot_burns_network.py
---------------------
Improved burn-injury GGM network figure + animated GIF.

Static figure (burns_network.pdf/png):
  Single full-canvas network. Nodes coloured by biological category,
  sized by degree. Edge opacity scaled by |Z|-score. Top hub genes labelled.

Animated GIF (burns_network.gif):
  Network builds up edge-by-edge, sorted from strongest |Z| to weakest.
  Fixed node positions. Node size grows with degree. Each frame shows
  current edge count and |Z| threshold. Reveals hub structure dramatically.

Usage:
    cd NODIS/
    python scripts/plot_burns_network.py
    python scripts/plot_burns_network.py --no-gif   # skip GIF (faster)
    python scripts/plot_burns_network.py --frames 50
"""

import argparse
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       9,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "burns")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---------------------------------------------------------------------------
# Biological categories — node fill colour
# ---------------------------------------------------------------------------
BIO_CATEGORIES = {
    # Immune / inflammation
    "immune": {
        "color": "#E8527A",  # rose-red
        "genes": {"IL10", "C1QB", "BTLA", "EMR3", "GPR174", "C1QC", "C1QA",
                  "LAIR1", "CD163", "MS4A4A", "VSIG4", "FCGR3A"},
    },
    # Apoptosis / cell fate
    "apoptosis": {
        "color": "#FF8C42",  # orange
        "genes": {"BBC3", "PIM2", "BCL2L1", "MCL1", "CASP3", "TP53"},
    },
    # Signalling kinases
    "signalling": {
        "color": "#9B59B6",  # purple
        "genes": {"RPS6KA5", "PPP1R12B", "MAP3K1", "MAPK14"},
    },
    # RNA / epigenetic
    "epigenetic": {
        "color": "#1ABC9C",  # teal
        "genes": {"MALAT1", "SAP30", "UBN1", "CPSF3L"},
    },
    # Metabolism / transport
    "metabolic": {
        "color": "#F39C12",  # amber
        "genes": {"SLC1A3", "GRAMD1C", "ABCA1"},
    },
}

# Genes to always label (hubs + biologically notable)
ALWAYS_LABEL = {
    "KIAA1257", "RPS6KA5", "GRAMD1C", "GPR174", "SLC1A3", "SAP30",
    "PPP1R12B", "EMR3", "C1QB", "MALAT1", "IL10", "BTLA", "BBC3", "PIM2",
}


def _gene_color(gene: str) -> str:
    for cat in BIO_CATEGORIES.values():
        if gene in cat["genes"]:
            return cat["color"]
    return "#B0C4DE"   # light steel blue — uncharacterised


def load_data():
    adj = pd.read_csv(os.path.join(RESULTS_DIR, "burns_nodis_adj_fdr05.csv"),
                      index_col=0)
    z   = pd.read_csv(os.path.join(RESULTS_DIR, "burns_nodis_zscores.csv"),
                      index_col=0)
    genes  = adj.index.tolist()
    degree = adj.values.sum(axis=1).astype(int)

    # Build sorted edge list: (i, j, |Z|)
    n = len(genes)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj.iloc[i, j] == 1:
                edges.append((i, j, abs(float(z.iloc[i, j]))))
    edges.sort(key=lambda e: -e[2])   # strongest first
    return adj, z, genes, degree, edges


def _compute_layout(genes, degree, edges):
    """
    Shell layout: hub nodes in a tight inner ring, high-degree in a middle
    ring, peripherals on the outer ring. Within each ring, nodes are spread
    evenly by angle. This prevents centre collapse regardless of edge density.
    """
    n = len(genes)
    mu, sd = degree.mean(), degree.std()
    thr_hub  = mu + 2 * sd
    thr_high = mu + sd

    hubs       = [i for i in range(n) if degree[i] >= thr_hub]
    high       = [i for i in range(n) if thr_high <= degree[i] < thr_hub]
    connected  = [i for i in range(n) if mu <= degree[i] < thr_high]
    peripheral = [i for i in range(n) if degree[i] < mu]

    pos = np.zeros((n, 2))
    rng = np.random.default_rng(42)

    def _ring(indices, radius, jitter=0.0):
        k = len(indices)
        if k == 0:
            return
        # Sort by degree descending so strongest are at top
        indices = sorted(indices, key=lambda i: -degree[i])
        # Random angle offset to avoid symmetry artefacts
        offset = rng.uniform(0, 2 * np.pi)
        for rank, i in enumerate(indices):
            angle = offset + 2 * np.pi * rank / k
            r = radius + rng.uniform(-jitter, jitter)
            pos[i, 0] = r * np.cos(angle)
            pos[i, 1] = r * np.sin(angle)

    _ring(hubs,       radius=0.30, jitter=0.06)
    _ring(high,       radius=0.65, jitter=0.08)
    _ring(connected,  radius=1.10, jitter=0.10)
    _ring(peripheral, radius=1.65, jitter=0.12)

    return pos


# ---------------------------------------------------------------------------
# Static figure
# ---------------------------------------------------------------------------

def draw_static(genes, degree, edges, pos, out_path, dpi=200):
    n = len(genes)
    node_colors = [_gene_color(g) for g in genes]
    node_sizes  = [18 + d ** 1.65 for d in degree]
    mu, sd      = degree.mean(), degree.std()
    thr_hub     = mu + 2 * sd
    thr_high    = mu + sd

    # Edge alpha scaled log of |Z|
    max_logz = np.log(max(e[2] for e in edges) + 1)
    min_logz = np.log(min(e[2] for e in edges) + 1)

    fig, ax = plt.subplots(figsize=(14, 11), dpi=dpi)
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")

    # --- Edges ---
    for i, j, absz in edges:
        alpha = 0.08 + 0.55 * (np.log(absz + 1) - min_logz) / (max_logz - min_logz + 1e-9)
        lw    = 0.3 + 1.8 * (np.log(absz + 1) - min_logz) / (max_logz - min_logz + 1e-9)
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color="#aaaacc", alpha=alpha, linewidth=lw, zorder=1)

    # --- Nodes ---
    for i, g in enumerate(genes):
        c  = node_colors[i]
        s  = node_sizes[i]
        d  = degree[i]
        zo = 5 if d >= thr_hub else (4 if d >= thr_high else 3)
        ew = 1.5 if d >= thr_hub else (0.8 if d >= thr_high else 0.3)
        ec = "white" if d >= thr_hub else ("white" if d >= thr_high else c)
        ax.scatter(pos[i, 0], pos[i, 1], s=s, c=c,
                   edgecolors=ec, linewidths=ew, zorder=zo, alpha=0.92)

    # --- Labels ---
    # Only label top-6 hubs + a few key bio genes to avoid crowding
    deg_series = pd.Series(degree, index=genes)
    to_label   = set(deg_series.nlargest(6).index)
    bio_notable = {"IL10", "C1QB", "MALAT1", "BTLA", "BBC3", "RPS6KA5"}
    to_label |= {g for g in bio_notable if g in genes}

    gene_idx = {g: i for i, g in enumerate(genes)}
    for g in to_label:
        if g not in gene_idx:
            continue
        i  = gene_idx[g]
        d  = degree[i]
        fw = "bold" if d >= thr_hub else "normal"
        fs = 7.5 if d >= thr_hub else 6.0
        c  = "white" if d >= thr_hub else "#dddddd"
        bc = _gene_color(g)
        ax.annotate(
            g,
            xy=(pos[i, 0], pos[i, 1]),
            xytext=(pos[i, 0] + 0.025, pos[i, 1] + 0.025),
            fontsize=fs, fontweight=fw, color=c, zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", fc=bc, ec="none",
                      alpha=0.75),
        )

    # --- Legend: bio categories ---
    handles = [
        mpatches.Patch(facecolor=cat["color"], edgecolor="white",
                       linewidth=0.5, label=name.capitalize())
        for name, cat in BIO_CATEGORIES.items()
    ]
    handles.append(
        mpatches.Patch(facecolor="#B0C4DE", edgecolor="none",
                       label="Uncharacterised")
    )
    leg = ax.legend(handles=handles, loc="lower left", fontsize=7.5,
                    frameon=True, framealpha=0.25, edgecolor="#555555",
                    facecolor="#111122", labelcolor="white",
                    handlelength=1.0, labelspacing=0.4, title="Category",
                    title_fontsize=8)
    leg.get_title().set_color("white")

    # --- Size legend ---
    for deg_ex, label in [(5, "deg=5"), (12, "deg=12"), (24, "deg=24 (hub)")]:
        s_ex = 30 + deg_ex ** 1.9
        ax.scatter([], [], s=s_ex, c="white", alpha=0.7, label=label)
    size_leg = ax.legend(loc="lower right", fontsize=7, frameon=True,
                         framealpha=0.25, edgecolor="#555555",
                         facecolor="#111122", labelcolor="white",
                         title="Node size", title_fontsize=8,
                         scatterpoints=1, labelspacing=0.6)
    size_leg.get_title().set_color("white")
    ax.add_artist(leg)

    # --- Stats annotation ---
    ax.text(0.01, 0.99,
            f"n = 584 samples  ·  p = {n} genes\n"
            f"Edges: {len(edges)} (FDR 5%)  ·  Mean degree: {mu:.1f}\n"
            f"Edge opacity ∝ log|Z|  (range {min(e[2] for e in edges):.1f}–{max(e[2] for e in edges):.0f})",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7, color="#aaaaaa",
            bbox=dict(boxstyle="round,pad=0.4", fc="#111122",
                      ec="#444444", alpha=0.8))

    ax.set_axis_off()
    ax.set_title(
        "NODIS Burn-Injury Co-expression Network  ·  GSE182616 (Acute Phase)\n"
        "De-sparsified GGM  ·  FDR 5%  ·  n/p = 3.56  [CAUTIONARY]",
        color="white", pad=10, fontsize=10.5, fontweight="bold"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Animated GIF: network builds up from strongest |Z| to weakest
# ---------------------------------------------------------------------------

def draw_gif(genes, degree, edges, pos, out_path, n_frames=90, fps=3):
    """30-second buildup GIF matching the knockout GIF style exactly."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    BG = "#0f0f1a"
    n  = len(genes)
    node_base_colors = [_gene_color(g) for g in genes]
    mu, sd   = degree.mean(), degree.std()
    thr_hub  = mu + 2 * sd
    thr_high = mu + sd

    max_z    = edges[0][2]
    min_z    = edges[-1][2]
    max_logz = np.log10(max_z + 1)
    min_logz = np.log10(min_z + 1)
    total    = len(edges)

    thresholds = np.logspace(np.log10(max_z), np.log10(min_z), n_frames)

    gene_idx = {g: i for i, g in enumerate(genes)}
    to_label = ({g for g in ALWAYS_LABEL if g in genes}
                | set(pd.Series(degree, index=genes).nlargest(8).index))

    # Pre-compute active edges and per-node degree for every frame
    print("  Pre-computing frame states …")
    frame_active  = []
    frame_cur_deg = []
    frame_counts  = []
    for thresh in thresholds:
        active = [(i, j, az) for i, j, az in edges if az >= thresh]
        cur_d  = np.zeros(n, dtype=int)
        for i, j, _ in active:
            cur_d[i] += 1
            cur_d[j] += 1
        frame_active.append(active)
        frame_cur_deg.append(cur_d)
        frame_counts.append(len(active))

    hist_edges = frame_counts   # for the right-panel plot

    # Figure — same layout as knockout: wide network left, narrow plot right
    fig = plt.figure(figsize=(17, 8.5), dpi=120)
    fig.patch.set_facecolor(BG)
    gs  = fig.add_gridspec(1, 2, width_ratios=[3.5, 1], wspace=0.04,
                            left=0.01, right=0.98, top=0.91, bottom=0.08)
    ax_net  = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    def _draw_frame(f):
        ax_net.clear()
        ax_plot.clear()

        active  = frame_active[f]
        cur_deg = frame_cur_deg[f]
        thresh  = thresholds[f]

        # ── Network panel ──────────────────────────────────────────────────
        ax_net.set_facecolor(BG)
        ax_net.set_axis_off()

        for i, j, absz in active:
            t_z   = (np.log10(absz + 1) - min_logz) / (max_logz - min_logz + 1e-9)
            alpha = 0.08 + 0.55 * t_z
            lw    = 0.3  + 2.0  * t_z
            ax_net.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color="#aaaacc", alpha=alpha, linewidth=lw,
                        zorder=1, solid_capstyle="round")

        for i, g in enumerate(genes):
            c      = node_base_colors[i]
            d_full = degree[i]
            d_cur  = cur_deg[i]
            # 2× base size, grows proportionally with edges gained
            deg_frac = max(d_cur / max(d_full, 1), 0.10)
            s_base   = 2.0 * (14 + d_full ** 1.65) * deg_frac
            zo = 5 if d_full >= thr_hub else (4 if d_full >= thr_high else 3)
            ew = 1.5 if d_full >= thr_hub else (0.7 if d_full >= thr_high else 0.3)
            ec = "white" if d_full >= thr_hub else c
            # soft glow
            ax_net.scatter(pos[i, 0], pos[i, 1], s=s_base * 3.5,
                           c=[c], alpha=0.10, edgecolors="none", zorder=zo - 1)
            # crisp node
            ax_net.scatter(pos[i, 0], pos[i, 1], s=s_base,
                           c=[c], edgecolors=ec, linewidths=ew,
                           zorder=zo, alpha=0.92)

        # Labels — same sizes as knockout GIF
        for g in to_label:
            if g not in gene_idx:
                continue
            i  = gene_idx[g]
            d  = degree[i]
            fw = "bold"  if d >= thr_hub  else "normal"
            fs = 9.5     if d >= thr_hub  else 7.5
            fc = "white" if d >= thr_hub  else "#cccccc"
            bc = node_base_colors[i]
            ax_net.annotate(
                g,
                xy=(pos[i, 0], pos[i, 1]),
                xytext=(pos[i, 0] + 0.028, pos[i, 1] + 0.028),
                fontsize=fs, fontweight=fw, color=fc, zorder=9,
                bbox=dict(boxstyle="round,pad=0.22", fc=bc, ec="none", alpha=0.80),
            )

        ax_net.set_title(
            f"NODIS Burn Network — Edges added by |Z|-score (strongest first)\n"
            f"|Z| ≥ {thresh:.1f}   ·   {len(active)} / {total} edges   ·   "
            f"FDR 5% threshold = {min_z:.1f}",
            color="white", pad=8, fontsize=10, fontweight="bold",
        )

        # Progress bar (same style as knockout)
        frac = len(active) / total
        BAR_L, BAR_R, BAR_Y, BAR_LW = 0.08, 0.92, 0.048, 9
        ax_net.plot([BAR_L, BAR_R], [BAR_Y, BAR_Y], color="#1e1e3a",
                    linewidth=BAR_LW, transform=ax_net.transAxes,
                    solid_capstyle="round", zorder=10)
        if frac > 0.002:
            ax_net.plot([BAR_L, BAR_L + (BAR_R - BAR_L) * frac],
                        [BAR_Y, BAR_Y], color="#4fc3f7",
                        linewidth=BAR_LW, transform=ax_net.transAxes,
                        solid_capstyle="round", zorder=11)
        ax_net.text(0.50, 0.008,
                    f"Edges shown:  {len(active)} / {total}",
                    transform=ax_net.transAxes, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#cccccc")

        ax_net.text(0.01, 0.99,
                    f"n = 584 samples\np = {n} genes\n"
                    f"|Z| threshold: {thresh:.1f}\nFDR 5% min: {min_z:.1f}",
                    transform=ax_net.transAxes, va="top", ha="left",
                    fontsize=8, color="#aaaaaa",
                    bbox=dict(boxstyle="round,pad=0.35", fc="#111122",
                              ec="#444444", alpha=0.8))

        # ── Plot panel ─────────────────────────────────────────────────────
        ax_plot.set_facecolor(BG)
        steps = list(range(n_frames))

        ax_plot.plot(steps, hist_edges, color="#4fc3f7", alpha=0.18, lw=0.9, zorder=1)
        ax_plot.plot(steps[:f + 1], hist_edges[:f + 1],
                     color="#4fc3f7", lw=2.2, zorder=3,
                     label=f"Edges  ({hist_edges[f]})")
        ax_plot.fill_between(steps[:f + 1], hist_edges[:f + 1], 0,
                              color="#4fc3f7", alpha=0.07, zorder=2)
        ax_plot.axvline(f, color="#666688", lw=0.9, ls="--", alpha=0.6, zorder=4)

        # FDR 5% final-edge reference line
        ax_plot.axhline(total, color="#FFD700", lw=0.8, ls=":",
                        alpha=0.6, zorder=1)
        ax_plot.text(1, total + 8, "FDR 5% total",
                     fontsize=8, color="#FFD700", va="bottom")

        if f == n_frames - 1:
            ax_plot.annotate(
                f"{total}\nedges",
                xy=(f, total),
                xytext=(f - 14, total - 80),
                fontsize=9, color="#4fc3f7",
                arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=0.9),
            )

        ax_plot.set_xlim(0, n_frames - 1)
        ax_plot.set_ylim(0, total * 1.10)
        ax_plot.set_xlabel("Simulation step", fontsize=9, color="#aaaaaa")
        ax_plot.set_ylabel("Edge count",      fontsize=9, color="#aaaaaa")
        ax_plot.set_title("Network build-up", fontsize=10.5, fontweight="bold",
                           color="white", pad=6)
        ax_plot.tick_params(colors="#888899", labelsize=8)
        for spine in ax_plot.spines.values():
            spine.set_edgecolor("#333355")

        # Edge legend (bigger)
        leg1 = ax_plot.legend(frameon=True, fontsize=9.5, loc="upper left",
                               framealpha=0.30, edgecolor="#555566",
                               facecolor="#111122", labelcolor="white",
                               handlelength=1.4)
        ax_plot.add_artist(leg1)

        # Category legend (bigger)
        handles = [mpatches.Patch(facecolor=cat["color"], edgecolor="none",
                                   label=name.capitalize())
                   for name, cat in BIO_CATEGORIES.items()]
        handles.append(mpatches.Patch(facecolor="#B0C4DE", edgecolor="none",
                                       label="Uncharact."))
        leg2 = ax_plot.legend(handles=handles, loc="lower right",
                               fontsize=8.5, frameon=True, framealpha=0.30,
                               edgecolor="#555566", facecolor="#111122",
                               labelcolor="white", handlelength=1.2,
                               labelspacing=0.45, title="Category",
                               title_fontsize=9)
        leg2.get_title().set_color("white")

    anim = FuncAnimation(fig, _draw_frame, frames=n_frames,
                         interval=1000 // fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=120,
              savefig_kwargs={"facecolor": BG})
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-static", default=os.path.join(FIGURES_DIR, "burns_network.pdf"))
    parser.add_argument("--out-gif",    default=os.path.join(FIGURES_DIR, "burns_network.gif"))
    parser.add_argument("--no-gif",     action="store_true")
    parser.add_argument("--frames",     type=int, default=90)
    parser.add_argument("--fps",        type=int, default=3)
    parser.add_argument("--dpi",        type=int, default=200)
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading network data …")
    adj, z, genes, degree, edges = load_data()
    print(f"  {len(genes)} genes, {len(edges)} edges, |Z| {edges[-1][2]:.1f}–{edges[0][2]:.0f}")

    print("Computing layout …")
    pos = _compute_layout(genes, degree, edges)

    print("Drawing static figure …")
    static_path = os.path.join(os.path.dirname(__file__), "..", args.out_static)
    draw_static(genes, degree, edges, pos, static_path, dpi=args.dpi)
    # also PNG
    png_path = static_path.replace(".pdf", ".png") if static_path.endswith(".pdf") else static_path + ".png"
    draw_static(genes, degree, edges, pos, png_path, dpi=150)

    if not args.no_gif:
        print(f"Rendering GIF ({args.frames} frames @ {args.fps} fps) …")
        gif_path = os.path.join(os.path.dirname(__file__), "..", args.out_gif)
        draw_gif(genes, degree, edges, pos, gif_path,
                 n_frames=args.frames, fps=args.fps)


if __name__ == "__main__":
    main()
