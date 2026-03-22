#!/usr/bin/env python3
"""
il10_knockout_gif.py  —  IL10 knockout cascade GIF (30 seconds, smooth)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       9,
    "figure.dpi":      120,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

def hex2rgb(h):
    return mc.to_rgb(h)

# ── Paths ──────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results", "burns")
FIGURES_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
BG = "#0f0f1a"

# ── Biological colour categories ───────────────────────────────────────────
BIO_CATEGORIES = {
    "immune":     {"color": "#E8527A",
                   "genes": {"IL10","C1QB","BTLA","EMR3","GPR174","C1QC",
                              "C1QA","LAIR1","CD163","MS4A4A","VSIG4","FCGR3A"}},
    "apoptosis":  {"color": "#FF8C42",
                   "genes": {"BBC3","PIM2","BCL2L1","MCL1","CASP3","TP53"}},
    "signalling": {"color": "#9B59B6",
                   "genes": {"RPS6KA5","PPP1R12B","MAP3K1","MAPK14"}},
    "epigenetic": {"color": "#1ABC9C",
                   "genes": {"MALAT1","SAP30","UBN1","CPSF3L"}},
    "metabolic":  {"color": "#F39C12",
                   "genes": {"SLC1A3","GRAMD1C","ABCA1"}},
}
ALWAYS_LABEL = {
    "KIAA1257","RPS6KA5","GRAMD1C","GPR174","SLC1A3","SAP30",
    "PPP1R12B","EMR3","C1QB","MALAT1","IL10","BTLA","BBC3","PIM2","FCGR3A",
}

def _gene_color(g):
    for cat in BIO_CATEGORIES.values():
        if g in cat["genes"]:
            return cat["color"]
    return "#B0C4DE"

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data …")
adj = pd.read_csv(os.path.join(RESULTS_DIR, "burns_nodis_adj_fdr05.csv"), index_col=0)
zsc = pd.read_csv(os.path.join(RESULTS_DIR, "burns_nodis_zscores.csv"),   index_col=0)

genes    = adj.index.tolist()
degree   = adj.values.sum(axis=1).astype(int)
n        = len(genes)
gene_idx = {g: i for i, g in enumerate(genes)}

all_edges = []
for i in range(n):
    for j in range(i + 1, n):
        if adj.iloc[i, j] == 1:
            all_edges.append((i, j, abs(float(zsc.iloc[i, j]))))
all_edges.sort(key=lambda e: -e[2])
N_EDGES_INIT = len(all_edges)
print(f"  {n} genes, {N_EDGES_INIT} edges")

# ── IL10 ───────────────────────────────────────────────────────────────────
HUB           = "IL10"
HUB_IDX       = gene_idx[HUB]
hub_color_base = _gene_color(HUB)

hub_edges = [(i, j, z) for i, j, z in all_edges if i == HUB_IDX or j == HUB_IDX]
hub_edges.sort(key=lambda e: e[2])   # weakest first
N_HUB     = len(hub_edges)
hub_nb_idx = {(j if i == HUB_IDX else i) for i, j, _ in hub_edges}
print(f"  IL10 degree: {N_HUB}")

secondary = []
for i, j, z in all_edges:
    if i in hub_nb_idx and j in hub_nb_idx:
        ps = sum(hz for hi, hj, hz in hub_edges
                 if (hj if hi == HUB_IDX else hi) in {i, j})
        secondary.append((i, j, z, ps))
secondary.sort(key=lambda x: x[3], reverse=True)
secondary = secondary[:min(30, len(secondary))]
N_SEC = len(secondary)
print(f"  Secondary edges: {N_SEC}")

# ── Shell ring layout ──────────────────────────────────────────────────────
print("Computing layout …")
mu, sd   = degree.mean(), degree.std()
thr_hub  = mu + 2 * sd
thr_high = mu + sd

hubs_ring  = [i for i in range(n) if degree[i] >= thr_hub]
high_ring  = [i for i in range(n) if thr_high <= degree[i] < thr_hub]
conn_ring  = [i for i in range(n) if mu <= degree[i] < thr_high]
peri_ring  = [i for i in range(n) if degree[i] < mu]

pos = np.zeros((n, 2))
rng = np.random.default_rng(42)

def _ring(indices, radius, jitter=0.0):
    if not indices: return
    indices = sorted(indices, key=lambda i: -degree[i])
    offset  = rng.uniform(0, 2 * np.pi)
    for rank, i in enumerate(indices):
        angle     = offset + 2 * np.pi * rank / len(indices)
        r         = radius + rng.uniform(-jitter, jitter)
        pos[i, 0] = r * np.cos(angle)
        pos[i, 1] = r * np.sin(angle)

_ring(hubs_ring, radius=0.30, jitter=0.06)
_ring(high_ring, radius=0.65, jitter=0.08)
_ring(conn_ring, radius=1.10, jitter=0.10)
_ring(peri_ring, radius=1.65, jitter=0.12)

# ── Animation parameters — 90 frames @ 3 fps = 30 s ──────────────────────
N_FRAMES = 90
FPS      = 3
P1_END   = 18    # intact / pulse       (0–17,  6 s)
P2_END   = 45    # silencing            (18–44, 9 s)
P3_END   = 72    # cascade              (45–71, 9 s)
                 # stabilise            (72–89, 6 s)

node_colors = [_gene_color(g) for g in genes]
max_z    = all_edges[0][2]
min_z    = all_edges[-1][2]
max_logz = np.log10(max_z + 1)
min_logz = np.log10(min_z + 1)

to_label = ({g for g in ALWAYS_LABEL if g in gene_idx}
            | set(pd.Series(degree, index=genes).nlargest(8).index))

heat_cmap = LinearSegmentedColormap.from_list(
    "heat", ["#2166ac", "#fdae61", "#d73027"]
)

# ── Easing ─────────────────────────────────────────────────────────────────
def ease_in_out(t):
    """Smooth cubic ease-in-out."""
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0, 1))

# ── Pre-compute per-frame state ────────────────────────────────────────────
def get_state(f):
    # IL10 fade (eased)
    if f < P1_END:
        pulse      = 0.80 + 0.20 * np.sin(f * np.pi * 2.5 / P1_END)
        il10_alpha = 1.0
        il10_scale = pulse
    elif f < P2_END:
        t_raw      = (f - P1_END) / (P2_END - P1_END)
        t          = ease_in_out(t_raw)
        il10_alpha = lerp(1.0, 0.10, t)
        il10_scale = lerp(1.0, 0.20, t)
    else:
        il10_alpha = 0.10
        il10_scale = 0.20

    # Hub edges removed (eased)
    if f < P1_END:
        n_hub_rm = 0
    elif f < P2_END:
        t        = ease_in_out((f - P1_END) / (P2_END - P1_END))
        n_hub_rm = int(np.round(t * N_HUB))
    else:
        n_hub_rm = N_HUB

    removed_hub = set()
    for k in range(n_hub_rm):
        u, v, _ = hub_edges[k]
        removed_hub |= {(u, v), (v, u)}

    # Secondary edges removed (eased)
    if f < P2_END:
        n_sec_rm = 0
    elif f < P3_END:
        t        = ease_in_out((f - P2_END) / (P3_END - P2_END))
        n_sec_rm = int(np.round(t * N_SEC))
    else:
        n_sec_rm = N_SEC

    removed_sec = set()
    for k in range(n_sec_rm):
        u, v, _, _ = secondary[k]
        removed_sec |= {(u, v), (v, u)}

    # Current degree per node (for dynamic node sizing)
    node_deg = degree.copy().astype(float)
    for k in range(n_hub_rm):
        u, v, _ = hub_edges[k]
        node_deg[u] = max(0, node_deg[u] - 1)
        node_deg[v] = max(0, node_deg[v] - 1)
    for k in range(n_sec_rm):
        u, v, _, _ = secondary[k]
        node_deg[u] = max(0, node_deg[u] - 1)
        node_deg[v] = max(0, node_deg[v] - 1)

    # Heat diffusion (smooth multi-wave)
    heat = np.zeros(n)
    if f >= P2_END:
        t = ease_in_out((f - P2_END) / max(P3_END - P2_END, 1))

        # Wave 1: direct IL10 neighbours
        w1 = min(t * 2.5, 1.0)
        for u, v, w in hub_edges:
            nb = v if u == HUB_IDX else u
            heat[nb] = min(1.0, (w / 6.0) * w1)

        # Wave 2: second-order (delay 0.25)
        if t > 0.25:
            t2 = ease_in_out((t - 0.25) / 0.75)
            for nb_i in hub_nb_idx:
                for u2, v2, w2 in all_edges:
                    if (u2 == nb_i or v2 == nb_i) and u2 != HUB_IDX and v2 != HUB_IDX:
                        other = v2 if u2 == nb_i else u2
                        heat[other] = min(1.0, heat[other] + heat[nb_i] * 0.45 * t2)

        # Wave 3: third-order (delay 0.60)
        if t > 0.60:
            t3    = ease_in_out((t - 0.60) / 0.40)
            heat2 = heat.copy()
            for u2, v2, _ in all_edges:
                if u2 != HUB_IDX and v2 != HUB_IDX:
                    heat2[u2] = min(1.0, heat2[u2] + heat[v2] * 0.18 * t3)
                    heat2[v2] = min(1.0, heat2[v2] + heat[u2] * 0.18 * t3)
            heat = heat2

    # After stabilise: gently decay heat
    if f >= P3_END:
        t_decay = ease_in_out((f - P3_END) / max(N_FRAMES - P3_END, 1))
        heat    = heat * (1.0 - t_decay * 0.55)

    return dict(
        il10_alpha  = il10_alpha,
        il10_scale  = il10_scale,
        n_hub_rm    = n_hub_rm,
        n_sec_rm    = n_sec_rm,
        removed_hub = removed_hub,
        removed_sec = removed_sec,
        node_deg    = node_deg,
        heat        = heat,
        total_edges = N_EDGES_INIT - n_hub_rm - n_sec_rm,
        il10_edges  = N_HUB - n_hub_rm,
    )

print("Pre-computing states …")
states     = [get_state(f) for f in range(N_FRAMES)]
hist_total = [s["total_edges"] for s in states]
hist_il10  = [s["il10_edges"]  for s in states]

# ── Figure ─────────────────────────────────────────────────────────────────
# Plot panel is narrow (width_ratio 1 vs 3.5 for network)
fig = plt.figure(figsize=(17, 8.5), dpi=120)
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(1, 2, width_ratios=[3.5, 1], wspace=0.04,
                        left=0.01, right=0.98, top=0.91, bottom=0.08)
ax_net  = fig.add_subplot(gs[0, 0])
ax_plot = fig.add_subplot(gs[0, 1])

phase_labels_map = [
    (0,      P1_END,   "Intact network"),
    (P1_END, P2_END,   "IL10 silencing"),
    (P2_END, P3_END,   "Cascade propagation"),
    (P3_END, N_FRAMES, "Network stabilised"),
]

# ── Draw function ──────────────────────────────────────────────────────────
def draw(f):
    st = states[f]
    ax_net.clear()
    ax_plot.clear()

    # ── NETWORK PANEL ─────────────────────────────────────────────────────
    ax_net.set_facecolor(BG)
    ax_net.set_axis_off()

    # Edges
    for u, v, absz in all_edges:
        pair = (u, v)
        is_hub_edge = (u == HUB_IDX or v == HUB_IDX)

        if pair in st["removed_hub"] or pair in st["removed_sec"]:
            if pair in st["removed_hub"] or (v, u) in st["removed_hub"]:
                ax_net.plot([pos[u,0], pos[v,0]], [pos[u,1], pos[v,1]],
                            color="#d73027", alpha=0.09, lw=0.45,
                            linestyle="--", zorder=1)
            continue

        t_z   = (np.log10(absz + 1) - min_logz) / (max_logz - min_logz + 1e-9)
        alpha = 0.07 + 0.50 * t_z
        lw    = 0.3  + 1.8  * t_z

        if is_hub_edge:
            alpha *= st["il10_alpha"]
            color  = "#aaaacc"
        else:
            hm = (st["heat"][u] + st["heat"][v]) / 2
            if hm > 0.05:
                r, g, b, _ = heat_cmap(hm)
                color = (r, g, b)
                alpha = min(1.0, alpha + hm * 0.35)
            else:
                color = "#aaaacc"

        ax_net.plot([pos[u,0], pos[v,0]], [pos[u,1], pos[v,1]],
                    color=color, alpha=alpha, lw=lw, zorder=1,
                    solid_capstyle="round")

    # Nodes — sized 2× base, shrinking by fraction of edges remaining
    for i, g in enumerate(genes):
        if i == HUB_IDX:
            continue
        c    = node_colors[i]
        d    = degree[i]
        h    = st["heat"][i]
        # Degree fraction: how many edges this node still has (min 10%)
        deg_frac = max(st["node_deg"][i] / max(d, 1), 0.10)
        # 2× initial size, then shrink smoothly with deg_frac
        s_base = 2.0 * (14 + d ** 1.65) * deg_frac

        # Heat colour blend
        if h > 0.05:
            rc, gc_c, bc_c, _ = heat_cmap(h)
            r0, g0, b0 = hex2rgb(c)
            blend  = min(h * 1.2, 1.0)
            draw_c = (lerp(r0, rc, blend*0.6),
                      lerp(g0, gc_c, blend*0.6),
                      lerp(b0, bc_c, blend*0.6))
        else:
            draw_c = c

        zo = 5 if d >= thr_hub else (4 if d >= thr_high else 3)
        ew = 1.5 if d >= thr_hub else (0.7 if d >= thr_high else 0.3)
        ec = "white" if d >= thr_hub else draw_c

        ax_net.scatter(pos[i,0], pos[i,1], s=s_base * 3.5,
                       c=[draw_c], alpha=0.10 + h * 0.15,
                       edgecolors="none", zorder=zo - 1)
        ax_net.scatter(pos[i,0], pos[i,1], s=s_base,
                       c=[draw_c], edgecolors=ec, linewidths=ew,
                       zorder=zo, alpha=0.92)

    # IL10 node
    il10_d = degree[HUB_IDX]
    il10_deg_frac = max(st["node_deg"][HUB_IDX] / max(il10_d, 1), 0.10)
    il10_s = 2.0 * (14 + il10_d ** 1.65) * il10_deg_frac * st["il10_scale"] * 2.0
    il10_a = st["il10_alpha"]

    if f < P1_END:
        glow_s = il10_s * (3.5 + 1.5 * np.sin(f * np.pi * 2.5 / P1_END))
        ax_net.scatter(pos[HUB_IDX,0], pos[HUB_IDX,1],
                       s=glow_s, c=["#FFD700"], alpha=0.20,
                       edgecolors="none", zorder=5)
    ax_net.scatter(pos[HUB_IDX,0], pos[HUB_IDX,1],
                   s=il10_s, c=[hub_color_base], alpha=il10_a,
                   edgecolors="white" if il10_a > 0.5 else "#888888",
                   linewidths=1.8, zorder=6)

    # Knockout ×
    if f >= P1_END:
        t_x = min(ease_in_out((f - P1_END) / 3.0), 1.0)
        r   = 0.068 * t_x
        kw  = dict(color="#d73027", lw=2.2, alpha=min(t_x, 0.92),
                   solid_capstyle="round", zorder=7)
        hx, hy = pos[HUB_IDX]
        ax_net.plot([hx-r, hx+r], [hy-r, hy+r], **kw)
        ax_net.plot([hx+r, hx-r], [hy-r, hy+r], **kw)

    # Gene labels — 2 font sizes larger than before
    for g in to_label:
        if g not in gene_idx: continue
        i  = gene_idx[g]
        d  = degree[i]
        h  = st["heat"][i]
        fw = "bold"   if d >= thr_hub  else "normal"
        fs = 9.5      if d >= thr_hub  else 7.5   # was 7.5 / 5.5
        fc = "white"  if d >= thr_hub  else "#cccccc"
        bc = node_colors[i]
        if g == HUB:
            fc = "white" if il10_a > 0.5 else "#777777"
            bc = "#FFD700" if f < P1_END else hub_color_base
        if h > 0.3 and g != HUB:
            fc = "white"
        ax_net.annotate(
            g,
            xy=(pos[i,0], pos[i,1]),
            xytext=(pos[i,0] + 0.028, pos[i,1] + 0.028),
            fontsize=fs, fontweight=fw, color=fc, zorder=9,
            bbox=dict(boxstyle="round,pad=0.22", fc=bc, ec="none", alpha=0.80),
        )

    # Phase title
    phase_str = ""
    for pa, pb, lbl in phase_labels_map:
        if pa <= f < pb:
            phase_str = lbl
    detail = ""
    if P1_END <= f < P2_END:
        detail = f"  ({st['n_hub_rm']}/{N_HUB} IL10 edges removed)"
    elif P2_END <= f < P3_END:
        detail = f"  (−{st['n_sec_rm']} secondary edges)"
    ax_net.set_title(
        f"IL10 Knockout Cascade — Burns Co-expression Network (NODIS, FDR 5%)\n"
        f"{phase_str}{detail}",
        color="white", pad=8, fontsize=10, fontweight="bold",
    )

    # Progress bar — track at y=0.048, fill in red, label below at y=0.008
    frac = (N_EDGES_INIT - st["total_edges"]) / N_EDGES_INIT
    BAR_L, BAR_R, BAR_Y, BAR_LW = 0.08, 0.92, 0.048, 9
    ax_net.plot([BAR_L, BAR_R], [BAR_Y, BAR_Y], color="#1e1e3a",
                linewidth=BAR_LW, transform=ax_net.transAxes,
                solid_capstyle="round", zorder=10)
    fill_r = BAR_L + (BAR_R - BAR_L) * frac
    if frac > 0.002:
        ax_net.plot([BAR_L, fill_r], [BAR_Y, BAR_Y], color="#d73027",
                    linewidth=BAR_LW, transform=ax_net.transAxes,
                    solid_capstyle="round", zorder=11)
    loss = N_EDGES_INIT - st["total_edges"]
    ax_net.text(0.50, 0.008,
                f"Edge loss:  {loss} / {N_EDGES_INIT}",
                transform=ax_net.transAxes, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#cccccc")   # bigger, below bar

    # Stats box
    ax_net.text(0.01, 0.99,
                f"n = 584 samples\np = 164 genes\n"
                f"IL10 degree: {N_HUB}\nFDR 5% threshold",
                transform=ax_net.transAxes, va="top", ha="left",
                fontsize=8, color="#aaaaaa",
                bbox=dict(boxstyle="round,pad=0.35", fc="#111122",
                          ec="#444444", alpha=0.8))

    # ── PLOT PANEL ────────────────────────────────────────────────────────
    ax_plot.set_facecolor(BG)
    steps = list(range(N_FRAMES))

    ax_plot.plot(steps, hist_total, color="#d73027", alpha=0.18, lw=0.9, zorder=1)
    ax_plot.plot(steps, hist_il10,  color="#4fc3f7", alpha=0.18, lw=0.9, zorder=1)
    ax_plot.plot(steps[:f+1], hist_total[:f+1],
                 color="#d73027", lw=2.2, zorder=3,
                 label=f"Total  ({hist_total[f]})")
    ax_plot.plot(steps[:f+1], hist_il10[:f+1],
                 color="#4fc3f7", lw=2.2, zorder=3,
                 label=f"IL10  ({hist_il10[f]})")
    ax_plot.fill_between(steps[:f+1], hist_total[:f+1],
                          min(hist_total) * 0.90,
                          color="#d73027", alpha=0.07, zorder=2)
    ax_plot.axvline(f, color="#666688", lw=0.9, ls="--", alpha=0.6, zorder=4)

    # Phase transitions — bigger font
    for ph_f, ph_lbl, ph_col in [
        (P1_END, "Silencing", "#FFD700"),
        (P2_END, "Cascade",   "#d73027"),
        (P3_END, "Stable",    "#2ca02c"),
    ]:
        ax_plot.axvline(ph_f, color=ph_col, lw=0.8, ls=":", alpha=0.55, zorder=1)
        ax_plot.text(ph_f + 0.5, N_EDGES_INIT + 3, ph_lbl,
                     fontsize=9, color=ph_col,        # was 6
                     va="bottom", style="italic", fontweight="bold")

    if f == N_FRAMES - 1:
        loss = N_EDGES_INIT - hist_total[-1]
        ax_plot.annotate(
            f"−{loss}\nedges",
            xy=(f, hist_total[f]),
            xytext=(f - 14, hist_total[f] - 60),
            fontsize=9, color="#d73027",
            arrowprops=dict(arrowstyle="->", color="#d73027", lw=0.9),
        )

    ax_plot.set_xlim(0, N_FRAMES - 1)
    ax_plot.set_ylim(min(hist_total) - 15, N_EDGES_INIT + 22)
    ax_plot.set_xlabel("Simulation step", fontsize=9, color="#aaaaaa")
    ax_plot.set_ylabel("Edge count",      fontsize=9, color="#aaaaaa")
    ax_plot.set_title("Connectivity", fontsize=10.5, fontweight="bold",
                       color="white", pad=6)
    ax_plot.tick_params(colors="#888899", labelsize=8)
    for spine in ax_plot.spines.values():
        spine.set_edgecolor("#333355")

    # Main legend — bigger font
    leg1 = ax_plot.legend(frameon=True, fontsize=9.5,   # was 7.5
                           loc="upper right",
                           framealpha=0.30, edgecolor="#555566",
                           facecolor="#111122", labelcolor="white",
                           handlelength=1.4, handletextpad=0.5)
    ax_plot.add_artist(leg1)

    # Category legend — bigger font
    handles = [mpatches.Patch(facecolor=cat["color"], edgecolor="none",
                               label=name.capitalize())
               for name, cat in BIO_CATEGORIES.items()]
    handles.append(mpatches.Patch(facecolor="#B0C4DE", edgecolor="none",
                                   label="Uncharact."))
    leg2 = ax_plot.legend(handles=handles, loc="lower left",
                           fontsize=8.5,               # was 6.5
                           frameon=True, framealpha=0.30, edgecolor="#555566",
                           facecolor="#111122", labelcolor="white",
                           handlelength=1.2, labelspacing=0.45,
                           title="Category", title_fontsize=9)  # was 7
    leg2.get_title().set_color("white")

# ── Render ─────────────────────────────────────────────────────────────────
print(f"Rendering {N_FRAMES} frames @ {FPS} fps ({N_FRAMES/FPS:.0f} s) …")
anim   = FuncAnimation(fig, draw, frames=N_FRAMES,
                        interval=1000 // FPS, blit=False)
out_gif = os.path.join(FIGURES_DIR, "il10_knockout_cascade.gif")
writer  = PillowWriter(fps=FPS)
anim.save(out_gif, writer=writer, dpi=120,
          savefig_kwargs={"facecolor": BG})

size_mb = os.path.getsize(out_gif) / 1e6
print(f"Done → {out_gif}  ({size_mb:.1f} MB)")
