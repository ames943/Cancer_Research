#!/usr/bin/env python3
"""
Master figure for IEEE BIBM paper.
Panel A: PERMANOVA/AUC dose-response n=39→79→118→283
Panel B: Phase 2 simulation heatmap (n_cohorts=3, uncorrected)

Output: results/figures/master_figure.{png,pdf}
Run from cancer_project/:  python3 scripts/master_figure.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

os.makedirs("results/figures", exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

fig = plt.figure(figsize=(12, 5.2))
gs  = GridSpec(1, 2, figure=fig, wspace=0.44, left=0.08, right=0.96,
               top=0.88, bottom=0.16)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A — Dose-response curve
# ═══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])
ax_a2 = ax_a.twinx()   # right y-axis for batch:signal ratio

n_vals     = [39,     79,     118,    283   ]
r2_vals    = [0.0267, 0.0110, 0.0068, 0.0039]
bs_x       = [79,     118,    283   ]         # batch:signal starts at n=79
bs_vals    = [7.7,    11.3,   27.3  ]

BLUE = "#1565C0"
RED  = "#C62828"

# ── Response R² line (left axis) ──
ax_a.plot(n_vals, r2_vals,
          color=BLUE, linewidth=2.2, zorder=4,
          marker="o", markersize=7, markerfacecolor=BLUE,
          markeredgecolor="white", markeredgewidth=1.2, label="Response R²")
ax_a.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)

# ── Batch:signal line (right axis) ──
ax_a2.plot(bs_x, bs_vals,
           color=RED, linewidth=2.2, zorder=4,
           marker="s", markersize=7, markerfacecolor=RED,
           markeredgecolor="white", markeredgewidth=1.2, label="Batch:signal ratio",
           linestyle="--")

# ── Vertical dashed line at n=118 ──
ax_a.axvline(118, color="#555555", linewidth=1.1, linestyle=":", zorder=2)
ax_a.text(122, 0.0235,
          "current\n3-cohort\nstudy",
          fontsize=7.5, color="#555555", va="top", ha="left", linespacing=1.4)

# ── "6.9× decline" annotation ──
ax_a.annotate(
    "",
    xy=(283, 0.0039), xytext=(39, 0.0267),
    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3,
                    connectionstyle="arc3,rad=-0.25"),
    zorder=5,
)
ax_a.text(145, 0.0175, "6.9× decline", fontsize=8, color=BLUE,
          ha="center", va="bottom", style="italic",
          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

# ── R² data point labels ──
label_kw = dict(fontsize=7.5, color=BLUE, ha="center", va="bottom")
ax_a.text(39,  0.0267 + 0.0019, "R²=0.0267", **label_kw)
ax_a.text(79,  0.0110 + 0.0014, "R²=0.0110", **label_kw)
ax_a.text(126, 0.0068 + 0.0009, "R²=0.0068", **label_kw)
ax_a.text(283, 0.0039 + 0.0015, "R²=0.0039", **label_kw)

# ── Batch:signal labels ──
bs_lkw = dict(fontsize=7.5, color=RED, ha="center", va="bottom")
ax_a2.text(79,  7.7  + 1.8, "7.7×",  **bs_lkw)
ax_a2.text(118, 11.3 + 1.8, "11.3×", **bs_lkw)
ax_a2.text(265, 27.3 + 1.5, "27.3×", **bs_lkw)

# ── Axes formatting ──
ax_a.set_xlabel("Total samples (n)", fontsize=10)
ax_a.set_ylabel("Response R² (Aitchison PERMANOVA)", color=BLUE, fontsize=10)
ax_a2.set_ylabel("Batch : Signal ratio", color=RED, fontsize=10)

ax_a.set_xticks([39, 79, 118, 283])
ax_a.set_xticklabels(["39\n(C1)", "79\n(C1+2)", "118\n(C1–3)", "283\n(C1–4)"])
ax_a.set_xlim(15, 310)
ax_a.set_ylim(-0.004, 0.036)
ax_a2.set_ylim(-1, 34)

ax_a.tick_params(axis="y", labelcolor=BLUE)
ax_a2.tick_params(axis="y", labelcolor=RED)
ax_a2.spines["right"].set_visible(True)
ax_a2.spines["right"].set_linewidth(0.8)
ax_a2.spines["top"].set_visible(False)

# ── Legend ──
h1 = mpatches.Patch(color=BLUE, label="Response R²")
h2 = mpatches.Patch(color=RED,  label="Batch:signal ratio", linestyle="--",
                     linewidth=2)
ax_a.legend(handles=[h1, h2], fontsize=8.5, frameon=True,
            loc="upper right", framealpha=0.9, edgecolor="lightgray")

ax_a.set_title("Dose-response: pooling dilutes biological signal", fontsize=11, pad=6)

# ── Panel label ──
ax_a.text(-0.13, 1.05, "A", transform=ax_a.transAxes,
          fontsize=14, fontweight="bold", va="top", ha="left")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B — Simulation heatmap
# ═══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1])

sim = pd.read_csv("results/ml/simulation/grid_results.tsv", sep="\t")

# Filter: n_cohorts=3, method='none', average across batch_f2
sub = sim[(sim["n_cohorts"] == 3) & (sim["method"] == "none")]
hm  = sub.groupby(["sig_f2", "n_per_cohort"])["mean_auc"].mean().reset_index()

sig_vals = [0.007, 0.027, 0.05, 0.10]
npc_vals = [20, 40, 80]

# Build matrix: rows=n_per_cohort (high→low), cols=sig_f2 (low→high)
Z = np.zeros((len(npc_vals), len(sig_vals)))
for i, npc in enumerate(npc_vals):
    for j, sf in enumerate(sig_vals):
        row = hm[(hm["n_per_cohort"] == npc) & (np.isclose(hm["sig_f2"], sf))]
        Z[i, j] = row["mean_auc"].values[0] if len(row) else np.nan

# Flip so n=80 at top
Z_plot    = Z[::-1, :]
npc_labels = [str(n) for n in reversed(npc_vals)]

im = ax_b.imshow(
    Z_plot, aspect="auto", cmap="RdYlGn",
    vmin=0.45, vmax=0.95,
    extent=[-0.5, len(sig_vals) - 0.5, -0.5, len(npc_vals) - 0.5],
    origin="lower",
)

# Colorbar
cbar = fig.colorbar(im, ax=ax_b, shrink=0.82, pad=0.03)
cbar.set_label("Mean AUC", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Cell value annotations
for i in range(len(npc_vals)):
    for j in range(len(sig_vals)):
        val = Z_plot[i, j]
        txt_color = "white" if (val > 0.80 or val < 0.48) else "black"
        ax_b.text(j, i, f"{val:.2f}", ha="center", va="center",
                  fontsize=8.5, fontweight="bold", color=txt_color)

# ── Real operating point star ──
# sig_f2=0.027 → col index 1; npc=40 → row index in Z_plot
# npc_vals reversed: [80,40,20] → npc=40 is index 1
real_col = sig_vals.index(0.027)      # = 1
real_row = list(reversed(npc_vals)).index(40)  # = 1
ax_b.scatter(real_col, real_row, marker="*", s=340, color="#C62828",
             zorder=5, linewidths=0.5, edgecolors="white")
ax_b.text(real_col + 0.08, real_row + 0.38,
          "real data\noperating point",
          fontsize=7.5, color="#C62828", ha="left", va="bottom",
          fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

# ── "correction helps" vertical dashed line (between col 2 and 3) ──
# sig_f2=0.10 is col index 3; draw line between col 2.5 and mark col 3
ax_b.axvline(2.5, color="#555555", linewidth=1.2, linestyle="--", zorder=4)
ax_b.text(2.55, len(npc_vals) - 0.55,
          "correction\nhelps ≥ here",
          fontsize=7.5, color="#555555", ha="left", va="top", linespacing=1.4)

# ── Axes ──
ax_b.set_xticks(range(len(sig_vals)))
ax_b.set_xticklabels([f"{v:.3f}" for v in sig_vals], fontsize=9)
ax_b.set_yticks(range(len(npc_vals)))
ax_b.set_yticklabels(npc_labels, fontsize=9)

ax_b.set_xlabel("Signal effect size (f²)", fontsize=10)
ax_b.set_ylabel("Samples per cohort", fontsize=10)
ax_b.set_title("Simulation: AUC by signal strength & cohort size\n(3 cohorts, uncorrected)",
               fontsize=11, pad=6)

# Remove heatmap spines
for spine in ax_b.spines.values():
    spine.set_visible(False)
ax_b.tick_params(length=0)

# ── x-axis secondary labels showing null calibration ──
ax_b.text(0, -0.58, "null", ha="center", va="top", fontsize=7,
          color="#777777", transform=ax_b.transData)
ax_b.text(1, -0.58, "≈ real", ha="center", va="top", fontsize=7,
          color="#C62828", transform=ax_b.transData)

# ── Panel label ──
ax_b.text(-0.18, 1.05, "B", transform=ax_b.transAxes,
          fontsize=14, fontweight="bold", va="top", ha="left")

# ── Shared figure title ──
fig.suptitle(
    "Gut microbiome batch effects dominate biological signal across "
    "all cohort sizes and correction methods",
    fontsize=11.5, fontweight="bold", y=0.98,
)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["png", "pdf"]:
    out = f"results/figures/master_figure.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}")

plt.close(fig)
print("Done.")
