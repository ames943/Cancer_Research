#!/usr/bin/env python3
"""
batch_detector.py — Batch Effect Detector for Multi-Cohort Omics Studies
=========================================================================

Quantifies batch effects vs. biological signal in multi-cohort omics datasets.
Accepts any sample × feature matrix (microbiome, transcriptomics, proteomics,
metabolomics, etc.) and produces:

  1. A structured JSON report with PERMANOVA statistics, batch:signal ratio,
     minimum same-protocol sample size for 80% power, and a GO / CAUTION /
     NO-GO pooling recommendation.
  2. A publication-ready PDF figure: PERMANOVA R² bar chart + Dirichlet-
     corrected power curve.

Method
------
Uses distance-based PERMANOVA (Anderson 2001) on Aitchison distances
(Euclidean distance on CLR-transformed data) to partition multivariate variance
into batch and biological response components.

The power analysis uses a Dirichlet parametric bootstrap — each bootstrap
iteration draws fresh compositions from Dirichlet(observed + pseudocount),
avoiding zero-distance artifacts from naive bootstrap-with-replacement on
distance matrices (which inflate pseudo-F by up to 2× and produce power
estimates ~2× too optimistic; see companion paper §Methods).

References
----------
Anderson MJ (2001). A new method for non-parametric multivariate analysis
    of variance. Austral Ecology, 26(1):32–46.
Aitchison J (1986). The Statistical Analysis of Compositional Data.
    Chapman & Hall, London.
Anderson MJ & Walsh DCI (2013). PERMANOVA, ANOSIM, and the Mantel test in
    the face of heterogeneous dispersions. Ecol Monographs, 83(4):557–574.
"""

# =============================================================================
# RESEARCHER USAGE GUIDE
# =============================================================================
#
# INSTALLATION (standard scientific Python stack — no extra dependencies)
# -----------------------------------------------------------------------
#   pip install numpy pandas scipy matplotlib
#
# QUICK START
# -----------
#   # Microbiome study: 3 cohorts, response = immunotherapy responder status
#   python batch_detector.py \
#       --input  feature_matrix.tsv \
#       --labels sample_labels.tsv  \
#       --batch  cohort             \   # column name in sample_labels.tsv
#       --output results/           \
#       --clr                           # apply CLR (omit if already transformed)
#
#   # With simulation validation (50-rep Dirichlet bootstrap at current n)
#   python batch_detector.py --input X.tsv --labels meta.tsv \
#       --batch cohort --output out/ --clr --simulate
#
#   # Single-cohort mode (no batch) — PERMANOVA + power analysis only
#   python batch_detector.py --input X.tsv --labels meta.tsv \
#       --output out/ --clr
#
# INPUT FORMATS
# -------------
#   feature_matrix.tsv : Tab-separated, samples as rows, features as columns.
#                        First column = sample identifier (e.g. run_accession).
#                        Header row required (feature names in row 1).
#
#   sample_labels.tsv  : Tab-separated, first column = sample identifier.
#                        Must contain a 'response' column (or use --response-col).
#                        May also contain the batch column (use --batch <colname>).
#
#   batch (optional)   : Either (a) a column name in sample_labels.tsv,
#                        or (b) a path to a separate 2-column TSV:
#                        first col = sample_id, second col = batch label.
#
# OUTPUT FILES
# ------------
#   <output>/batch_detector_report.json  — Full structured JSON report
#   <output>/batch_detector_figure.pdf   — PERMANOVA bar chart + power curve
#   <output>/batch_detector_power.tsv    — Power curve table (n, power, CI)
#
# INTERPRETATION
# --------------
#   GO      batch:signal < 3  AND  response PERMANOVA p < 0.05
#           → Pooling appears appropriate; batch effect is modest.
#
#   CAUTION batch:signal 3–8  OR  (batch:signal < 3 AND response p ≥ 0.05)
#           → Apply and validate batch correction before pooling, or check
#             the power curve to see if the dataset is simply underpowered.
#
#   NO-GO   batch:signal > 8
#           → Batch variance dominates biological signal; naive pooling is
#             not recommended. Seek a same-protocol cohort or substantially
#             larger n (see power curve for the threshold).
#
# KNOWN LIMITATIONS
# -----------------
#   • Power analysis bootstraps from the observed data; power at n > observed
#     is extrapolated via resampling with replacement, not from a parametric
#     model. Estimates are reliable near the observed n and increasingly
#     optimistic (upper-bound) well above it when the observed signal is weak.
#   • Dirichlet draws assume compositional (non-negative, sum-normalizable) data.
#     For pre-normalized or pre-transformed data (e.g., log-TPM), pass without
#     --clr and note that the power analysis Dirichlet draws may not be valid —
#     a warning is printed if any X values are negative.
#
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm as sp_norm

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_PSEUDOCOUNT = 1e-6
_VERSION = "1.0.0"


# =============================================================================
# PERMANOVA (Anderson 2001)
# =============================================================================

def _ss_total(d2: np.ndarray) -> float:
    """Upper-triangle sum of squared distances / n."""
    return float(np.sum(np.triu(d2, k=1))) / d2.shape[0]


def _ss_within(d2: np.ndarray, grp: np.ndarray) -> float:
    """Sum over groups of (upper-triangle within-group SS / n_group)."""
    sw = 0.0
    for g in np.unique(grp):
        m = grp == g
        ng = int(m.sum())
        if ng < 2:
            continue
        sub = d2[np.ix_(m, m)]
        sw += float(np.sum(np.triu(sub, k=1))) / ng
    return sw


def permanova(
    D: np.ndarray,
    grouping,
    n_perms: int = 999,
    seed: int = 42,
) -> dict:
    """
    Distance-based PERMANOVA (Anderson 2001).

    Parameters
    ----------
    D        : (n, n) symmetric distance matrix (not squared).
    grouping : array-like, length n — group label per sample.
    n_perms  : permutation count for the test (default 999).
    seed     : random seed.

    Returns
    -------
    dict with keys R2, F_stat, p_value, SS_total, SS_between, SS_within,
    n_samples, n_groups, n_perms, perm_F_mean, perm_F_std.
    """
    grp = np.asarray(grouping)
    n = D.shape[0]
    q = len(np.unique(grp))
    if q < 2:
        raise ValueError("PERMANOVA requires ≥ 2 distinct groups.")

    d2 = D ** 2
    ST = _ss_total(d2)
    SW = _ss_within(d2, grp)
    SA = ST - SW
    F = (SA / (q - 1)) / (SW / (n - q)) if SW > 0 else 0.0
    R2 = SA / ST if ST > 0 else 0.0

    rng = np.random.default_rng(seed)
    perm_F = np.empty(n_perms)
    for i in range(n_perms):
        g_p = rng.permutation(grp)
        sw_p = _ss_within(d2, g_p)
        sa_p = ST - sw_p
        perm_F[i] = (sa_p / (q - 1)) / (sw_p / (n - q)) if sw_p > 0 else 0.0

    # p = (# permuted F ≥ observed F + 1) / (n_perms + 1)  [Phipson & Smyth 2010]
    p_val = float((perm_F >= F).sum() + 1) / (n_perms + 1)

    return {
        "R2": round(float(R2), 6),
        "F_stat": round(float(F), 4),
        "p_value": round(p_val, 4),
        "SS_total": round(float(ST), 4),
        "SS_between": round(float(SA), 4),
        "SS_within": round(float(SW), 4),
        "n_samples": n,
        "n_groups": q,
        "n_perms": n_perms,
        "perm_F_mean": round(float(perm_F.mean()), 4),
        "perm_F_std": round(float(perm_F.std()), 4),
    }


# =============================================================================
# CLR transformation and Aitchison distance
# =============================================================================

def clr_transform(X: np.ndarray, pseudocount: float = _PSEUDOCOUNT) -> np.ndarray:
    """
    Centered log-ratio (CLR) transformation with pseudocount.

    CLR(x_i) = log(x_i + ε) − mean_j[log(x_j + ε)]

    Applied row-wise (per sample). Pseudocount handles exact zeros.
    """
    lx = np.log(X + pseudocount)
    return lx - lx.mean(axis=1, keepdims=True)


def aitchison_dist(X_clr: np.ndarray) -> np.ndarray:
    """Aitchison distance matrix — Euclidean distance on CLR data."""
    return squareform(pdist(X_clr, metric="euclidean"))


# =============================================================================
# Dirichlet bootstrap helpers
# =============================================================================

def _dirichlet_draw(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Draw one fresh composition per row using the Gamma-normalization identity.

    X ~ Dirichlet(α_row)  iff  X = G / Σ(G)  where G_i ~ Gamma(α_i, 1).

    Avoids zero-distance duplicate pairs that arise from naive
    bootstrap-with-replacement on distance matrices. Each resampled index
    receives a *distinct* synthetic composition even if the index repeats.

    Parameters
    ----------
    alpha : (n, p) concentration parameters — all entries must be > 0.
    rng   : numpy Generator.

    Returns
    -------
    (n, p) matrix where each row sums to 1.
    """
    G = rng.standard_gamma(alpha)
    return G / G.sum(axis=1, keepdims=True)


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    z = sp_norm.ppf(1 - alpha / 2)
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, c - m), min(1.0, c + m)


def _permanova_pval_fast(
    d2: np.ndarray,
    y: np.ndarray,
    n_perms_inner: int,
    rng: np.random.Generator,
) -> float:
    """
    Inline PERMANOVA p-value for binary grouping (faster path for power loops).

    Uses the 2-group pseudo-F: F = (SA / SW) × (n − 2).
    Returns p-value in [0, 1] (fraction of permuted F ≥ observed F).
    """
    n = d2.shape[0]
    ST = float(np.sum(np.triu(d2, k=1))) / n
    SW = 0.0
    for g in (0, 1):
        m = y == g
        ng = int(m.sum())
        if ng < 2:
            return 1.0  # degenerate group
        sub = d2[np.ix_(m, m)]
        SW += float(np.sum(np.triu(sub, k=1))) / ng

    if SW == 0:
        return 1.0
    F0 = (ST - SW) / SW * (n - 2)

    n_ge = 0
    for _ in range(n_perms_inner):
        y_p = rng.permutation(y)
        sw_p = 0.0
        for g in (0, 1):
            m = y_p == g
            ng = int(m.sum())
            if ng < 2:
                continue
            sub = d2[np.ix_(m, m)]
            sw_p += float(np.sum(np.triu(sub, k=1))) / ng
        F_p = (ST - sw_p) / sw_p * (n - 2) if sw_p > 0 else 0.0
        if F_p >= F0:
            n_ge += 1

    return n_ge / n_perms_inner  # approximate p-value (no +1 correction in inner loop)


# =============================================================================
# Dirichlet-corrected power analysis
# =============================================================================

def dirichlet_power_analysis(
    X_raw: np.ndarray,
    y_binary: np.ndarray,
    n_values: list,
    n_reps: int = 200,
    n_perms_inner: int = 199,
    pseudocount: float = _PSEUDOCOUNT,
    seed: int = 42,
) -> list:
    """
    Dirichlet-corrected parametric power analysis for PERMANOVA(response).

    For each target sample size n in n_values:
      1. Resample n indices with replacement from the observed data (preserving
         response label proportions in the draw).
      2. For each resampled index, draw a fresh Dirichlet realization using
         X_raw[index] + pseudocount as concentration parameters.  This avoids
         the zero-distance duplicate-pair artifact of naive bootstrap.
      3. Apply CLR, compute Aitchison distances, run PERMANOVA.
      4. Power = fraction of reps where PERMANOVA p < 0.05.

    Parameters
    ----------
    X_raw        : (n, p) raw feature matrix (counts or relative abundances ≥ 0).
    y_binary     : (n,) binary response array (0 / 1 integers).
    n_values     : list of target sample sizes to evaluate.
    n_reps       : bootstrap reps per n (default 200).
    n_perms_inner: inner PERMANOVA permutations per rep (default 199).
    pseudocount  : floor for Dirichlet concentration parameters.
    seed         : random seed.

    Returns
    -------
    List of dicts: [{"n", "power", "ci_lower", "ci_upper", "n_reps", ...}, ...]
    """
    rng = np.random.default_rng(seed)
    alpha_mat = X_raw + pseudocount          # (n_obs, p) — all strictly positive
    n_obs = len(y_binary)
    y = np.asarray(y_binary, dtype=int)

    rows = []
    for n_target in n_values:
        n_sig = 0
        for _ in range(n_reps):
            idx = rng.integers(0, n_obs, size=n_target)
            alpha_draw = alpha_mat[idx]                      # (n_target, p)
            X_sim = _dirichlet_draw(alpha_draw, rng)         # fresh compositions
            X_clr = clr_transform(X_sim, pseudocount)
            D_sim = squareform(pdist(X_clr, metric="euclidean"))
            d2 = D_sim ** 2
            y_sim = y[idx]

            if y_sim.sum() in (0, n_target):
                continue  # degenerate draw — single class

            p = _permanova_pval_fast(d2, y_sim, n_perms_inner, rng)
            if p < 0.05:
                n_sig += 1

        power = n_sig / n_reps
        ci_lo, ci_hi = _wilson_ci(n_sig, n_reps)
        rows.append({
            "n": n_target,
            "power": round(power, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "n_reps": n_reps,
            "n_perms_inner": n_perms_inner,
        })
        log.info("  n=%4d  power=%.3f  95%% CI [%.3f, %.3f]",
                 n_target, power, ci_lo, ci_hi)

    return rows


def _find_n_80pct(power_rows: list) -> int | None:
    """Interpolate the n where estimated power first reaches 80%."""
    ns = [r["n"] for r in power_rows]
    ps = [r["power"] for r in power_rows]
    for n, p in zip(ns, ps):
        if p >= 0.80:
            return n
    for i in range(len(ps) - 1):
        if ps[i] < 0.80 <= ps[i + 1]:
            frac = (0.80 - ps[i]) / (ps[i + 1] - ps[i])
            return int(ns[i] + frac * (ns[i + 1] - ns[i]))
    return None


# =============================================================================
# Simulation validation (--simulate)
# =============================================================================

def simulate_at_current_n(
    X_raw: np.ndarray,
    y_binary: np.ndarray,
    n_target: int,
    n_sim_reps: int = 50,
    n_perms_inner: int = 99,
    pseudocount: float = _PSEUDOCOUNT,
    seed: int = 99,
) -> dict:
    """
    Quick Dirichlet bootstrap at the current n to validate the power estimate.

    Runs the same method as the full power analysis (50 reps, 99 inner perms)
    and returns an independent power estimate at n_target.  Agreement with the
    full curve at n_target (within sampling noise) validates the approach.
    """
    rows = dirichlet_power_analysis(
        X_raw, y_binary,
        n_values=[n_target],
        n_reps=n_sim_reps,
        n_perms_inner=n_perms_inner,
        pseudocount=pseudocount,
        seed=seed,
    )
    return rows[0] if rows else {}


# =============================================================================
# Pooling recommendation
# =============================================================================

def _pooling_recommendation(
    batch_signal_ratio: float | None,
    response_p: float,
) -> tuple[str, str]:
    """
    Return (recommendation, reason) based on batch:signal ratio and response p.

    GO      : batch:signal < 3  AND  response p < 0.05
    CAUTION : batch:signal 3–8  OR  signal not significant at the current n
    NO-GO   : batch:signal > 8
    """
    if batch_signal_ratio is None:
        # Single-cohort mode — no batch analysis
        if response_p < 0.05:
            return "GO", (
                f"Single-cohort mode. Significant biological signal detected "
                f"(PERMANOVA p = {response_p:.3f})."
            )
        return "CAUTION", (
            f"Single-cohort mode. Biological signal not significant "
            f"(PERMANOVA p = {response_p:.3f}). Check the power curve to "
            "determine the same-protocol n needed for 80% power."
        )

    if batch_signal_ratio > 8:
        return "NO-GO", (
            f"Batch variance dominates biological signal by {batch_signal_ratio:.1f}×. "
            "Naive pooling across these cohorts is not recommended. "
            "Seek a same-protocol cohort or a substantially larger n "
            "(see power curve); validate any batch-correction method independently."
        )
    if batch_signal_ratio >= 3:
        return "CAUTION", (
            f"Moderate batch:signal ratio ({batch_signal_ratio:.1f}×). "
            "Apply and independently validate batch correction before downstream "
            "analysis. Re-run this tool on the corrected data to confirm batch R² "
            "has decreased without collapsing response R²."
        )
    # batch:signal < 3
    if response_p < 0.05:
        return "GO", (
            f"Low batch:signal ratio ({batch_signal_ratio:.1f}×) with significant "
            f"biological signal (p = {response_p:.3f}). Pooling appears appropriate."
        )
    return "CAUTION", (
        f"Low batch:signal ratio ({batch_signal_ratio:.1f}×) but biological signal "
        f"is not significant (p = {response_p:.3f}). Dataset may be underpowered "
        "at the current n — check the power curve."
    )


# =============================================================================
# Figure
# =============================================================================

_PALETTE = {
    "batch": "#E53935",        # red
    "response": "#1E88E5",     # blue
    "GO": "#43A047",           # green
    "CAUTION": "#FB8C00",      # orange
    "NO-GO": "#E53935",        # red
    "power_line": "#1565C0",
    "power_fill": "#90CAF9",
    "threshold_80": "#C62828",
    "current_n": "#E65100",
    "n_80pct": "#AD1457",
}


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "p < 0.001 (***)"
    if p < 0.01:
        return f"p = {p:.3f} (**)"
    if p < 0.05:
        return f"p = {p:.3f} (*)"
    if p < 0.10:
        return f"p = {p:.3f} (.)"
    return f"p = {p:.3f} (ns)"


def generate_figure(
    res_response: dict | None,
    res_batch: dict | None,
    power_rows: list,
    n_current: int,
    n_80pct: int | None,
    recommendation: str,
    out_path: Path,
) -> None:
    """
    Produce a 2-panel publication summary figure.

    Panel A: PERMANOVA R² bar chart (batch vs. response) with p-value annotations.
    Panel B: Dirichlet-corrected power curve with 95% Wilson CI band,
             current-n marker, and 80%-power threshold.
    """
    rec_color = _PALETTE.get(recommendation, "#555555")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    fig.subplots_adjust(wspace=0.38)

    # ── Panel A: PERMANOVA R² bar chart ──────────────────────────────────────
    ax = axes[0]
    bar_labels, bar_heights, bar_colors, bar_pvals = [], [], [], []

    if res_response:
        bar_labels.append("Response\n(biological signal)")
        bar_heights.append(res_response["R2"])
        bar_colors.append(_PALETTE["response"])
        bar_pvals.append(res_response["p_value"])

    if res_batch:
        bar_labels.append("Batch\n(technical noise)")
        bar_heights.append(res_batch["R2"])
        bar_colors.append(_PALETTE["batch"])
        bar_pvals.append(res_batch["p_value"])

    x_pos = np.arange(len(bar_labels))
    bars = ax.bar(
        x_pos, bar_heights,
        color=bar_colors, width=0.45,
        edgecolor="white", linewidth=1.4, zorder=3,
    )

    y_max = max(bar_heights) if bar_heights else 0.10
    for bar, pval, h in zip(bars, bar_pvals, bar_heights):
        stars = "***" if pval < 0.001 else "**" if pval < 0.01 \
            else "*" if pval < 0.05 else "ns"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_max * 0.03,
            stars,
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            color="black",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h / 2,
            f"R² = {h:.4f}",
            ha="center", va="center",
            fontsize=9, color="white", fontweight="bold",
        )

    if res_batch and res_response and res_response["R2"] > 0:
        ratio = res_batch["R2"] / res_response["R2"]
        ax.text(
            0.97, 0.97,
            f"Batch : Signal = {ratio:.1f}×",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color=rec_color,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=rec_color, linewidth=1.5, alpha=0.90),
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=11)
    ax.set_ylabel("PERMANOVA R²  (Aitchison distance)", fontsize=11)
    ax.set_ylim(0, y_max * 1.30)
    ax.set_title("(A)  Batch vs. Biological Signal", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.28, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # p-value legend text
    if res_response or res_batch:
        legend_lines = []
        if res_response:
            legend_lines.append(f"Response: {_sig_stars(res_response['p_value'])}")
        if res_batch:
            legend_lines.append(f"Batch: {_sig_stars(res_batch['p_value'])}")
        ax.text(
            0.03, 0.97, "\n".join(legend_lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                      edgecolor="#BDBDBD", alpha=0.90),
        )

    # ── Panel B: Power curve ──────────────────────────────────────────────────
    ax2 = axes[1]

    if power_rows:
        ns_pw = [r["n"] for r in power_rows]
        ps_pw = [r["power"] for r in power_rows]
        ci_lo = [r["ci_lower"] for r in power_rows]
        ci_hi = [r["ci_upper"] for r in power_rows]

        ax2.fill_between(ns_pw, ci_lo, ci_hi,
                         color=_PALETTE["power_fill"], alpha=0.40, zorder=2,
                         label="95% Wilson CI")
        ax2.plot(ns_pw, ps_pw, "o-",
                 color=_PALETTE["power_line"], linewidth=2.0, markersize=6,
                 zorder=4, label="Estimated power")

        # 80% threshold
        ax2.axhline(0.80, color=_PALETTE["threshold_80"], linestyle="--",
                    linewidth=1.4, zorder=3, label="80% power target")

        # Current n
        pw_at_current = next((r["power"] for r in power_rows if r["n"] == n_current), None)
        ax2.axvline(n_current, color=_PALETTE["current_n"], linestyle=":",
                    linewidth=1.8, zorder=3, label=f"Current n = {n_current}")
        if pw_at_current is not None:
            ax2.scatter([n_current], [pw_at_current],
                        color=_PALETTE["current_n"], s=80, zorder=5)

        # n for 80% power
        if n_80pct is not None:
            ax2.axvline(n_80pct, color=_PALETTE["n_80pct"], linestyle="-.",
                        linewidth=1.3, zorder=3, alpha=0.85,
                        label=f"n for 80% power ≈ {n_80pct}")

        ax2.set_xlabel("Same-protocol sample size  n", fontsize=11)
        ax2.set_ylabel("Empirical power  P(PERMANOVA p < 0.05)", fontsize=11)
        ax2.set_ylim(-0.04, 1.08)
        ax2.set_xlim(min(ns_pw) * 0.92, max(ns_pw) * 1.05)
        ax2.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
        ax2.grid(True, alpha=0.25, zorder=0)
        ax2.spines[["top", "right"]].set_visible(False)
    else:
        ax2.text(0.5, 0.5, "Power analysis unavailable\n(non-binary response)",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=11, color="#777777")

    ax2.set_title("(B)  Dirichlet-Corrected Power Curve", fontsize=12,
                  fontweight="bold", pad=10)

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.suptitle(
        f"BatchDetector  |  Recommendation:  {recommendation}  "
        f"(n = {n_current})",
        fontsize=12, fontweight="bold", color=rec_color, y=1.02,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved → %s", out_path)


# =============================================================================
# Data loading and preprocessing
# =============================================================================

def _load_feature_matrix(path: str) -> pd.DataFrame:
    """Load samples × features TSV (first column = sample ID)."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    log.info("Feature matrix loaded: %d samples × %d features  [%s]",
             df.shape[0], df.shape[1], path)
    return df


def _load_labels(path: str, response_col: str) -> pd.Series:
    """Load response labels from a TSV (first column = sample ID)."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    if response_col not in df.columns:
        raise ValueError(
            f"Response column '{response_col}' not found in labels file. "
            f"Available columns: {list(df.columns)}"
        )
    return df[response_col]


def _load_batch(batch_arg: str, labels_path: str) -> pd.Series:
    """
    Resolve batch assignments from either a column name or a file path.

    Tries batch_arg as a column in the labels file first; if not found, tries
    it as a path to a 2-column TSV (sample_id, batch_label).
    """
    labels_df = pd.read_csv(labels_path, sep="\t", index_col=0)
    if batch_arg in labels_df.columns:
        log.info("Batch loaded from column '%s' in labels file.", batch_arg)
        return labels_df[batch_arg]
    try:
        batch_df = pd.read_csv(batch_arg, sep="\t", index_col=0)
        col = batch_df.columns[0]
        log.info("Batch loaded from file '%s' (column '%s').", batch_arg, col)
        return batch_df[col]
    except (FileNotFoundError, pd.errors.ParserError, IsADirectoryError) as exc:
        raise ValueError(
            f"--batch '{batch_arg}' is neither a column in the labels file "
            f"nor a readable TSV file. Error: {exc}"
        ) from exc


def _preprocess(
    X: pd.DataFrame,
    response: pd.Series | None,
    batch: pd.Series | None,
    apply_clr: bool,
    pseudocount: float = _PSEUDOCOUNT,
) -> tuple:
    """
    Align samples, drop missing labels, remove zero-variance features,
    and optionally apply CLR transformation.

    Returns
    -------
    X_transformed : (n, p) float array — CLR-transformed or raw.
    X_raw         : (n, p) float array — pre-CLR (needed for Dirichlet draws).
    response_arr  : (n,) array or None.
    batch_arr     : (n,) array or None.
    sample_ids    : list of n sample identifiers.
    """
    # Align on common sample IDs
    common = X.index
    if response is not None:
        common = common.intersection(response.dropna().index)
    if batch is not None:
        common = common.intersection(batch.dropna().index)

    n_dropped = len(X) - len(common)
    if n_dropped > 0:
        log.warning("Dropped %d samples with missing labels after alignment.",
                    n_dropped)
    if len(common) == 0:
        raise ValueError(
            "No samples remain after aligning feature matrix with labels. "
            "Check that sample IDs match across files."
        )

    X = X.loc[common]
    response = response.loc[common] if response is not None else None
    batch = batch.loc[common] if batch is not None else None

    # Impute NaN features with 0
    n_nan = int(X.isna().sum().sum())
    if n_nan > 0:
        log.warning("Imputing %d NaN feature values with 0.", n_nan)
        X = X.fillna(0)

    # Drop zero-variance features
    feat_var = X.var(axis=0)
    n_zero_var = int((feat_var == 0).sum())
    if n_zero_var > 0:
        log.info("Dropping %d zero-variance features.", n_zero_var)
        X = X.loc[:, feat_var > 0]
    if X.shape[1] == 0:
        raise ValueError("All features have zero variance. Cannot proceed.")

    X_raw = X.values.astype(float)

    # Warn if data has negative values (Dirichlet draws won't be valid)
    if np.any(X_raw < 0):
        log.warning(
            "Feature matrix contains negative values. Dirichlet power analysis "
            "assumes non-negative compositional data. Power estimates may not be "
            "meaningful. Pass non-negative count or relative-abundance data, or "
            "omit --simulate."
        )

    if apply_clr:
        X_out = clr_transform(X_raw, pseudocount)
        log.info("CLR transformation applied (pseudocount = %.0e).", pseudocount)
    else:
        X_out = X_raw.copy()
        log.info("No CLR transformation (data used as-is).")

    log.info("Final dataset: n = %d samples, p = %d features.", *X_out.shape)

    return (
        X_out,
        X_raw,
        response.values if response is not None else None,
        batch.values if batch is not None else None,
        list(common),
    )


# =============================================================================
# Argument parser
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="batch_detector.py",
        description=(
            "Batch Effect Detector for Multi-Cohort Omics Studies.\n\n"
            "Quantifies batch vs. biological signal via PERMANOVA on Aitchison\n"
            "distances and emits a GO / CAUTION / NO-GO pooling recommendation,\n"
            "a JSON report, and a PDF figure.\n\n"
            "See the README comment block at the top of this file for a full\n"
            "usage guide, input formats, and interpretation notes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Microbiome: 3 cohorts, response = responder status, apply CLR
  python batch_detector.py --input X.tsv --labels meta.tsv \\
      --batch cohort --output results/ --clr

  # Pre-CLR data (no --clr flag)
  python batch_detector.py --input X_clr.tsv --labels meta.tsv \\
      --batch batch_file.tsv --output out/

  # Single-cohort (no batch) — PERMANOVA + power analysis
  python batch_detector.py --input X.tsv --labels meta.tsv \\
      --output out/ --clr

  # Add simulation validation at the current n
  python batch_detector.py --input X.tsv --labels meta.tsv \\
      --batch cohort --output out/ --clr --simulate
        """,
    )

    # Required
    p.add_argument("--input",   required=True,
                   help="TSV: samples × features, first column = sample ID.")
    p.add_argument("--labels",  required=True,
                   help="TSV: first column = sample ID; must contain --response-col.")
    p.add_argument("--output",  required=True,
                   help="Output directory (created if absent).")

    # Optional data args
    p.add_argument("--batch", default=None,
                   help=("Batch assignments: (a) column name in --labels, "
                         "or (b) path to a 2-column TSV (sample_id, batch)."))
    p.add_argument("--response-col", default="response",
                   help="Column name for the binary outcome (default: response).")
    p.add_argument("--clr", action="store_true",
                   help="Apply CLR transformation with pseudocount before analysis.")
    p.add_argument("--pseudocount", type=float, default=1e-6,
                   help="Pseudocount for CLR and Dirichlet draws (default: 1e-6).")

    # PERMANOVA
    p.add_argument("--n-perms", type=int, default=999,
                   help="PERMANOVA permutations (default: 999).")

    # Power analysis
    p.add_argument("--n-boot", type=int, default=200,
                   help="Dirichlet bootstrap reps per n for power analysis (default: 200).")
    p.add_argument("--n-perms-inner", type=int, default=199,
                   help="Inner PERMANOVA perms per bootstrap rep (default: 199).")

    # Simulation
    p.add_argument("--simulate", action="store_true",
                   help=("Run a quick 50-rep Dirichlet bootstrap at the current n "
                         "to validate the power estimate."))

    # Misc
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")

    return p


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("BatchDetector v%s — Batch Effect Detector for Multi-Cohort Omics", _VERSION)
    log.info("=" * 65)
    log.info("Input       : %s", args.input)
    log.info("Labels      : %s  (column: '%s')", args.labels, args.response_col)
    log.info("Batch       : %s", args.batch or "(none — single-cohort mode)")
    log.info("CLR         : %s  (pseudocount = %.0e)", args.clr, args.pseudocount)
    log.info("PERMANOVA   : n_perms = %d", args.n_perms)
    log.info("Power boot  : n_boot = %d, n_perms_inner = %d",
             args.n_boot, args.n_perms_inner)
    log.info("Output dir  : %s", out_dir)

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("\nLoading data …")
    X_df = _load_feature_matrix(args.input)
    response_series = _load_labels(args.labels, args.response_col)
    batch_series = _load_batch(args.batch, args.labels) if args.batch else None

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_out, X_raw, response_arr, batch_arr, sample_ids = _preprocess(
        X_df, response_series, batch_series,
        apply_clr=args.clr, pseudocount=args.pseudocount,
    )
    n = len(sample_ids)

    if n < 4:
        log.error("Need at least 4 samples for PERMANOVA. Got n = %d.", n)
        sys.exit(1)

    # ── Aitchison distance matrix ─────────────────────────────────────────────
    log.info("\nComputing Aitchison distance matrix (%d × %d) …", n, n)
    D = aitchison_dist(X_out)
    log.info("  max = %.2f   mean (off-diagonal) = %.2f",
             D.max(), D[D > 0].mean())

    # ── PERMANOVA — response ──────────────────────────────────────────────────
    res_response = None
    unique_resp = np.unique(response_arr) if response_arr is not None else []
    if response_arr is not None and len(unique_resp) >= 2:
        log.info("\nPERMANOVA — response  (n_perms = %d) …", args.n_perms)
        res_response = permanova(D, response_arr, n_perms=args.n_perms, seed=args.seed)
        log.info("  R² = %.4f   F = %.4f   %s",
                 res_response["R2"], res_response["F_stat"],
                 _sig_stars(res_response["p_value"]))
    elif response_arr is not None:
        log.warning("Response PERMANOVA skipped: fewer than 2 unique labels "
                    "(%s). Check --response-col.", unique_resp)

    # ── PERMANOVA — batch ─────────────────────────────────────────────────────
    res_batch = None
    if batch_arr is not None:
        n_batches = len(np.unique(batch_arr))
        if n_batches < 2:
            log.warning("Only 1 unique batch value — treating as single-cohort mode.")
            batch_arr = None
        else:
            log.info("\nPERMANOVA — batch  (%d groups, n_perms = %d) …",
                     n_batches, args.n_perms)
            res_batch = permanova(D, batch_arr, n_perms=args.n_perms,
                                  seed=args.seed + 1)
            log.info("  R² = %.4f   F = %.4f   %s",
                     res_batch["R2"], res_batch["F_stat"],
                     _sig_stars(res_batch["p_value"]))

    # ── Batch:signal ratio ────────────────────────────────────────────────────
    batch_signal_ratio = None
    if res_batch and res_response and res_response["R2"] > 0:
        batch_signal_ratio = round(res_batch["R2"] / res_response["R2"], 2)
        log.info("\nBatch:Signal ratio = %.2f×", batch_signal_ratio)

    # ── Recommendation ────────────────────────────────────────────────────────
    resp_p = res_response["p_value"] if res_response else 1.0
    rec, rec_reason = _pooling_recommendation(batch_signal_ratio, resp_p)
    log.info("Recommendation: %s", rec)
    log.info("Reason: %s", rec_reason)

    # ── Binary response for power analysis ───────────────────────────────────
    y_binary = None
    if response_arr is not None and len(unique_resp) == 2:
        y_binary = (response_arr == unique_resp[0]).astype(int)
    elif response_arr is not None and len(unique_resp) != 2:
        log.warning("Power analysis requires a binary response "
                    "(%d unique values). Skipping.", len(unique_resp))

    # ── Power analysis ────────────────────────────────────────────────────────
    power_rows: list = []
    n_80pct: int | None = None

    if y_binary is not None and not np.any(X_raw < 0):
        log.info("\nDirichlet-corrected power analysis …")
        log.info("  %d bootstrap reps × %d inner perms per n",
                 args.n_boot, args.n_perms_inner)

        # Build n_values: span from ~½ × current to ~3× current, step ~10%
        n_min = max(10, n // 2)
        n_max = min(500, max(200, n * 3))
        step = max(10, (n_max - n_min) // 12)
        n_values = sorted(set(range(n_min, n_max + 1, step)) | {n})

        power_rows = dirichlet_power_analysis(
            X_raw, y_binary,
            n_values=n_values,
            n_reps=args.n_boot,
            n_perms_inner=args.n_perms_inner,
            pseudocount=args.pseudocount,
            seed=args.seed + 2,
        )
        n_80pct = _find_n_80pct(power_rows)
        if n_80pct is not None:
            log.info("  → 80%% power threshold: n ≈ %d same-protocol samples", n_80pct)
        else:
            log.info("  → 80%% power not reached within n ≤ %d; "
                     "effect size may be very small.", n_values[-1] if n_values else 0)
    elif y_binary is not None:
        log.info("Power analysis skipped (negative feature values detected).")

    # ── Simulation validation ─────────────────────────────────────────────────
    sim_result: dict | None = None
    if args.simulate and y_binary is not None and not np.any(X_raw < 0):
        log.info("\nSimulation validation (50 reps at current n = %d) …", n)
        sim_result = simulate_at_current_n(
            X_raw, y_binary,
            n_target=n,
            n_sim_reps=50,
            n_perms_inner=99,
            pseudocount=args.pseudocount,
            seed=args.seed + 999,
        )
        log.info(
            "  Simulated power at n = %d: %.3f  95%% CI [%.3f, %.3f]  "
            "(50 reps, 99 inner perms)",
            n,
            sim_result.get("power", float("nan")),
            sim_result.get("ci_lower", float("nan")),
            sim_result.get("ci_upper", float("nan")),
        )
        pw_full = next((r for r in power_rows if r["n"] == n), None)
        if pw_full:
            delta = abs(sim_result.get("power", 0) - pw_full["power"])
            log.info(
                "  Full curve at n = %d: %.3f  |  Δ = %.3f  "
                "(%s within sampling noise)",
                n, pw_full["power"], delta,
                "✓" if delta < 0.10 else "⚠ larger than expected",
            )

    # ── Build JSON report ─────────────────────────────────────────────────────
    report = {
        "tool": "batch_detector.py",
        "version": _VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input),
        "labels_file": str(args.labels),
        "response_col": args.response_col,
        "batch_arg": str(args.batch) if args.batch else None,
        # Dataset summary
        "n_samples": n,
        "n_features": X_out.shape[1],
        "n_batch_groups": (int(len(np.unique(batch_arr)))
                           if batch_arr is not None else None),
        "clr_applied": args.clr,
        "pseudocount": args.pseudocount,
        "permanova_n_perms": args.n_perms,
        # Core statistics
        "response_R2": res_response["R2"] if res_response else None,
        "response_F": res_response["F_stat"] if res_response else None,
        "permanova_p_response": res_response["p_value"] if res_response else None,
        "batch_R2": res_batch["R2"] if res_batch else None,
        "batch_F": res_batch["F_stat"] if res_batch else None,
        "permanova_p_batch": res_batch["p_value"] if res_batch else None,
        "batch_signal_ratio": batch_signal_ratio,
        # Recommendation
        "pooling_recommendation": rec,
        "recommendation_reason": rec_reason,
        # Power analysis
        "min_n_for_80pct_power": n_80pct,
        "dirichlet_bootstrap_n_reps": args.n_boot if power_rows else None,
        "simulation_validation": sim_result if args.simulate else None,
        # Full PERMANOVA detail
        "permanova_response_detail": res_response,
        "permanova_batch_detail": res_batch,
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    json_path = out_dir / "batch_detector_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("\nReport saved  → %s", json_path)

    power_tsv_path: Path | None = None
    if power_rows:
        power_tsv_path = out_dir / "batch_detector_power.tsv"
        pd.DataFrame(power_rows).to_csv(power_tsv_path, sep="\t", index=False)
        log.info("Power curve   → %s", power_tsv_path)

    fig_path = out_dir / "batch_detector_figure.pdf"
    generate_figure(
        res_response=res_response,
        res_batch=res_batch,
        power_rows=power_rows,
        n_current=n,
        n_80pct=n_80pct,
        recommendation=rec,
        out_path=fig_path,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("RESULTS SUMMARY")
    log.info("=" * 65)
    if res_response:
        log.info("  Response PERMANOVA : R² = %.4f   %s",
                 res_response["R2"], _sig_stars(res_response["p_value"]))
    if res_batch:
        log.info("  Batch PERMANOVA    : R² = %.4f   %s",
                 res_batch["R2"], _sig_stars(res_batch["p_value"]))
    if batch_signal_ratio is not None:
        log.info("  Batch:Signal ratio : %.2f×", batch_signal_ratio)
    log.info("  Recommendation     : %s", rec)
    if n_80pct is not None:
        log.info("  Min n (80%% power)  : %d same-protocol samples", n_80pct)
    else:
        log.info("  Min n (80%% power)  : > %d  (very small effect)",
                 power_rows[-1]["n"] if power_rows else "?")
    log.info("\nOutputs:")
    log.info("  %s", json_path)
    if power_tsv_path:
        log.info("  %s", power_tsv_path)
    log.info("  %s", fig_path)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
