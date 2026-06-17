#!/usr/bin/env python3
"""
Phase 4: DerSimonian-Laird random-effects meta-analysis across microbiome cohorts.

For each cohort, fits ElasticNet via LOOCV and extracts per-genus mean coefficient
and SE across folds. Applies DerSimonian-Laird random-effects meta-analysis across
genera present in ≥2 cohorts, with BH FDR correction.

Outputs results/ml/phase4/:
  meta_analysis_results.tsv   — pooled effects, heterogeneity, FDR q-values
  per_cohort_loocv_coefs.tsv  — per-cohort LOOCV mean coef + SE per genus
  forest_plot.png             — top-20 genera by |pooled_effect|
  funnel_plot.png             — effect size vs SE (publication bias check)

Usage:
    cd cancer_project/
    python scripts/phase4_meta_analysis.py
"""

import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
CLR_PATH    = "results/ml/n118_3cohort/X_genus_clr.tsv"
RAW_PATH    = "results/ml/n118_3cohort/X_genus_raw.tsv"
LABELS_PATH = "metadata/response_labels_3cohort.tsv"
LEE_CLR     = "results/ml/lee2022/X_genus_clr.tsv"   # C4 — include if present
OUT_DIR     = Path("results/ml/phase4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
# EN: most common hyperparams from nested_cv (C=1.0 in 49/118, l1=0.3 in 53/118)
EN_C            = 1.0
EN_L1           = 0.3
PREVALENCE_FRAC = 0.10    # genus must appear in ≥10% of samples
VAR_KEEP_FRAC   = 0.50    # keep top-50% by within-cohort CLR variance
TOP_N_PBR       = 100     # top-N by |point-biserial r| for feature selection
MIN_SE          = 1e-4    # SE floor (avoids division by zero in meta-analysis)
SEED            = 42

# ── Cohort colors ──────────────────────────────────────────────────────────────
COHORT_COLORS = {
    "cohort1": "#1E88E5",   # blue   — Frankel 2017 / HiSeq
    "cohort2": "#FB8C00",   # orange — NovaSeq cohort
    "cohort3": "#43A047",   # green  — Matson 2018 / NextSeq
    "cohort4": "#8E24AA",   # purple — Lee 2022
}
COHORT_LABELS = {
    "cohort1": "C1 Frankel (n=39)",
    "cohort2": "C2 NovaSeq (n=40)",
    "cohort3": "C3 Matson  (n=39)",
    "cohort4": "C4 Lee 2022",
}


def ts():
    return time.strftime("%H:%M:%S")


# ── Helpers ────────────────────────────────────────────────────────────────────

def select_features(clr_c: pd.DataFrame, raw_c: pd.DataFrame,
                    resp_c: pd.Series) -> list:
    """Prevalence → top-50% variance → top-100 |point-biserial r|."""
    n = len(clr_c)
    min_prev = max(2, int(np.ceil(PREVALENCE_FRAC * n)))
    pres = (raw_c > 0).sum(axis=0)
    prev_feats = pres[pres >= min_prev].index.tolist()
    vv = clr_c[prev_feats].var(axis=0)
    hv_feats = vv[vv >= vv.quantile(1.0 - VAR_KEEP_FRAC)].index.tolist()
    y_bin = (resp_c == "R").astype(int).values
    pbr = {c: abs(stats.pointbiserialr(y_bin, clr_c[c].values)[0])
           for c in hv_feats}
    return pd.Series(pbr).sort_values(ascending=False).head(TOP_N_PBR).index.tolist()


def fit_en(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit ElasticNet logistic regression; return coefficient vector."""
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga",
        C=EN_C, l1_ratio=EN_L1,
        class_weight="balanced",
        max_iter=5000, tol=1e-3, random_state=SEED,
    )
    clf.fit(X, y)
    return clf.coef_[0]


def loocv_coefs(clr_c: pd.DataFrame, raw_c: pd.DataFrame,
                resp_c: pd.Series, cname: str) -> tuple:
    """
    LOOCV coefficient extraction for one cohort.

    For each LOOCV fold (leave-one-sample-out):
      1. Feature selection on training set (n-1 samples).
      2. Fit ElasticNet on selected features.
      3. Record coefficient for each genus (0 if not selected).

    Returns
    -------
    mean_coefs : Series indexed by genus — mean coefficient across folds.
    se_coefs   : Series indexed by genus — SE (std/sqrt(n)) across folds.
    fold_mat   : DataFrame (n_folds × n_genera) of per-fold coefficients.
    """
    genera = clr_c.columns.tolist()
    n = len(clr_c)
    fold_mat = np.zeros((n, len(genera)))
    genus_idx = {g: i for i, g in enumerate(genera)}

    print(f"[{ts()}]   LOOCV {cname} (n={n}, {n} folds) …", flush=True)
    for fold in range(n):
        mask_train = np.ones(n, dtype=bool)
        mask_train[fold] = False
        clr_tr = clr_c.iloc[mask_train]
        raw_tr = raw_c.iloc[mask_train]
        resp_tr = resp_c.iloc[mask_train]

        if resp_tr.nunique() < 2:
            continue  # degenerate fold — leave coefficients as 0

        sel = select_features(clr_tr, raw_tr, resp_tr)
        X_tr = clr_tr[sel].values.astype(float)
        y_tr = resp_tr.values

        try:
            coef_sel = fit_en(X_tr, y_tr)
        except Exception:
            continue

        for feat, c in zip(sel, coef_sel):
            if feat in genus_idx:
                fold_mat[fold, genus_idx[feat]] = c

        if (fold + 1) % 10 == 0:
            print(f"[{ts()}]     fold {fold+1}/{n}", flush=True)

    mean_coefs = pd.Series(fold_mat.mean(axis=0), index=genera)
    se_coefs   = pd.Series(
        np.maximum(fold_mat.std(axis=0) / np.sqrt(n), MIN_SE),
        index=genera,
    )
    fold_df = pd.DataFrame(fold_mat, columns=genera,
                           index=clr_c.index)
    return mean_coefs, se_coefs, fold_df


# ── DerSimonian-Laird random-effects meta-analysis ────────────────────────────

def dsl_meta(effects: np.ndarray, variances: np.ndarray) -> dict:
    """
    DerSimonian-Laird random-effects meta-analysis.

    Parameters
    ----------
    effects   : per-cohort point estimates (k,).
    variances : per-cohort sampling variances = SE² (k,).

    Returns
    -------
    dict with theta_re, se_re, ci_lo, ci_hi, tau2, I2, Q, Q_p, z, p_z.
    """
    w_fe = 1.0 / variances
    theta_fe = np.sum(w_fe * effects) / np.sum(w_fe)
    Q = float(np.sum(w_fe * (effects - theta_fe) ** 2))
    k = len(effects)
    df_Q = k - 1
    p_Q = float(1 - stats.chi2.cdf(Q, df=df_Q)) if df_Q > 0 else 1.0

    c_factor = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
    tau2 = float(max(0.0, (Q - df_Q) / c_factor)) if c_factor > 0 else 0.0
    I2 = float(max(0.0, (Q - df_Q) / Q * 100)) if Q > df_Q else 0.0

    w_re = 1.0 / (variances + tau2)
    theta_re = float(np.sum(w_re * effects) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    z = theta_re / se_re if se_re > 0 else 0.0
    p_z = float(2 * (1 - stats.norm.cdf(abs(z))))

    return dict(
        theta_re=theta_re, se_re=se_re,
        ci_lo=theta_re - 1.96 * se_re,
        ci_hi=theta_re + 1.96 * se_re,
        tau2=tau2, I2=I2,
        Q=Q, Q_p=p_Q, z=z, p_z=p_z,
    )


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    q = pvals * m / ranks
    q_mono = np.minimum.accumulate(q[order][::-1])[::-1]
    q_out = np.empty(m)
    q_out[order] = np.minimum(q_mono, 1.0)
    return q_out


# ── Forest plot ────────────────────────────────────────────────────────────────

def forest_plot(meta_df: pd.DataFrame, cohort_coefs: dict, out_path: Path) -> None:
    """
    Forest plot for top-20 genera by |pooled_effect|.

    Shows per-cohort 95% CI bars (colored) + pooled diamond per genus.
    """
    top20 = meta_df.reindex(
        meta_df["pooled_effect"].abs().nlargest(20).index
    ).copy()
    top20 = top20.sort_values("pooled_effect", ascending=True)  # positive at top

    cohort_names = list(cohort_coefs.keys())
    n_cohorts = len(cohort_names)
    n_genera = len(top20)

    # Layout: each genus gets (n_cohorts + 1.5) vertical slots
    slot = n_cohorts + 1.5
    fig_height = max(8, n_genera * slot * 0.22 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_ticks, y_labels = [], []
    all_x = []

    for gi, (_, row) in enumerate(top20.iterrows()):
        genus = row["genus"]
        base_y = gi * slot

        # Per-cohort CI bars
        for ci, cname in enumerate(cohort_names):
            if cname not in cohort_coefs:
                continue
            mean_c = cohort_coefs[cname]["mean"]
            se_c   = cohort_coefs[cname]["se"]
            if genus not in mean_c.index:
                continue
            m = mean_c[genus]
            s = max(se_c[genus], MIN_SE)
            if abs(m) < 1e-6:
                continue  # zero/unselected — skip
            y_pos = base_y + ci * 0.55
            lo, hi = m - 1.96 * s, m + 1.96 * s
            color = COHORT_COLORS.get(cname, "gray")
            ax.plot([lo, hi], [y_pos, y_pos], color=color, linewidth=1.8, zorder=3)
            ax.scatter([m], [y_pos], color=color, s=28, zorder=4, marker="s")
            all_x.extend([lo, hi])

        # Pooled diamond (at base_y + n_cohorts * 0.55 + 0.3)
        y_dia = base_y + n_cohorts * 0.55 + 0.15
        pe, se_re = row["pooled_effect"], row["SE"]
        dia_lo, dia_hi = pe - 1.96 * se_re, pe + 1.96 * se_re
        dia_h = 0.28
        diamond_x = [dia_lo, pe, dia_hi, pe, dia_lo]
        diamond_y = [y_dia, y_dia + dia_h, y_dia, y_dia - dia_h, y_dia]
        ax.fill(diamond_x, diamond_y, color="#37474F", zorder=4, alpha=0.85)
        ax.plot(diamond_x, diamond_y, color="#37474F", linewidth=0.8, zorder=5)
        all_x.extend([dia_lo, dia_hi])

        # Genus label
        q = row["q_value"]
        sig_tag = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
        dir_tag = "R+" if pe > 0 else "NR+"
        y_mid = base_y + (n_cohorts * 0.55) / 2
        y_ticks.append(y_mid)
        y_labels.append(f"{genus}  {sig_tag}")

        # Right-side annotation
        ax.text(
            1.01, (y_mid) / (n_genera * slot),
            f"β={pe:+.3f}  q={q:.3f}  I²={row['I2']:.0f}%  {dir_tag}",
            transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=7.5, color="#333333",
        )

    # Vertical zero line
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2)

    # X-axis range
    if all_x:
        x_pad = (max(all_x) - min(all_x)) * 0.08
        ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad + (max(all_x) - min(all_x)) * 0.35)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylim(-slot * 0.5, n_genera * slot)
    ax.set_xlabel("ElasticNet coefficient (LOOCV mean ± 1.96 SE)", fontsize=11)
    ax.set_title(
        "Forest Plot — Top 20 Genera by |Pooled Effect|\n"
        "DerSimonian-Laird random-effects meta-analysis across gut microbiome cohorts",
        fontsize=11, fontweight="bold",
    )

    # Cohort legend
    handles = [mpatches.Patch(color=COHORT_COLORS[c], label=COHORT_LABELS.get(c, c))
               for c in cohort_names if c in COHORT_COLORS]
    handles.append(mpatches.Patch(color="#37474F", label="Pooled (DerSimonian-Laird)"))
    ax.legend(handles=handles, fontsize=8.5, loc="lower right",
              framealpha=0.9, bbox_to_anchor=(0.62, 0.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.20, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{ts()}] Forest plot saved → {out_path}", flush=True)


# ── Funnel plot ────────────────────────────────────────────────────────────────

def funnel_plot(meta_df: pd.DataFrame, out_path: Path) -> None:
    """
    Funnel plot: pooled effect size vs SE (inverted y-axis).

    Symmetric funnel = no publication bias. Asymmetry may indicate
    selective reporting or small-study effects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    x = meta_df["pooled_effect"].values
    y = meta_df["SE"].values

    # Color by FDR significance
    colors = np.where(meta_df["q_value"].values < 0.05, "#E53935", "#90A4AE")
    ax.scatter(x, y, c=colors, s=28, alpha=0.75, zorder=3)

    # Pseudo-funnel lines (95% CI around mean pooled effect)
    mean_pe = np.average(x, weights=1.0 / y ** 2)  # precision-weighted mean
    se_range = np.linspace(0, y.max() * 1.05, 200)
    ax.plot(mean_pe + 1.96 * se_range, se_range,
            "--", color="#455A64", linewidth=1.2, alpha=0.6, label="Pseudo-95% CI")
    ax.plot(mean_pe - 1.96 * se_range, se_range,
            "--", color="#455A64", linewidth=1.2, alpha=0.6)
    ax.axvline(mean_pe, color="#455A64", linewidth=1.0, linestyle=":",
               alpha=0.5, label=f"Weighted mean = {mean_pe:.4f}")

    ax.set_xlabel("Pooled ElasticNet coefficient (effect size)", fontsize=11)
    ax.set_ylabel("Standard error (SE) — smaller SE at top", fontsize=11)
    ax.invert_yaxis()
    ax.set_title(
        "Funnel Plot — Publication Bias Check\n"
        f"{len(meta_df)} genera in meta-analysis  |  "
        f"FDR q<0.05: {(meta_df['q_value'] < 0.05).sum()}",
        fontsize=11, fontweight="bold",
    )

    sig_patch = mpatches.Patch(color="#E53935", label="FDR q < 0.05")
    ns_patch  = mpatches.Patch(color="#90A4AE", label="Not significant")
    ax.legend(handles=[sig_patch, ns_patch], fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.20, zorder=0)

    # Annotate top genera by |effect|
    for idx in meta_df["pooled_effect"].abs().nlargest(5).index:
        row = meta_df.loc[idx]
        ax.annotate(
            row["genus"],
            xy=(row["pooled_effect"], row["SE"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#333333",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{ts()}] Funnel plot saved → {out_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

print(f"[{ts()}] ═══ Phase 4: DerSimonian-Laird Meta-Analysis ═══", flush=True)
print(f"[{ts()}] Loading 3-cohort data …", flush=True)

clr    = pd.read_csv(CLR_PATH,    sep="\t", index_col="run_accession")
raw    = pd.read_csv(RAW_PATH,    sep="\t", index_col="run_accession")
labels = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")

response = labels.reindex(clr.index)["response"]
cohort   = labels.reindex(clr.index)["cohort"]
keep     = response.notna()
clr      = clr.loc[keep]; raw = raw.loc[keep]
response = response.loc[keep]; cohort = cohort.loc[keep]

print(f"[{ts()}]   n={len(clr)} samples, {clr.shape[1]} genera", flush=True)

cohort_map = {
    "cohort1": clr.index[clr.index.str.startswith("SRR5930")].tolist(),
    "cohort2": clr.index[clr.index.str.startswith("SRR11413")].tolist(),
    "cohort3": clr.index[clr.index.str.startswith("SRR6000")].tolist(),
}

# Check for C4 (Lee 2022) — include only if CLR matrix exists
if Path(LEE_CLR).exists():
    try:
        lee_clr = pd.read_csv(LEE_CLR, sep="\t", index_col="run_accession")
        lee_labels_path = "metadata/lee2022_labels.tsv"
        lee_labels = pd.read_csv(lee_labels_path, sep="\t").set_index("run_accession")
        lee_resp = lee_labels.reindex(lee_clr.index)["response"].dropna()
        lee_clr = lee_clr.loc[lee_resp.index]
        cohort_map["cohort4"] = lee_clr.index.tolist()
        print(f"[{ts()}] C4 Lee 2022 found: n={len(lee_clr)}", flush=True)
        # Extend main CLR and raw with C4
        lee_raw_path = "results/ml/lee2022/X_genus_raw.tsv"
        if Path(lee_raw_path).exists():
            lee_raw = pd.read_csv(lee_raw_path, sep="\t", index_col="run_accession")
            lee_raw = lee_raw.loc[lee_resp.index]
        else:
            lee_raw = pd.DataFrame(0, index=lee_clr.index, columns=clr.columns)
        clr = pd.concat([clr, lee_clr.reindex(columns=clr.columns, fill_value=0)])
        raw = pd.concat([raw, lee_raw.reindex(columns=raw.columns, fill_value=0)])
        response = pd.concat([response, lee_resp])
    except Exception as e:
        print(f"[{ts()}] C4 skipped (load error: {e})", flush=True)
else:
    print(f"[{ts()}] C4 Lee 2022 not available ({LEE_CLR} missing — download in progress). "
          "Meta-analysis will use C1+C2+C3 only.", flush=True)


# ── Per-cohort LOOCV ──────────────────────────────────────────────────────────

cohort_coefs = {}   # cname → {"mean": Series, "se": Series, "fold_df": DataFrame}

for cname, idx_list in cohort_map.items():
    clr_c  = clr.loc[idx_list]
    raw_c  = raw.loc[idx_list]
    resp_c = response.loc[idx_list]
    n_c    = len(idx_list)
    print(f"\n[{ts()}] {cname} (n={n_c}) — LOOCV feature selection + EN fitting …",
          flush=True)
    mean_c, se_c, fold_df = loocv_coefs(clr_c, raw_c, resp_c, cname)
    n_nz = (mean_c.abs() > 1e-6).sum()
    print(f"[{ts()}]   {cname}: nonzero mean coef in {n_nz} genera", flush=True)
    cohort_coefs[cname] = {"mean": mean_c, "se": se_c, "fold_df": fold_df, "n": n_c}


# ── Save per-cohort coefficients ──────────────────────────────────────────────

print(f"\n[{ts()}] Saving per-cohort LOOCV coefficients …", flush=True)
genera = clr.columns.tolist()
pc_rows = []
for genus in genera:
    row = {"genus": genus}
    for cname, cd in cohort_coefs.items():
        m = float(cd["mean"].get(genus, 0.0))
        s = float(cd["se"].get(genus, np.nan))
        row[f"mean_{cname}"] = round(m, 6)
        row[f"se_{cname}"]   = round(s, 6) if not np.isnan(s) else None
        row[f"nz_{cname}"]   = int(abs(m) > 1e-6)
    pc_rows.append(row)

pc_df = pd.DataFrame(pc_rows)
pc_path = OUT_DIR / "per_cohort_loocv_coefs.tsv"
pc_df.to_csv(pc_path, sep="\t", index=False)
print(f"[{ts()}] Saved: {pc_path}", flush=True)


# ── DerSimonian-Laird meta-analysis ───────────────────────────────────────────

print(f"\n[{ts()}] Running DerSimonian-Laird meta-analysis …", flush=True)
cohort_names = list(cohort_coefs.keys())

meta_rows = []
for genus in genera:
    eff, var, coh_contrib = [], [], []
    for cname in cohort_names:
        cd = cohort_coefs[cname]
        m = float(cd["mean"].get(genus, 0.0))
        s = float(cd["se"].get(genus, np.nan))
        if abs(m) > 1e-6 and not np.isnan(s) and s > 0:
            eff.append(m)
            var.append(s ** 2)
            coh_contrib.append(cname)

    if len(eff) < 2:
        continue

    eff = np.array(eff)
    var = np.array(var)
    res = dsl_meta(eff, var)

    # Direction consistency: all nonzero estimates agree in sign?
    dir_consistent = bool(np.all(eff > 0) or np.all(eff < 0))

    meta_rows.append({
        "genus":             genus,
        "n_cohorts":         len(eff),
        "cohorts":           ",".join(coh_contrib),
        "pooled_effect":     round(res["theta_re"], 6),
        "SE":                round(res["se_re"], 6),
        "ci_lo_95":          round(res["ci_lo"], 6),
        "ci_hi_95":          round(res["ci_hi"], 6),
        "tau2":              round(res["tau2"], 6),
        "I2":                round(res["I2"], 1),
        "Q":                 round(res["Q"], 4),
        "Q_p":               round(res["Q_p"], 4),
        "z_stat":            round(res["z"], 4),
        "p_raw":             round(res["p_z"], 6),
        "direction_consistent": dir_consistent,
        "direction":         "R+" if res["theta_re"] > 0 else "NR+",
    })

meta_df = pd.DataFrame(meta_rows)
print(f"[{ts()}]   Genera eligible for meta-analysis: {len(meta_df)}", flush=True)

if len(meta_df) == 0:
    print(f"[{ts()}] WARNING: No genera had nonzero mean coefficients in ≥2 cohorts.",
          flush=True)
else:
    # BH FDR correction
    meta_df["q_value"] = bh_fdr(meta_df["p_raw"].values).round(6)
    meta_df["fdr_sig"] = meta_df["q_value"] < 0.05
    meta_df = meta_df.sort_values("p_raw").reset_index(drop=True)

    out_tsv = OUT_DIR / "meta_analysis_results.tsv"
    meta_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[{ts()}] Saved: {out_tsv}", flush=True)

    # ── Figures ───────────────────────────────────────────────────────────────
    forest_plot(meta_df, cohort_coefs,
                out_path=OUT_DIR / "forest_plot.png")
    funnel_plot(meta_df,
                out_path=OUT_DIR / "funnel_plot.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_sig     = int(meta_df["fdr_sig"].sum())
    n_dir_con = int(meta_df["direction_consistent"].sum())
    mean_I2   = round(float(meta_df["I2"].mean()), 1)
    n_hetero  = int((meta_df["Q_p"] < 0.05).sum())
    top5 = meta_df.reindex(meta_df["pooled_effect"].abs().nlargest(5).index)

    print()
    print("═" * 65)
    print("PHASE 4 — META-ANALYSIS SUMMARY")
    print("═" * 65)
    print(f"  Cohorts included       : {len(cohort_names)}  ({', '.join(cohort_names)})")
    print(f"  Genera tested total    : {len(meta_df)}")
    print(f"  FDR q < 0.05           : {n_sig}")
    print(f"  Directionally consistent: {n_dir_con} / {len(meta_df)}")
    print(f"  Mean I²                : {mean_I2}%")
    print(f"  Significant heterogeneity (Q p<0.05): {n_hetero}")
    print()
    print("  Top 5 genera by |pooled_effect|:")
    print(f"  {'Genus':<28} {'Effect':>8} {'SE':>7} {'I²':>6} {'q-value':>9} {'Dir':>4} {'Consistent':>10}")
    print("  " + "-" * 75)
    for _, row in top5.iterrows():
        con_flag = "yes" if row["direction_consistent"] else "no"
        print(f"  {row['genus']:<28} {row['pooled_effect']:>+8.4f} "
              f"{row['SE']:>7.4f} {row['I2']:>5.1f}% "
              f"{row['q_value']:>9.4f}  {row['direction']:>4}  {con_flag:>10}")
    print()
    if n_sig == 0:
        print("  INTERPRETATION: No genus reached FDR significance.")
        print("  This confirms the Phase 3b finding (cross-cohort inversion p=0.213):")
        print("  heterogeneous effect directions across cohorts preclude reliable")
        print("  pooling. The null result is robust at every level of analysis.")
    print("═" * 65)
    print(f"\n[{ts()}] Phase 4 meta-analysis complete. Outputs in {OUT_DIR}/",
          flush=True)
