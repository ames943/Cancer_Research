#!/usr/bin/env python3
"""
Phase 0 Fix: Dirichlet bootstrap for corrected PERMANOVA CIs and power analysis.

Naive bootstrap-with-replacement on a distance matrix reuses rows/columns
of the same matrix, creating zero-distance duplicate-sample pairs.  That
deflates SS_within and inflates pseudo-F and R², so point estimates fall
*below* their own bootstrapped lower CI bounds.

Fix: parametric (Dirichlet) bootstrap.  For each sample, the observed
genus-level percentage vector defines Dirichlet concentration parameters
(after applying a 1e-6 pseudocount to zeros).  Each bootstrap iteration
draws a fresh composition per sample, CLR-transforms, recomputes the
Aitchison distance matrix, and runs PERMANOVA -- no duplicate pairs.

Outputs
-------
results/ml/batch_diagnostics/bootstrap_cis_naive_biased.tsv   old biased CIs
results/ml/batch_diagnostics/bootstrap_cis.tsv                Dirichlet-corrected CIs
results/ml/power_analysis/power_curve_corrected.tsv
results/ml/power_analysis/power_curve_corrected.png
results/ml/batch_diagnostics/permdisp_interpretation.md

Usage
-----
    cd cancer_project/      # the inner directory containing scripts/, results/, metadata/
    python3 scripts/phase0_dirichlet_bootstrap.py
"""

import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm as sp_norm
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────────
N118_CLR  = "results/ml/n118_3cohort/X_genus_clr.tsv"
N118_RAW  = "results/ml/n118_3cohort/X_genus_raw.tsv"
LABELS_3C = "metadata/response_labels_3cohort.tsv"

LOOCV_N39  = "results/ml/loocv_n39_enet_results.tsv"
LOOCV_N79U = "results/ml/loocv_linear_results_ElasticNet_LogReg.tsv"
LOOCV_N79C = "results/ml/loocv_combat_results_ElasticNet_LogReg.tsv"
LOOCV_N118 = "results/ml/n118_3cohort/loocv_3cohort_results_RandomForest.tsv"

OUT_DIAG  = Path("results/ml/batch_diagnostics")
OUT_POWER = Path("results/ml/power_analysis")
OUT_DIAG.mkdir(parents=True, exist_ok=True)
OUT_POWER.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEED          = 42
N_BOOT        = 1000
N_REPS_POWER  = 200
N_PERMS_POWER = 199
TARGET_NS     = [39, 60, 80, 100, 150, 200, 300]
PSEUDOCOUNT   = 1e-6

rng       = np.random.default_rng(SEED)
rng_power = np.random.default_rng(SEED + 999)


# ── Helpers ────────────────────────────────────────────────────────────────────
def clr_transform(X_ra: np.ndarray) -> np.ndarray:
    """CLR-transform a (n, p) non-negative relative-abundance matrix."""
    lx = np.log(X_ra + PSEUDOCOUNT)
    return lx - lx.mean(axis=1, keepdims=True)


def aitchison_dist(clr: np.ndarray) -> np.ndarray:
    return squareform(pdist(clr, metric="euclidean"))


def _ss_total(d2: np.ndarray) -> float:
    return float(np.sum(np.triu(d2, k=1))) / d2.shape[0]


def _ss_within(d2: np.ndarray, grp: np.ndarray) -> float:
    sw = 0.0
    for g in np.unique(grp):
        m = grp == g
        ng = int(m.sum())
        if ng < 2:
            continue
        sub = d2[np.ix_(m, m)]
        sw += float(np.sum(np.triu(sub, k=1))) / ng
    return sw


def permanova_r2(D: np.ndarray, grp: np.ndarray) -> float:
    d2 = D ** 2
    ST = _ss_total(d2)
    return (ST - _ss_within(d2, grp)) / ST if ST > 0 else 0.0


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    z = sp_norm.ppf(1 - alpha / 2)
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, c - m), min(1.0, c + m)


def dirichlet_draw(alpha_matrix: np.ndarray, rng_: np.random.Generator) -> np.ndarray:
    """
    Draw one fresh Dirichlet sample per row of alpha_matrix.

    Uses the Gamma-normalization identity: X ~ Dirichlet(alpha) iff
    X = G / sum(G) where G_i ~ Gamma(alpha_i, 1) independently.

    alpha_matrix: (n, p) -- all entries > 0
    Returns:      (n, p) -- each row sums to 1
    """
    gammas = rng_.standard_gamma(alpha_matrix)          # (n, p) independent draws
    return gammas / gammas.sum(axis=1, keepdims=True)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
clr_n118 = pd.read_csv(N118_CLR, sep="\t", index_col="run_accession")
raw_n118 = pd.read_csv(N118_RAW, sep="\t", index_col="run_accession")
labels   = pd.read_csv(LABELS_3C, sep="\t").set_index("run_accession")

response  = labels["response"].reindex(clr_n118.index)
cohort_id = pd.Series(
    [1 if a.startswith("SRR5930") else 2 if a.startswith("SRR11413") else 3
     for a in clr_n118.index],
    index=clr_n118.index,
)

c1_mask  = cohort_id == 1
c12_mask = cohort_id.isin([1, 2])

resp_39  = response.loc[c1_mask]
coh_39   = cohort_id.loc[c1_mask]
resp_79  = response.loc[c12_mask]
coh_79   = cohort_id.loc[c12_mask]
resp_118 = response.loc[clr_n118.index]
coh_118  = cohort_id.loc[clr_n118.index]

# Concentration parameters: observed percentage + pseudocount on zeros.
# X_genus_raw stores Kraken2 percentage values; adding PSEUDOCOUNT makes every
# entry strictly positive so all Dirichlet draws are nonzero.
alpha_39  = raw_n118.loc[c1_mask].values  + PSEUDOCOUNT   # (39,  p)
alpha_79  = raw_n118.loc[c12_mask].values + PSEUDOCOUNT   # (79,  p)
alpha_118 = raw_n118.values               + PSEUDOCOUNT   # (118, p)

print(f"  Cohort 1 (n=39) : {alpha_39.shape[1]} genera")
print(f"  Cohort 1+2 (n=79)")
print(f"  All 3 cohorts (n=118)")

# Pre-compute Aitchison distances from real CLR data (for point estimates)
print("\nComputing point-estimate PERMANOVA R² from observed CLR data …")
D_39  = aitchison_dist(clr_n118.loc[c1_mask].values)
D_79  = aitchison_dist(clr_n118.loc[c12_mask].values)
D_118 = aitchison_dist(clr_n118.values)

boot_configs = [
    # (metric,                 n_label,  alpha_mat,  group_arr,        D_real)
    ("PERMANOVA_response_R2", "n=39",   alpha_39,   resp_39.values,   D_39 ),
    ("PERMANOVA_response_R2", "n=79",   alpha_79,   resp_79.values,   D_79 ),
    ("PERMANOVA_response_R2", "n=118",  alpha_118,  resp_118.values,  D_118),
    ("PERMANOVA_batch_R2",    "n=79",   alpha_79,   coh_79.values,    D_79 ),
    ("PERMANOVA_batch_R2",    "n=118",  alpha_118,  coh_118.values,   D_118),
]

for metric, n_label, _, grp, D in boot_configs:
    print(f"  {metric:30s} {n_label}: R² = {permanova_r2(D, np.asarray(grp)):.6f}")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 0: Preserve naive biased CIs for transparency
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 0: Preserving naive biased CIs …")
naive_src = OUT_DIAG / "bootstrap_cis.tsv"
naive_dst = OUT_DIAG / "bootstrap_cis_naive_biased.tsv"
if naive_src.exists():
    shutil.copy2(naive_src, naive_dst)
    print(f"  Backed up: {naive_src} → {naive_dst}")
else:
    print(f"  WARNING: {naive_src} not found — skipping backup")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: Dirichlet bootstrap for PERMANOVA R² CIs
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 1: Dirichlet bootstrap PERMANOVA R² CIs  (N={N_BOOT})")
print("=" * 65)
print("Each iteration draws a fresh Dirichlet composition per sample,")
print("CLR-transforms, and recomputes the Aitchison distance matrix.")

boot_rows = []

for metric, n_label, alpha, grp, D_real in boot_configs:
    grp_arr = np.asarray(grp)
    obs_r2  = permanova_r2(D_real, grp_arr)
    boot_r2s = np.empty(N_BOOT)

    n_samp = len(alpha)
    print(f"\n  {metric} {n_label}  (n={n_samp}) …", flush=True)
    for b in range(N_BOOT):
        X_ra        = dirichlet_draw(alpha, rng)     # (n, p) fresh compositions
        X_clr       = clr_transform(X_ra)            # (n, p) CLR
        D_b         = aitchison_dist(X_clr)          # (n, n) Aitchison distances
        boot_r2s[b] = permanova_r2(D_b, grp_arr)
        if (b + 1) % 250 == 0:
            print(f"    {b + 1}/{N_BOOT} iterations …", flush=True)

    ci_lo, ci_hi = np.percentile(boot_r2s, [2.5, 97.5])
    inside = ci_lo <= obs_r2 <= ci_hi
    flag   = "✓" if inside else "⚠ OUTSIDE"
    print(f"  → R²={obs_r2:.6f}  95% CI [{ci_lo:.6f}, {ci_hi:.6f}]  {flag}")
    boot_rows.append({
        "metric":         metric,
        "n":              n_label,
        "point_estimate": round(obs_r2, 6),
        "ci_lower":       round(ci_lo, 6),
        "ci_upper":       round(ci_hi, 6),
        "n_boot":         N_BOOT,
        "method":         "dirichlet",
    })


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2: AUC bootstrap CI verification
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: AUC bootstrap CI verification  (patient-level resampling)")
print("=" * 65)
print("AUC CIs resample (true_label, predicted_prob) pairs at the patient")
print("level -- no distance matrix involved, so duplicate-distance bias")
print("cannot occur.  Verifying point estimates fall within CIs.")

auc_configs = [
    ("LOOCV_AUC_ENet", "n=39",             LOOCV_N39 ),
    ("LOOCV_AUC_ENet", "n=79_uncorrected", LOOCV_N79U),
    ("LOOCV_AUC_ENet", "n=79_combat",      LOOCV_N79C),
    ("LOOCV_AUC_RF",   "n=118_combat",     LOOCV_N118),
]

auc_note_lines = []
for metric, n_label, fpath in auc_configs:
    df      = pd.read_csv(fpath, sep="\t")
    y_true  = (df["actual"] == "R").astype(int).values
    y_score = df["predicted_prob_R"].values
    obs_auc = roc_auc_score(y_true, y_score)

    boot_aucs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx          = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys       = y_true[idx], y_score[idx]
        boot_aucs[b] = roc_auc_score(yt, ys) if 0 < yt.sum() < len(yt) else 0.5

    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    inside = ci_lo <= obs_auc <= ci_hi
    status = "no correction needed" if inside else "INVESTIGATE"
    print(f"  {'✓' if inside else '⚠'} {metric} {n_label}: "
          f"AUC={obs_auc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  — {status}")
    auc_note_lines.append(
        f"- {metric} {n_label}: AUC={obs_auc:.4f}  CI=[{ci_lo:.4f},{ci_hi:.4f}]"
        f"  {'✓ no correction needed' if inside else '⚠ investigate'}"
    )
    boot_rows.append({
        "metric":         metric,
        "n":              n_label,
        "point_estimate": round(obs_auc, 6),
        "ci_lower":       round(ci_lo, 6),
        "ci_upper":       round(ci_hi, 6),
        "n_boot":         N_BOOT,
        "method":         "patient_resample",
    })

# Write updated bootstrap_cis.tsv
boot_df  = pd.DataFrame(boot_rows)
boot_out = OUT_DIAG / "bootstrap_cis.tsv"
boot_df[["metric", "n", "point_estimate", "ci_lower", "ci_upper",
         "n_boot", "method"]].to_csv(boot_out, sep="\t", index=False)
print(f"\nSaved: {boot_out}")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3: Power analysis with Dirichlet bootstrap
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 3: Power analysis — Dirichlet bootstrap")
print(f"        {N_REPS_POWER} reps × {N_PERMS_POWER} internal PERMANOVA perms")
print("=" * 65)
print("For each target n, n indices are drawn with replacement from cohort 1")
print("(n=39).  Each drawn index gets a FRESH Dirichlet realization, so even")
print("repeated indices produce distinct synthetic samples (no zero distances).")

# Cohort-1 concentration parameters and response labels
y_c1 = (resp_39 == "R").values.astype(np.int8)  # (39,) binary

print(f"\n{'n':>6}  {'power':>7}  {'95% CI Wilson':>18}  ({N_REPS_POWER} reps × {N_PERMS_POWER} perms)")
print("-" * 52)

power_rows_corr = []
power_vals_corr = []

for n_target in TARGET_NS:
    n_sig = 0

    for _ in range(N_REPS_POWER):
        # Resample n_target patient indices WITH replacement from cohort 1
        idx_draw   = rng_power.integers(0, len(alpha_39), size=n_target)
        alpha_draw = alpha_39[idx_draw]                  # (n_target, p) -- may repeat rows
        X_ra       = dirichlet_draw(alpha_draw, rng_power)   # fresh draw per row
        X_clr      = clr_transform(X_ra)
        D_s        = aitchison_dist(X_clr)
        y_s        = y_c1[idx_draw]

        if y_s.sum() == 0 or y_s.sum() == n_target:
            continue                                      # degenerate: single class

        d2  = D_s ** 2
        ST  = _ss_total(d2)
        SW0 = _ss_within(d2, y_s)
        SA0 = ST - SW0
        den = n_target - 2
        F0  = SA0 / SW0 * den if SW0 > 0 else 0.0

        # Inner permutation test (label shuffle only; no new Dirichlet draws)
        n_ge = 0
        for _ in range(N_PERMS_POWER):
            y_p  = rng_power.permutation(y_s)
            SW_p = _ss_within(d2, y_p)
            F_p  = (ST - SW_p) / SW_p * den if SW_p > 0 else 0.0
            if F_p >= F0:
                n_ge += 1

        if n_ge / N_PERMS_POWER < 0.05:
            n_sig += 1

    power = n_sig / N_REPS_POWER
    ci_lo, ci_hi = wilson_ci(n_sig, N_REPS_POWER)
    print(f"  {n_target:>4}  {power:>7.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)
    power_rows_corr.append({
        "n":                 n_target,
        "power":             round(power, 4),
        "ci_lower":          round(ci_lo, 4),
        "ci_upper":          round(ci_hi, 4),
        "n_reps":            N_REPS_POWER,
        "n_perms_internal":  N_PERMS_POWER,
        "method":            "dirichlet",
    })
    power_vals_corr.append(power)

corr_df = pd.DataFrame(power_rows_corr)
corr_df.to_csv(OUT_POWER / "power_curve_corrected.tsv", sep="\t", index=False)
print(f"\nSaved: {OUT_POWER}/power_curve_corrected.tsv")

# 80% power crossing
n_80pct = None
for i, (nt, pwr) in enumerate(zip(TARGET_NS, power_vals_corr)):
    if pwr >= 0.80:
        n_80pct = nt
        break
if n_80pct is None:
    for i in range(len(power_vals_corr) - 1):
        if power_vals_corr[i] < 0.80 <= power_vals_corr[i + 1]:
            frac    = (0.80 - power_vals_corr[i]) / (power_vals_corr[i + 1] - power_vals_corr[i])
            n_80pct = int(TARGET_NS[i] + frac * (TARGET_NS[i + 1] - TARGET_NS[i]))
            break

# Load old (naive biased) power curve for comparison plot
old_df = pd.read_csv(OUT_POWER / "power_curve.tsv", sep="\t")

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(TARGET_NS, power_vals_corr, "o-", color="#2196F3", linewidth=2,
        markersize=7, label="Dirichlet bootstrap (corrected)")
ax.fill_between(TARGET_NS,
                [r["ci_lower"] for r in power_rows_corr],
                [r["ci_upper"] for r in power_rows_corr],
                alpha=0.18, color="#2196F3", label="95% Wilson CI (corrected)")
ax.plot(old_df["n"], old_df["power"], "s--", color="#9E9E9E", linewidth=1.5,
        markersize=6, alpha=0.75, label="Naive bootstrap (biased upper bound)")
ax.axhline(0.80, color="#F44336", linestyle="--", linewidth=1.2, label="80% power target")
ax.axhline(0.05, color="gray",    linestyle=":",  linewidth=1.0, label="α = 0.05")
ax.set_xlabel("Sample size  n", fontsize=12)
ax.set_ylabel("Empirical power  P(PERMANOVA p < 0.05)", fontsize=12)
ax.set_title(
    "PERMANOVA(response) power — Dirichlet vs. naive bootstrap\n"
    f"Effect basis: cohort-1 (n=39), observed R² ≈ 0.027   "
    f"({N_REPS_POWER} reps × {N_PERMS_POWER} perms)",
    fontsize=11,
)
ax.set_ylim(-0.02, 1.05)
ax.set_xticks(TARGET_NS)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_POWER / "power_curve_corrected.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_POWER}/power_curve_corrected.png")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4: PERMDISP interpretation note
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: Writing permdisp_interpretation.md …")

permdisp_md = """\
# PERMDISP Interpretation Note

## Overview

Aitchison-distance PERMDISP (Anderson 2006) was run alongside each PERMANOVA
to assess whether dispersion homogeneity holds across the grouping variables
tested (response: R vs. NR; batch: cohort identity).  Significant PERMDISP
means the PERMANOVA R² conflates centroid shift with variance differences,
and should be interpreted with caution.

---

## Response grouping (R vs. NR)

Within-group dispersions were **homogeneous** at every cohort size tested:

| Dataset           |  n  | F-stat | p-value | Homogeneous? |
|-------------------|-----|--------|---------|--------------|
| Cohort 1 only     |  39 |  1.27  |  0.341  | Yes          |
| Cohorts 1+2       |  79 |  1.99  |  0.192  | Yes          |
| All 3 cohorts     | 118 |  0.10  |  0.779  | Yes          |

**Implication:** The response PERMANOVA results are not confounded by dispersion
differences between R and NR.  The monotonically declining response R²
(0.027 → 0.011 → 0.007 as cohorts are added) reflects a genuine shrinkage
of the centroid signal, not a masking artefact from unequal within-group
spread.  PERMANOVA p-values for response (0.418 / 0.695 / 0.847) are valid.

---

## Batch/cohort grouping

Within-batch dispersions were **significantly heterogeneous** in every
multi-cohort dataset:

| Dataset       |  n  | Groups | F-stat | p-value | Homogeneous? |
|---------------|-----|--------|--------|---------|--------------|
| Cohorts 1+2   |  79 |   2    |  9.59  |  0.005  | **No**       |
| All 3 cohorts | 118 |   3    |  5.63  |  0.006  | **No**       |

Within-group dispersions (mean Aitchison distance to group centroid):

- n=79:  Cohort 1 = 42.49,  Cohort 2 = 47.08
- n=118: Cohort 1 = 42.49,  Cohort 2 = 47.08,  Cohort 3 = 44.16

Cohort 2 (NovaSeq) has noticeably greater within-cohort compositional spread
than Cohort 1 (HiSeq) or Cohort 3 (NextSeq), consistent with sequencing-
platform effects on measured diversity.

**Implication:** The batch PERMANOVA R² (0.085 at n=79, 0.077 at n=118) reflects
a mixture of two distinct effects: (1) centroid shift between cohorts (mean
compositional differences) and (2) differences in within-cohort compositional
spread.  The reported R² values are therefore **upper bounds** on the pure
centroid-shift effect.  Mechanistically, the heterogeneity likely arises from
platform-specific read-depth and diversity profiles (HiSeq / NovaSeq / NextSeq),
which differ in both mean composition estimates and variance structure.

### Mitigating factor: near-balanced design

Anderson & Walsh (2013) demonstrated that PERMANOVA maintains acceptable Type I
error rates under dispersion heterogeneity when group sizes are approximately
balanced.  Our cohort sizes (39 / 40 / 39) are near-perfectly balanced,
attenuating the risk that heterogeneous dispersions inflate the batch R² through
inflated Type I error.  Internal comparisons of response R² across cohort sizes
are therefore still interpretable as a measure of signal dilution, even in the
presence of batch dispersion heterogeneity.

---

## Forward reference: Phase 1 batch-correction comparison

The batch PERMDISP heterogeneity has direct mechanistic implications for
batch-correction method selection.

ComBat-seq (Johnson et al. 2007; Leek et al. extension) applies a
location-and-scale correction: it shifts cohort mean compositions toward a
common reference and rescales within-cohort variances.  However, its scale
correction operates on marginal feature-level variances, not on the full
compositional dispersion structure captured by Aitchison distances.  When
the observed dispersion heterogeneity is **structural** -- arising from genuine
platform-level differences in which taxa are detected and at what variability --
ComBat may:

- **Over-correct**: remove biologically meaningful within-cohort variance
  structure along with the batch mean shift, leaving post-correction data
  that is artificially uniform (consistent with the collapse of response AUC
  below the permuted null after per-fold ComBat at n=79/118).
- **Under-correct**: fail to equalize dispersions even after centering means,
  leaving residual dispersion differences that continue to confound downstream
  distance-based analyses.

This motivates the **Phase 1 correction-method comparison** (ConQuR, MMUPHin,
percentile normalization, cohort-as-covariate), which will assess:

1. Post-correction batch R² (PERMANOVA) -- does correction reduce batch signal?
2. Post-correction response R² -- is biological signal preserved or destroyed?
3. Post-correction PERMDISP(batch) -- does the dispersion heterogeneity resolve?
4. LOOCV AUC and permutation p-value -- end-to-end predictive utility.

A well-performing correction should reduce batch PERMANOVA R² and ideally
bring PERMDISP(batch) to p ≥ 0.05, while leaving response R² and LOOCV AUC
unchanged or improved relative to uncorrected pooling.  Methods that treat
dispersion heterogeneity explicitly (e.g., rank-based percentile normalization)
may outperform mean-centering approaches for this dataset.

---

## References

Anderson, M.J. (2006). Distance-based tests for homogeneity of multivariate
dispersions. *Biometrics*, 62(1), 245–253.

Anderson, M.J. & Walsh, D.C.I. (2013). PERMANOVA, ANOSIM, and the Mantel test
in the face of heterogeneous dispersions: What null hypothesis are you testing?
*Ecological Monographs*, 83(4), 557–574. https://doi.org/10.1890/12-2010.1

Johnson, W.E., Li, C. & Rabinovic, A. (2007). Adjusting batch effects in
microarray expression data using empirical Bayes methods. *Biostatistics*,
8(1), 118–127.
"""

with open(OUT_DIAG / "permdisp_interpretation.md", "w") as fh:
    fh.write(permdisp_md)
print(f"Saved: {OUT_DIAG}/permdisp_interpretation.md")


# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print("\n── 1. Dirichlet bootstrap PERMANOVA R² CIs ─────────────────────")
perm_boot = [r for r in boot_rows if r["metric"].startswith("PERMANOVA")]
all_inside = True
for r in perm_boot:
    inside = r["ci_lower"] <= r["point_estimate"] <= r["ci_upper"]
    if not inside:
        all_inside = False
    print(f"  {'✓' if inside else '⚠'} {r['metric']:30s} {r['n']:6s}: "
          f"R²={r['point_estimate']:.4f}  95% CI [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
          f"  {'inside' if inside else 'OUTSIDE -- check'}")
if all_inside:
    print("  All point estimates fall inside their Dirichlet-bootstrap CIs. ✓")

print("\n── 2. AUC bootstrap CIs (patient-level resampling, no fix needed) ──")
for line in auc_note_lines:
    print(f"  {line}")

print("\n── 3. Power analysis (Dirichlet bootstrap) ─────────────────────")
for r in power_rows_corr:
    print(f"  n={r['n']:>4}: power={r['power']:.3f}  CI=[{r['ci_lower']:.3f},{r['ci_upper']:.3f}]")
if n_80pct:
    print(f"\n  → 80% power first reached at approximately n = {n_80pct}")
else:
    print(f"\n  → 80% power NOT reached within n ≤ {TARGET_NS[-1]}")
    print(f"     Observed R² ≈ 0.027 is a small effect; n > 300 likely required.")

print(f"\n── 4. PERMDISP interpretation ───────────────────────────────────")
print(f"  {OUT_DIAG}/permdisp_interpretation.md")

print("\nOutputs:")
for f in [
    OUT_DIAG / "bootstrap_cis_naive_biased.tsv",
    OUT_DIAG / "bootstrap_cis.tsv",
    OUT_POWER / "power_curve_corrected.tsv",
    OUT_POWER / "power_curve_corrected.png",
    OUT_DIAG / "permdisp_interpretation.md",
]:
    print(f"  {f}")
