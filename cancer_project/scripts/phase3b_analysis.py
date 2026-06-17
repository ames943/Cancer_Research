#!/usr/bin/env python3
"""
Phase 3b: FDR-corrected feature stability + cross-cohort sign-inversion test.

Step 1 — Per-cohort LOOCV fold stability:
  Cohort 1 (n=39): loaded from existing feature_stability_n39.tsv
  Cohort 2 (n=40): new per-cohort LOOCV (no batch correction, single cohort)
  Cohort 3 (n=39): new per-cohort LOOCV (no batch correction, single cohort)
  Combined BH FDR correction: pooled binomial test across all three cohorts.

Step 2 — Cross-cohort sign inversion permutation test:
  Observed inversion rate: genera appearing in >=2 full-cohort EN fits with sign flip.
  Null: shuffle response labels within each cohort, refit EN, recount inversions.
  N_PERM=1000 permutations. p-value = fraction with null >= observed.

Step 3 — Summary table:
  results/ml/phase3b/feature_stability_summary.tsv
  Columns: genus, freq_nonzero_c1, sign_consistency_c1, freq_nonzero_c2,
           sign_consistency_c2, freq_nonzero_c3, sign_consistency_c3,
           cross_cohort_sign_consistent, bh_qvalue
"""

import os, warnings, time, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
CLR_PATH    = "results/ml/n118_3cohort/X_genus_clr.tsv"
RAW_PATH    = "results/ml/n118_3cohort/X_genus_raw.tsv"
LABELS_PATH = "metadata/response_labels_3cohort.tsv"
EXISTING_N39 = "results/ml/feature_stability/feature_stability_n39.tsv"
EXISTING_CC  = "results/ml/feature_stability/feature_stability_cross_cohort.tsv"
OUT_DIR = "results/ml/phase3b"
os.makedirs(OUT_DIR, exist_ok=True)

# ── hyperparameters ───────────────────────────────────────────────────────────
PREVALENCE_FRAC        = 0.10
VARIANCE_KEEP_FRACTION = 0.50
TOP_N                  = 100
EN_C                   = 1.0
EN_L1                  = 0.5
FDR_ALPHA              = 0.05
N_PERM                 = 1000
RNG                    = np.random.default_rng(42)

# ── load data ─────────────────────────────────────────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Loading data …", flush=True)
clr    = pd.read_csv(CLR_PATH,    sep="\t", index_col="run_accession")
raw    = pd.read_csv(RAW_PATH,    sep="\t", index_col="run_accession")
labels = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")
response = labels.reindex(clr.index)["response"]
keep     = response.notna()
clr      = clr.loc[keep]; raw = raw.loc[keep]; response = response.loc[keep]

FEATURES = clr.columns.tolist()
N_FEAT   = len(FEATURES)
feat_idx = {f: i for i, f in enumerate(FEATURES)}

c1_idx = clr.index[clr.index.str.startswith("SRR5930")].tolist()
c2_idx = clr.index[clr.index.str.startswith("SRR11413")].tolist()
c3_idx = clr.index[clr.index.str.startswith("SRR6000")].tolist()
print(f"  C1 n={len(c1_idx)}, C2 n={len(c2_idx)}, C3 n={len(c3_idx)}", flush=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def select_features(tr_clr, tr_raw, tr_labels):
    n        = len(tr_clr)
    min_prev = max(2, int(np.ceil(PREVALENCE_FRAC * n)))
    pres     = (tr_raw > 0).sum(axis=0)
    prev_cols = pres[pres >= min_prev].index.tolist()
    vv       = tr_clr[prev_cols].var(axis=0)
    high_var = vv[vv >= vv.quantile(1.0 - VARIANCE_KEEP_FRACTION)].index.tolist()
    y_bin    = (tr_labels == "R").astype(int).values
    pbs      = {c: abs(stats.pointbiserialr(y_bin, tr_clr[c].values)[0])
                for c in high_var}
    return pd.Series(pbs).sort_values(ascending=False).head(TOP_N).index.tolist()


def fit_en(X_tr, y_tr):
    m = LogisticRegression(
        penalty="elasticnet", solver="saga",
        C=EN_C, l1_ratio=EN_L1,
        class_weight="balanced", max_iter=5000, tol=1e-3, random_state=42,
    )
    m.fit(X_tr, y_tr)
    return m


def full_cohort_en_coefs(samples):
    """Fit EN on all samples; return coefficient vector over FEATURES."""
    clr_c  = clr.loc[samples]
    raw_c  = raw.loc[samples]
    resp_c = response.loc[samples]
    sel    = select_features(clr_c, raw_c, resp_c)
    X      = clr_c[sel].values
    y      = resp_c.values
    if len(np.unique(y)) < 2:
        return np.zeros(N_FEAT)
    m      = fit_en(X, y)
    coefs  = m.coef_[0] if m.coef_.shape[0] == 1 else m.coef_[list(m.classes_).index("R")]
    vec    = np.zeros(N_FEAT)
    for i, s in enumerate(sel):
        vec[feat_idx[s]] = coefs[i]
    return vec


def full_cohort_en_coefs_shuffled(samples, rng):
    """Fit EN with shuffled labels; return coefficient vector."""
    clr_c  = clr.loc[samples]
    raw_c  = raw.loc[samples]
    resp_c = response.loc[samples].copy()
    resp_c.values[:] = rng.permutation(resp_c.values)
    sel    = select_features(clr_c, raw_c, resp_c)
    X      = clr_c[sel].values
    y      = resp_c.values
    if len(np.unique(y)) < 2:
        return np.zeros(N_FEAT)
    m      = fit_en(X, y)
    coefs  = m.coef_[0] if m.coef_.shape[0] == 1 else m.coef_[list(m.classes_).index("R")]
    vec    = np.zeros(N_FEAT)
    for i, s in enumerate(sel):
        vec[feat_idx[s]] = coefs[i]
    return vec


def count_inversions(coef_c1, coef_c2, coef_c3):
    """
    Count genera nonzero in >=2 cohorts. Return (n_analyzed, n_inverted).
    """
    n_analyzed = 0
    n_inverted = 0
    for gi in range(N_FEAT):
        nz = [(v, ci) for v, ci in [(coef_c1[gi], 1), (coef_c2[gi], 2), (coef_c3[gi], 3)]
              if abs(v) > 1e-6]
        if len(nz) < 2:
            continue
        n_analyzed += 1
        signs = [np.sign(v) for v, _ in nz]
        if any(s > 0 for s in signs) and any(s < 0 for s in signs):
            n_inverted += 1
    return n_analyzed, n_inverted


def bh_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m     = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    q     = pvals * m / ranks
    q_mono = np.minimum.accumulate(q[order][::-1])[::-1]
    q_out  = np.empty(m)
    q_out[order] = np.minimum(q_mono, 1.0)
    return q_out


# ── LOOCV coef collection ─────────────────────────────────────────────────────

def run_loocv_collect_coefs(samples, tag):
    n        = len(samples)
    clr_s    = clr.loc[samples]
    raw_s    = raw.loc[samples]
    resp_s   = response.loc[samples]
    coef_mat = np.zeros((n, N_FEAT))
    sel_mat  = np.zeros((n, N_FEAT), dtype=bool)

    for fi, held in enumerate(samples):
        tr_idx  = [s for s in samples if s != held]
        tr_clr  = clr_s.loc[tr_idx]
        tr_raw  = raw_s.loc[tr_idx]
        tr_resp = resp_s.loc[tr_idx]

        sel = select_features(tr_clr, tr_raw, tr_resp)
        for s in sel:
            sel_mat[fi, feat_idx[s]] = True

        X_tr = tr_clr[sel].values
        y_tr = tr_resp.values
        if len(np.unique(y_tr)) < 2:
            continue

        m     = fit_en(X_tr, y_tr)
        coefs = m.coef_[0] if m.coef_.shape[0] == 1 else m.coef_[list(m.classes_).index("R")]
        for i, s in enumerate(sel):
            coef_mat[fi, feat_idx[s]] = coefs[i]

        if (fi + 1) % 10 == 0 or fi == n - 1:
            print(f"    [{tag}] fold {fi+1:3d}/{n}  sel={len(sel)}", flush=True)

    return coef_mat, sel_mat


def stability_stats(coef_mat, sel_mat, n_folds):
    """Return per-genus (freq_nonzero, sign_consistency, mean_coef, n_pos, n_nz, raw_p)."""
    rows = []
    for gi in range(N_FEAT):
        coefs_sel = coef_mat[sel_mat[:, gi], gi]
        coefs_nz  = coefs_sel[np.abs(coefs_sel) > 1e-6]
        n_nz = len(coefs_nz)
        n_pos = int((coefs_nz > 0).sum()) if n_nz > 0 else 0
        sign_frac = float(n_pos / n_nz) if n_nz > 0 else float("nan")
        mean_coef = float(coefs_nz.mean()) if n_nz > 0 else 0.0
        # binomial p for sign consistency (if enough data)
        if n_nz >= max(2, int(0.10 * n_folds)):
            pval = binomtest(n_pos, n_nz, p=0.5, alternative="two-sided").pvalue
        else:
            pval = 1.0
        rows.append((FEATURES[gi], n_nz, sign_frac, mean_coef, n_pos, n_nz, pval))
    return rows  # list of tuples per genus


# ── STEP 1: Per-cohort LOOCV ──────────────────────────────────────────────────

print(f"\n[{time.strftime('%H:%M:%S')}] Loading existing C1 stability (n=39) …", flush=True)
c1_stab = pd.read_csv(EXISTING_N39, sep="\t").set_index("genus")

print(f"[{time.strftime('%H:%M:%S')}] Running C2 LOOCV (n={len(c2_idx)}) …", flush=True)
c2_coef_mat, c2_sel_mat = run_loocv_collect_coefs(c2_idx, "C2")
c2_stats = stability_stats(c2_coef_mat, c2_sel_mat, len(c2_idx))

print(f"[{time.strftime('%H:%M:%S')}] Running C3 LOOCV (n={len(c3_idx)}) …", flush=True)
c3_coef_mat, c3_sel_mat = run_loocv_collect_coefs(c3_idx, "C3")
c3_stats = stability_stats(c3_coef_mat, c3_sel_mat, len(c3_idx))

# Build per-cohort dataframes
c2_df = pd.DataFrame(c2_stats, columns=["genus","freq_nz","sign_frac","mean_coef","n_pos","n_nz","raw_p"]).set_index("genus")
c3_df = pd.DataFrame(c3_stats, columns=["genus","freq_nz","sign_frac","mean_coef","n_pos","n_nz","raw_p"]).set_index("genus")

# ── Combined BH correction: pool n_pos, n_nz across all three cohorts ─────────
# For each genus: pool nonzero fold counts from C1 + C2 + C3

print(f"\n[{time.strftime('%H:%M:%S')}] Computing combined BH FDR correction …", flush=True)

combined_pvals = []
for genus in FEATURES:
    # C1 from loaded file
    if genus in c1_stab.index:
        c1_row = c1_stab.loc[genus]
        n_nz_c1 = int(c1_row["folds_nonzero"])
        n_pos_c1 = round(float(c1_row["sign_pos_frac"]) * n_nz_c1) if n_nz_c1 > 0 and not np.isnan(c1_row["sign_pos_frac"]) else 0
    else:
        n_nz_c1, n_pos_c1 = 0, 0

    gi = feat_idx[genus]
    # C2
    coefs_nz_c2 = c2_coef_mat[c2_sel_mat[:, gi], gi]
    coefs_nz_c2 = coefs_nz_c2[np.abs(coefs_nz_c2) > 1e-6]
    n_nz_c2 = len(coefs_nz_c2)
    n_pos_c2 = int((coefs_nz_c2 > 0).sum())
    # C3
    coefs_nz_c3 = c3_coef_mat[c3_sel_mat[:, gi], gi]
    coefs_nz_c3 = coefs_nz_c3[np.abs(coefs_nz_c3) > 1e-6]
    n_nz_c3 = len(coefs_nz_c3)
    n_pos_c3 = int((coefs_nz_c3 > 0).sum())

    n_nz_total  = n_nz_c1 + n_nz_c2 + n_nz_c3
    n_pos_total = n_pos_c1 + n_pos_c2 + n_pos_c3
    min_folds   = max(2, int(0.10 * (len(c1_idx) + len(c2_idx) + len(c3_idx))))
    if n_nz_total >= min_folds:
        pval = binomtest(n_pos_total, n_nz_total, p=0.5, alternative="two-sided").pvalue
    else:
        pval = 1.0
    combined_pvals.append(pval)

combined_qvals = bh_correction(np.array(combined_pvals))


# ── Load cross-cohort sign inversion data ─────────────────────────────────────
cc_df = pd.read_csv(EXISTING_CC, sep="\t").set_index("genus")


# ── Build summary table ───────────────────────────────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Building summary table …", flush=True)

rows = []
for gi, genus in enumerate(FEATURES):
    # C1
    if genus in c1_stab.index:
        r = c1_stab.loc[genus]
        freq_nz_c1   = int(r["folds_nonzero"])
        sign_cons_c1 = float(r["sign_pos_frac"]) if not np.isnan(r["sign_pos_frac"]) else float("nan")
    else:
        freq_nz_c1, sign_cons_c1 = 0, float("nan")

    # C2
    c2_r = c2_df.loc[genus] if genus in c2_df.index else None
    freq_nz_c2   = int(c2_r["freq_nz"]) if c2_r is not None else 0
    sign_cons_c2 = float(c2_r["sign_frac"]) if c2_r is not None and not np.isnan(c2_r["sign_frac"]) else float("nan")

    # C3
    c3_r = c3_df.loc[genus] if genus in c3_df.index else None
    freq_nz_c3   = int(c3_r["freq_nz"]) if c3_r is not None else 0
    sign_cons_c3 = float(c3_r["sign_frac"]) if c3_r is not None and not np.isnan(c3_r["sign_frac"]) else float("nan")

    # Cross-cohort sign consistency (from full-cohort EN fits)
    if genus in cc_df.index:
        cc_row = cc_df.loc[genus]
        cross_sign_consistent = bool(cc_row["sign_consistent"])
    else:
        cross_sign_consistent = None  # only 1 or 0 cohorts had nonzero coef

    bh_q = round(float(combined_qvals[gi]), 6)

    rows.append({
        "genus":                     genus,
        "freq_nonzero_c1":           freq_nz_c1,
        "sign_consistency_c1":       round(sign_cons_c1, 3) if not np.isnan(sign_cons_c1) else float("nan"),
        "freq_nonzero_c2":           freq_nz_c2,
        "sign_consistency_c2":       round(sign_cons_c2, 3) if not np.isnan(sign_cons_c2) else float("nan"),
        "freq_nonzero_c3":           freq_nz_c3,
        "sign_consistency_c3":       round(sign_cons_c3, 3) if not np.isnan(sign_cons_c3) else float("nan"),
        "cross_cohort_sign_consistent": cross_sign_consistent,
        "bh_qvalue":                 bh_q,
    })

summary_df = pd.DataFrame(rows).sort_values("bh_qvalue").reset_index(drop=True)
out_path = f"{OUT_DIR}/feature_stability_summary.tsv"
summary_df.to_csv(out_path, sep="\t", index=False)
print(f"  → {out_path}  ({len(summary_df)} rows)", flush=True)


# ── STEP 2: Permutation test on cross-cohort sign inversion rate ──────────────
print(f"\n[{time.strftime('%H:%M:%S')}] Computing full-cohort EN coefficients (observed) …", flush=True)

obs_c1 = full_cohort_en_coefs(c1_idx)
obs_c2 = full_cohort_en_coefs(c2_idx)
obs_c3 = full_cohort_en_coefs(c3_idx)
obs_analyzed, obs_inverted = count_inversions(obs_c1, obs_c2, obs_c3)
obs_inv_rate = obs_inverted / obs_analyzed if obs_analyzed > 0 else float("nan")
print(f"  Observed: {obs_inverted}/{obs_analyzed} genera inverted = {obs_inv_rate:.3f}", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] Running permutation test (N={N_PERM}) …", flush=True)
null_inv_rates = []
for pi in range(N_PERM):
    p_c1 = full_cohort_en_coefs_shuffled(c1_idx, RNG)
    p_c2 = full_cohort_en_coefs_shuffled(c2_idx, RNG)
    p_c3 = full_cohort_en_coefs_shuffled(c3_idx, RNG)
    na, ni = count_inversions(p_c1, p_c2, p_c3)
    null_inv_rates.append(ni / na if na > 0 else float("nan"))
    if (pi + 1) % 100 == 0:
        print(f"    perm {pi+1}/{N_PERM} …", flush=True)

null_arr = np.array([r for r in null_inv_rates if not np.isnan(r)])
perm_p = float(np.mean(null_arr >= obs_inv_rate))
null_mean = float(np.mean(null_arr))
null_std  = float(np.std(null_arr))
print(f"  Null: mean={null_mean:.3f} ± {null_std:.3f}", flush=True)
print(f"  Permutation p = {perm_p:.4f}", flush=True)

# Save permutation results
perm_df = pd.DataFrame({
    "metric": ["obs_genera_analyzed", "obs_inversions", "obs_inversion_rate",
               "null_mean_inversion_rate", "null_std_inversion_rate",
               "permutation_p", "n_permutations"],
    "value":  [obs_analyzed, obs_inverted, round(obs_inv_rate, 4),
               round(null_mean, 4), round(null_std, 4),
               round(perm_p, 4), N_PERM],
})
perm_df.to_csv(f"{OUT_DIR}/inversion_permutation_test.tsv", sep="\t", index=False)

null_dist_df = pd.DataFrame({"null_inversion_rate": null_arr})
null_dist_df.to_csv(f"{OUT_DIR}/inversion_null_distribution.tsv", sep="\t", index=False)


# ── STEP 3: Plain-language summary ───────────────────────────────────────────
print(f"\n[{time.strftime('%H:%M:%S')}] === PHASE 3B RESULTS ===\n", flush=True)

# FDR summary
n_fdr_sig = (summary_df["bh_qvalue"] < FDR_ALPHA).sum()
top_stable = summary_df[summary_df["bh_qvalue"] < FDR_ALPHA].head(20)

print("── STEP 1: Within-cohort fold stability ─────────────────────────────────", flush=True)
print(f"  Total genera tested: {len(summary_df)}", flush=True)
print(f"  FDR-significant (BH q < {FDR_ALPHA}, combined across C1+C2+C3): {n_fdr_sig}", flush=True)
print(f"", flush=True)

if n_fdr_sig > 0:
    # Count sign directions among FDR-sig genera
    sig_df = summary_df[summary_df["bh_qvalue"] < FDR_ALPHA].copy()
    # Direction = stable-positive if sign_consistency_c1 > 0.8 (or best available cohort)
    def classify_dir(row):
        fracs = [row["sign_consistency_c1"], row["sign_consistency_c2"], row["sign_consistency_c3"]]
        fracs = [f for f in fracs if not np.isnan(f)]
        if not fracs:
            return "unknown"
        mean_frac = np.mean(fracs)
        if mean_frac > 0.8:
            return "stable-positive (R+)"
        elif mean_frac < 0.2:
            return "stable-negative (NR+)"
        else:
            return "sign-mixed (inversion)"
    sig_df["direction"] = sig_df.apply(classify_dir, axis=1)
    dir_counts = sig_df["direction"].value_counts()
    print("  Direction breakdown among FDR-significant genera:", flush=True)
    for d, c in dir_counts.items():
        print(f"    {d}: {c}", flush=True)
    print(f"", flush=True)
    print("  Top 15 FDR-significant genera:", flush=True)
    display_cols = ["genus","freq_nonzero_c1","sign_consistency_c1",
                    "freq_nonzero_c2","sign_consistency_c2",
                    "freq_nonzero_c3","sign_consistency_c3","bh_qvalue"]
    print(sig_df[display_cols].head(15).to_string(index=False), flush=True)
else:
    print("  No FDR-significant genera found.", flush=True)

print(f"", flush=True)
print("── STEP 2: Cross-cohort sign inversion test ─────────────────────────────", flush=True)
print(f"  Genera nonzero in >=2 full-cohort EN fits:  {obs_analyzed}", flush=True)
print(f"  Sign-consistent (same direction):            {obs_analyzed - obs_inverted} ({100*(1-obs_inv_rate):.1f}%)", flush=True)
print(f"  Sign-inverted (direction flip):              {obs_inverted} ({100*obs_inv_rate:.1f}%)", flush=True)
print(f"  Null inversion rate (shuffled labels):       {null_mean:.3f} ± {null_std:.3f}", flush=True)
print(f"  Permutation p-value:                         {perm_p:.4f}", flush=True)
print(f"", flush=True)

print("── CONCLUSION ───────────────────────────────────────────────────────────", flush=True)
if perm_p < 0.05:
    verdict = "REAL INVERSION"
    detail = (
        f"The {obs_inverted}/{obs_analyzed} ({100*obs_inv_rate:.1f}%) sign-inversion rate is "
        f"significantly higher than the null expectation "
        f"({100*null_mean:.1f}% ± {100*null_std:.1f}%, permutation p={perm_p:.4f}). "
        f"Below-chance AUCs in Phase 0.5 cross-cohort transfer reflect GENUINE "
        f"feature-direction inversion across cohorts — the same genus predicts "
        f"response in one cohort and non-response in another. This is a real "
        f"biological or technical inconsistency that batch correction cannot resolve."
    )
else:
    verdict = "NOISE"
    detail = (
        f"The {obs_inverted}/{obs_analyzed} ({100*obs_inv_rate:.1f}%) sign-inversion rate "
        f"is NOT significantly higher than the null expectation "
        f"({100*null_mean:.1f}% ± {100*null_std:.1f}%, permutation p={perm_p:.4f}). "
        f"Below-chance AUCs in Phase 0.5 cross-cohort transfer most likely reflect "
        f"PURE NOISE — coefficients are unstable across cohorts because there is no "
        f"consistent signal to learn, not because of systematic direction reversal. "
        f"The number of genera with apparent sign inversion is consistent with what "
        f"you would see if labels were randomly shuffled."
    )

print(f"  Verdict: {verdict}", flush=True)
print(f"", flush=True)
# Wrap at 80 chars
import textwrap
for line in textwrap.wrap(detail, width=78):
    print(f"  {line}", flush=True)

if n_fdr_sig > 0:
    print(f"", flush=True)
    print(f"  Additionally: {n_fdr_sig} genera show FDR-significant sign consistency "
          f"(BH q < {FDR_ALPHA}) when pooling fold coefficients across all three cohorts. "
          f"These are candidates for the Phase 4 meta-analytic effect combination.", flush=True)

print(f"\n[{time.strftime('%H:%M:%S')}] Outputs:", flush=True)
print(f"  {OUT_DIR}/feature_stability_summary.tsv", flush=True)
print(f"  {OUT_DIR}/inversion_permutation_test.tsv", flush=True)
print(f"  {OUT_DIR}/inversion_null_distribution.tsv", flush=True)
print(f"[{time.strftime('%H:%M:%S')}] Phase 3b complete.", flush=True)
