#!/usr/bin/env python3
"""
Steps 1-4: C4 (Lee 2022) integration.

1. Build C4-only CLR matrix  → results/ml/lee2022/X_genus_{raw,clr}.tsv
2. Build n=283 (4-cohort)    → results/ml/n283_4cohort/X_genus_{raw,clr}.tsv
3. PERMANOVA (999 perms)     → response + batch on n=283
4. PERMDISP (999 perms)      → response + batch on n=283

Run from cancer_project/:
    python3 scripts/steps_1_4_c4_integration.py
"""

import csv
import glob
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

EXCLUDED_GENERA = {"Homo"}
PSEUDOCOUNT     = 1e-6


def ts():
    return time.strftime("%H:%M:%S")


# ── Kraken2 parser ────────────────────────────────────────────────────────────

def parse_kraken_report(fp: str) -> dict:
    """Return genus→pct dict from a single Kraken2 report."""
    genus_data = {}
    with open(fp) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6 or parts[3] != "G":
                continue
            pct     = float(parts[0].strip())
            stripped = parts[5].strip()
            if not stripped:
                continue
            if stripped.startswith("Candidatus "):
                name = "Candidatus_" + stripped.split()[1]
            elif " (" in stripped:
                name = stripped.split(" (")[0]
            else:
                name = stripped.split()[0]
            if name in EXCLUDED_GENERA:
                continue
            genus_data[name] = genus_data.get(name, 0.0) + pct
    return genus_data


# ── Build raw + CLR matrices ──────────────────────────────────────────────────

def build_matrices(samples: dict, out_dir: Path) -> tuple:
    """
    Given {sample_id: {genus: pct}} build raw + CLR TSV files.
    Returns (raw_df, clr_df) as DataFrames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_genera = sorted({g for gd in samples.values() for g in gd})

    sample_ids = sorted(samples)

    # raw
    raw_out = out_dir / "X_genus_raw.tsv"
    with open(raw_out, "w") as f:
        f.write("run_accession\t" + "\t".join(all_genera) + "\n")
        for sid in sample_ids:
            row = [sid] + [str(samples[sid].get(g, 0.0)) for g in all_genera]
            f.write("\t".join(row) + "\n")

    # CLR
    clr_out = out_dir / "X_genus_clr.tsv"
    with open(clr_out, "w") as f:
        f.write("run_accession\t" + "\t".join(all_genera) + "\n")
        for sid in sample_ids:
            vals     = [samples[sid].get(g, 0.0) for g in all_genera]
            log_vals = [math.log(v + PSEUDOCOUNT) for v in vals]
            mean_log = sum(log_vals) / len(log_vals)
            clr_vals = [v - mean_log for v in log_vals]
            row = [sid] + [f"{v:.6f}" for v in clr_vals]
            f.write("\t".join(row) + "\n")

    print(f"  Raw → {raw_out}   ({len(sample_ids)} samples × {len(all_genera)} genera)")
    print(f"  CLR → {clr_out}")

    raw_df = pd.read_csv(raw_out, sep="\t", index_col="run_accession")
    clr_df = pd.read_csv(clr_out, sep="\t", index_col="run_accession")
    return raw_df, clr_df


# ── PERMANOVA ─────────────────────────────────────────────────────────────────

def _ss_total(d2):
    return float(np.sum(np.triu(d2, k=1))) / d2.shape[0]


def _ss_within(d2, grp):
    sw = 0.0
    for g in np.unique(grp):
        mask = grp == g
        n_g  = int(mask.sum())
        if n_g < 2:
            continue
        sub = d2[np.ix_(mask, mask)]
        sw += float(np.sum(np.triu(sub, k=1))) / n_g
    return sw


def permanova(D, grouping, n_perms=999, seed=42):
    grp = np.asarray(grouping)
    n   = D.shape[0]
    d2  = D ** 2
    q   = len(np.unique(grp))

    SS_T = _ss_total(d2)
    SS_W = _ss_within(d2, grp)
    SS_A = SS_T - SS_W
    F    = (SS_A / (q - 1)) / (SS_W / (n - q))
    R2   = SS_A / SS_T

    rng    = np.random.default_rng(seed)
    perm_F = np.empty(n_perms)
    for i in range(n_perms):
        g_perm    = rng.permutation(grp)
        sw_perm   = _ss_within(d2, g_perm)
        sa_perm   = SS_T - sw_perm
        perm_F[i] = (sa_perm / (q - 1)) / (sw_perm / (n - q))

    p_val = float((perm_F >= F).sum() + 1) / (n_perms + 1)
    return dict(
        n_samples=n, n_groups=q,
        SS_total=round(SS_T, 4), SS_between=round(SS_A, 4), SS_within=round(SS_W, 4),
        R2=round(float(R2), 6), F_stat=round(float(F), 4),
        p_value=round(p_val, 4), n_perms=n_perms,
        cohens_f2=round(float(R2 / (1 - R2)), 6),
    )


# ── PERMDISP ──────────────────────────────────────────────────────────────────

def permdisp(D, grouping, n_perms=999, seed=42):
    """
    PERMDISP (Anderson 2006): test homogeneity of multivariate dispersions.
    Uses distance to group centroid in PCoA space (correct implementation).
    """
    grp  = np.asarray(grouping)
    n    = D.shape[0]
    groups = np.unique(grp)

    # PCoA (classical MDS) to get coordinates
    D2   = D ** 2
    A    = -0.5 * D2
    ones = np.ones((n, 1))
    H    = np.eye(n) - ones @ ones.T / n
    B    = H @ A @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Keep positive eigenvalues only
    pos_mask = eigvals > 1e-10
    coords   = eigvecs[:, pos_mask] * np.sqrt(eigvals[pos_mask])

    # Distance to group centroid
    def dist_to_centroid(coords, grp):
        d = np.zeros(n)
        for g in groups:
            mask      = grp == g
            centroid  = coords[mask].mean(axis=0)
            diffs     = coords[mask] - centroid
            d[mask]   = np.sqrt((diffs ** 2).sum(axis=1))
        return d

    d_obs = dist_to_centroid(coords, grp)

    # Group dispersions
    group_disp = {}
    for g in groups:
        mask = grp == g
        group_disp[str(g)] = round(float(d_obs[mask].mean()), 4)

    # F-statistic (Levene-style on distances)
    grand_mean = d_obs.mean()
    SS_A = sum(
        (grp == g).sum() * (d_obs[grp == g].mean() - grand_mean) ** 2
        for g in groups
    )
    SS_W = sum(
        ((d_obs[grp == g] - d_obs[grp == g].mean()) ** 2).sum()
        for g in groups
    )
    q     = len(groups)
    F_obs = (SS_A / (q - 1)) / (SS_W / (n - q))

    rng    = np.random.default_rng(seed)
    perm_F = np.empty(n_perms)
    for i in range(n_perms):
        g_perm = rng.permutation(grp)
        d_p    = dist_to_centroid(coords, g_perm)
        gm_p   = d_p.mean()
        ssa_p  = sum(
            (g_perm == g).sum() * (d_p[g_perm == g].mean() - gm_p) ** 2
            for g in groups
        )
        ssw_p  = sum(
            ((d_p[g_perm == g] - d_p[g_perm == g].mean()) ** 2).sum()
            for g in groups
        )
        perm_F[i] = (ssa_p / (q - 1)) / (ssw_p / (n - q))

    p_val = float((perm_F >= F_obs).sum() + 1) / (n_perms + 1)
    homogeneous = p_val > 0.05

    return dict(
        n=n, n_groups=q,
        F_stat=round(float(F_obs), 4),
        p_value=round(p_val, 4),
        n_perms=n_perms,
        group_dispersions=group_disp,
        dispersion_homogeneous=homogeneous,
    )


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

print(f"\n[{ts()}] ── STEP 1: Build C4 (Lee 2022) matrix ──────────────────────────")

# Load C4 labels
c4_labels = {}
with open("metadata/lee2022_labels.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        c4_labels[row["run_accession"]] = row["response"]

print(f"  C4 labeled samples: {len(c4_labels)}  "
      f"(R={sum(v=='R' for v in c4_labels.values())}  "
      f"NR={sum(v=='NR' for v in c4_labels.values())})")

# Parse C4 Kraken reports
c4_reports = sorted(glob.glob("results/kraken_reports/lee2022/*_report.txt"))
print(f"  C4 Kraken reports found: {len(c4_reports)}")

c4_samples = {}
skipped_c4 = []
for fp in c4_reports:
    sid = os.path.basename(fp).replace("_report.txt", "")
    if sid not in c4_labels:
        skipped_c4.append(sid)
        continue
    c4_samples[sid] = parse_kraken_report(fp)

print(f"  Parsed: {len(c4_samples)} samples  (skipped {len(skipped_c4)} not in labels)")

out_lee = Path("results/ml/lee2022")
raw_c4, clr_c4 = build_matrices(c4_samples, out_lee)


print(f"\n[{ts()}] ── STEP 2: Build n=283 (4-cohort) matrix ─────────────────────")

# Load C1/C2/C3 labels
c123_labels = {}
with open("metadata/response_labels_3cohort.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        c123_labels[row["run_accession"]] = row["response"]

all_labels  = {**c123_labels, **c4_labels}
all_known   = set(all_labels)

# Parse C1/C2/C3 reports (flat directory)
c123_reports = sorted(glob.glob("results/kraken_reports/*_report.txt"))
print(f"  C1/C2/C3 reports found: {len(c123_reports)}")

all_samples = {}
for fp in c123_reports:
    sid = os.path.basename(fp).replace("_report.txt", "")
    if sid in all_known:
        all_samples[sid] = parse_kraken_report(fp)

# Add C4
for sid, gd in c4_samples.items():
    all_samples[sid] = gd

print(f"  Total samples loaded: {len(all_samples)}  "
      f"(expected 283)")

missing = all_known - set(all_samples)
if missing:
    print(f"  WARNING: {len(missing)} labeled samples missing reports: {sorted(missing)[:5]}")

out_n283 = Path("results/ml/n283_4cohort")
raw_n283, clr_n283 = build_matrices(all_samples, out_n283)

# Save combined labels file
def cohort_of(sid):
    if sid.startswith("SRR5930"):   return "cohort1"
    if sid.startswith("SRR11413"):  return "cohort2"
    if sid.startswith("SRR6000"):   return "cohort3"
    return "cohort4"

labels_n283 = [
    {"run_accession": sid, "response": all_labels[sid], "cohort": cohort_of(sid)}
    for sid in sorted(all_samples)
]
labels_df = pd.DataFrame(labels_n283)
labels_df.to_csv("results/ml/n283_4cohort/response_labels_n283.tsv", sep="\t", index=False)
print(f"  Labels → results/ml/n283_4cohort/response_labels_n283.tsv")

# Cohort breakdown
cohort_counts = Counter(cohort_of(sid) for sid in all_samples)
for c, cnt in sorted(cohort_counts.items()):
    resp = [all_labels[sid] for sid in all_samples if cohort_of(sid) == c]
    r_cnt = resp.count("R"); nr_cnt = resp.count("NR")
    print(f"    {c}: n={cnt}  (R={r_cnt}, NR={nr_cnt})")


print(f"\n[{ts()}] ── STEP 3: PERMANOVA on n=283 ────────────────────────────────")

# Align clr + labels
response_s = pd.Series({sid: all_labels[sid] for sid in clr_n283.index})
batch_s    = pd.Series({sid: cohort_of(sid)  for sid in clr_n283.index})

# Drop any samples missing labels
keep = response_s.notna()
clr_perm = clr_n283.loc[keep]
resp_perm = response_s.loc[keep]
batch_perm = batch_s.loc[keep]

n = len(clr_perm)
print(f"  Samples for PERMANOVA: {n}")
print(f"  Response: {resp_perm.value_counts().to_dict()}")
print(f"  Batch:    {batch_perm.value_counts().to_dict()}")

print(f"\n[{ts()}]   Computing Aitchison distance matrix …")
X      = clr_perm.values.astype(float)
dist1d = pdist(X, metric="euclidean")
D      = squareform(dist1d)
print(f"  Distance matrix: {D.shape}  max={D.max():.2f}  mean={D[D>0].mean():.2f}")

print(f"\n[{ts()}]   PERMANOVA — batch (999 perms) …")
res_batch = permanova(D, batch_perm.values, n_perms=999, seed=42)
res_batch["factor"] = "batch"

print(f"\n[{ts()}]   PERMANOVA — response (999 perms) …")
res_resp  = permanova(D, resp_perm.values,  n_perms=999, seed=43)
res_resp["factor"] = "response"

print(f"\n{'='*65}")
print(f"PERMANOVA — Aitchison distance, n={n}, 4 cohorts, 999 perms")
print(f"{'='*65}")
print(f"  {'Factor':<12} {'R²':>8} {'F':>8} {'p':>8}  {'cohens_f2':>10}")
print(f"  {'-'*60}")
for res in [res_batch, res_resp]:
    sig = " ***" if res["p_value"] < 0.001 else \
          " **"  if res["p_value"] < 0.01  else \
          " *"   if res["p_value"] < 0.05  else " ns"
    print(f"  {res['factor']:<12} {res['R2']:>8.4f} {res['F_stat']:>8.4f} "
          f"{res['p_value']:>8.4f}{sig}   f²={res['cohens_f2']:.4f}")

if res_batch["R2"] > 0 and res_resp["R2"] > 0:
    ratio = res_batch["R2"] / res_resp["R2"]
    print(f"\n  Batch:signal ratio = {ratio:.1f}×")
    print(f"  Response R² trend: n=39→0.0267, n=79→0.0110, n=118→0.0068, n=283→{res_resp['R2']:.4f}")

# Save PERMANOVA results
perm_out = Path("results/ml/n283_4cohort/permanova_n283_results.tsv")
perm_rows = []
for res in [res_batch, res_resp]:
    perm_rows.append({
        "analysis": "pooled_n283",
        "n": n, "factor": res["factor"],
        "R2": res["R2"], "F_stat": res["F_stat"],
        "p_value": res["p_value"], "n_perms": res["n_perms"],
        "cohens_f2": res["cohens_f2"],
        "note": "4-cohort pooled (Frankel+NovaSeq+Matson+Lee2022), no batch correction",
    })
pd.DataFrame(perm_rows).to_csv(perm_out, sep="\t", index=False)
print(f"\n  Saved: {perm_out}")

# Append to comparison table
OLD_CMP = "results/ml/batch_diagnostics/permanova_comparison.tsv"
new_rows = [
    {
        "analysis": "pooled_n283",
        "cohort":   "SRR5930 (Frankel) + SRR11413 + SRR6000 (Matson) + PRJEB43119 (Lee 2022) — 4 cohorts",
        "n": n, "factor": "response",
        "R2": res_resp["R2"], "F_stat": res_resp["F_stat"],
        "p_value": res_resp["p_value"], "n_perms": 999,
        "note": "4-cohort pooled; response R² continues monotonic decline",
    },
    {
        "analysis": "pooled_n283",
        "cohort":   "SRR5930 (Frankel) + SRR11413 + SRR6000 (Matson) + PRJEB43119 (Lee 2022) — 4 cohorts",
        "n": n, "factor": "batch",
        "R2": res_batch["R2"], "F_stat": res_batch["F_stat"],
        "p_value": res_batch["p_value"], "n_perms": 999,
        "note": "4-cohort batch effect (4 groups); batch >> signal",
    },
]
try:
    old_df = pd.read_csv(OLD_CMP, sep="\t")
    old_df = old_df[old_df["analysis"] != "pooled_n283"]
    new_df = pd.concat([old_df, pd.DataFrame(new_rows)], ignore_index=True)
except FileNotFoundError:
    new_df = pd.DataFrame(new_rows)

fields = ["analysis","cohort","n","factor","R2","F_stat","p_value","n_perms","note"]
for col in fields:
    if col not in new_df.columns:
        new_df[col] = ""
new_df[fields].to_csv(OLD_CMP, sep="\t", index=False)
print(f"  Updated: {OLD_CMP}")


print(f"\n[{ts()}] ── STEP 4: PERMDISP on n=283 ─────────────────────────────────")

print(f"[{ts()}]   PERMDISP — response (999 perms) …")
pd_resp  = permdisp(D, resp_perm.values,  n_perms=999, seed=44)
pd_resp["factor"]  = "response"
pd_resp["dataset"] = "n283"

print(f"[{ts()}]   PERMDISP — batch (999 perms) …")
pd_batch = permdisp(D, batch_perm.values, n_perms=999, seed=45)
pd_batch["factor"]  = "batch"
pd_batch["dataset"] = "n283"

print(f"\n{'='*65}")
print(f"PERMDISP — n={n}, 4 cohorts, 999 perms")
print(f"{'='*65}")
for pd_res in [pd_resp, pd_batch]:
    hom = "homogeneous" if pd_res["dispersion_homogeneous"] else "HETEROGENEOUS"
    sig = " ***" if pd_res["p_value"] < 0.001 else \
          " **"  if pd_res["p_value"] < 0.01  else \
          " *"   if pd_res["p_value"] < 0.05  else " ns"
    print(f"  {pd_res['factor']:<12}  F={pd_res['F_stat']:.4f}  "
          f"p={pd_res['p_value']:.4f}{sig}  → {hom}")
    print(f"              dispersions: {pd_res['group_dispersions']}")

# Append to permdisp_results.tsv
PDISP_OUT = "results/ml/batch_diagnostics/permdisp_results.tsv"
disp_rows = []
for pd_res in [pd_resp, pd_batch]:
    disp_rows.append({
        "dataset":               pd_res["dataset"],
        "n":                     pd_res["n"],
        "factor":                pd_res["factor"],
        "n_groups":              pd_res["n_groups"],
        "F_stat":                pd_res["F_stat"],
        "p_value":               pd_res["p_value"],
        "n_perms":               pd_res["n_perms"],
        "group_dispersions":     str(pd_res["group_dispersions"]),
        "dispersion_homogeneous": pd_res["dispersion_homogeneous"],
    })
try:
    old_pd = pd.read_csv(PDISP_OUT, sep="\t")
    old_pd = old_pd[old_pd["dataset"] != "n283"]
    new_pd = pd.concat([old_pd, pd.DataFrame(disp_rows)], ignore_index=True)
except FileNotFoundError:
    new_pd = pd.DataFrame(disp_rows)

new_pd.to_csv(PDISP_OUT, sep="\t", index=False)
print(f"\n  Saved: {PDISP_OUT}")

# PCA figure
print(f"\n[{ts()}]   Generating PCA figure …")
pca    = PCA(n_components=2, random_state=42)
scores = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_ * 100

batch_colors = {
    "cohort1": "#2196F3",   # blue   Frankel
    "cohort2": "#FF5722",   # orange NovaSeq
    "cohort3": "#9C27B0",   # purple Matson
    "cohort4": "#4CAF50",   # green  Lee 2022
}
batch_labels_map = {
    "cohort1": "C1 Frankel/HiSeq (n=39)",
    "cohort2": "C2 NovaSeq (n=40)",
    "cohort3": "C3 Matson/NextSeq (n=39)",
    "cohort4": "C4 Lee 2022 (n=165)",
}
resp_colors = {"R": "#4CAF50", "NR": "#F44336"}
batch_arr = batch_perm.values
resp_arr  = resp_perm.values

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for b in ["cohort1","cohort2","cohort3","cohort4"]:
    mask = batch_arr == b
    ax.scatter(scores[mask, 0], scores[mask, 1],
               c=batch_colors[b], label=batch_labels_map[b],
               alpha=0.75, s=40, edgecolors="white", linewidths=0.3)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11)
ax.set_title(f"PCA by Batch\nPERMANOVA: R²={res_batch['R2']:.4f}, p={res_batch['p_value']:.4f}", fontsize=11)
ax.legend(fontsize=8, framealpha=0.85)
ax.grid(True, alpha=0.2)

ax = axes[1]
for r in ["R","NR"]:
    mask = resp_arr == r
    ax.scatter(scores[mask, 0], scores[mask, 1],
               c=resp_colors[r], label=("Responder (R)" if r=="R" else "Non-Responder (NR)"),
               alpha=0.75, s=40, edgecolors="white", linewidths=0.3)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11)
ax.set_title(f"PCA by Response\nPERMANOVA: R²={res_resp['R2']:.4f}, p={res_resp['p_value']:.4f}", fontsize=11)
ax.legend(fontsize=9, framealpha=0.85)
ax.grid(True, alpha=0.2)

ratio_str = f"{res_batch['R2']/res_resp['R2']:.1f}×" if res_resp["R2"] > 0 else "N/A"
fig.suptitle(
    f"Aitchison PCA (n={n}, 4 cohorts)\n"
    f"Batch explains {res_batch['R2']*100:.2f}% vs Response {res_resp['R2']*100:.2f}%  "
    f"[ratio {ratio_str}]",
    fontsize=12, y=1.01,
)
fig.tight_layout()
pca_out = "results/ml/n283_4cohort/pca_4cohort_batch_vs_response.png"
fig.savefig(pca_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  PCA → {pca_out}")

print(f"\n[{ts()}] ══ ALL STEPS 1-4 COMPLETE ════════════════════════════════════")
print(f"\nSUMMARY")
print(f"{'─'*65}")
print(f"  Step 1: C4 matrix  → {out_lee}  ({len(c4_samples)} samples)")
print(f"  Step 2: n=283 mat  → {out_n283}  ({len(all_samples)} samples)")
print()
print(f"  PERMANOVA (n=283, 999 perms, Aitchison):")
print(f"    Response  R²={res_resp['R2']:.4f}  F={res_resp['F_stat']:.4f}  p={res_resp['p_value']:.4f}")
print(f"    Batch     R²={res_batch['R2']:.4f}  F={res_batch['F_stat']:.4f}  p={res_batch['p_value']:.4f}")
if res_resp["R2"] > 0:
    print(f"    Ratio     {res_batch['R2']/res_resp['R2']:.1f}× batch:signal")
print()
print(f"  PERMDISP (n=283, 999 perms):")
print(f"    Response  F={pd_resp['F_stat']:.4f}  p={pd_resp['p_value']:.4f}  "
      f"→ {'homogeneous' if pd_resp['dispersion_homogeneous'] else 'HETEROGENEOUS'}")
print(f"    Batch     F={pd_batch['F_stat']:.4f}  p={pd_batch['p_value']:.4f}  "
      f"→ {'homogeneous' if pd_batch['dispersion_homogeneous'] else 'HETEROGENEOUS'}")
print()
print(f"  Response R² dose-response curve:")
for label, r2 in [("n=39 (C1)",  0.0267), ("n=79 (C1+2)", 0.0110),
                  ("n=118 (C1-3)", 0.0068), (f"n=283 (C1-4)", res_resp["R2"])]:
    bar = "█" * int(r2 * 1000)
    print(f"    {label:<15} R²={r2:.4f}  {bar}")
print(f"\n  Files: {pca_out}")
print(f"         {perm_out}")
print(f"         {PDISP_OUT}")
