#!/usr/bin/env python3
"""
Phase 3c: Taxonomic-level sensitivity analysis.

Tests whether the batch >> response PERMANOVA pattern observed at genus level
also holds at species and phylum level, using the same 118 samples and the
same Kraken2 reports that produced the genus-level matrix.

Steps:
  1. Parse Kraken2 reports at species (rank S) and phylum (rank P) to build
     CLR feature matrices, applying the same name-cleaning and pseudocount
     logic as build_matrix.py.  Genus-level matrix loaded from existing file.
  2. PERMANOVA (Aitchison distance, 999 perms) for batch and response at
     each taxonomic level — identical implementation to batch_diagnostics_3cohort.py.
  3. PERMDISP (Anderson 2006) for batch and response at each level.
  4. Summary TSV: results/ml/phase3c/taxonomic_sensitivity_summary.tsv
  5. Bar chart: results/ml/phase3c/taxonomic_sensitivity_figure.png

Usage:
    cd cancer_project/
    python3 scripts/phase3c_taxonomic_sensitivity.py
"""

import glob
import math
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
REPORT_GLOB = "results/kraken_reports/*_report.txt"
CLR_GENUS   = "results/ml/n118_3cohort/X_genus_clr.tsv"
LABELS_PATH = "metadata/response_labels_3cohort.tsv"
OUT_DIR     = "results/ml/phase3c"
os.makedirs(OUT_DIR, exist_ok=True)

N_PERMS = 999
SEED    = 42
PC      = 1e-6  # CLR pseudocount

EXCLUDED_PHYLA   = {"Chordata", "Vertebrata"}   # host phyla
EXCLUDED_SPECIES = {"Homo"}                      # first word of species name

# ── load reference sample list + labels ───────────────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Loading labels and genus CLR …", flush=True)
genus_clr = pd.read_csv(CLR_GENUS, sep="\t", index_col="run_accession")
labels    = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")

SAMPLES  = genus_clr.index.tolist()   # 118 samples, canonical order
response = labels.reindex(SAMPLES)["response"]
batch    = pd.Series(
    ["C1" if s.startswith("SRR5930")  else
     "C2" if s.startswith("SRR11413") else "C3"
     for s in SAMPLES],
    index=SAMPLES,
)
print(f"  n={len(SAMPLES)}  response={response.value_counts().to_dict()}  "
      f"batch={batch.value_counts().to_dict()}", flush=True)


# ── name parser ───────────────────────────────────────────────────────────────

def parse_taxon_name(raw_field: str, rank: str):
    """Return a clean taxon name for a Kraken2 name field at the given rank."""
    stripped = raw_field.strip()
    if not stripped:
        return None
    # Strip parenthetical author citations e.g. "Genus (ex Smith et al. 2020)"
    if " (" in stripped:
        stripped = stripped.split(" (")[0].strip()
    words = stripped.split()
    if not words:
        return None

    if rank == "P":   # phylum: first word
        return words[0]

    if rank == "G":   # genus: first word (Candidatus joined)
        if stripped.startswith("Candidatus ") and len(words) >= 2:
            return "Candidatus_" + words[1]
        return words[0]

    if rank == "S":   # species: full binomial (two words)
        if stripped.startswith("Candidatus ") and len(words) >= 3:
            return f"Candidatus_{words[1]}_{words[2]}"
        if len(words) >= 2:
            return f"{words[0]} {words[1]}"
        return words[0]

    return None


# ── matrix builder ────────────────────────────────────────────────────────────

def build_clr_matrix(rank: str, label: str) -> pd.DataFrame:
    """
    Parse all Kraken2 reports for the given rank code and return a CLR
    DataFrame aligned to the canonical 118-sample order (SAMPLES).
    """
    report_files = sorted(glob.glob(REPORT_GLOB))
    sample_data: dict[str, dict[str, float]] = {}
    all_taxa: set[str] = set()

    for fp in report_files:
        sample = os.path.basename(fp).replace("_report.txt", "")
        if sample not in SAMPLES:
            continue

        taxa_counts: dict[str, float] = {}
        with open(fp) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6 or parts[3] != rank:
                    continue
                pct  = float(parts[0].strip())
                name = parse_taxon_name(parts[5], rank)
                if name is None:
                    continue
                first_word = name.split()[0].rstrip("_")
                if rank == "P" and first_word in EXCLUDED_PHYLA:
                    continue
                if rank == "S" and first_word in EXCLUDED_SPECIES:
                    continue
                taxa_counts[name] = taxa_counts.get(name, 0.0) + pct
                all_taxa.add(name)

        sample_data[sample] = taxa_counts

    all_taxa_sorted = sorted(all_taxa)
    print(f"  [{label}] {len(sample_data)} samples, {len(all_taxa_sorted)} taxa", flush=True)

    rows = []
    for sample in SAMPLES:
        taxa_c = sample_data.get(sample, {})
        vals   = [taxa_c.get(t, 0.0) for t in all_taxa_sorted]
        lv     = [math.log(v + PC) for v in vals]
        mean_l = sum(lv) / len(lv)
        rows.append([v - mean_l for v in lv])

    return pd.DataFrame(rows, index=SAMPLES, columns=all_taxa_sorted)


# ── PERMANOVA (matches batch_diagnostics_3cohort.py exactly) ──────────────────

def _ss_total(d2: np.ndarray) -> float:
    return float(np.sum(np.triu(d2, k=1))) / d2.shape[0]


def _ss_within(d2: np.ndarray, grp: np.ndarray) -> float:
    sw = 0.0
    for g in np.unique(grp):
        mask = grp == g
        n_g  = int(mask.sum())
        if n_g < 2:
            continue
        sub = d2[np.ix_(mask, mask)]
        sw += float(np.sum(np.triu(sub, k=1))) / n_g
    return sw


def run_permanova(D: np.ndarray, grouping, n_perms: int = 999, seed: int = 42) -> dict:
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
        gp     = rng.permutation(grp)
        sw_p   = _ss_within(d2, gp)
        sa_p   = SS_T - sw_p
        perm_F[i] = (sa_p / (q - 1)) / (sw_p / (n - q))

    p_val = float((perm_F >= F).sum()) / n_perms
    return dict(R2=round(float(R2), 4), F_stat=round(float(F), 4),
                p_value=round(p_val, 4), n_groups=q)


# ── PERMDISP (matches statistical_foundations.py exactly) ────────────────────

def run_permdisp(X: np.ndarray, grouping, n_perms: int = 999, seed: int = 42) -> dict:
    """Anderson (2006) PERMDISP — Euclidean distances to group centroid."""
    grp    = np.asarray(grouping)
    groups = np.unique(grp)
    q      = len(groups)
    if q < 2:
        return {"F_stat": np.nan, "p_value": np.nan, "n_groups": q,
                "dispersion_homogeneous": None}

    def _d2centroid(X, grp):
        d = np.zeros(len(grp))
        for g in np.unique(grp):
            m = grp == g
            d[m] = np.linalg.norm(X[m] - X[m].mean(axis=0), axis=1)
        return d

    def _anova_f(d, grp):
        groups = np.unique(grp); N = len(d); k = len(groups)
        gm  = d.mean()
        ssb = sum(((d[grp == g]).mean() - gm) ** 2 * (grp == g).sum() for g in groups)
        ssw = sum(((d[grp == g]) - (d[grp == g]).mean()).var() * (grp == g).sum()
                  for g in groups)
        return np.nan if ssw == 0 else (ssb / (k - 1)) / (ssw / (N - k))

    d_obs = _d2centroid(X, grp)
    F_obs = _anova_f(d_obs, grp)

    rng     = np.random.default_rng(seed)
    perm_Fs = [_anova_f(_d2centroid(X, rng.permutation(grp)), rng.permutation(grp))
               for _ in range(n_perms)]
    valid   = np.array([f for f in perm_Fs if not np.isnan(f)])
    p_val   = float((valid >= F_obs).sum()) / len(valid) if len(valid) > 0 else np.nan

    group_disp = {str(g): round(float(d_obs[grp == g].mean()), 4) for g in groups}
    return {
        "F_stat":                round(float(F_obs), 4) if not np.isnan(F_obs) else np.nan,
        "p_value":               round(p_val, 4),
        "n_groups":              q,
        "group_dispersions":     str(group_disp),
        "dispersion_homogeneous": bool(p_val >= 0.05) if not np.isnan(p_val) else None,
    }


# ── build species and phylum matrices ─────────────────────────────────────────

print(f"\n[{time.strftime('%H:%M:%S')}] Building phylum CLR matrix …", flush=True)
phylum_clr = build_clr_matrix("P", "phylum")
phylum_clr.to_csv(f"{OUT_DIR}/X_phylum_clr.tsv", sep="\t", index_label="run_accession")

print(f"[{time.strftime('%H:%M:%S')}] Building species CLR matrix …", flush=True)
species_clr = build_clr_matrix("S", "species")
species_clr.to_csv(f"{OUT_DIR}/X_species_clr.tsv", sep="\t", index_label="run_accession")

# ── run PERMANOVA + PERMDISP at each level ────────────────────────────────────

levels = [
    ("phylum",  phylum_clr),
    ("genus",   genus_clr),
    ("species", species_clr),
]

all_rows = []

for level_name, clr_df in levels:
    n_feat = clr_df.shape[1]
    X      = clr_df.values.astype(float)
    print(f"\n[{time.strftime('%H:%M:%S')}] === {level_name.upper()} "
          f"(n_features={n_feat}) ===", flush=True)

    print(f"  Computing Aitchison distance matrix …", flush=True)
    D = squareform(pdist(X, metric="euclidean"))

    for factor_name, grouping in [("batch", batch.values), ("response", response.values)]:
        seed_off = 0 if factor_name == "batch" else 1

        # PERMANOVA
        t0 = time.time()
        print(f"  PERMANOVA({factor_name}, {N_PERMS} perms) …", flush=True)
        pr = run_permanova(D, grouping, n_perms=N_PERMS, seed=SEED + seed_off)
        sig = ("***" if pr["p_value"] < 0.001 else "**"  if pr["p_value"] < 0.01 else
               "*"   if pr["p_value"] < 0.05  else "ns")
        print(f"    R²={pr['R2']:.4f}  F={pr['F_stat']:.4f}  "
              f"p={pr['p_value']:.4f} {sig}  ({time.time()-t0:.1f}s)", flush=True)

        all_rows.append({
            "level":      level_name,
            "n_features": n_feat,
            "test":       "PERMANOVA",
            "factor":     factor_name,
            "R2":         pr["R2"],
            "F_stat":     pr["F_stat"],
            "p_value":    pr["p_value"],
            "n_groups":   pr["n_groups"],
            "n_perms":    N_PERMS,
            "dispersion_homogeneous": "",
        })

        # PERMDISP
        t0 = time.time()
        print(f"  PERMDISP({factor_name}, {N_PERMS} perms) …", flush=True)
        dr = run_permdisp(X, grouping, n_perms=N_PERMS, seed=SEED + seed_off + 10)
        sig_d = ("***" if dr["p_value"] < 0.001 else "**"  if dr["p_value"] < 0.01 else
                 "*"   if dr["p_value"] < 0.05  else "ns")
        hom = "homogeneous" if dr["dispersion_homogeneous"] else "HETEROGENEOUS"
        print(f"    F={dr['F_stat']}  p={dr['p_value']:.4f} {sig_d}  "
              f"→ {hom}  ({time.time()-t0:.1f}s)", flush=True)

        all_rows.append({
            "level":      level_name,
            "n_features": n_feat,
            "test":       "PERMDISP",
            "factor":     factor_name,
            "R2":         "",
            "F_stat":     dr["F_stat"],
            "p_value":    dr["p_value"],
            "n_groups":   dr["n_groups"],
            "n_perms":    N_PERMS,
            "dispersion_homogeneous": dr["dispersion_homogeneous"],
        })

# ── save summary TSV ──────────────────────────────────────────────────────────

summary_df = pd.DataFrame(all_rows)
out_tsv    = f"{OUT_DIR}/taxonomic_sensitivity_summary.tsv"
summary_df.to_csv(out_tsv, sep="\t", index=False)
print(f"\n[{time.strftime('%H:%M:%S')}] Saved: {out_tsv}", flush=True)

# ── bar chart ─────────────────────────────────────────────────────────────────

perm_df   = summary_df[summary_df["test"] == "PERMANOVA"].copy()
level_ord = ["phylum", "genus", "species"]

batch_r2 = {r["level"]: r["R2"]      for _, r in perm_df[perm_df["factor"] == "batch"].iterrows()}
resp_r2  = {r["level"]: r["R2"]      for _, r in perm_df[perm_df["factor"] == "response"].iterrows()}
batch_p  = {r["level"]: r["p_value"] for _, r in perm_df[perm_df["factor"] == "batch"].iterrows()}
resp_p   = {r["level"]: r["p_value"] for _, r in perm_df[perm_df["factor"] == "response"].iterrows()}

x = np.arange(len(level_ord))
w = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_b = ax.bar(x - w/2, [batch_r2.get(l, 0) for l in level_ord], w,
                color="#C62828", label="Batch (cohort)", alpha=0.88, edgecolor="white")
bars_r = ax.bar(x + w/2, [resp_r2.get(l,  0) for l in level_ord], w,
                color="#1565C0", label="Response (R/NR)", alpha=0.88, edgecolor="white")

def _annotate_bars(bars, p_dict, levels):
    for bar, lvl in zip(bars, levels):
        p   = p_dict.get(lvl, 1.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001, sig,
                ha="center", va="bottom", fontsize=9, color="black")

_annotate_bars(bars_b, batch_p,  level_ord)
_annotate_bars(bars_r, resp_p,   level_ord)

# Batch:response ratio labels above bar pairs
for i, lvl in enumerate(level_ord):
    bv = batch_r2.get(lvl, 0)
    rv = resp_r2.get(lvl,  0)
    if rv > 0:
        ratio = bv / rv
        ypos  = max(bv, rv) + 0.007
        ax.text(i, ypos, f"{ratio:.0f}×", ha="center", fontsize=8, color="#444",
                fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels([l.capitalize() for l in level_ord], fontsize=12)
ax.set_ylabel("PERMANOVA R² (Aitchison distance)", fontsize=11)
ax.set_title(
    "Batch vs. Response Variance Explained — Taxonomic Level Sensitivity\n"
    "n=118, 3 cohorts, 999 permutations  (italic = batch÷response ratio)",
    fontsize=10)
ax.legend(fontsize=10, framealpha=0.85)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, max(max(batch_r2.values()), 0.12) * 1.25)
fig.tight_layout()

out_fig = f"{OUT_DIR}/taxonomic_sensitivity_figure.png"
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_fig}", flush=True)

# ── plain-language summary ────────────────────────────────────────────────────

print(f"\n[{time.strftime('%H:%M:%S')}] ══ PHASE 3C SUMMARY ════════════════════════════════\n",
      flush=True)

perm_only = summary_df[summary_df["test"] == "PERMANOVA"].copy()
disp_only = summary_df[summary_df["test"] == "PERMDISP"].copy()

print("PERMANOVA (Aitchison distance, 999 perms, n=118):", flush=True)
print(f"  {'Level':<10} {'Factor':<10} {'n_feat':>7}  {'R²':>7} {'F':>8} "
      f"{'p-value':>10}  sig", flush=True)
print("  " + "-" * 60, flush=True)
for lvl in level_ord:
    for fac in ["batch", "response"]:
        row = perm_only[(perm_only["level"] == lvl) & (perm_only["factor"] == fac)]
        if row.empty:
            continue
        r = row.iloc[0]
        sig = ("***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else
               "*"   if r["p_value"] < 0.05  else "ns")
        print(f"  {lvl:<10} {fac:<10} {int(r['n_features']):>7}  {r['R2']:>7.4f} "
              f"{r['F_stat']:>8.4f} {r['p_value']:>10.4f}  {sig}", flush=True)
    b  = perm_only[(perm_only["level"] == lvl) & (perm_only["factor"] == "batch")]
    rr = perm_only[(perm_only["level"] == lvl) & (perm_only["factor"] == "response")]
    if not b.empty and not rr.empty and float(rr.iloc[0]["R2"]) > 0:
        ratio = float(b.iloc[0]["R2"]) / float(rr.iloc[0]["R2"])
        print(f"  {'':10} {'→ batch:resp':10}  {'ratio =':>7} {ratio:>7.1f}×", flush=True)
    print("", flush=True)

print("PERMDISP (dispersion homogeneity):", flush=True)
print(f"  {'Level':<10} {'Factor':<10} {'F':>8} {'p-value':>10}  Homogeneous?", flush=True)
print("  " + "-" * 55, flush=True)
for lvl in level_ord:
    for fac in ["batch", "response"]:
        row = disp_only[(disp_only["level"] == lvl) & (disp_only["factor"] == fac)]
        if row.empty:
            continue
        r   = row.iloc[0]
        hom = "Yes" if r["dispersion_homogeneous"] else "NO (heterogeneous)"
        print(f"  {lvl:<10} {fac:<10} {r['F_stat']:>8} {r['p_value']:>10.4f}  {hom}",
              flush=True)
print("", flush=True)

print("── KEY FINDING ──────────────────────────────────────────────────────────", flush=True)
levels_batch_wins = []
for lvl in level_ord:
    b  = perm_only[(perm_only["level"] == lvl) & (perm_only["factor"] == "batch")]
    rr = perm_only[(perm_only["level"] == lvl) & (perm_only["factor"] == "response")]
    if not b.empty and not rr.empty:
        levels_batch_wins.append(float(b.iloc[0]["R2"]) > float(rr.iloc[0]["R2"]))

if all(levels_batch_wins):
    print("  Batch R² > Response R² at ALL three taxonomic levels.", flush=True)
    print("  The batch >> response dominance is NOT genus-specific — it is a", flush=True)
    print("  structural property of this multi-cohort dataset independent of", flush=True)
    print("  taxonomic resolution. Changing aggregation level cannot recover signal.", flush=True)
else:
    n_ok = sum(levels_batch_wins)
    print(f"  Batch R² > Response R² at {n_ok}/3 taxonomic levels. See table.", flush=True)

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 3c complete. All outputs in {OUT_DIR}/",
      flush=True)
