"""
Process cBioPortal melanoma immunotherapy studies into clean feature matrices.

Fixes vs the original one-liner:
  - Gene name extracted from `keyword` field (SUMMARY projection omits hugoGeneSymbol)
  - Riaz 2017: paired pre/on-treatment biopsies — filter to `_pre` samples only
  - Liu 2019: response (BR) is at PATIENT level — join via patientId
  - All response mappings corrected to actual value strings
"""
import json, os
import pandas as pd
import numpy as np
from collections import Counter

os.makedirs("results/ml/tumor", exist_ok=True)

CODING_MB = 38.0
TOP_N     = 30
DRIVERS   = ["BRAF","NRAS","NF1","PTEN","TP53","CDKN2A","KIT",
             "RAC1","MAP2K1","MAP2K2","IDH1","ARID2","SETD2"]


# ── helpers ────────────────────────────────────────────────────────────────

def pivot_clinical(raw, id_field="sampleId"):
    """Long-format clinical JSON → wide DataFrame keyed on id_field.
    Always preserves both sampleId and patientId when present."""
    rows = {}
    for rec in raw:
        pid = rec.get(id_field)
        if pid is None:
            continue
        if pid not in rows:
            rows[pid] = {id_field: pid}
            # carry both IDs if available
            for extra in ("sampleId", "patientId"):
                if extra != id_field and rec.get(extra):
                    rows[pid][extra] = rec[extra]
        rows[pid][rec["clinicalAttributeId"]] = rec["value"]
    return pd.DataFrame(list(rows.values()))


def gene_from_keyword(kw):
    """'BRAF V600E missense' → 'BRAF'"""
    if not kw:
        return None
    return kw.split()[0]


def build_mutation_matrix(muts, label):
    """Binary gene × sample matrix + TMB, from list of mutation dicts."""
    records = {}
    gene_counts = Counter()
    n_skipped = 0
    for m in muts:
        sid  = m.get("sampleId")
        gene = gene_from_keyword(m.get("keyword"))
        if not sid or not gene:
            n_skipped += 1
            continue
        records.setdefault(sid, {"sample_id": sid, "_n_mut": 0})
        records[sid]["_n_mut"] += 1
        records[sid][f"mut_{gene}"] = 1

    print(f"  {label}: {len(records)} samples, {len(muts)} records "
          f"({n_skipped} skipped no-gene)", flush=True)

    df = pd.DataFrame(list(records.values())).fillna(0)
    df.rename(columns={"_n_mut": "n_mutations"}, inplace=True)
    df["TMB"] = df["n_mutations"] / CODING_MB

    # frequency across samples
    mut_cols = [c for c in df.columns if c.startswith("mut_")]
    freq = df[mut_cols].sum().sort_values(ascending=False)
    top_genes   = [c.removeprefix("mut_") for c in freq.index[:TOP_N]]
    keep_genes  = list(dict.fromkeys(top_genes + DRIVERS))  # dedup, preserve order
    keep_cols   = [f"mut_{g}" for g in keep_genes if f"mut_{g}" in df.columns]

    print(f"  Top 5 mutated genes: "
          + ", ".join(f"{c.removeprefix('mut_')}({int(freq[c])})" for c in freq.index[:5]),
          flush=True)

    return df[["sample_id", "n_mutations", "TMB"] + keep_cols]


# ── Riaz 2017 ──────────────────────────────────────────────────────────────
print("\n=== RIAZ 2017 (mel_iatlas_riaz_nivolumab_2017) ===", flush=True)

with open("data/cbioportal/riaz2017_clinical_raw.json") as f:
    riaz_clin_df = pivot_clinical(json.load(f), id_field="sampleId")

with open("data/cbioportal/riaz2017_mutations_raw.json") as f:
    riaz_mut_df = build_mutation_matrix(json.load(f), "Riaz")

# Keep only pre-treatment samples (sample IDs ending in _pre)
riaz_clin_pre = riaz_clin_df[riaz_clin_df["sampleId"].str.endswith("_pre")].copy()
riaz_mut_pre  = riaz_mut_df[riaz_mut_df["sample_id"].str.endswith("_pre")].copy()
print(f"  Pre-treatment samples (clinical): {len(riaz_clin_pre)}", flush=True)
print(f"  Pre-treatment samples (mutations): {len(riaz_mut_pre)}", flush=True)

# Response: RESPONSE col — CR/PR → R, SD/PD → NR
RIAZ_R = {"Complete Response": "R", "Partial Response": "R",
           "Stable Disease": "NR", "Progressive Disease": "NR"}
if "RESPONSE" in riaz_clin_pre.columns:
    riaz_clin_pre["response"] = riaz_clin_pre["RESPONSE"].map(RIAZ_R)
else:
    print("  WARNING: RESPONSE column missing in Riaz clinical data", flush=True)

riaz = riaz_clin_pre.merge(riaz_mut_pre,
                            left_on="sampleId", right_on="sample_id", how="inner")
riaz = riaz.dropna(subset=["response"])
riaz["study"]   = "riaz2017"
riaz["therapy"] = "anti-PD1"

r_n  = (riaz["response"] == "R").sum()
nr_n = (riaz["response"] == "NR").sum()
print(f"  Final: {len(riaz)} patients  R={r_n}  NR={nr_n}", flush=True)
riaz.to_csv("results/ml/tumor/riaz2017_features.tsv", sep="\t", index=False)


# ── Liu 2019 ───────────────────────────────────────────────────────────────
print("\n=== LIU 2019 (mel_dfci_2019) ===", flush=True)

# sample-level clinical (tumor features: TMB, purity, etc.)
with open("data/cbioportal/liu2019_clinical_raw.json") as f:
    liu_sample_df = pivot_clinical(json.load(f), id_field="sampleId")

# patient-level clinical (response: BR field)
with open("data/cbioportal/liu2019_clinical_patient_raw.json") as f:
    liu_patient_df = pivot_clinical(json.load(f), id_field="patientId")

with open("data/cbioportal/liu2019_mutations_raw.json") as f:
    liu_mut_df = build_mutation_matrix(json.load(f), "Liu")

# BR values: CR/PR → R; Mixed/SD/PD → NR
LIU_R = {"Complete Response": "R", "Partial Response": "R",
          "Mixed Response": "NR", "Stable Disease": "NR",
          "Progressive Disease": "NR"}
if "BR" in liu_patient_df.columns:
    liu_patient_df["response"] = liu_patient_df["BR"].map(LIU_R)
else:
    print("  WARNING: BR column missing in Liu patient-level data", flush=True)

# For mutation data, match on patientId (each patient has one WES in this study)
# Mutation records carry both sampleId and patientId — get the patientId map
with open("data/cbioportal/liu2019_mutations_raw.json") as f:
    liu_muts_raw = json.load(f)
sid_to_pid = {m["sampleId"]: m["patientId"] for m in liu_muts_raw if m.get("sampleId")}
liu_mut_df["patientId"] = liu_mut_df["sample_id"].map(sid_to_pid)

# merge: sample-level features + patient-level response
liu = liu_sample_df.merge(liu_patient_df[["patientId", "response"]],
                          on="patientId", how="inner")
liu = liu.merge(liu_mut_df, on="patientId", how="inner")
liu = liu.dropna(subset=["response"])
liu["study"]   = "liu2019"
liu["therapy"] = "anti-PD1"

r_n  = (liu["response"] == "R").sum()
nr_n = (liu["response"] == "NR").sum()
print(f"  Final: {len(liu)} patients  R={r_n}  NR={nr_n}", flush=True)
liu.to_csv("results/ml/tumor/liu2019_features.tsv", sep="\t", index=False)


# ── Hugo 2016 ──────────────────────────────────────────────────────────────
print("\n=== HUGO 2016 (mel_ucla_2016) ===", flush=True)

with open("data/cbioportal/hugo2016_clinical_raw.json") as f:
    hugo_clin_df = pivot_clinical(json.load(f), id_field="sampleId")

with open("data/cbioportal/hugo2016_mutations_raw.json") as f:
    hugo_mut_df = build_mutation_matrix(json.load(f), "Hugo")

# TREATMENT_RESPONSE: 'Responder' / 'Non-responder'
HUGO_R = {"Responder": "R", "Non-responder": "NR"}
if "TREATMENT_RESPONSE" in hugo_clin_df.columns:
    hugo_clin_df["response"] = hugo_clin_df["TREATMENT_RESPONSE"].map(HUGO_R)
elif "DURABLE_CLINICAL_BENEFIT" in hugo_clin_df.columns:
    hugo_clin_df["response"] = hugo_clin_df["DURABLE_CLINICAL_BENEFIT"].map(
        {"Yes": "R", "No": "NR", "DCB": "R", "NDB": "NR"})
else:
    print("  WARNING: no response column found in Hugo clinical data", flush=True)

hugo = hugo_clin_df.merge(hugo_mut_df,
                           left_on="sampleId", right_on="sample_id", how="inner")
hugo = hugo.dropna(subset=["response"])
hugo["study"]   = "hugo2016"
hugo["therapy"] = "anti-PD1"

r_n  = (hugo["response"] == "R").sum()
nr_n = (hugo["response"] == "NR").sum()
print(f"  Final: {len(hugo)} patients  R={r_n}  NR={nr_n}", flush=True)
hugo.to_csv("results/ml/tumor/hugo2016_features.tsv", sep="\t", index=False)


# ── Combined ───────────────────────────────────────────────────────────────
print("\n=== COMBINED ===", flush=True)

combined = pd.concat([riaz, liu, hugo], ignore_index=True, sort=False)
mut_cols = [c for c in combined.columns if c.startswith("mut_")]
combined[mut_cols] = combined[mut_cols].fillna(0)
combined.to_csv("results/ml/tumor/combined_tumor_features.tsv", sep="\t", index=False)

print(f"Combined: {len(combined)} patients total", flush=True)
print(f"  R={( combined['response']=='R').sum()}  "
      f"NR={(combined['response']=='NR').sum()}", flush=True)
print(f"  Features: TMB + {len(mut_cols)} gene binary flags", flush=True)
print(f"  Studies: {dict(combined['study'].value_counts())}", flush=True)

# TMB summary per study
print("\nTMB (mut/Mb) per study:", flush=True)
for study, grp in combined.groupby("study"):
    print(f"  {study}: median={grp['TMB'].median():.1f}  "
          f"range=[{grp['TMB'].min():.1f}, {grp['TMB'].max():.1f}]", flush=True)

print("\nDONE", flush=True)
