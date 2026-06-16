#!/usr/bin/env python3
"""
TCGA-SKCM tumor feature extraction from GDC open-access data.

Steps:
  1. Query GDC API for all TCGA-SKCM masked somatic mutation MAF file UUIDs
     + their case (patient) IDs.
  2. Download all MAF files in a single batch POST to /data (returns tar.gz).
  3. Parse MAFs: compute TMB per patient, extract top-mutated gene flags.
  4. Download clinical data via GDC cases API (TSV).
  5. Report patient overlap with existing immunotherapy response labels.
  6. Write results/ml/tcga_skcm/tcga_skcm_tumor_features.tsv
"""

import io
import json
import os
import sys
import tarfile
import time
import gzip
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT_DIR = PROJECT_DIR / "results" / "ml" / "tcga_skcm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_3COHORT = PROJECT_DIR / "metadata" / "response_labels_3cohort.tsv"
LABELS_ALL     = PROJECT_DIR / "metadata" / "response_labels.tsv"

GDC_FILES_EP   = "https://api.gdc.cancer.gov/files"
GDC_CASES_EP   = "https://api.gdc.cancer.gov/cases"
GDC_DATA_EP    = "https://api.gdc.cancer.gov/data"

CODING_MB      = 38.0   # hg38 coding genome size used for TMB denominator
TOP_N_GENES    = 20     # number of most-mutated genes to keep as binary features
BATCH_SIZE     = 100    # file IDs per download POST
RETRY_BATCH    = 20     # smaller batch size on retry


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Step 1: collect all TCGA-SKCM MAF file IDs + case IDs ─────────────────

def get_maf_file_ids():
    """Return list of {file_id, case_id, case_submitter_id} for all MAFs."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": ["TCGA-SKCM"]}},
            {"op": "in", "content": {"field": "data_type",
                                     "value": ["Masked Somatic Mutation"]}},
        ]
    }
    fields = "file_id,file_name,cases.case_id,cases.submitter_id"
    records = []
    page, size = 0, 500

    log("Querying GDC for TCGA-SKCM masked somatic mutation files …")
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": fields,
            "size": size,
            "from": page * size,
        }
        r = requests.get(GDC_FILES_EP, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()["data"]
        hits = data["hits"]
        if not hits:
            break
        for h in hits:
            fid = h["file_id"]
            fname = h["file_name"]
            for c in h.get("cases", []):
                records.append({
                    "file_id":           fid,
                    "file_name":         fname,
                    "case_id":           c["case_id"],
                    "case_submitter_id": c["submitter_id"],
                })
        total = data["pagination"]["total"]
        fetched = page * size + len(hits)
        log(f"  {fetched}/{total} file records fetched")
        if fetched >= total:
            break
        page += 1
        time.sleep(0.3)

    log(f"  → {len(records)} case-file associations ({len(set(r['file_id'] for r in records))} unique files)")
    return records


# ── Step 2: batch-download MAFs from GDC /data ────────────────────────────

def download_batch(file_ids, label):
    """POST to /data and return in-memory tar.gz bytes."""
    log(f"  Downloading {len(file_ids)} MAF files ({label}) …")
    payload = {"ids": file_ids}
    r = requests.post(
        GDC_DATA_EP,
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=360,
    )
    r.raise_for_status()
    buf = io.BytesIO()
    for chunk in r.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
    buf.seek(0)
    return buf


def parse_maf_bytes(gz_bytes):
    """Parse a single gzipped MAF file, return DataFrame of coding mutations."""
    with gzip.open(io.BytesIO(gz_bytes), "rt", errors="replace") as f:
        lines = []
        for line in f:
            if line.startswith("#"):
                continue
            lines.append(line)
    if not lines:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)
    # keep coding / non-silent variants (standard MAF classification)
    keep_vc = {
        "Missense_Mutation", "Nonsense_Mutation", "Splice_Site",
        "Frame_Shift_Del", "Frame_Shift_Ins",
        "In_Frame_Del", "In_Frame_Ins",
        "Translation_Start_Site", "Nonstop_Mutation",
    }
    if "Variant_Classification" in df.columns:
        df = df[df["Variant_Classification"].isin(keep_vc)]
    return df


def _parse_tar_into(buf, batch, fid_to_case, case_muts):
    """Parse a tar.gz buffer and accumulate mutation DataFrames into case_muts."""
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".maf.gz"):
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            gz_bytes = fobj.read()
            file_uuid = member.name.split("/")[0]
            case_id = fid_to_case.get(file_uuid)
            if case_id is None:
                for fid in batch:
                    if fid in member.name:
                        case_id = fid_to_case[fid]
                        break
            if case_id is None:
                continue
            df = parse_maf_bytes(gz_bytes)
            if case_id not in case_muts:
                case_muts[case_id] = []
            case_muts[case_id].append(df)


def download_and_parse_all(records):
    """
    Download all MAF files in batches, parse, return
    {case_submitter_id -> DataFrame of mutations}.
    """
    fid_to_case = {r["file_id"]: r["case_submitter_id"] for r in records}
    file_ids = list(fid_to_case.keys())

    case_muts = {}   # case_id -> list of DataFrames

    for i in range(0, len(file_ids), BATCH_SIZE):
        batch = file_ids[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(file_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        label = f"batch {batch_num}/{total_batches}"

        try:
            buf = download_batch(batch, label)
            _parse_tar_into(buf, batch, fid_to_case, case_muts)
        except Exception as e:
            log(f"  WARNING: {label} failed: {e}  — retrying in sub-batches of {RETRY_BATCH} …")
            for j in range(0, len(batch), RETRY_BATCH):
                sub = batch[j: j + RETRY_BATCH]
                sub_label = f"{label} sub {j//RETRY_BATCH+1}/{(len(batch)+RETRY_BATCH-1)//RETRY_BATCH}"
                for attempt in range(3):
                    try:
                        sub_buf = download_batch(sub, sub_label)
                        _parse_tar_into(sub_buf, sub, fid_to_case, case_muts)
                        break
                    except Exception as e2:
                        log(f"    attempt {attempt+1} failed: {e2}")
                        time.sleep(8 * (attempt + 1))
                time.sleep(3)

        log(f"  {label} → {len(case_muts)} cases accumulated")
        time.sleep(1)

    # concatenate per-case DataFrames (skip cases where all dfs are empty)
    final = {}
    for case_id, dfs in case_muts.items():
        non_empty = [d for d in dfs if len(d) > 0]
        if non_empty:
            final[case_id] = pd.concat(non_empty, ignore_index=True)
        else:
            final[case_id] = pd.DataFrame()

    log(f"Parsed mutations for {len(final)} cases ({sum(1 for d in final.values() if len(d)>0)} with coding mutations)")
    return final


# ── Step 3: compute features ───────────────────────────────────────────────

def compute_features(case_muts):
    """Return feature DataFrame (patients × features)."""
    # TMB
    tmb = {cid: len(df) / CODING_MB for cid, df in case_muts.items()}

    # per-case gene hit counts
    gene_hits = defaultdict(lambda: defaultdict(int))
    for cid, df in case_muts.items():
        if "Hugo_Symbol" in df.columns:
            for gene in df["Hugo_Symbol"].dropna():
                gene_hits[gene][cid] += 1

    # global mutation frequency across patients
    n_cases = len(case_muts)
    gene_freq = {g: len(hits) / n_cases for g, hits in gene_hits.items()}
    top_genes = sorted(gene_freq, key=gene_freq.get, reverse=True)[:TOP_N_GENES]
    log(f"Top {TOP_N_GENES} mutated genes (by patient frequency):")
    for g in top_genes:
        freq = gene_freq[g]
        log(f"    {g:<20s}  {freq*100:5.1f}% of patients")

    # build feature matrix
    all_cases = sorted(case_muts.keys())
    rows = []
    for cid in all_cases:
        row = {"patient_id": cid, "TMB": tmb[cid], "n_mutations": len(case_muts[cid])}
        for g in top_genes:
            row[f"mut_{g}"] = 1 if gene_hits[g].get(cid, 0) > 0 else 0
        rows.append(row)

    feat_df = pd.DataFrame(rows).set_index("patient_id")
    return feat_df, top_genes, gene_freq


# ── Step 4: clinical data ─────────────────────────────────────────────────

def download_clinical():
    """Fetch clinical data for all TCGA-SKCM cases via GDC cases API."""
    log("Downloading TCGA-SKCM clinical data …")
    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": ["TCGA-SKCM"],
        }
    }
    fields = (
        "submitter_id,case_id,"
        "diagnoses.vital_status,diagnoses.days_to_death,"
        "diagnoses.days_to_last_follow_up,"
        "diagnoses.tumor_stage,diagnoses.morphology,"
        "treatments.treatment_type,treatments.therapeutic_agents,"
        "treatments.treatment_outcome,treatments.days_to_treatment_start,"
        "demographic.gender,demographic.race,demographic.ethnicity,"
        "demographic.year_of_birth"
    )
    records = []
    page, size = 0, 500
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": fields,
            "size": size,
            "from": page * size,
            "format": "JSON",
        }
        r = requests.get(GDC_CASES_EP, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()["data"]
        hits = data["hits"]
        if not hits:
            break
        records.extend(hits)
        total = data["pagination"]["total"]
        fetched = page * size + len(hits)
        if fetched >= total:
            break
        page += 1
        time.sleep(0.3)

    log(f"  Retrieved {len(records)} clinical records")

    # flatten to rows
    rows = []
    for c in records:
        row = {
            "case_id":        c.get("case_id", ""),
            "submitter_id":   c.get("submitter_id", ""),
        }
        diag = c.get("diagnoses", [{}])[0] if c.get("diagnoses") else {}
        row["vital_status"]           = diag.get("vital_status", "")
        row["days_to_death"]          = diag.get("days_to_death", np.nan)
        row["days_to_last_follow_up"] = diag.get("days_to_last_follow_up", np.nan)
        row["tumor_stage"]            = diag.get("tumor_stage", "")

        dem = c.get("demographic", {}) or {}
        row["gender"] = dem.get("gender", "")
        row["race"]   = dem.get("race", "")

        # treatments — immunotherapy flag
        treats = c.get("treatments", []) or []
        immuno_agents = []
        for t in treats:
            agent = (t.get("therapeutic_agents") or "").lower()
            ttype = (t.get("treatment_type") or "").lower()
            if any(kw in agent or kw in ttype for kw in
                   ["pembrolizumab", "nivolumab", "ipilimumab", "pd-1",
                    "pd-l1", "ctla-4", "immunotherapy", "checkpoint"]):
                immuno_agents.append(t.get("therapeutic_agents", ttype))
        row["immunotherapy_agents"] = "; ".join(immuno_agents)
        row["had_immunotherapy"]    = 1 if immuno_agents else 0

        rows.append(row)

    clin_df = pd.DataFrame(rows)
    return clin_df


# ── Step 5: overlap with existing response labels ──────────────────────────

def check_overlap(feat_df, clin_df):
    log("\n" + "="*60)
    log("PATIENT OVERLAP ANALYSIS")
    log("="*60)

    tcga_ids = set(feat_df.index)
    log(f"TCGA-SKCM patients with mutation features: {len(tcga_ids)}")

    # load existing response labels
    if LABELS_3COHORT.exists():
        lab3 = pd.read_csv(LABELS_3COHORT, sep="\t")
        log(f"Existing 3-cohort labeled patients: {len(lab3)}")
    if LABELS_ALL.exists():
        lab_all = pd.read_csv(LABELS_ALL, sep="\t")
        log(f"Existing all-labeled patients: {len(lab_all)}")

    # TCGA IDs look like TCGA-XX-XXXX; our run IDs are SRR-accessions — no overlap possible
    skcm_submitters = set(feat_df.index)
    log("\nOur microbiome cohort IDs are SRR accessions (e.g., SRR5930493).")
    log("TCGA-SKCM IDs are TCGA case submitter IDs (e.g., TCGA-D3-A2J6).")
    log("These are entirely different patient cohorts — ZERO overlap expected.")
    log("  → Direct integration requires a study that collected BOTH")
    log("    pre-treatment gut microbiome samples AND tumor WES on the same patients.")

    # immunotherapy info from clinical
    if not clin_df.empty:
        n_immuno = clin_df["had_immunotherapy"].sum()
        log(f"\nTCGA-SKCM clinical records with recorded immunotherapy: {n_immuno}/{len(clin_df)}")
        if n_immuno > 0:
            log("  Note: TCGA treatment records are incomplete; most pre-ICI era patients.")
            agents = clin_df[clin_df["had_immunotherapy"]==1]["immunotherapy_agents"].value_counts()
            log("  Agents recorded:")
            for ag, ct in agents.head(10).items():
                log(f"    {ag}: {ct}")

    log("\nConclusion for multi-omic integration:")
    log("  TCGA-SKCM provides tumor genomic context for melanoma generally,")
    log("  but cannot be directly joined to the gut microbiome response data.")
    log("  The feature matrix is still valuable for:")
    log("  (a) characterizing the melanoma mutation landscape (TMB distribution,")
    log("      BRAF/NRAS/NF1 frequencies, UV signature prevalence)")
    log("  (b) future work if a dataset with paired gut-microbiome + tumor WES exists.")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("TCGA-SKCM Tumor Feature Extraction Pipeline")
    log("=" * 60)

    # Step 1
    records = get_maf_file_ids()
    if not records:
        log("ERROR: No files found. Exiting.")
        sys.exit(1)

    # save file manifest
    manifest_path = OUT_DIR / "gdc_file_manifest.tsv"
    pd.DataFrame(records).to_csv(manifest_path, sep="\t", index=False)
    log(f"File manifest saved: {manifest_path}")

    # Step 2 + 3
    log("\nSTEP 2/3: Downloading and parsing MAF files …")
    case_muts = download_and_parse_all(records)

    if not case_muts:
        log("ERROR: No mutations parsed. Exiting.")
        sys.exit(1)

    # Step 3: feature extraction
    log("\nSTEP 3: Computing tumor features …")
    feat_df, top_genes, gene_freq = compute_features(case_muts)

    out_features = OUT_DIR / "tcga_skcm_tumor_features.tsv"
    feat_df.to_csv(out_features, sep="\t")
    log(f"Feature matrix saved: {out_features}  ({feat_df.shape[0]} patients × {feat_df.shape[1]} features)")

    # TMB summary
    log(f"\nTMB summary (mut/Mb, coding nonsynonymous):")
    log(f"  median = {feat_df['TMB'].median():.2f}")
    log(f"  mean   = {feat_df['TMB'].mean():.2f}")
    log(f"  range  = [{feat_df['TMB'].min():.2f}, {feat_df['TMB'].max():.2f}]")
    log(f"  TMB>10 = {(feat_df['TMB']>10).sum()} patients ({(feat_df['TMB']>10).mean()*100:.1f}%)")

    # Step 4: clinical
    log("\nSTEP 4: Downloading clinical data …")
    try:
        clin_df = download_clinical()
        clin_path = OUT_DIR / "tcga_skcm_clinical.tsv"
        clin_df.to_csv(clin_path, sep="\t", index=False)
        log(f"Clinical data saved: {clin_path}")
    except Exception as e:
        log(f"Clinical download failed: {e}")
        clin_df = pd.DataFrame()

    # Step 5: overlap report
    check_overlap(feat_df, clin_df)

    # gene frequency summary
    log("\nTop 20 gene mutation frequencies across TCGA-SKCM:")
    log(f"  {'Gene':<20s}  {'% patients':>10s}  {'abs count':>10s}")
    n = len(case_muts)
    for g in top_genes:
        ct = int(round(gene_freq[g] * n))
        log(f"  {g:<20s}  {gene_freq[g]*100:9.1f}%  {ct:>10d}")

    log(f"\nAll outputs in: {OUT_DIR}")
    log("DONE")


if __name__ == "__main__":
    main()
