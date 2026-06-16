import requests, json, os, pandas as pd

os.makedirs("data/cbioportal", exist_ok=True)

studies = {
    "mel_iatlas_riaz_nivolumab_2017": "riaz2017",
    "mel_dfci_2019":                  "liu2019",
    "mel_ucla_2016":                  "hugo2016",
}

# confirmed molecular profile IDs from /api/studies/{id}/molecular-profiles
MUT_PROFILE  = "{study_id}_mutations"
CNA_PROFILES = {                          # only where gistic exists
    "mel_dfci_2019": "mel_dfci_2019_gistic",
    "mel_ucla_2016": "mel_ucla_2016_gistic",
    # mel_iatlas_riaz_nivolumab_2017 has no CNA profile
}

for study_id, label in studies.items():
    print(f"\n=== {label} ({study_id}) ===", flush=True)

    # ── Clinical data (response labels + TMB) ──────────────────────────────
    r = requests.get(
        f"https://www.cbioportal.org/api/studies/{study_id}/clinical-data",
        params={"clinicalDataType": "SAMPLE", "pageSize": 10000},
        timeout=60,
    )
    clinical = r.json()
    with open(f"data/cbioportal/{label}_clinical_raw.json", "w") as f:
        json.dump(clinical, f)
    print(f"  Clinical records: {len(clinical)}", flush=True)

    # ── Mutation data ──────────────────────────────────────────────────────
    # Bulk fetch requires POST to /mutations/fetch with sampleListId in body
    profile_id  = MUT_PROFILE.format(study_id=study_id)
    sample_list = f"{study_id}_sequenced"
    muts = []
    page = 0
    page_size = 10000
    while True:
        r2 = requests.post(
            f"https://www.cbioportal.org/api/molecular-profiles/{profile_id}/mutations/fetch",
            params={"pageSize": page_size, "pageNumber": page, "projection": "SUMMARY"},
            headers={"Content-Type": "application/json"},
            data=json.dumps({"sampleListId": sample_list}),
            timeout=180,
        )
        r2.raise_for_status()
        page_data = r2.json()
        muts.extend(page_data)
        if len(page_data) < page_size:
            break
        page += 1
    with open(f"data/cbioportal/{label}_mutations_raw.json", "w") as f:
        json.dump(muts, f)
    print(f"  Mutation records: {len(muts)}", flush=True)

    # ── CNA data (gistic, where available) ────────────────────────────────
    if study_id in CNA_PROFILES:
        cna_profile  = CNA_PROFILES[study_id]
        cna_samplist = f"{study_id}_cna"
        try:
            r3 = requests.get(
                f"https://www.cbioportal.org/api/molecular-profiles/{cna_profile}/discrete-copy-number",
                params={"sampleListId": cna_samplist, "pageSize": 500000},
                timeout=180,
            )
            cna = r3.json()
            with open(f"data/cbioportal/{label}_cna_raw.json", "w") as f:
                json.dump(cna, f)
            print(f"  CNA records: {len(cna)}", flush=True)
        except Exception as e:
            print(f"  CNA: error ({e})", flush=True)
    else:
        print(f"  CNA: no profile for this study", flush=True)

print("\nDONE", flush=True)
