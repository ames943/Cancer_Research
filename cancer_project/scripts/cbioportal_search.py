import requests, json, os

os.makedirs("data/cbioportal", exist_ok=True)

studies = [
    ("skcm_vanderbilt_mskcc_2015", "Van Allen 2015 anti-CTLA4"),
    ("mel_ucla_2016",               "Hugo 2016 anti-PD-1"),           # was skcm_ucla_2016
    ("mel_iatlas_riaz_nivolumab_2017", "Riaz 2017 anti-PD-1"),        # was skcm_mskcc_2020
    ("mel_dfci_2019",               "Liu 2019 anti-PD-1"),
    ("skcm_dfci_2015",              "Van Allen 2015 anti-CTLA4 (Science)"),  # extra
    ("mel_iatlas_gide_2019",        "Gide 2019 nivo+ipi"),             # extra
]

for study_id, label in studies:
    try:
        # Get available clinical attributes
        r2 = requests.get(
            f"https://www.cbioportal.org/api/studies/{study_id}/clinical-attributes",
            timeout=30
        )
        attrs = [a["clinicalAttributeId"] for a in r2.json()]

        # Get sample count
        r3 = requests.get(
            f"https://www.cbioportal.org/api/studies/{study_id}/samples",
            params={"pageSize": 500},
            timeout=30
        )
        samples = r3.json()

        with open(f"data/cbioportal/{study_id}_samples.json", "w") as f:
            json.dump(samples, f)
        with open(f"data/cbioportal/{study_id}_attrs.json", "w") as f:
            json.dump(attrs, f)

        print(f"\n{label} ({study_id}): {len(samples)} samples")
        print(f"  attrs: {attrs}")
    except Exception as e:
        print(f"{label}: ERROR {e}")

print("\nDONE")
