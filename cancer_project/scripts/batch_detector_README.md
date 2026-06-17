# batch_detector.py — Batch Effect Detector for Multi-Cohort Omics Studies

A command-line tool that quantifies batch effects vs. biological signal in
multi-cohort omics datasets and issues a **GO / CAUTION / NO-GO** pooling
recommendation. Works with any sample × feature matrix: microbiome, RNA-seq,
proteomics, metabolomics, etc.

---

## Installation

Standard scientific Python stack — no extra dependencies.

```bash
pip install numpy pandas scipy matplotlib
```

Tested with Python 3.8+, numpy ≥ 1.22, pandas ≥ 1.4, scipy ≥ 1.7,
matplotlib ≥ 3.5.

---

## Method

1. **Optional CLR transformation** (`--clr`): centered log-ratio with
   pseudocount (default 1e-6), standard for compositional microbiome data.
2. **Aitchison distance matrix**: Euclidean distance on CLR-transformed data.
3. **PERMANOVA** (Anderson 2001, 999 permutations) for both batch and
   biological response factors → R² and p-value per factor.
4. **Batch:signal ratio** = batch R² / response R².
5. **Dirichlet-corrected power analysis**: parametric bootstrap that draws
   fresh compositions from Dirichlet(observed + pseudocount) for each
   resampled index. Avoids zero-distance duplicate-pair artifacts from naive
   bootstrap-with-replacement on distance matrices, which inflate pseudo-F
   by up to 2× and produce power estimates ~2× too optimistic.
6. **Pooling recommendation** based on batch:signal ratio thresholds.

---

## Usage

```bash
# Basic: 3-cohort microbiome study with CLR
python batch_detector.py \
    --input  feature_matrix.tsv \
    --labels sample_labels.tsv  \
    --batch  cohort             \
    --output results/           \
    --clr

# Batch column in a separate file
python batch_detector.py \
    --input  X.tsv         \
    --labels meta.tsv      \
    --batch  batch_ids.tsv \
    --output out/          \
    --clr

# Single-cohort mode (no --batch) — PERMANOVA + power analysis only
python batch_detector.py \
    --input  X.tsv    \
    --labels meta.tsv \
    --output out/     \
    --clr

# With simulation validation (50-rep Dirichlet bootstrap at current n)
python batch_detector.py \
    --input  X.tsv    \
    --labels meta.tsv \
    --batch  cohort   \
    --output out/     \
    --clr --simulate

# Tune permutation count and bootstrap reps (faster run for quick checks)
python batch_detector.py \
    --input X.tsv --labels meta.tsv --batch cohort --output out/ --clr \
    --n-perms 499 --n-boot 100 --n-perms-inner 99
```

---

## Input formats

### feature_matrix.tsv

Tab-separated. **Samples as rows**, features as columns. First column is the
sample identifier (any unique string). Header row required.

```
sample_id    genus_A    genus_B    genus_C
SRR001       12.3       0.0        4.5
SRR002        0.1       8.7        2.2
...
```

### sample_labels.tsv

Tab-separated. First column is the sample identifier. Must contain a
`response` column (or specify a different name with `--response-col`).
May also contain the batch column.

```
sample_id    response    cohort
SRR001       R           cohort1
SRR002       NR          cohort2
...
```

### batch (optional)

Either:
- **(a)** A column name in `sample_labels.tsv` (e.g. `--batch cohort`)
- **(b)** A path to a 2-column TSV (sample_id in first column, batch label
  in second column)

```
sample_id    batch
SRR001       cohort1
SRR002       cohort2
```

---

## Output files

| File | Contents |
|------|----------|
| `batch_detector_report.json` | Full structured report: PERMANOVA statistics, batch:signal ratio, power estimate, recommendation, all input parameters. |
| `batch_detector_figure.pdf` | Two-panel figure: (A) PERMANOVA R² bar chart with significance annotations; (B) Dirichlet-corrected power curve with current-n marker and 80%-power threshold. |
| `batch_detector_power.tsv` | Power curve table: n, power, ci_lower, ci_upper. |

---

## Interpretation

### Pooling recommendation

| Label | Condition | Action |
|-------|-----------|--------|
| **GO** | batch:signal < 3 **and** response p < 0.05 | Pooling appears appropriate. |
| **CAUTION** | batch:signal 3–8 **or** response p ≥ 0.05 | Apply validated batch correction, or check power curve. |
| **NO-GO** | batch:signal > 8 | Batch dominates signal; naive pooling is not recommended. |

### Batch:signal ratio

The ratio of batch PERMANOVA R² to response PERMANOVA R². Values >> 1
mean technical variance (platform, lab, protocol) exceeds biological
variance (treatment response, phenotype). Ratios above 8× have been
empirically associated with complete loss of predictive signal across
correction methods in published immunotherapy microbiome studies.

### Power curve

Shows the probability that PERMANOVA(response) reaches p < 0.05 as a
function of same-protocol sample size, computed via Dirichlet bootstrap
from the observed data. The curve answers: **"How many same-protocol
patients do I need for 80% power at the effect size observed in this
dataset?"** Note: if the observed effect is null (R² near 0), power
stays low regardless of n — more same-protocol samples help only if the
biological signal is real.

---

## Applied example — gut-tumor microbiome (melanoma anti-PD-1)

Running on n=118 samples across 3 cohorts (Frankel 2017, NovaSeq cohort,
Matson 2018) with CLR transformation:

```
Response PERMANOVA : R² = 0.0068   p = 0.847 (ns)
Batch PERMANOVA    : R² = 0.0768   p = 0.001 (***)
Batch:Signal ratio : 11.3×
Recommendation     : NO-GO
Min n (80% power)  : ~80 same-protocol samples
```

Interpretation: batch variance dominates biological signal by 11.3×; none
of 6 batch-correction methods (including ComBat-seq) restored significant
signal; Dirichlet power analysis confirms the dataset is structurally
underpowered regardless of correction approach.

---

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--clr` | off | Apply CLR transformation (recommended for compositional data). |
| `--n-perms` | 999 | PERMANOVA permutations. 999 is standard; use 9999 for final results. |
| `--n-boot` | 200 | Bootstrap reps per n in power analysis. ≥200 recommended for final results; use 50–100 for quick checks. |
| `--n-perms-inner` | 199 | Inner PERMANOVA perms per bootstrap rep. 199 is sufficient for the power loop. |
| `--simulate` | off | Run 50-rep Dirichlet bootstrap at current n as a validation check. |
| `--response-col` | `response` | Column name for the outcome variable in `--labels`. |
| `--pseudocount` | 1e-6 | Floor added before log (CLR) and used as Dirichlet floor. |
| `--seed` | 42 | Random seed for reproducibility. |

---

## References

Anderson MJ (2001). A new method for non-parametric multivariate analysis
of variance. *Austral Ecology*, 26(1):32–46.

Aitchison J (1986). *The Statistical Analysis of Compositional Data*.
Chapman & Hall, London.

Anderson MJ & Walsh DCI (2013). PERMANOVA, ANOSIM, and the Mantel test in
the face of heterogeneous dispersions: What null hypothesis are you testing?
*Ecological Monographs*, 83(4):557–574.

Phipson B & Smyth GK (2010). Permutation P-values should never be zero:
calculating exact P-values when permutations are randomly drawn. *Statistical
Applications in Genetics and Molecular Biology*, 9(1):Article 39.
