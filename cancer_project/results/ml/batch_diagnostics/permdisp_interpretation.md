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

## Batch R² Dirichlet bootstrap — a note on CI validity

Dirichlet bootstrap CIs for **response R²** contain their point estimates at
all cohort sizes (n=39/79/118), confirming those estimates are well-characterised
by the bootstrap.

For **batch R²**, the observed values fall *above* the Dirichlet bootstrap CI
upper bounds (n=79: R²=0.085 > CI upper 0.079; n=118: R²=0.077 > CI upper
0.072).  The explanation is structural: the Dirichlet bootstrap resamples each
sample's composition **independently**, but the batch effect in the real data is
**correlated within cohorts** — every sample from Cohort 2 (NovaSeq) shares the
same platform-level composition shift, which is not captured by per-sample
independent Dirichlet draws.  Resampling individual samples therefore inflates
within-cohort spread relative to the real data, deflating the bootstrapped R²
below the observed value.

Consequence: the Dirichlet bootstrap CIs for batch R² are **lower bounds** on
the true uncertainty, and the observed R² values are robust lower bounds on the
batch effect size.  For the paper, we report these CIs with this caveat; a
cohort-level bootstrap (resampling entire cohorts) would be unworkable at
k=2–3 cohorts.  The key scientific message — that batch R² (0.077–0.085,
p<0.001) is 11× the response R² at n=118 — does not depend on CI validity:
the PERMANOVA permutation test p-values are valid regardless.

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
