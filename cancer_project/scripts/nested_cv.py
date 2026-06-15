#!/usr/bin/env python3
"""
Phase 3a: Nested LOOCV with inner 5-fold hyperparameter tuning.

Outer loop : LOOCV (n=118 folds, one held-out sample each)
Inner loop : 5-fold stratified CV to tune hyperparameters
Per-fold   : batch correction (additive mean-centering, 3 cohorts) then
             feature selection (prevalence + top-50% variance + top-100 |PBr|)
             — fully leak-free, no information from held-out sample.

Models / grids
  ElasticNet  : alpha ∈ {0.01,0.1,1.0}  × l1_ratio ∈ {0.3,0.5,0.7}   → 9 configs
  RandomForest: n_estimators ∈ {200,500} × max_depth ∈ {5,10,None}     → 6 configs
  XGBoost     : n_estimators ∈ {100,200} × max_depth ∈ {2,3}
                × learning_rate ∈ {0.05,0.1}                            → 8 configs

Self-check: times the first TIMING_PROBE_N outer folds (discarded, not in results).
If projected total > MAX_RUNTIME_HOURS the XGBoost grid collapses to a single
fixed config (n_estimators=150, max_depth=3, lr=0.1); decision logged.

After nested CV: permutation test (N=100) per model using consensus best params
(mode across outer folds) — outer LOOCV with fixed params, no inner tuning.

Outputs (all under results/ml/nested_cv/):
  nested_cv_results.tsv        per-fold predictions + best params + inner AUC
  nested_cv_summary.tsv        per-model AUC/accuracy/sens/spec
  nested_cv_best_params.tsv    best params per outer fold per model
  permutation_test_nested.tsv  perm test summary (p-values)
  perm_aucs_nested.tsv         per-permutation AUC for each model
"""

import os, sys, time, warnings, json, collections
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import xgboost as xgb

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ──────────────────────────────────────────────────────────────────────
CLR_PATH    = "results/ml/n118_3cohort/X_genus_clr.tsv"
RAW_PATH    = "results/ml/n118_3cohort/X_genus_raw.tsv"
LABELS_PATH = "metadata/response_labels_3cohort.tsv"
OUT_DIR     = "results/ml/nested_cv"
os.makedirs(OUT_DIR, exist_ok=True)

# ── settings ───────────────────────────────────────────────────────────────────
SEED                   = 42
INNER_K                = 5
N_PERMS                = 100
TIMING_PROBE_N         = 3      # outer folds timed before deciding XGB grid
MAX_RUNTIME_HOURS      = 3.0
RF_N_JOBS              = 4      # parallelism for RF tree fitting
PREVALENCE_FRAC        = 0.10
VARIANCE_KEEP_FRACTION = 0.50
TOP_N                  = 100
XGB_FALLBACK           = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1}

# ── hyperparameter grids ───────────────────────────────────────────────────────
EN_GRID = [
    {"C": round(1.0 / alpha, 6), "l1_ratio": l1}
    for alpha in [0.01, 0.1, 1.0]
    for l1 in [0.3, 0.5, 0.7]
]
RF_GRID = [
    {"n_estimators": ne, "max_depth": md}
    for ne in [200, 500]
    for md in [5, 10, None]
]
XGB_GRID_FULL = [
    {"n_estimators": ne, "max_depth": md, "learning_rate": lr}
    for ne in [100, 200]
    for md in [2, 3]
    for lr in [0.05, 0.1]
]

# ── load data ──────────────────────────────────────────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Loading data …", flush=True)
clr    = pd.read_csv(CLR_PATH,    sep="\t", index_col="run_accession")
raw    = pd.read_csv(RAW_PATH,    sep="\t", index_col="run_accession")
labels = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")

response = labels.reindex(clr.index)["response"]
batch    = pd.Series(
    [
        "cohort1" if a.startswith("SRR5930")  else
        "cohort2" if a.startswith("SRR11413") else
        "cohort3"
        for a in clr.index
    ],
    index=clr.index,
)

keep = response.notna()
if not keep.all():
    print(f"  WARNING: dropping {(~keep).sum()} samples with missing labels", flush=True)
    clr = clr.loc[keep]; raw = raw.loc[keep]
    response = response.loc[keep]; batch = batch.loc[keep]

SAMPLES  = clr.index.tolist()
N        = len(SAMPLES)
FEATURES = clr.columns.tolist()

print(f"  n={N}, features={len(FEATURES)}, "
      f"R={(response=='R').sum()}, NR={(response=='NR').sum()}", flush=True)
print(f"  cohorts: {batch.value_counts().to_dict()}", flush=True)
print(f"  EN grid: {len(EN_GRID)}, RF grid: {len(RF_GRID)}, "
      f"XGB grid (full): {len(XGB_GRID_FULL)}", flush=True)

# ── per-fold helpers ───────────────────────────────────────────────────────────
def batch_correct_fold(tr_clr: pd.DataFrame, te_clr: pd.DataFrame,
                       tr_batch: pd.Series, te_batch: pd.Series):
    gm = tr_clr.mean(axis=0)
    offsets = {b: tr_clr.loc[tr_batch == b].mean(axis=0) - gm
               for b in tr_batch.unique()}
    corr_tr = tr_clr.copy()
    for b, off in offsets.items():
        corr_tr.loc[tr_batch == b] = tr_clr.loc[tr_batch == b].values - off.values
    tb = te_batch.iloc[0]
    corr_te = (te_clr.values - offsets[tb].values) if tb in offsets else te_clr.values.copy()
    return corr_tr, corr_te  # DataFrame, ndarray

def feature_select_df(corr_tr: pd.DataFrame, tr_raw: pd.DataFrame,
                      tr_labels: pd.Series) -> list:
    n = len(corr_tr)
    min_prev   = max(2, int(np.ceil(PREVALENCE_FRAC * n)))
    pres       = (tr_raw > 0).sum(axis=0)
    prevalent  = pres[pres >= min_prev].index.tolist()
    vv         = corr_tr[prevalent].var(axis=0)
    high_var   = vv[vv >= vv.quantile(1.0 - VARIANCE_KEEP_FRACTION)].index.tolist()
    y_bin      = (tr_labels == "R").astype(int).values
    pbs        = {col: abs(stats.pointbiserialr(y_bin, corr_tr[col].values)[0])
                  for col in high_var}
    return pd.Series(pbs).sort_values(ascending=False).head(TOP_N).index.tolist()


def inner_cv_select(tr_clr: pd.DataFrame, tr_raw: pd.DataFrame,
                    tr_response: pd.Series, tr_batch: pd.Series,
                    en_grid, rf_grid, xgb_grid, fold_seed: int):
    """
    5-fold inner CV on outer-training set.
    Returns (best_en_params, best_rf_params, best_xgb_params,
             best_en_inner_auc, best_rf_inner_auc, best_xgb_inner_auc).
    """
    y_bin_full = (tr_response == "R").astype(int).values
    skf = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=fold_seed)

    en_fold_aucs  = [[] for _ in en_grid]
    rf_fold_aucs  = [[] for _ in rf_grid]
    xgb_fold_aucs = [[] for _ in xgb_grid]

    for inner_tr_i, inner_va_i in skf.split(tr_clr.values, y_bin_full):
        # Slice to inner train / val using integer positions
        itr_clr  = tr_clr.iloc[inner_tr_i]
        iva_clr  = tr_clr.iloc[inner_va_i]
        itr_raw  = tr_raw.iloc[inner_tr_i]
        itr_resp = tr_response.iloc[inner_tr_i]
        iva_resp = tr_response.iloc[inner_va_i]
        itr_bat  = tr_batch.iloc[inner_tr_i]
        iva_bat  = tr_batch.iloc[inner_va_i]

        # Inner batch correction
        icorr_tr, icorr_va = batch_correct_fold(itr_clr, iva_clr, itr_bat, iva_bat)

        # Inner feature selection (on inner training only, with inner training labels)
        icorr_tr_df = pd.DataFrame(icorr_tr, index=itr_clr.index, columns=itr_clr.columns)
        isel        = feature_select_df(icorr_tr_df, itr_raw, itr_resp)

        iX_tr  = icorr_tr_df[isel].values
        iX_va  = icorr_va[:, [FEATURES.index(c) for c in isel]]
        iy_tr  = itr_resp.values
        iy_va  = iva_resp.values
        iy_bin_va = (pd.Series(iy_va) == "R").astype(int).values

        if len(np.unique(iy_bin_va)) < 2:
            # All one class in inner val — skip this split for AUC
            continue

        # ── ElasticNet ──
        for gi, params in enumerate(en_grid):
            try:
                m = LogisticRegression(
                    penalty="elasticnet", solver="saga",
                    C=params["C"], l1_ratio=params["l1_ratio"],
                    class_weight="balanced", max_iter=5000, tol=1e-3,
                    random_state=SEED,
                )
                m.fit(iX_tr, iy_tr)
                classes   = list(m.classes_)
                r_prob_va = m.predict_proba(iX_va)[:, classes.index("R")]
                en_fold_aucs[gi].append(roc_auc_score(iy_bin_va, r_prob_va))
            except Exception:
                pass

        # ── RandomForest ──
        for gi, params in enumerate(rf_grid):
            try:
                m = RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    class_weight="balanced", random_state=SEED, n_jobs=RF_N_JOBS,
                )
                m.fit(iX_tr, iy_tr)
                classes   = list(m.classes_)
                r_prob_va = m.predict_proba(iX_va)[:, classes.index("R")]
                rf_fold_aucs[gi].append(roc_auc_score(iy_bin_va, r_prob_va))
            except Exception:
                pass

        # ── XGBoost ──
        iy_bin_tr = (pd.Series(iy_tr) == "R").astype(int).values
        for gi, params in enumerate(xgb_grid):
            try:
                m = xgb.XGBClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    eval_metric="logloss", n_jobs=1, verbosity=0, random_state=SEED,
                )
                m.fit(iX_tr, iy_bin_tr)
                r_prob_va = m.predict_proba(iX_va)[:, 1]
                xgb_fold_aucs[gi].append(roc_auc_score(iy_bin_va, r_prob_va))
            except Exception:
                pass

    def best_in_grid(grid, fold_aucs):
        means = [np.mean(a) if a else -1.0 for a in fold_aucs]
        bi    = int(np.argmax(means))
        return grid[bi], float(means[bi])

    best_en,  auc_en  = best_in_grid(en_grid,  en_fold_aucs)
    best_rf,  auc_rf  = best_in_grid(rf_grid,  rf_fold_aucs)
    best_xgb, auc_xgb = best_in_grid(xgb_grid, xgb_fold_aucs)
    return best_en, best_rf, best_xgb, auc_en, auc_rf, auc_xgb


def run_one_outer_fold(held_out_idx: str, resp_vec: pd.Series, xgb_grid):
    """
    One outer LOOCV fold:
      1. Batch correct outer training, apply to held-out.
      2. Feature select on outer training.
      3. Inner 5-fold CV to pick best params per model.
      4. Refit on full outer training, predict held-out.
    Returns (result_dict, best_params_dict).
    """
    train_idx = pd.Index([s for s in SAMPLES if s != held_out_idx])

    tr_clr  = clr.loc[train_idx]
    te_clr  = clr.loc[[held_out_idx]]
    tr_raw  = raw.loc[train_idx]
    tr_resp = resp_vec.loc[train_idx]
    tr_bat  = batch.loc[train_idx]
    te_bat  = batch.loc[[held_out_idx]]

    # Outer batch correction
    corr_tr_df, corr_te_arr = batch_correct_fold(tr_clr, te_clr, tr_bat, te_bat)

    # Outer feature selection
    sel = feature_select_df(corr_tr_df, tr_raw, tr_resp)
    X_tr  = corr_tr_df[sel].values
    y_tr  = tr_resp.values
    X_te  = corr_te_arr[:, [FEATURES.index(c) for c in sel]]

    # Inner CV: select best hyperparams
    fold_seed = SEED + SAMPLES.index(held_out_idx)
    best_en, best_rf, best_xgb, auc_en, auc_rf, auc_xgb = inner_cv_select(
        corr_tr_df, tr_raw, tr_resp, tr_bat,
        EN_GRID, RF_GRID, xgb_grid, fold_seed,
    )

    preds = {}

    # ── ElasticNet final fit ──
    m_en = LogisticRegression(
        penalty="elasticnet", solver="saga",
        C=best_en["C"], l1_ratio=best_en["l1_ratio"],
        class_weight="balanced", max_iter=5000, tol=1e-3, random_state=SEED,
    )
    m_en.fit(X_tr, y_tr)
    classes_en = list(m_en.classes_)
    r_prob_en  = float(m_en.predict_proba(X_te)[0, classes_en.index("R")])
    preds["ElasticNet"] = r_prob_en

    # ── RandomForest final fit ──
    m_rf = RandomForestClassifier(
        n_estimators=best_rf["n_estimators"], max_depth=best_rf["max_depth"],
        class_weight="balanced", random_state=SEED, n_jobs=RF_N_JOBS,
    )
    m_rf.fit(X_tr, y_tr)
    classes_rf = list(m_rf.classes_)
    r_prob_rf  = float(m_rf.predict_proba(X_te)[0, classes_rf.index("R")])
    preds["RandomForest"] = r_prob_rf

    # ── XGBoost final fit ──
    y_bin_tr = (pd.Series(y_tr) == "R").astype(int).values
    m_xgb = xgb.XGBClassifier(
        n_estimators=best_xgb["n_estimators"], max_depth=best_xgb["max_depth"],
        learning_rate=best_xgb["learning_rate"],
        eval_metric="logloss", n_jobs=1, verbosity=0, random_state=SEED,
    )
    m_xgb.fit(X_tr, y_bin_tr)
    r_prob_xgb = float(m_xgb.predict_proba(X_te)[0, 1])
    preds["XGBoost"] = r_prob_xgb

    actual = resp_vec[held_out_idx]
    result = {
        "run_accession": held_out_idx,
        "actual":         actual,
        "n_features":     len(sel),
        # ElasticNet
        "EN_prob_R":       round(r_prob_en, 4),
        "EN_pred":         "R" if r_prob_en >= 0.5 else "NR",
        "EN_correct":      "YES" if ("R" if r_prob_en >= 0.5 else "NR") == actual else "NO",
        "EN_inner_auc":    round(auc_en, 4),
        "EN_best_C":       best_en["C"],
        "EN_best_l1":      best_en["l1_ratio"],
        # RandomForest
        "RF_prob_R":       round(r_prob_rf, 4),
        "RF_pred":         "R" if r_prob_rf >= 0.5 else "NR",
        "RF_correct":      "YES" if ("R" if r_prob_rf >= 0.5 else "NR") == actual else "NO",
        "RF_inner_auc":    round(auc_rf, 4),
        "RF_best_n_est":   best_rf["n_estimators"],
        "RF_best_depth":   str(best_rf["max_depth"]),
        # XGBoost
        "XGB_prob_R":      round(r_prob_xgb, 4),
        "XGB_pred":        "R" if r_prob_xgb >= 0.5 else "NR",
        "XGB_correct":     "YES" if ("R" if r_prob_xgb >= 0.5 else "NR") == actual else "NO",
        "XGB_inner_auc":   round(auc_xgb, 4),
        "XGB_best_n_est":  best_xgb["n_estimators"],
        "XGB_best_depth":  best_xgb["max_depth"],
        "XGB_best_lr":     best_xgb["learning_rate"],
    }
    best_params = {
        "ElasticNet":   best_en,
        "RandomForest": best_rf,
        "XGBoost":      best_xgb,
    }
    return result, best_params


def run_one_outer_fold_fixed(held_out_idx: str, resp_vec: pd.Series,
                              params_en, params_rf, params_xgb):
    """
    Outer fold with fixed hyperparams (used in permutation test — no inner CV).
    Batch correction and feature selection still use resp_vec (permuted labels).
    """
    train_idx = pd.Index([s for s in SAMPLES if s != held_out_idx])

    tr_clr  = clr.loc[train_idx]
    te_clr  = clr.loc[[held_out_idx]]
    tr_raw  = raw.loc[train_idx]
    tr_resp = resp_vec.loc[train_idx]
    tr_bat  = batch.loc[train_idx]
    te_bat  = batch.loc[[held_out_idx]]

    corr_tr_df, corr_te_arr = batch_correct_fold(tr_clr, te_clr, tr_bat, te_bat)
    sel = feature_select_df(corr_tr_df, tr_raw, tr_resp)
    X_tr  = corr_tr_df[sel].values
    y_tr  = tr_resp.values
    X_te  = corr_te_arr[:, [FEATURES.index(c) for c in sel]]

    m_en = LogisticRegression(
        penalty="elasticnet", solver="saga",
        C=params_en["C"], l1_ratio=params_en["l1_ratio"],
        class_weight="balanced", max_iter=5000, tol=1e-3, random_state=SEED,
    )
    m_en.fit(X_tr, y_tr)
    classes_en = list(m_en.classes_)
    r_prob_en  = float(m_en.predict_proba(X_te)[0, classes_en.index("R")])

    m_rf = RandomForestClassifier(
        n_estimators=params_rf["n_estimators"], max_depth=params_rf["max_depth"],
        class_weight="balanced", random_state=SEED, n_jobs=RF_N_JOBS,
    )
    m_rf.fit(X_tr, y_tr)
    classes_rf = list(m_rf.classes_)
    r_prob_rf  = float(m_rf.predict_proba(X_te)[0, classes_rf.index("R")])

    y_bin_tr = (pd.Series(y_tr) == "R").astype(int).values
    m_xgb = xgb.XGBClassifier(
        n_estimators=params_xgb["n_estimators"], max_depth=params_xgb["max_depth"],
        learning_rate=params_xgb["learning_rate"],
        eval_metric="logloss", n_jobs=1, verbosity=0, random_state=SEED,
    )
    m_xgb.fit(X_tr, y_bin_tr)
    r_prob_xgb = float(m_xgb.predict_proba(X_te)[0, 1])

    return r_prob_en, r_prob_rf, r_prob_xgb


def compute_metrics(prob_R_list, actual_list):
    y_true  = np.array([(1 if a == "R" else 0) for a in actual_list])
    y_prob  = np.array(prob_R_list)
    y_pred  = (y_prob >= 0.5).astype(int)
    acc     = float((y_true == y_pred).mean())
    auc     = float(roc_auc_score(y_true, y_prob))
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return dict(roc_auc=round(auc, 4), accuracy=round(acc, 4),
                sensitivity=round(sens, 4), specificity=round(spec, 4),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))


def consensus_params(param_list):
    """Mode of a list of param dicts — picks the most frequent config."""
    counts = collections.Counter(json.dumps(p, sort_keys=True) for p in param_list)
    return json.loads(counts.most_common(1)[0][0])


# ══════════════════════════════════════════════════════════════════════════════
# TIMING PROBE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{time.strftime('%H:%M:%S')}] Timing probe: running {TIMING_PROBE_N} "
      f"outer folds with full grids …", flush=True)

t_probe_start = time.time()
for probe_idx in SAMPLES[:TIMING_PROBE_N]:
    run_one_outer_fold(probe_idx, response, XGB_GRID_FULL)
t_probe = time.time() - t_probe_start

t_per_fold  = t_probe / TIMING_PROBE_N
t_projected = t_per_fold * N
print(f"  {TIMING_PROBE_N} folds in {t_probe:.1f}s  →  "
      f"{t_per_fold:.1f}s/fold  →  projected total: {t_projected/3600:.2f}h", flush=True)

if t_projected / 3600 > MAX_RUNTIME_HOURS:
    XGB_GRID = [XGB_FALLBACK]
    print(f"  [AUTO] Projected runtime ({t_projected/3600:.2f}h) > {MAX_RUNTIME_HOURS}h — "
          f"XGBoost grid REDUCED to single fixed config: {XGB_FALLBACK}", flush=True)
    print(f"  [AUTO] All 118 outer folds will use reduced XGB grid.", flush=True)
else:
    XGB_GRID = XGB_GRID_FULL
    print(f"  [OK] Runtime within budget — using full XGB grid ({len(XGB_GRID_FULL)} configs).",
          flush=True)

print(f"  Inner fits per outer fold: "
      f"EN={len(EN_GRID)*INNER_K}, RF={len(RF_GRID)*INNER_K}, "
      f"XGB={len(XGB_GRID)*INNER_K}  total={( len(EN_GRID)+len(RF_GRID)+len(XGB_GRID))*INNER_K}",
      flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN NESTED LOOCV
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{time.strftime('%H:%M:%S')}] Starting nested LOOCV ({N} outer folds) …",
      flush=True)

all_results  = []
best_params_per_fold = {"ElasticNet": [], "RandomForest": [], "XGBoost": []}

t_main_start = time.time()
for fold_num, held_out in enumerate(SAMPLES, start=1):
    result, best_params = run_one_outer_fold(held_out, response, XGB_GRID)
    all_results.append(result)
    for mname in ["ElasticNet", "RandomForest", "XGBoost"]:
        best_params_per_fold[mname].append(best_params[mname])

    if fold_num % 10 == 0 or fold_num == 1 or fold_num == N:
        elapsed = time.time() - t_main_start
        eta     = (N - fold_num) * (elapsed / fold_num)
        print(f"  fold {fold_num:3d}/{N}  "
              f"EN={result['EN_prob_R']:.3f}({result['EN_pred']})  "
              f"RF={result['RF_prob_R']:.3f}({result['RF_pred']})  "
              f"XGB={result['XGB_prob_R']:.3f}({result['XGB_pred']})  "
              f"actual={result['actual']}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
              flush=True)

t_main = time.time() - t_main_start
print(f"\n[{time.strftime('%H:%M:%S')}] Nested LOOCV done in {t_main/60:.1f}min", flush=True)

# ── compute metrics ────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
actuals    = results_df["actual"].tolist()

metrics = {}
for mname, prob_col in [("ElasticNet","EN_prob_R"), ("RandomForest","RF_prob_R"), ("XGBoost","XGB_prob_R")]:
    metrics[mname] = compute_metrics(results_df[prob_col].tolist(), actuals)

print(f"\n{'='*70}", flush=True)
print("NESTED LOOCV RESULTS — n=118, 3-cohort, per-fold ComBat + feature selection", flush=True)
print(f"Inner 5-fold CV per outer fold | XGB grid: {'full' if len(XGB_GRID)>1 else 'fixed'}", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'Model':<16} {'AUC':>7} {'Acc':>7} {'Sens':>7} {'Spec':>7}  TP/FP/TN/FN", flush=True)
print(f"  {'-'*65}", flush=True)
for mn in ["ElasticNet", "RandomForest", "XGBoost"]:
    m  = metrics[mn]
    mk = " <- best" if m["roc_auc"] == max(v["roc_auc"] for v in metrics.values()) else ""
    print(f"  {mn:<16} {m['roc_auc']:>7.4f} {m['accuracy']:>7.4f} "
          f"{m['sensitivity']:>7.4f} {m['specificity']:>7.4f}  "
          f"{m['TP']}/{m['FP']}/{m['TN']}/{m['FN']}{mk}", flush=True)

# Inner CV AUC averages
print(f"\n  Inner CV AUC (mean across outer folds):", flush=True)
for mname, col in [("ElasticNet","EN_inner_auc"), ("RandomForest","RF_inner_auc"), ("XGBoost","XGB_inner_auc")]:
    print(f"    {mname:<16}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}", flush=True)

# ── save per-fold results ──────────────────────────────────────────────────────
results_df.to_csv(f"{OUT_DIR}/nested_cv_results.tsv", sep="\t", index=False)

summary_rows = [{"model": mn, **metrics[mn]} for mn in ["ElasticNet", "RandomForest", "XGBoost"]]
pd.DataFrame(summary_rows).to_csv(f"{OUT_DIR}/nested_cv_summary.tsv", sep="\t", index=False)

# Best params per outer fold
param_rows = []
for fi, result in enumerate(all_results):
    param_rows.append({
        "fold": fi + 1,
        "run_accession":   result["run_accession"],
        "EN_C":            result["EN_best_C"],
        "EN_l1_ratio":     result["EN_best_l1"],
        "RF_n_estimators": result["RF_best_n_est"],
        "RF_max_depth":    result["RF_best_depth"],
        "XGB_n_estimators":result["XGB_best_n_est"],
        "XGB_max_depth":   result["XGB_best_depth"],
        "XGB_learning_rate":result["XGB_best_lr"],
    })
pd.DataFrame(param_rows).to_csv(f"{OUT_DIR}/nested_cv_best_params.tsv", sep="\t", index=False)

print(f"\n  Saved: {OUT_DIR}/nested_cv_results.tsv", flush=True)
print(f"  Saved: {OUT_DIR}/nested_cv_summary.tsv", flush=True)
print(f"  Saved: {OUT_DIR}/nested_cv_best_params.tsv", flush=True)

# ── consensus hyperparams for permutation test ─────────────────────────────────
cons_en  = consensus_params(best_params_per_fold["ElasticNet"])
cons_rf  = consensus_params(best_params_per_fold["RandomForest"])
cons_xgb = consensus_params(best_params_per_fold["XGBoost"])
print(f"\n  Consensus params for permutation test:", flush=True)
print(f"    ElasticNet  : C={cons_en['C']}, l1_ratio={cons_en['l1_ratio']}", flush=True)
print(f"    RandomForest: n_est={cons_rf['n_estimators']}, depth={cons_rf['max_depth']}", flush=True)
print(f"    XGBoost     : n_est={cons_xgb['n_estimators']}, depth={cons_xgb['max_depth']}, "
      f"lr={cons_xgb['learning_rate']}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# PERMUTATION TEST (N=100)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{time.strftime('%H:%M:%S')}] Permutation test: N={N_PERMS}, "
      f"fixed params (no inner tuning per perm) …", flush=True)
print(f"  Note: feature selection uses permuted labels (point-biserial on shuffled y) — "
      f"no label leakage from real run.", flush=True)

obs_auc = {mn: metrics[mn]["roc_auc"] for mn in ["ElasticNet", "RandomForest", "XGBoost"]}
print(f"  Observed AUC: EN={obs_auc['ElasticNet']:.4f}, "
      f"RF={obs_auc['RandomForest']:.4f}, XGB={obs_auc['XGBoost']:.4f}", flush=True)

rng = np.random.default_rng(SEED)
true_arr = response.values.copy()

perm_en_aucs  = []
perm_rf_aucs  = []
perm_xgb_aucs = []

t_perm_start = time.time()
for perm_i in range(1, N_PERMS + 1):
    shuffled_arr = rng.permutation(true_arr)
    shuffled_resp = pd.Series(shuffled_arr, index=response.index)

    en_probs  = []
    rf_probs  = []
    xgb_probs = []

    for held_out in SAMPLES:
        p_en, p_rf, p_xgb = run_one_outer_fold_fixed(
            held_out, shuffled_resp, cons_en, cons_rf, cons_xgb)
        en_probs.append(p_en)
        rf_probs.append(p_rf)
        xgb_probs.append(p_xgb)

    y_true_perm = [(1 if a == "R" else 0) for a in shuffled_arr]
    try:    auc_en  = roc_auc_score(y_true_perm, en_probs)
    except: auc_en  = 0.5
    try:    auc_rf  = roc_auc_score(y_true_perm, rf_probs)
    except: auc_rf  = 0.5
    try:    auc_xgb = roc_auc_score(y_true_perm, xgb_probs)
    except: auc_xgb = 0.5

    perm_en_aucs.append(auc_en)
    perm_rf_aucs.append(auc_rf)
    perm_xgb_aucs.append(auc_xgb)

    if perm_i % 10 == 0 or perm_i == 1:
        elapsed = time.time() - t_perm_start
        eta     = (N_PERMS - perm_i) * (elapsed / perm_i)
        print(f"  perm {perm_i:3d}/{N_PERMS}  "
              f"EN={auc_en:.4f} RF={auc_rf:.4f} XGB={auc_xgb:.4f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
              flush=True)

t_perm = time.time() - t_perm_start
print(f"\n[{time.strftime('%H:%M:%S')}] Permutation test done in {t_perm/60:.1f}min", flush=True)

# ── permutation test results ───────────────────────────────────────────────────
perm_aucs_dict = {
    "ElasticNet":   np.array(perm_en_aucs),
    "RandomForest": np.array(perm_rf_aucs),
    "XGBoost":      np.array(perm_xgb_aucs),
}

print(f"\n{'='*70}", flush=True)
print("PERMUTATION TEST RESULTS", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Fixed consensus params | feature selection on permuted labels", flush=True)
print(f"  {'Model':<16} {'Obs AUC':>8} {'Perm mean':>10} {'Perm std':>9} "
      f"{'p-value':>8}  Decision", flush=True)
print(f"  {'-'*65}", flush=True)

summary_rows = []
for mn in ["ElasticNet", "RandomForest", "XGBoost"]:
    pa  = perm_aucs_dict[mn]
    obs = obs_auc[mn]
    pv  = float((pa >= obs).mean())
    dec = ("SIGNIFICANT" if pv < 0.05 else
           "MARGINAL"    if pv < 0.10 else
           "not significant")
    print(f"  {mn:<16} {obs:>8.4f} {pa.mean():>10.4f} {pa.std():>9.4f} "
          f"{pv:>8.4f}  {dec}", flush=True)
    summary_rows.append({
        "model":         mn,
        "observed_auc":  round(obs, 4),
        "perm_auc_mean": round(float(pa.mean()), 4),
        "perm_auc_std":  round(float(pa.std()),  4),
        "perm_auc_min":  round(float(pa.min()),  4),
        "perm_auc_max":  round(float(pa.max()),  4),
        "n_perm_gte_obs":int((pa >= obs).sum()),
        "n_perms":       N_PERMS,
        "empirical_pval":round(pv, 4),
        "xgb_grid_used": "full" if len(XGB_GRID) > 1 else "fixed_fallback",
        "cons_en_C":     cons_en["C"],
        "cons_en_l1":    cons_en["l1_ratio"],
        "cons_rf_n_est": cons_rf["n_estimators"],
        "cons_rf_depth": str(cons_rf["max_depth"]),
        "cons_xgb_n_est":cons_xgb["n_estimators"],
        "cons_xgb_depth":cons_xgb["max_depth"],
        "cons_xgb_lr":   cons_xgb["learning_rate"],
    })

pd.DataFrame(summary_rows).to_csv(
    f"{OUT_DIR}/permutation_test_nested.tsv", sep="\t", index=False)

perm_df = pd.DataFrame({
    "perm_index":  range(1, N_PERMS + 1),
    "ElasticNet":  perm_en_aucs,
    "RandomForest":perm_rf_aucs,
    "XGBoost":     perm_xgb_aucs,
})
perm_df.to_csv(f"{OUT_DIR}/perm_aucs_nested.tsv", sep="\t", index=False)

t_total = time.time() - t_probe_start
print(f"\n  Saved: {OUT_DIR}/permutation_test_nested.tsv", flush=True)
print(f"  Saved: {OUT_DIR}/perm_aucs_nested.tsv", flush=True)
print(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE. Total wall time: {t_total/60:.1f}min", flush=True)
