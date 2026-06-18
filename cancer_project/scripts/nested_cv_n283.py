#!/usr/bin/env python3
"""
Nested LOOCV for n=283 (4-cohort: Frankel + NovaSeq + Matson + Lee 2022).

Per-fold additive mean-centering batch correction (4 cohorts).
Same hyperparameter grids + timing-probe + auto XGB reduction as nested_cv.py.
Outputs: results/ml/n283_4cohort/nested_cv_{results,summary,perm}.tsv

Run from cancer_project/:
    nohup python3 scripts/nested_cv_n283.py > logs/nested_cv_n283.log 2>&1 &
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

CLR_PATH    = "results/ml/n283_4cohort/X_genus_clr.tsv"
RAW_PATH    = "results/ml/n283_4cohort/X_genus_raw.tsv"
LABELS_PATH = "results/ml/n283_4cohort/response_labels_n283.tsv"
OUT_DIR     = "results/ml/n283_4cohort"
os.makedirs(OUT_DIR, exist_ok=True)

SEED                   = 42
INNER_K                = 5
N_PERMS                = 100
TIMING_PROBE_N         = 3
MAX_RUNTIME_HOURS      = 6.0
RF_N_JOBS              = 4
PREVALENCE_FRAC        = 0.10
VARIANCE_KEEP_FRACTION = 0.50
TOP_N                  = 100
XGB_FALLBACK           = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1}

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


def cohort_of(sid):
    if sid.startswith("SRR5930"):  return "cohort1"
    if sid.startswith("SRR11413"): return "cohort2"
    if sid.startswith("SRR6000"):  return "cohort3"
    return "cohort4"


print(f"[{time.strftime('%H:%M:%S')}] Loading n=283 data …", flush=True)
clr    = pd.read_csv(CLR_PATH,    sep="\t", index_col="run_accession")
raw    = pd.read_csv(RAW_PATH,    sep="\t", index_col="run_accession")
labels = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")

response = labels.reindex(clr.index)["response"]
batch    = pd.Series({sid: cohort_of(sid) for sid in clr.index}, name="cohort")

keep = response.notna()
if not keep.all():
    print(f"  WARNING: dropping {(~keep).sum()} samples", flush=True)
    clr = clr.loc[keep]; raw = raw.loc[keep]
    response = response.loc[keep]; batch = batch.loc[keep]

SAMPLES  = clr.index.tolist()
N        = len(SAMPLES)
FEATURES = clr.columns.tolist()

print(f"  n={N}, features={len(FEATURES)}, "
      f"R={(response=='R').sum()}, NR={(response=='NR').sum()}", flush=True)
print(f"  cohorts: {batch.value_counts().to_dict()}", flush=True)


# ── per-fold helpers ──────────────────────────────────────────────────────────

def batch_correct_fold(tr_clr, te_clr, tr_batch, te_batch):
    gm = tr_clr.mean(axis=0)
    offsets = {b: tr_clr.loc[tr_batch == b].mean(axis=0) - gm
               for b in tr_batch.unique()}
    corr_tr = tr_clr.copy()
    for b, off in offsets.items():
        corr_tr.loc[tr_batch == b] = tr_clr.loc[tr_batch == b].values - off.values
    tb = te_batch.iloc[0]
    corr_te = (te_clr.values - offsets[tb].values) if tb in offsets else te_clr.values.copy()
    return corr_tr, corr_te


def feature_select_df(clr_tr, raw_tr, resp_tr):
    n         = len(clr_tr)
    min_prev  = max(2, int(np.ceil(PREVALENCE_FRAC * n)))
    pres      = (raw_tr > 0).sum(axis=0)
    prevalent = pres[pres >= min_prev].index.tolist()
    vv        = clr_tr[prevalent].var(axis=0)
    high_var  = vv[vv >= vv.quantile(1.0 - VARIANCE_KEEP_FRACTION)].index.tolist()
    y_bin     = (resp_tr == "R").astype(int).values
    pbs       = {col: abs(stats.pointbiserialr(y_bin, clr_tr[col].values)[0])
                 for col in high_var}
    return pd.Series(pbs).sort_values(ascending=False).head(TOP_N).index.tolist()


def inner_cv_select(tr_clr, tr_raw, tr_response, tr_batch,
                    en_grid, rf_grid, xgb_grid, fold_seed):
    y_bin_full = (tr_response == "R").astype(int).values
    skf = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=fold_seed)

    en_aucs  = [[] for _ in en_grid]
    rf_aucs  = [[] for _ in rf_grid]
    xgb_aucs = [[] for _ in xgb_grid]

    for itr_i, iva_i in skf.split(tr_clr.values, y_bin_full):
        itr_clr  = tr_clr.iloc[itr_i];  iva_clr  = tr_clr.iloc[iva_i]
        itr_raw  = tr_raw.iloc[itr_i]
        itr_resp = tr_response.iloc[itr_i]; iva_resp = tr_response.iloc[iva_i]
        itr_bat  = tr_batch.iloc[itr_i];   iva_bat  = tr_batch.iloc[iva_i]

        icorr_tr, icorr_va = batch_correct_fold(itr_clr, iva_clr, itr_bat, iva_bat)
        icorr_tr_df = pd.DataFrame(icorr_tr, index=itr_clr.index, columns=itr_clr.columns)
        isel = feature_select_df(icorr_tr_df, itr_raw, itr_resp)

        iX_tr = icorr_tr_df[isel].values
        # For inner val, use raw CLR at those feature positions
        feat_idx = [FEATURES.index(c) for c in isel]
        iX_va    = icorr_va[:, feat_idx]
        iy_tr    = itr_resp.values
        iy_bin_va = (iva_resp == "R").astype(int).values

        if len(np.unique(iy_bin_va)) < 2:
            continue

        for gi, params in enumerate(en_grid):
            try:
                m = LogisticRegression(
                    penalty="elasticnet", solver="saga", C=params["C"],
                    l1_ratio=params["l1_ratio"], class_weight="balanced",
                    max_iter=5000, tol=1e-3, random_state=SEED,
                )
                m.fit(iX_tr, iy_tr)
                classes = list(m.classes_)
                en_aucs[gi].append(roc_auc_score(iy_bin_va,
                                    m.predict_proba(iX_va)[:, classes.index("R")]))
            except Exception:
                pass

        for gi, params in enumerate(rf_grid):
            try:
                m = RandomForestClassifier(
                    n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                    class_weight="balanced", random_state=SEED, n_jobs=RF_N_JOBS,
                )
                m.fit(iX_tr, iy_tr)
                classes = list(m.classes_)
                rf_aucs[gi].append(roc_auc_score(iy_bin_va,
                                    m.predict_proba(iX_va)[:, classes.index("R")]))
            except Exception:
                pass

        iy_bin_tr = (itr_resp == "R").astype(int).values
        for gi, params in enumerate(xgb_grid):
            try:
                m = xgb.XGBClassifier(
                    n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    eval_metric="logloss", n_jobs=1, verbosity=0, random_state=SEED,
                )
                m.fit(iX_tr, iy_bin_tr)
                xgb_aucs[gi].append(roc_auc_score(iy_bin_va,
                                    m.predict_proba(iX_va)[:, 1]))
            except Exception:
                pass

    def best_in_grid(grid, aucs):
        means = [np.mean(a) if a else -1.0 for a in aucs]
        bi = int(np.argmax(means))
        return grid[bi], float(means[bi])

    return (*best_in_grid(en_grid, en_aucs),
            *best_in_grid(rf_grid, rf_aucs),
            *best_in_grid(xgb_grid, xgb_aucs))


def run_one_outer_fold(held_out_idx, resp_vec, xgb_grid):
    train_idx = pd.Index([s for s in SAMPLES if s != held_out_idx])
    tr_clr  = clr.loc[train_idx];  te_clr  = clr.loc[[held_out_idx]]
    tr_raw  = raw.loc[train_idx]
    tr_resp = resp_vec.loc[train_idx]
    tr_bat  = batch.loc[train_idx]; te_bat  = batch.loc[[held_out_idx]]

    corr_tr, corr_te = batch_correct_fold(tr_clr, te_clr, tr_bat, te_bat)
    corr_tr_df = pd.DataFrame(corr_tr, index=tr_clr.index, columns=tr_clr.columns)

    sel   = feature_select_df(corr_tr_df, tr_raw, tr_resp)
    X_tr  = corr_tr_df[sel].values
    y_tr  = tr_resp.values
    X_te  = corr_te[:, [FEATURES.index(c) for c in sel]]

    fold_seed = SEED + SAMPLES.index(held_out_idx)
    best_en, auc_en, best_rf, auc_rf, best_xgb, auc_xgb = inner_cv_select(
        corr_tr_df, tr_raw, tr_resp, tr_bat,
        EN_GRID, RF_GRID, xgb_grid, fold_seed,
    )

    m_en = LogisticRegression(
        penalty="elasticnet", solver="saga", C=best_en["C"],
        l1_ratio=best_en["l1_ratio"], class_weight="balanced",
        max_iter=5000, tol=1e-3, random_state=SEED,
    )
    m_en.fit(X_tr, y_tr)
    classes_en = list(m_en.classes_)
    r_en = float(m_en.predict_proba(X_te)[0, classes_en.index("R")])

    m_rf = RandomForestClassifier(
        n_estimators=best_rf["n_estimators"], max_depth=best_rf["max_depth"],
        class_weight="balanced", random_state=SEED, n_jobs=RF_N_JOBS,
    )
    m_rf.fit(X_tr, y_tr)
    classes_rf = list(m_rf.classes_)
    r_rf = float(m_rf.predict_proba(X_te)[0, classes_rf.index("R")])

    y_bin_tr = (pd.Series(y_tr) == "R").astype(int).values
    m_xgb = xgb.XGBClassifier(
        n_estimators=best_xgb["n_estimators"], max_depth=best_xgb["max_depth"],
        learning_rate=best_xgb["learning_rate"],
        eval_metric="logloss", n_jobs=1, verbosity=0, random_state=SEED,
    )
    m_xgb.fit(X_tr, y_bin_tr)
    r_xgb = float(m_xgb.predict_proba(X_te)[0, 1])

    actual = resp_vec[held_out_idx]
    result = {
        "run_accession": held_out_idx, "actual": actual, "n_features": len(sel),
        "EN_prob_R":  round(r_en, 4),  "EN_pred":  "R" if r_en  >= 0.5 else "NR",
        "EN_correct":  "YES" if ("R" if r_en  >= 0.5 else "NR") == actual else "NO",
        "EN_inner_auc": round(auc_en, 4),  "EN_best_C": best_en["C"],  "EN_best_l1": best_en["l1_ratio"],
        "RF_prob_R":  round(r_rf, 4),  "RF_pred":  "R" if r_rf  >= 0.5 else "NR",
        "RF_correct":  "YES" if ("R" if r_rf  >= 0.5 else "NR") == actual else "NO",
        "RF_inner_auc": round(auc_rf, 4),  "RF_best_n_est": best_rf["n_estimators"],
        "RF_best_depth": str(best_rf["max_depth"]),
        "XGB_prob_R": round(r_xgb, 4), "XGB_pred": "R" if r_xgb >= 0.5 else "NR",
        "XGB_correct": "YES" if ("R" if r_xgb >= 0.5 else "NR") == actual else "NO",
        "XGB_inner_auc": round(auc_xgb, 4), "XGB_best_n_est": best_xgb["n_estimators"],
        "XGB_best_depth": best_xgb["max_depth"], "XGB_best_lr": best_xgb["learning_rate"],
    }
    return result, {"ElasticNet": best_en, "RandomForest": best_rf, "XGBoost": best_xgb}


def run_one_outer_fold_fixed(held_out_idx, resp_vec, params_en, params_rf, params_xgb):
    train_idx = pd.Index([s for s in SAMPLES if s != held_out_idx])
    tr_clr  = clr.loc[train_idx];  te_clr  = clr.loc[[held_out_idx]]
    tr_raw  = raw.loc[train_idx];  tr_resp  = resp_vec.loc[train_idx]
    tr_bat  = batch.loc[train_idx]; te_bat  = batch.loc[[held_out_idx]]

    corr_tr, corr_te = batch_correct_fold(tr_clr, te_clr, tr_bat, te_bat)
    corr_tr_df = pd.DataFrame(corr_tr, index=tr_clr.index, columns=tr_clr.columns)
    sel  = feature_select_df(corr_tr_df, tr_raw, tr_resp)
    X_tr = corr_tr_df[sel].values; y_tr = tr_resp.values
    X_te = corr_te[:, [FEATURES.index(c) for c in sel]]

    m_en = LogisticRegression(
        penalty="elasticnet", solver="saga", C=params_en["C"],
        l1_ratio=params_en["l1_ratio"], class_weight="balanced",
        max_iter=5000, tol=1e-3, random_state=SEED,
    )
    m_en.fit(X_tr, y_tr)
    classes_en = list(m_en.classes_)
    r_en = float(m_en.predict_proba(X_te)[0, classes_en.index("R")])

    m_rf = RandomForestClassifier(
        n_estimators=params_rf["n_estimators"], max_depth=params_rf["max_depth"],
        class_weight="balanced", random_state=SEED, n_jobs=RF_N_JOBS,
    )
    m_rf.fit(X_tr, y_tr)
    classes_rf = list(m_rf.classes_)
    r_rf = float(m_rf.predict_proba(X_te)[0, classes_rf.index("R")])

    y_bin_tr = (pd.Series(y_tr) == "R").astype(int).values
    m_xgb = xgb.XGBClassifier(
        n_estimators=params_xgb["n_estimators"], max_depth=params_xgb["max_depth"],
        learning_rate=params_xgb["learning_rate"],
        eval_metric="logloss", n_jobs=1, verbosity=0, random_state=SEED,
    )
    m_xgb.fit(X_tr, y_bin_tr)
    r_xgb = float(m_xgb.predict_proba(X_te)[0, 1])
    return r_en, r_rf, r_xgb


def compute_metrics(prob_R_list, actual_list):
    y_true = np.array([(1 if a == "R" else 0) for a in actual_list])
    y_prob = np.array(prob_R_list)
    y_pred = (y_prob >= 0.5).astype(int)
    acc    = float((y_true == y_pred).mean())
    auc    = float(roc_auc_score(y_true, y_prob))
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return dict(roc_auc=round(auc,4), accuracy=round(acc,4),
                sensitivity=round(sens,4), specificity=round(spec,4),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))


def consensus_params(param_list):
    counts = collections.Counter(json.dumps(p, sort_keys=True) for p in param_list)
    return json.loads(counts.most_common(1)[0][0])


# ── timing probe ──────────────────────────────────────────────────────────────
print(f"\n[{time.strftime('%H:%M:%S')}] Timing probe: {TIMING_PROBE_N} folds …", flush=True)
t0 = time.time()
for probe_idx in SAMPLES[:TIMING_PROBE_N]:
    run_one_outer_fold(probe_idx, response, XGB_GRID_FULL)
t_probe = time.time() - t0
t_per = t_probe / TIMING_PROBE_N
t_proj = t_per * N
print(f"  {t_per:.1f}s/fold → projected total: {t_proj/3600:.2f}h", flush=True)

if t_proj / 3600 > MAX_RUNTIME_HOURS:
    XGB_GRID = [XGB_FALLBACK]
    print(f"  [AUTO] Reducing XGB grid to single config: {XGB_FALLBACK}", flush=True)
else:
    XGB_GRID = XGB_GRID_FULL
    print(f"  [OK] Full XGB grid ({len(XGB_GRID_FULL)} configs)", flush=True)

# ── nested LOOCV ──────────────────────────────────────────────────────────────
print(f"\n[{time.strftime('%H:%M:%S')}] Nested LOOCV ({N} folds) …", flush=True)
all_results = []
best_params_per_fold = {"ElasticNet": [], "RandomForest": [], "XGBoost": []}
t_start = time.time()

for fold_num, held_out in enumerate(SAMPLES, start=1):
    result, best_params = run_one_outer_fold(held_out, response, XGB_GRID)
    all_results.append(result)
    for mname in ["ElasticNet", "RandomForest", "XGBoost"]:
        best_params_per_fold[mname].append(best_params[mname])

    if fold_num % 30 == 0 or fold_num == 1 or fold_num == N:
        elapsed = time.time() - t_start
        eta     = (N - fold_num) * (elapsed / fold_num)
        print(f"  fold {fold_num:3d}/{N}  "
              f"EN={result['EN_prob_R']:.3f}  RF={result['RF_prob_R']:.3f}  "
              f"XGB={result['XGB_prob_R']:.3f}  actual={result['actual']}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

t_main = time.time() - t_start
print(f"\n[{time.strftime('%H:%M:%S')}] Nested LOOCV done in {t_main/60:.1f} min", flush=True)

results_df = pd.DataFrame(all_results)
actuals    = results_df["actual"].tolist()
metrics    = {}
for mname, col in [("ElasticNet","EN_prob_R"), ("RandomForest","RF_prob_R"), ("XGBoost","XGB_prob_R")]:
    metrics[mname] = compute_metrics(results_df[col].tolist(), actuals)

print(f"\n{'='*65}", flush=True)
print("NESTED LOOCV — n=283, 4-cohort, per-fold batch correction", flush=True)
print(f"{'='*65}", flush=True)
print(f"  {'Model':<16} {'AUC':>7} {'Acc':>7} {'Sens':>7} {'Spec':>7}", flush=True)
for mn in ["ElasticNet", "RandomForest", "XGBoost"]:
    m = metrics[mn]
    print(f"  {mn:<16} {m['roc_auc']:>7.4f} {m['accuracy']:>7.4f} "
          f"{m['sensitivity']:>7.4f} {m['specificity']:>7.4f}", flush=True)

results_df.to_csv(f"{OUT_DIR}/nested_cv_n283_results.tsv", sep="\t", index=False)
summary_rows = [{"model": mn, **metrics[mn]} for mn in ["ElasticNet","RandomForest","XGBoost"]]
pd.DataFrame(summary_rows).to_csv(f"{OUT_DIR}/nested_cv_n283_summary.tsv", sep="\t", index=False)

cons_en  = consensus_params(best_params_per_fold["ElasticNet"])
cons_rf  = consensus_params(best_params_per_fold["RandomForest"])
cons_xgb = consensus_params(best_params_per_fold["XGBoost"])
print(f"\n  Consensus: EN C={cons_en['C']} l1={cons_en['l1_ratio']}  "
      f"RF n={cons_rf['n_estimators']} d={cons_rf['max_depth']}  "
      f"XGB n={cons_xgb['n_estimators']} d={cons_xgb['max_depth']} lr={cons_xgb['learning_rate']}", flush=True)

# ── permutation test ──────────────────────────────────────────────────────────
print(f"\n[{time.strftime('%H:%M:%S')}] Permutation test (N={N_PERMS}) …", flush=True)
rng = np.random.default_rng(SEED + 100)
perm_aucs = {"ElasticNet": [], "RandomForest": [], "XGBoost": []}
obs_aucs  = {mn: metrics[mn]["roc_auc"] for mn in metrics}

for perm_i in range(N_PERMS):
    perm_resp = response.copy()
    perm_resp.iloc[:] = rng.permutation(response.values)

    en_probs, rf_probs, xgb_probs = [], [], []
    for held_out in SAMPLES:
        r_en, r_rf, r_xgb = run_one_outer_fold_fixed(
            held_out, perm_resp, cons_en, cons_rf, cons_xgb,
        )
        en_probs.append(r_en); rf_probs.append(r_rf); xgb_probs.append(r_xgb)

    y_true = np.array([(1 if a == "R" else 0) for a in response.values])
    perm_aucs["ElasticNet"].append(roc_auc_score(y_true, en_probs))
    perm_aucs["RandomForest"].append(roc_auc_score(y_true, rf_probs))
    perm_aucs["XGBoost"].append(roc_auc_score(y_true, xgb_probs))

    if (perm_i + 1) % 10 == 0:
        print(f"  perm {perm_i+1}/{N_PERMS}  "
              f"EN={perm_aucs['ElasticNet'][-1]:.3f}  "
              f"RF={perm_aucs['RandomForest'][-1]:.3f}  "
              f"XGB={perm_aucs['XGBoost'][-1]:.3f}", flush=True)

print(f"\n[{time.strftime('%H:%M:%S')}] Permutation test done", flush=True)
perm_summary = []
for mn in ["ElasticNet", "RandomForest", "XGBoost"]:
    pa = np.array(perm_aucs[mn])
    p  = float((pa >= obs_aucs[mn]).sum() + 1) / (N_PERMS + 1)
    perm_summary.append({"model": mn, "observed_auc": obs_aucs[mn],
                          "perm_mean": round(pa.mean(),4), "perm_std": round(pa.std(),4),
                          "p_value": round(p,4), "n_perms": N_PERMS})
    print(f"  {mn:<16}: obs_AUC={obs_aucs[mn]:.4f}  perm_mean={pa.mean():.4f}±{pa.std():.4f}  p={p:.4f}", flush=True)

pd.DataFrame(perm_summary).to_csv(f"{OUT_DIR}/nested_cv_n283_perm_summary.tsv", sep="\t", index=False)
pd.DataFrame(perm_aucs).to_csv(f"{OUT_DIR}/nested_cv_n283_perm_aucs.tsv", sep="\t", index=False)

print(f"\n  Saved: {OUT_DIR}/nested_cv_n283_results.tsv", flush=True)
print(f"  Saved: {OUT_DIR}/nested_cv_n283_summary.tsv", flush=True)
print(f"  Saved: {OUT_DIR}/nested_cv_n283_perm_summary.tsv", flush=True)
print(f"\n[{time.strftime('%H:%M:%S')}] DONE — n=283 nested CV complete", flush=True)
