import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings, os, time
warnings.filterwarnings("ignore")

os.makedirs("results/ml/tumor", exist_ok=True)

df = pd.read_csv("results/ml/tumor/combined_tumor_features.tsv", sep="\t")
df = df.dropna(subset=["response", "TMB"])

# Feature cols: TMB + gene flags
feature_cols = ["TMB", "n_mutations"] + [c for c in df.columns if c.startswith("mut_")]
X = df[feature_cols].fillna(0).values.astype(float)
y = (df["response"] == "R").astype(int).values
studies = df["study"].values

print(f"Dataset: {len(df)} patients, {X.shape[1]} features")
print(f"R={y.sum()}, NR={(1-y).sum()}")
print(f"Studies: {dict(pd.Series(studies).value_counts())}")

sw = (1 - y).sum() / y.sum()  # scale_pos_weight for XGBoost

models = {
    "ElasticNet": LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=0.5, C=0.1,
        class_weight="balanced", max_iter=2000),
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        scale_pos_weight=sw,
        random_state=42, eval_metric="auc",
        verbosity=0),
}

# ── 1. LOOCV on full combined dataset ──────────────────────────────────────
print("\n--- LOOCV (combined n={}) ---".format(len(y)))
loocv_results = {}
for name, model in models.items():
    t0 = time.time()
    preds = np.zeros(len(y))
    for i in range(len(y)):
        idx_train = np.arange(len(y)) != i
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[idx_train])
        X_te = scaler.transform(X[[i]])
        model.fit(X_tr, y[idx_train])
        preds[i] = model.predict_proba(X_te)[0][1]
    auc = roc_auc_score(y, preds)
    loocv_results[name] = auc
    print(f"  {name}: AUC={auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ── 2. Leave-one-study-out cross-validation ────────────────────────────────
print("\n--- Leave-One-Study-Out ---")
study_list = sorted(df["study"].unique())
loso_rows = []
for name, model in models.items():
    study_aucs = []
    for held_out in study_list:
        train_mask = studies != held_out
        test_mask  = studies == held_out
        if test_mask.sum() < 5 or len(np.unique(y[test_mask])) < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])
        model.fit(X_tr, y[train_mask])
        preds = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y[test_mask], preds)
        study_aucs.append(auc)
        loso_rows.append({"model": name, "held_out": held_out,
                          "n_test": int(test_mask.sum()), "AUC": auc})
        print(f"  {name} | hold-out {held_out}: AUC={auc:.4f} "
              f"(n={test_mask.sum()}, R={y[test_mask].sum()})", flush=True)
    mean_auc = float(np.mean(study_aucs)) if study_aucs else float("nan")
    print(f"  {name} mean cross-study AUC: {mean_auc:.4f}", flush=True)

pd.DataFrame(loso_rows).to_csv(
    "results/ml/tumor/tumor_loso_results.tsv", sep="\t", index=False)

# ── 3. Permutation test — XGBoost LOOCV, N=200 ───────────────────────────
print("\n--- Permutation test (XGBoost LOOCV, N=200) ---", flush=True)
t_perm_start = time.time()

xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    scale_pos_weight=sw, random_state=42, eval_metric="auc", verbosity=0)

# observed LOOCV AUC (already computed above, reuse if XGBoost was run)
obs_preds = np.zeros(len(y))
for i in range(len(y)):
    idx_train = np.arange(len(y)) != i
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[idx_train])
    X_te = scaler.transform(X[[i]])
    xgb.fit(X_tr, y[idx_train])
    obs_preds[i] = xgb.predict_proba(X_te)[0][1]
obs_auc = roc_auc_score(y, obs_preds)
print(f"  Observed XGBoost LOOCV AUC: {obs_auc:.4f}", flush=True)

null_aucs = []
rng = np.random.RandomState(42)
N_PERM = 200
for perm in range(N_PERM):
    y_perm = rng.permutation(y)
    perm_preds = np.zeros(len(y))
    for i in range(len(y)):
        idx_train = np.arange(len(y)) != i
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[idx_train])
        X_te = scaler.transform(X[[i]])
        xgb.fit(X_tr, y_perm[idx_train])
        perm_preds[i] = xgb.predict_proba(X_te)[0][1]
    null_aucs.append(roc_auc_score(y_perm, perm_preds))
    if (perm + 1) % 20 == 0:
        elapsed = time.time() - t_perm_start
        eta = elapsed / (perm + 1) * (N_PERM - perm - 1)
        p_so_far = np.mean(np.array(null_aucs) >= obs_auc)
        print(f"  {perm+1}/{N_PERM} perms done  "
              f"null_mean={np.mean(null_aucs):.4f}  "
              f"p_so_far={p_so_far:.3f}  ETA={eta/60:.1f}min", flush=True)

p_val = np.mean(np.array(null_aucs) >= obs_auc)
print(f"\nObserved AUC:       {obs_auc:.4f}")
print(f"Null mean ± std:    {np.mean(null_aucs):.4f} ± {np.std(null_aucs):.4f}")
print(f"Permutation p-value: {p_val:.4f}  (N={N_PERM})")
print(f"Total permutation time: {(time.time()-t_perm_start)/60:.1f} min")

# ── save results ──────────────────────────────────────────────────────────
pd.DataFrame({
    "model": ["XGBoost"], "eval": ["LOOCV"],
    "observed_auc": [obs_auc],
    "null_mean": [np.mean(null_aucs)], "null_std": [np.std(null_aucs)],
    "p_value": [p_val], "N_perm": [N_PERM],
}).to_csv("results/ml/tumor/tumor_permutation_results.tsv", sep="\t", index=False)

pd.DataFrame({"null_auc": null_aucs}).to_csv(
    "results/ml/tumor/tumor_null_aucs.tsv", sep="\t", index=False)

print("\nDONE - results saved to results/ml/tumor/")
