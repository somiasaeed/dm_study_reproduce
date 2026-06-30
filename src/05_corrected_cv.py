"""
05_corrected_cv.py
==================
STEP 5 - The leakage CORRECTION: leaky (Muchlinski) vs corrected (Kapoor &
Narayanan) cross-validated AUC, for all four models.

Behaviour depends on whether Step 00 located a raw (non-imputed) Sambanis file
(recorded in data/RAW_STATUS.txt):

  * RAW DATA PRESENT -> principled comparison:
        LEAKY     = Muchlinski's procedure: impute the WHOLE dataset once,
                    with the target as a helper, THEN do k-fold CV.
        CORRECTED = Kapoor & Narayanan's fix: within each fold, impute the
                    TRAINING rows only; fill the test rows from the
                    feature relationships learned on train (no target).
    We compare the corrected AUCs to K&N's Table A1 (0.54/0.57/0.68/0.64).

  * NO RAW DATA      -> synthetic-missingness fallback (clearly labelled):
        Re-introduce heavy missingness into the already-imputed data and run
        the corrected per-fold procedure. This estimates how each model
        performs once leakage is removed, but is NOT the exact Table A1
        (which needs K&N's R `mice`/`rfImpute` pipeline on the raw data).

Efficiency: imputations (the expensive part) are computed ONCE and shared
across all four models; only the (cheap) model fits differ per model.

Outputs: results/corrected_auc.csv, figures/fig_corrected_cv.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score

from data import DATA_DIR, load_training_data, MODEL_SPECS, VARS_90
from models import make_logreg, make_random_forest_downsampled

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 666
N_PEACE_SUBSAMPLE = 800    # subsample peace rows so IterativeImputer is fast
N_FOLDS = 5                # 5-fold CV for tractable per-fold imputation
RF_TREES = 100
MAX_ITER = 3

# Kapoor & Narayanan (2022), Table A1 -- the target corrected AUCs.
PAPER_TABLE_A1 = {
    "Fearon & Laitin (2003)": 0.54,
    "Collier & Hoeffler (2004)": 0.57,
    "Hegre & Sambanis (2006)": 0.68,
    "Muchlinski et al. (2016)": 0.64,
}


def raw_status():
    p = os.path.join(DATA_DIR, "RAW_STATUS.txt")
    if not os.path.exists(p):
        return False, None
    with open(p) as f:
        first = f.readline().strip()
    if first == "FOUND":
        path = f.readline().strip()
        return (os.path.exists(path)), path
    return False, None


def subsample(X, y, seed):
    rng = np.random.RandomState(seed)
    war = np.where(y == 1)[0]
    peace = np.where(y == 0)[0]
    keep = np.concatenate([war, rng.choice(peace, size=min(N_PEACE_SUBSAMPLE,
                                                           len(peace)),
                                           replace=False)])
    return X[keep], y[keep]


def impute_joint(M):
    return IterativeImputer(random_state=SEED, max_iter=MAX_ITER,
                            initial_strategy="mean").fit_transform(M)


def make_leaky_data(X, y, inject):
    """Muchlinski's global imputation: impute the whole (subsampled) dataset
    once with the target as a helper -> the leakage. Computed ONCE."""
    X = X.astype(float).copy()
    if inject > 0:
        rng = np.random.RandomState(SEED)
        mask = rng.random(X.shape) < (inject / 100.0)
        X[mask] = np.nan
    Xs, ys = subsample(X, y, SEED)
    return impute_joint(np.column_stack([Xs, ys]))[:, :-1], ys


def make_corrected_folds(X, y, inject_test):
    """Per-fold train-only imputation (the correction), computed ONCE.
    Train rows may use the target (labelled); test features are filled from
    a feature-only imputer (no target). Returns list of (Xtr_i, ytr, Xte_i, yte)."""
    X, y = subsample(X.astype(float), y, SEED)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr, yte = y[tr], y[te]
        if inject_test > 0:
            rng = np.random.RandomState(SEED)
            mask = rng.random(Xte.shape) < (inject_test / 100.0)
            Xte[mask] = np.nan
        Xtr_i = impute_joint(np.column_stack([Xtr, ytr]))[:, :-1]
        imp_feat = IterativeImputer(random_state=SEED, max_iter=MAX_ITER,
                                    initial_strategy="mean").fit(Xtr_i)
        folds.append((Xtr_i, ytr, imp_feat.transform(Xte), yte))
    return folds


def cv_auc(X, y, feat_idx, factory):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in cv.split(X, y):
        m = factory()
        m.fit(X[tr][:, feat_idx], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te][:, feat_idx])[:, 1]))
    return float(np.mean(aucs))


def folds_auc(folds, feat_idx, factory):
    aucs = []
    for Xtr_i, ytr, Xte_i, yte in folds:
        m = factory()
        m.fit(Xtr_i[:, feat_idx], ytr)
        aucs.append(roc_auc_score(yte, m.predict_proba(Xte_i[:, feat_idx])[:, 1]))
    return float(np.mean(aucs))


def feature_indices(all_feats, wanted):
    return [i for i, f in enumerate(all_feats) if f in wanted]


def run_models(X, y, all_feats, inject):
    f_logit = make_logreg
    f_rf = lambda: make_random_forest_downsampled(n_estimators=RF_TREES)

    print("  imputing leaky (global) ...", flush=True)
    X_leaky, ys_leaky = make_leaky_data(X, y, inject)
    print("  imputing corrected (per-fold) ...", flush=True)
    folds = make_corrected_folds(X, y, inject)

    rows = []
    for name, features in MODEL_SPECS.items():
        idx = feature_indices(all_feats, features)
        leaky = cv_auc(X_leaky, ys_leaky, idx, f_logit)
        corr = folds_auc(folds, idx, f_logit)
        rows.append({"Model": name, "Type": "Logistic Regression",
                     "Leaky AUC": round(leaky, 3),
                     "Corrected AUC": round(corr, 3),
                     "K&N Table A1": PAPER_TABLE_A1[name]})
        print(f"  {name:28s}  leaky {leaky:.3f} -> corrected {corr:.3f} "
              f"(K&N {PAPER_TABLE_A1[name]:.2f})", flush=True)

    idx_all = list(range(len(all_feats)))
    leaky = cv_auc(X_leaky, ys_leaky, idx_all, f_rf)
    corr = folds_auc(folds, idx_all, f_rf)
    rows.append({"Model": "Muchlinski et al. (2016)", "Type": "Random Forest",
                 "Leaky AUC": round(leaky, 3),
                 "Corrected AUC": round(corr, 3),
                 "K&N Table A1": PAPER_TABLE_A1["Muchlinski et al. (2016)"]})
    print(f"  {'Muchlinski et al. (2016)':28s}  leaky {leaky:.3f} -> "
          f"corrected {corr:.3f} (K&N {PAPER_TABLE_A1['Muchlinski et al. (2016)']:.2f})",
          flush=True)
    return rows


def run_synthetic_models(X, y, all_feats):
    """Synthetic-missingness fallback (NO raw data available).

    Reconstructs Muchlinski's mistake the same way Step 3 does: hide a fraction
    of the TEST-set features, then compare joint (leaky) vs train-only
    (corrected) imputation -- the ONLY train/test imputation asymmetry. Both
    pipelines train on the clean training data; only the test imputation
    differs, so any AUC gap is attributable to leakage. Done once per model.
    """
    HIDE = 90.0
    X, y = subsample(X.astype(float), y, SEED)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    tr, te = next(sss.split(X, y))
    Xtr, Xte = X[tr].copy(), X[te].copy()
    ytr, yte = y[tr], y[te]
    rng = np.random.RandomState(SEED)
    mask = rng.random(Xte.shape) < (HIDE / 100.0)
    Xte_hid = Xte.astype(float); Xte_hid[mask] = np.nan

    # LEAKY: impute train + hidden-test TOGETHER with the target as helper.
    M = np.vstack([np.column_stack([Xtr, ytr]),
                   np.column_stack([Xte_hid, yte])])
    Mi = impute_joint(M)
    n_tr = len(tr)
    XtrL, XteL = Mi[:n_tr, :-1], Mi[n_tr:, :-1]
    # CORRECTED: train imputed with target (labelled rows); test features from
    # a feature-only imputer learned on the (imputed) training data.
    XtrC = impute_joint(np.column_stack([Xtr, ytr]))[:, :-1]
    imp_feat = IterativeImputer(random_state=SEED, max_iter=MAX_ITER,
                                initial_strategy="mean").fit(XtrC)
    XteC = imp_feat.transform(Xte_hid)

    f_logit = make_logreg
    f_rf = lambda: make_random_forest_downsampled(n_estimators=RF_TREES)

    def eval_pair(feat_idx, factory):
        ml = factory(); ml.fit(XtrL[:, feat_idx], ytr)
        leaky = roc_auc_score(yte, ml.predict_proba(XteL[:, feat_idx])[:, 1])
        mc = factory(); mc.fit(XtrC[:, feat_idx], ytr)
        corr = roc_auc_score(yte, mc.predict_proba(XteC[:, feat_idx])[:, 1])
        return leaky, corr

    rows = []
    for name, features in MODEL_SPECS.items():
        idx = feature_indices(all_feats, features)
        leaky, corr = eval_pair(idx, f_logit)
        rows.append({"Model": name, "Type": "Logistic Regression",
                     "Leaky AUC": round(leaky, 3),
                     "Corrected AUC": round(corr, 3),
                     "K&N Table A1": PAPER_TABLE_A1[name]})
        print(f"  {name:28s}  leaky {leaky:.3f} -> corrected {corr:.3f} "
              f"(K&N {PAPER_TABLE_A1[name]:.2f})", flush=True)

    idx_all = list(range(len(all_feats)))
    leaky, corr = eval_pair(idx_all, f_rf)
    rows.append({"Model": "Muchlinski et al. (2016)", "Type": "Random Forest",
                 "Leaky AUC": round(leaky, 3),
                 "Corrected AUC": round(corr, 3),
                 "K&N Table A1": PAPER_TABLE_A1["Muchlinski et al. (2016)"]})
    print(f"  {'Muchlinski et al. (2016)':28s}  leaky {leaky:.3f} -> "
          f"corrected {corr:.3f} (K&N {PAPER_TABLE_A1['Muchlinski et al. (2016)']:.2f})",
          flush=True)
    return rows


def plot(rows, mode):
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    w = 0.28
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w, df["Leaky AUC"], w, label="Leaky (Muchlinski's method)",
           color="#d62728", edgecolor="black", linewidth=0.4)
    ax.bar(x, df["Corrected AUC"], w, label="Corrected (this study)",
           color="#2ca02c", edgecolor="black", linewidth=0.4)
    ax.bar(x + w, df["K&N Table A1"], w, label="K&N Table A1 (target)",
           color="#7f7f7f", edgecolor="black", linewidth=0.4)
    for i, r in df.iterrows():
        ax.text(i - w, r["Leaky AUC"] + 0.01, f'{r["Leaky AUC"]:.2f}', ha="center", fontsize=8)
        ax.text(i, r["Corrected AUC"] + 0.01, f'{r["Corrected AUC"]:.2f}', ha="center", fontsize=8)
        ax.text(i + w, r["K&N Table A1"] + 0.01, f'{r["K&N Table A1"]:.2f}', ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (20", "\n(20") for n in df["Model"]], fontsize=8)
    ax.set_ylabel("Cross-validated AUC")
    ax.set_ylim(0.4, 1.0)
    title = "Leakage correction: leaky vs corrected CV AUC"
    title += ("\n(raw Sambanis data; principled per-fold correction)"
              if mode == "raw" else
              "\n(synthetic-missingness fallback -- NOT the exact Table A1)")
    ax.set_title(title)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(FIG_DIR, "fig_corrected_cv.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    return out


def main():
    found, path = raw_status()
    print("=" * 64)
    print("STEP 5: Leakage correction -- leaky vs corrected CV AUC")
    print("=" * 64)

    if found:
        print(f"Raw data found -> {path}  (principled correction)\n")
        raw = pd.read_csv(path)
        use_cols = [c for c in VARS_90 if c in raw.columns]
        raw = raw[use_cols].dropna(subset=["warstds"])
        raw["warstds"] = raw["warstds"].astype(int)
        y = raw["warstds"].values
        X = raw.drop(columns=["warstds"]).values.astype(float)
        all_feats = [c for c in use_cols if c != "warstds"]
        rows = run_models(X, y, all_feats, inject=0.0)
        mode = "raw"
    else:
        print("No raw data -- using synthetic-missingness fallback.\n")
        df, y = load_training_data()
        all_feats = [c for c in df.columns if c != "warstds"]
        X = df[all_feats].values.astype(float)
        rows = run_synthetic_models(X, y, all_feats)
        mode = "synthetic"

    res = pd.DataFrame(rows)
    res.insert(0, "method", mode)
    res.to_csv(os.path.join(RESULTS_DIR, "corrected_auc.csv"), index=False)
    print(f"\nSaved -> results/corrected_auc.csv")
    out = plot(rows, mode)
    print(f"Figure saved -> {out}")
    if mode == "synthetic":
        print("\nNOTE: these corrected AUCs are SYNTHETIC estimates (no raw")
        print("data). The exact K&N Table A1 needs the raw non-imputed Sambanis")
        print("data + their R imputation pipeline. The qualitative result")
        print("(corrected << leaky) is what matters.")


if __name__ == "__main__":
    main()
