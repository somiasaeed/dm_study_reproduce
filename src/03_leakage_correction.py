"""
03_leakage_correction.py
========================
STEP 3 - Show the leakage mechanism on the REAL civil-war data.

`SambnisImp.csv` is already fully imputed (no missing values), so we cannot
"un-impute" it. Instead we reconstruct Muchlinski's exact mistake on the real
data:

    1. Take the real 90-variable civil-war data and split it (train/test).
    2. Hide a fraction of feature values in the TEST set (mimicking the fact
       that Muchlinski's out-of-sample test set was ~95% missing).
    3. LEAKY pipeline     -> impute train + test TOGETHER, using the target
       `warstds` as a helper column (Muchlinski's error). Then train a Random
       Forest and report test AUC.
    4. CORRECTED pipeline -> impute using ONLY the training data (the target
       is NOT used to impute the test set). Train RF, report test AUC.

Expected result: the LEAKY AUC is clearly higher than the CORRECTED AUC,
especially as more test data is missing -- the same pattern as the paper's
Muchlinski correction (RF AUC 0.95 -> 0.64).

Output: results/real_data_leakage.csv, figures/fig_real_data_leakage.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score

from data import load_training_data

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 666
MISSING_LEVELS = [0, 30, 60, 90]   # % of test feature values hidden
N_PEACE_SUBSAMPLE = 1500           # subsample peace rows for speed (keep all wars)


def subsample(df, y, seed):
    """Keep ALL war rows + a subsample of peace rows (so IterativeImputer is
    fast but both classes are well represented). Returns (df, y) subsampled."""
    rng = np.random.RandomState(seed)
    war_idx = np.where(y == 1)[0]
    peace_idx = np.where(y == 0)[0]
    peace_keep = rng.choice(peace_idx, size=min(N_PEACE_SUBSAMPLE, len(peace_idx)),
                            replace=False)
    keep = np.concatenate([war_idx, peace_keep])
    return df.iloc[keep].reset_index(drop=True), y[keep]


def hide_test_features(X_test, pct, rng):
    """Set `pct`% of feature cells in the test matrix to NaN."""
    if pct == 0:
        return X_test.copy()
    mask = rng.random(X_test.shape) < (pct / 100.0)
    X = X_test.copy().astype(float)
    X[mask] = np.nan
    return X


def leaky_impute(Xtr, Xte, ytr, yte, rng):
    """Muchlinski's error: impute train+test TOGETHER, with the target as a
    helper column. The test set's own `warstds` is used to fill its features."""
    # build [features, warstds] for train and test, then stack
    Mtr = np.column_stack([Xtr, ytr])
    Mte = np.column_stack([Xte, yte])
    Mall = np.vstack([Mtr, Mte])
    imp = IterativeImputer(random_state=SEED, max_iter=5, initial_strategy="mean")
    Mfilled = imp.fit_transform(Mall)        # fit on EVERYTHING
    n_tr = Mtr.shape[0]
    return Mfilled[:n_tr, :-1], Mfilled[n_tr:, :-1]


def corrected_impute(Xtr, Xte, ytr):
    """No leakage: imputer is fit on the TRAINING data only, and the target
    is NOT used to impute the test set."""
    # train: use target as helper (allowed, these are labelled rows)
    Mtr = np.column_stack([Xtr, ytr])
    imp = IterativeImputer(random_state=SEED, max_iter=5, initial_strategy="mean")
    imp.fit(Mtr)
    Xtr_filled = imp.transform(Mtr)[:, :-1]
    # test: impute features WITHOUT the target (drop target column entirely)
    imp_feat = IterativeImputer(random_state=SEED, max_iter=5, initial_strategy="mean")
    imp_feat.fit(Xtr)                        # learn feature relationships on train
    Xte_filled = imp_feat.transform(Xte)     # apply to test, no target involved
    return Xtr_filled, Xte_filled


def rf_auc(Xtr, Xte, ytr, yte):
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                max_features="sqrt", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p)


def main():
    df, y = load_training_data()
    df, y = subsample(df, y, SEED)
    print("=" * 64)
    print("STEP 3: Real-data leakage on the civil-war dataset")
    print("=" * 64)
    print(f"Subsampled data: {len(y)} rows  ({(y==1).sum()} wars, "
          f"{(y==0).sum()} peace)\n")

    feats = [c for c in df.columns if c != "warstds"]
    X = df[feats].values.astype(float)
    yv = y.astype(float)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    tr_idx, te_idx = next(sss.split(X, yv))
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = yv[tr_idx], yv[te_idx]
    print(f"Train: {len(tr_idx)} ({int(ytr.sum())} war) | "
          f"Test: {len(te_idx)} ({int(yte.sum())} war)\n")

    rng = np.random.RandomState(SEED)
    rows = []
    for pct in MISSING_LEVELS:
        Xte_hid = hide_test_features(Xte, pct, rng)
        # leaky
        XtrL, XteL = leaky_impute(Xtr, Xte_hid, ytr, yte, rng)
        auc_leaky = rf_auc(XtrL, XteL, ytr, yte)
        # corrected
        XtrC, XteC = corrected_impute(Xtr, Xte_hid, ytr)
        auc_corr = rf_auc(XtrC, XteC, ytr, yte)

        rows.append({"test_missing_pct": pct,
                     "leaky_auc": round(auc_leaky, 3),
                     "corrected_auc": round(auc_corr, 3)})
        print(f"  test missing {pct:2d}% -> LEAKY AUC {auc_leaky:.3f} | "
              f"CORRECTED AUC {auc_corr:.3f} | "
              f"gap {auc_leaky-auc_corr:+.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_DIR, "real_data_leakage.csv"), index=False)

    # ---- plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res["test_missing_pct"], res["leaky_auc"], "o-",
            color="#d62728", linewidth=2, label="Leaky (Muchlinski's method)")
    ax.plot(res["test_missing_pct"], res["corrected_auc"], "s-",
            color="#2ca02c", linewidth=2, label="Corrected (train-only imputation)")
    ax.set_xlabel("Proportion of test feature values hidden (%)")
    ax.set_ylabel("Random-Forest test AUC")
    ax.set_title("Leakage on the real civil-war data\n"
                 "(joint train+test imputation inflates out-of-sample AUC)")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(FIG_DIR, "fig_real_data_leakage.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nFigure saved -> {out}")
    print(f"Saved table  -> results/real_data_leakage.csv")


if __name__ == "__main__":
    main()
