"""
09_leakage_importance.py
========================
STEP 9 - A NOVEL analysis: HOW does data leakage distort the model?

Steps 2-3 show that joint train+test imputation inflates out-of-sample AUC.
Here we go one step further and show it also corrupts the model's
*interpretation*: under leakage, the features whose values were reconstructed
from the target get artificially inflated importance. A policymaker reading
Muchlinski's variable-importance plot would therefore draw the wrong
conclusions about what "causes" civil war.

Procedure (mirrors the real Muchlinski situation, where the training data was
also imputed jointly with the test data):

    1. Subsample the civil-war data (all wars + 1500 peace rows).
    2. Split train/test (70/30).
    3. Hide 90% of feature values in BOTH train and test (the heavily-missing
       regime that triggered the original imputation).
    4. LEAKY      -> impute train+test TOGETHER with the target as a helper.
                     The training features are then partly reconstructed from
                     their own labels.
    5. CORRECTED  -> impute using TRAIN data only; test filled from the
                     learned feature relationships (no target).
    6. Train a down-sampled RF on each, record feature importances, and
       compare rankings.

Outputs: figures/fig_importance_leaky_vs_corrected.png,
         results/importance_comparison.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score

from data import load_training_data
from models import make_random_forest_downsampled

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 666
HIDE_PCT = 90            # % of feature values hidden in train AND test
N_PEACE_SUBSAMPLE = 1500


def subsample(df, y, seed):
    rng = np.random.RandomState(seed)
    war_idx = np.where(y == 1)[0]
    peace_idx = np.where(y == 0)[0]
    keep_peace = rng.choice(peace_idx, size=min(N_PEACE_SUBSAMPLE, len(peace_idx)),
                            replace=False)
    keep = np.concatenate([war_idx, keep_peace])
    return df.iloc[keep].reset_index(drop=True), y[keep]


def leaky_impute(Xtr, Xte, ytr, yte):
    """Impute train+test TOGETHER with the target as a helper column
    (Muchlinski's error)."""
    Mtr = np.column_stack([Xtr, ytr])
    Mte = np.column_stack([Xte, yte])
    Mall = np.vstack([Mtr, Mte])
    imp = IterativeImputer(random_state=SEED, max_iter=5,
                           initial_strategy="mean")
    M = imp.fit_transform(Mall)
    n_tr = Mtr.shape[0]
    return M[:n_tr, :-1], M[n_tr:, :-1]


def corrected_impute(Xtr, Xte, ytr):
    """Impute using TRAIN only; the target is NOT used to fill the test set."""
    Mtr = np.column_stack([Xtr, ytr])
    imp = IterativeImputer(random_state=SEED, max_iter=5,
                           initial_strategy="mean")
    imp.fit(Mtr)
    Xtr_filled = imp.transform(Mtr)[:, :-1]
    imp_feat = IterativeImputer(random_state=SEED, max_iter=5,
                                initial_strategy="mean")
    imp_feat.fit(Xtr)
    Xte_filled = imp_feat.transform(Xte)
    return Xtr_filled, Xte_filled


def train_rf_importance(Xtr, Xte, ytr, yte, feats):
    rf = make_random_forest_downsampled(n_estimators=300)
    rf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    imp = np.mean([t.feature_importances_ for t in rf.trees_], axis=0)
    return auc, imp


def main():
    df, y = load_training_data()
    df, y = subsample(df, y, SEED)
    feats = [c for c in df.columns if c != "warstds"]
    X = df[feats].values.astype(float)
    yv = y.astype(float)
    print("=" * 64)
    print("STEP 9: How data leakage distorts feature importance")
    print("=" * 64)
    print(f"Subsampled data: {len(yv)} rows ({int((yv==1).sum())} war)\n")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    tr_idx, te_idx = next(sss.split(X, yv))
    Xtr0, Xte0 = X[tr_idx].copy(), X[te_idx].copy()
    ytr, yte = yv[tr_idx], yv[te_idx]

    # hide HIDE_PCT% of feature values in BOTH train and test
    rng = np.random.RandomState(SEED)
    for Xpart in (Xtr0, Xte0):
        mask = rng.random(Xpart.shape) < (HIDE_PCT / 100.0)
        Xpart[mask] = np.nan

    # LEAKY
    XtrL, XteL = leaky_impute(Xtr0, Xte0, ytr, yte)
    aucL, impL = train_rf_importance(XtrL, XteL, ytr, yte, feats)
    # CORRECTED
    XtrC, XteC = corrected_impute(Xtr0, Xte0, ytr)
    aucC, impC = train_rf_importance(XtrC, XteC, ytr, yte, feats)

    print(f"  Leaky      AUC = {aucL:.3f}")
    print(f"  Corrected  AUC = {aucC:.3f}   (gap {aucL-aucC:+.3f})")

    # rank by leaky importance
    order = np.argsort(impL)[::-1]
    res = pd.DataFrame({
        "feature": [feats[i] for i in order],
        "leaky_importance": impL[order],
        "corrected_importance": impC[order],
        "inflation_ratio": impL[order] / np.where(impC[order] == 0,
                                                  np.nan, impC[order]),
    })
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    res.to_csv(os.path.join(RESULTS_DIR, "importance_comparison.csv"),
               index=False)

    # rank correlation between the two rankings
    from scipy.stats import spearmanr
    rho, _ = spearmanr(impL, impC)
    print(f"\n  Spearman rank correlation of importances: {rho:.3f}  "
          f"(low = leakage re-orders which features 'matter')")

    # --- figure: scatter leaky vs corrected importance -----------------
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(impC, impL, s=20, color="#4c72b0", alpha=0.7)
    lim = max(impL.max(), impC.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--",
            label="y = x (importance unchanged)")
    # label the 8 features most inflated by leakage
    ratio = impL / np.where(impC == 0, np.nan, impC)
    top_inflate = np.argsort(np.nan_to_num(ratio, nan=-1))[::-1][:8]
    for i in top_inflate:
        ax.annotate(feats[i], (impC[i], impL[i]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Feature importance (CORRECTED imputation)")
    ax.set_ylabel("Feature importance (LEAKY imputation)")
    ax.set_title("Leakage inflates the apparent importance of\n"
                 "label-reconstructed features (above the dashed line)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    out = os.path.join(FIG_DIR, "fig_importance_leaky_vs_corrected.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nFigure saved -> {out}")
    print("Saved table  -> results/importance_comparison.csv")
    print("\nInterpretation: points well ABOVE the y=x line are features whose")
    print("importance is inflated by leakage -- they look predictive only")
    print("because their missing values were rebuilt from the war/peace label.")


if __name__ == "__main__":
    main()
