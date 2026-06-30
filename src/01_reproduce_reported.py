"""
01_reproduce_reported.py
========================
STEP 1 of the replication.

Reproduces the (leaky) results reported by Muchlinski et al. (2016) on the
rfImpute-imputed data `SambnisImp.csv`. We train the four models and report:

    (a) 10-fold cross-validated AUC   (what caret's train() prints in R)
    (b) in-sample AUC                 (the ROC plots in the paper, lines 200-222)

For each logistic specification the paper trains TWO estimators -- the ordinary
`glm` logit AND Firth's penalized (`plr`) logit -- so we report both. For the
Random Forest we report BOTH the class_weight approximation and the explicit
sampsize=c(30,90) per-tree down-sampling (the most faithful port).

Because the training data was imputed with train+test together (the leakage),
these numbers are the INFLATED ones. They should land near the paper's reported
values (Random Forest AUC ~ 0.90-0.95).

Output: writes results/reported_auc.csv (one row per model/method) and prints a table.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from data import load_training_data, MODEL_SPECS
from models import (make_logreg, make_firth_logit,
                    make_random_forest, make_random_forest_downsampled)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 666


def cv_auc(X, y, model, n_splits=10):
    """Stratified 10-fold CV AUC (mirrors caret twoClassSummary)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean(), scores.std()


def insample_auc(X, y, model):
    """AUC on the training data itself (the paper's ROC-curve AUCs)."""
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, p)


def main():
    df, y = load_training_data()
    print("=" * 70)
    print("STEP 1: Reproducing Muchlinski et al. (2016) reported AUCs")
    print("(trained on the rfImpute-imputed, LEAKED training data)")
    print("=" * 70)
    print(f"Rows: {len(y)}   |   War rate: {y.mean()*100:.2f}%\n")

    rows = []

    # --- The three logistic-regression sub-models ----------------------
    # Each run twice: ordinary glm + Firth's penalized (plr) logit.
    for name, features in MODEL_SPECS.items():
        feats = [f for f in features if f in df.columns]
        X = df[feats].values
        for method, factory in [("glm", make_logreg), ("plr (Firth)", make_firth_logit)]:
            cv_mean, cv_std = cv_auc(X, y, factory())
            in_auc = insample_auc(X, y, factory())
            rows.append({
                "Model": name,
                "Method": method,
                "Type": "Logistic Regression",
                "CV AUC (mean)": round(cv_mean, 3),
                "CV AUC (std)": round(cv_std, 3),
                "In-sample AUC": round(in_auc, 3),
            })
            print(f"  {name:26s} [{method:11s}]  CV AUC = {cv_mean:.3f} "
                  f"(+/-{cv_std:.3f})  in-sample = {in_auc:.3f}")

    # --- Muchlinski's Random Forest (all ~90 variables) ----------------
    X_all = df.drop(columns=["warstds"]).values
    for method, factory in [
        ("rf (class_weight)", make_random_forest),
        ("rf (sampsize=30,90)", make_random_forest_downsampled),
    ]:
        cv_mean, cv_std = cv_auc(X_all, y, factory(n_estimators=300))
        in_auc = insample_auc(X_all, y, factory(n_estimators=300))
        rows.append({
            "Model": "Muchlinski et al. (2016)",
            "Method": method,
            "Type": "Random Forest",
            "CV AUC (mean)": round(cv_mean, 3),
            "CV AUC (std)": round(cv_std, 3),
            "In-sample AUC": round(in_auc, 3),
        })
        print(f"  {'Muchlinski et al. (2016)':26s} [{method:19s}]  CV AUC = {cv_mean:.3f} "
              f"(+/-{cv_std:.3f})  in-sample = {in_auc:.3f}")

    res = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "reported_auc.csv")
    res.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print("\nThese AUCs are INFLATED because the data was imputed with train")
    print("and test together. The Random Forest should look much stronger than")
    print("Logistic Regression here -- the gap will shrink once we fix leakage.")
    print("\nNote: 'rf (sampsize=30,90)' is the most faithful port of the R code's")
    print("sampsize=c(30,90); 'rf (class_weight)' is the weighted approximation.")


if __name__ == "__main__":
    main()
