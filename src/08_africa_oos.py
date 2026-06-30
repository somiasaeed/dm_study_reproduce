"""
08_africa_oos.py
================
STEP 8 - A genuine OUT-OF-SAMPLE test on the Africa data (2001-2014).

The paper's own Africa "analysis" (R lines 291-359) is incoherent: it trains
models on the Sambanis data, then compares random *training-set* predictions
against Africa labels (sampling 737 of the 7140 training predictions to make
the lengths match -- R lines 315-319). That is not an out-of-sample test.

We instead do the test the paper *should* have done: train on the Sambanis
data, predict on the Africa rows, score against the Africa labels.

The obstacle is that `AfricaImp.csv` uses *QOG* variable codes, not the
Sambanis names. We map the ~8 QOG features to their closest Sambanis
equivalents (with caveats -- different sources/units), train a model on the
Sambanis data restricted to that mapped subset, and evaluate on Africa.

    QOG feature           -> Sambanis feature (transform)
    -------------------------------------------------------------
    gle_rgdpc (GDP/cap)   -> ln_gdpen   (log)
    imf_gdpgr (GDP growth)-> gdpgrowth  (identity)
    p_polity  (Polity IV) -> pol4       (identity)
    al_ethnic (eth. frac) -> ef         (identity, Alesina vs Fearon-Laitin)
    al_religion(rel frac) -> relfrac    (identity)
    ross_oil_netexpc(oil) -> sxpnew     (identity, oil net exports)
    une_gerst (sec school)-> seceduc    (identity, enrollment)
    eu_nama_aux_pem_POP   -> lpopns     (log, population)

This is a *best-effort* mapping: only ~8 features, several are approximate
proxies. It cannot reproduce the paper's (flawed) reported Africa AUCs
(RF 0.60, FL 0.43, CH 0.55, HS 0.40) exactly, but it is a real hold-out test.

Outputs: results/africa_oos.csv, figures/fig_africa_oos.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             roc_curve)

from data import load_training_data, load_africa_test

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 666
Z_CLIP = 5.0   # clip Africa z-scores to [-5,5] (robustness vs unit mismatch)

# (sambanis_name, qog_name, transform). The transform is applied to the QOG
# (Africa) value ONLY -- to bring levels into the log-units the Sambanis
# variables already use. The Sambanis values are used as-is.
MAPPING = [
    ("ln_gdpen", "gle_rgdpc", np.log),          # GDP/cap (levels -> log)
    ("gdpgrowth", "imf_gdpgr", lambda v: v),    # GDP growth %
    ("pol4", "p_polity", lambda v: v),          # Polity IV score
    ("ef", "al_ethnic", lambda v: v),           # ethnic fractionalization
    ("relfrac", "al_religion", lambda v: v),    # religious fractionalization
    ("sxpnew", "ross_oil_netexpc", lambda v: v),# primary-commodity/oil exports
    ("seceduc", "une_gerst", lambda v: v),      # secondary-school enrollment
    ("lpopns", "eu_nama_aux_pem_POP", np.log),  # population (levels -> log)
]


def mapped_matrix(df, source):
    """Build an (n, k) matrix of mapped features.

    source='sambanis' -> use the Sambanis columns directly (already in the
                         right units, e.g. ln_gdpen is already a log).
    source='qog'      -> apply the per-variable transform to the QOG columns
                         so they match the Sambanis units (e.g. take log of
                         raw GDP/capita and population).
    """
    cols = []
    names = []
    for samb, qog, tf in MAPPING:
        col = df[samb] if source == "sambanis" else df[qog]
        val = col.astype(float) if source == "sambanis" else tf(col.astype(float))
        cols.append(np.asarray(val, dtype=float))
        names.append(samb)
    return np.column_stack(cols), names


def main():
    train_df, y_train = load_training_data()
    africa = load_africa_test()
    y_africa = africa["warstds"].astype(int).values
    print("=" * 64)
    print("STEP 8: Out-of-sample test on Africa (2001-2014)")
    print("=" * 64)
    print(f"Train (Sambanis): {len(y_train)} rows, {(y_train==1).sum()} wars")
    print(f"Africa test:      {len(y_africa)} rows, {(y_africa==1).sum()} wars\n")

    Xtr, names = mapped_matrix(train_df, "sambanis")
    Xte, _ = mapped_matrix(africa, "qog")

    # The QOG and Sambanis variables sit on different raw scales, so we
    # standardise with the TRAINING distribution (fit-on-train) and apply to
    # both sides -- the standard ML practice for cross-distribution scoring.
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    # clip extreme Africa z-scores that arise from genuine unit mismatches
    Xte = np.clip(Xte, -Z_CLIP, Z_CLIP)

    # drop Africa rows whose mapped features are non-finite
    finite = np.isfinite(Xte).all(axis=1)
    n_dropped = (~finite).sum()
    Xte, y_africa_f = Xte[finite], y_africa[finite]
    print(f"Mapped features ({len(names)}): {names}")
    if n_dropped:
        print(f"Dropped {n_dropped} Africa rows with non-finite mapped values.")
    print(f"Usable Africa test rows: {len(y_africa_f)} "
          f"({(y_africa_f==1).sum()} wars)\n")

    rows = []
    probs = {}
    # Reduced logistic regression on the mapped subset (balanced for imbalance)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced",
                            random_state=SEED)
    lr.fit(Xtr, y_train)
    p_lr = lr.predict_proba(Xte)[:, 1]
    probs["Logistic Reg. (mapped)"] = p_lr

    # Small Random Forest on the same mapped subset
    rf = RandomForestClassifier(n_estimators=1000, class_weight="balanced_subsample",
                                max_features="sqrt", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, y_train)
    p_rf = rf.predict_proba(Xte)[:, 1]
    probs["Random Forest (mapped)"] = p_rf

    for name, p in probs.items():
        pred = (p >= 0.5).astype(int)
        auc = roc_auc_score(y_africa_f, p)
        prec = precision_score(y_africa_f, pred, zero_division=0)
        rec = recall_score(y_africa_f, pred, zero_division=0)
        rows.append({"model": name, "auc": round(auc, 3),
                     "precision@0.5": round(prec, 3),
                     "recall@0.5": round(rec, 3),
                     "n_test": len(y_africa_f), "n_wars": int((y_africa_f==1).sum())})
        print(f"  {name:26s}  AUC = {auc:.3f}  "
              f"precision = {prec:.3f}  recall = {rec:.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "africa_oos.csv"),
                              index=False)

    # --- ROC figure -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, p), c in zip(probs.items(), ["#1f77b4", "#d62728"]):
        fpr, tpr, _ = roc_curve(y_africa_f, p)
        a = roc_auc_score(y_africa_f, p)
        ax.plot(fpr, tpr, color=c, linewidth=2, label=f"{name} ({a:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-sample ROC on Africa (2001-2014)\n"
                 "proper hold-out test (paper's own procedure was incoherent)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    out = os.path.join(FIG_DIR, "fig_africa_oos.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nFigure saved -> {out}")
    print("Saved table  -> results/africa_oos.csv")


if __name__ == "__main__":
    main()
