"""
07_diagnostics.py
=================
STEP 7 - Reproduce the diagnostic figures in the paper (R lines 153-265):

  (1) ROC curves for the four models (Fearon-Laitin, Collier-Hoeffler,
      Hegre-Sambanis logit, and the Random Forest). Like the paper we plot
      them in-sample (the R code predicts on data.full itself).
  (2) The Random Forest's out-of-bag (OOB) error rate -- R lines 153-157.
  (3) Separation plots for the four models (R lines 245-265): sort country-
      years by predicted war-probability and mark where actual wars sit.

Outputs: figures/fig_roc_curves.png, figures/fig_separation.png,
         results/oob_error.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

from data import load_training_data, MODEL_SPECS
from models import make_logreg, make_random_forest_downsampled

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SHORT = {
    "Fearon & Laitin (2003)": "Fearon & Laitin (LR)",
    "Collier & Hoeffler (2004)": "Collier & Hoeffler (LR)",
    "Hegre & Sambanis (2006)": "Hegre & Sambanis (LR)",
}


def separation_plot(ax, probs, y, title):
    """Greenhill-style separation plot: sort by predicted prob, draw the
    probability line and mark actual wars as ticks at the top."""
    order = np.argsort(probs)
    p = probs[order]
    yy = y[order]
    n = len(p)
    ax.plot(np.arange(n), p, color="black", linewidth=0.8)
    ax.fill_between(np.arange(n), 0, p, color="lightgrey", alpha=0.5)
    wars = np.where(yy == 1)[0]
    ax.eventplot(wars, lineoffsets=1.0, linelengths=0.08,
                 colors="red", orientation="horizontal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("rank (by predicted P(war))", fontsize=8)
    ax.set_ylabel("P(war)", fontsize=8)
    ax.set_ylim(-0.02, 1.08)
    ax.tick_params(labelsize=7)


def main():
    df, y = load_training_data()
    print("=" * 64)
    print("STEP 7: Diagnostic figures (ROC, OOB error, separation plots)")
    print("=" * 64)

    probs = {}

    # --- The three logistic models (ordinary glm, in-sample) ------------
    for name, features in MODEL_SPECS.items():
        feats = [f for f in features if f in df.columns]
        X = df[feats].values
        m = make_logreg()
        m.fit(X, y)
        probs[name] = m.predict_proba(X)[:, 1]

    # --- The Random Forest (sampsize down-sampling, in-sample + OOB) ----
    X_all = df.drop(columns=["warstds"]).values
    rf = make_random_forest_downsampled(n_estimators=1000)
    rf.fit(X_all, y)
    probs["Muchlinski et al. (2016)"] = rf.predict_proba(X_all)[:, 1]

    # OOB error / AUC for the RF
    oob_proba = rf.oob_decision_function_[:, 1]
    oob_mask = ~np.isnan(oob_proba)
    oob_auc = roc_auc_score(y[oob_mask], oob_proba[oob_mask])
    pd.DataFrame([{
        "metric": "Random Forest OOB",
        "oob_error_rate": round(1 - rf.oob_score_, 4),
        "oob_auc": round(oob_auc, 4),
        "n_oob_samples": rf.oob_n_,
    }]).to_csv(os.path.join(RESULTS_DIR, "oob_error.csv"), index=False)
    print(f"RF OOB error rate = {1 - rf.oob_score_:.3f}  |  OOB AUC = {oob_auc:.3f}")

    # --- (1) ROC curves -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, p), c in zip(probs.items(), colors):
        fpr, tpr, _ = roc_curve(y, p)
        a = auc(fpr, tpr)
        label = SHORT.get(name, "Muchlinski (RF)")
        ax.plot(fpr, tpr, color=c, linewidth=2,
                label=f"{label} ({a:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (in-sample)\nreplicating paper Fig. 2-3")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    out1 = os.path.join(FIG_DIR, "fig_roc_curves.png")
    fig.tight_layout()
    fig.savefig(out1, dpi=140)
    print(f"ROC figure saved -> {out1}")

    # --- (3) Separation plots -------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 9))
    for ax, (name, p) in zip(axes, probs.items()):
        title = SHORT.get(name, "Random Forest")
        separation_plot(ax, p, y, title)
    fig.suptitle("Separation plots: do predicted probabilities sort the "
                 "rare wars to the right?\n(replicating paper Figs. 4-7)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out2 = os.path.join(FIG_DIR, "fig_separation.png")
    fig.savefig(out2, dpi=140)
    print(f"Separation-plot figure saved -> {out2}")


if __name__ == "__main__":
    main()
