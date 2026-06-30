"""
06_causal_mechanisms.py
=======================
STEP 6 - Reproduce the paper's "causal mechanism" analyses (R lines 160-189
and 268-288), which Muchlinski runs on the *Amelia*-imputed data (data2).

Two outputs:

  (1) Variable-importance dotplot  -- which features most drive the Random
      Forest's predictions. R uses MeanDecreaseGini; we report both the Gini
      (MDI) importance and permutation importance, and compare the top-20
      ranking to the hardcoded list in the paper (R lines 177-186):
      GDP Growth, GDP per Capita, Life Expectancy, W. Europe & US dummy,
      Infant Mortality, Trade, Mountainous Terrain, Illiteracy, Population,
      Linguistic heterogeneity, Anocracy, Median regional polity, Primary
      commodity exports squared, Democracy, Military power, Population
      density, Political instability, Ethnic fractionalization, Secondary
      education, Primary commodity exports.

  (2) Partial-dependence grid (3x3) -- how the predicted probability of civil
      war varies with each driver, marginalising over the others. Mirrors the
      nine partialPlot() calls in the R code.

Outputs: figures/fig_var_importance.png, figures/fig_partial_dependence.png,
         results/var_importance.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from data import load_amelia
from models import make_random_forest_downsampled

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 666

# The paper's reported top-20 drivers (R lines 180-186), as the variable
# names they correspond to in the data. Used only for comparison.
PAPER_TOP_VARS = [
    "gdpgrowth", "ln_gdpen", "life", "geo1", "infant", "trade", "lmtnest",
    "illiteracy", "lpopns", "ef", "anoc", "proxregc", "sxpsq", "pol4",
    "milper", "popdense", "inst3", "ef", "seceduc", "sxpnew",
]

# Partial-dependence grid: variable + readable label (R lines 271-288).
PDP_VARS = [
    ("gdpgrowth", "GDP Growth Rate"),
    ("ln_gdpen", "GDP per Capita (log)"),
    ("life", "Life Expectancy"),
    ("infant", "Infant Mortality Rate"),
    ("lmtnest", "Mountainous Terrain (log)"),
    ("pol4sq", "Polity IV squared"),
    ("lpopns", "Population (log)"),
    ("trade", "Trade"),
    ("geo1", "W. Europe & U.S. dummy"),
]


def main():
    df, y = load_amelia()
    feats = [c for c in df.columns if c != "warstds"]
    X = df[feats].values.astype(float)
    print("=" * 64)
    print("STEP 6: Causal mechanisms on the Amelia-imputed data")
    print(f"({len(y)} rows, {(y==1).sum()} wars, {len(feats)} features)")
    print("=" * 64)

    # --- Train the Random Forest (sampsize=c(30,90) down-sampling) ------
    print("Training down-sampled RF (300 trees) ...")
    rf = make_random_forest_downsampled(n_estimators=300)
    rf.fit(X, y)
    in_auc = roc_auc_score(y, rf.predict_proba(X)[:, 1])
    print(f"In-sample AUC = {in_auc:.3f}")

    # --- (1) Variable importance ----------------------------------------
    # MDI = mean Gini importance averaged over the 1000 trees.
    mdi = np.mean([t.feature_importances_ for t in rf.trees_], axis=0)
    # Permutation importance (on a subsample for speed).
    rng = np.random.RandomState(SEED)
    peace = np.where(y == 0)[0]
    war = np.where(y == 1)[0]
    sub = np.concatenate([war, rng.choice(peace, size=1500, replace=False)])
    perm = permutation_importance(rf, X[sub], y[sub], scoring="roc_auc",
                                  n_repeats=3, random_state=SEED, n_jobs=-1)
    perm_imp = perm.importances_mean

    imp = pd.DataFrame({
        "feature": feats,
        "mdi_importance": mdi,
        "perm_importance": perm_imp,
    }).sort_values("mdi_importance", ascending=False).reset_index(drop=True)
    imp.insert(0, "rank_mdi", np.arange(1, len(imp) + 1))
    imp.to_csv(os.path.join(RESULTS_DIR, "var_importance.csv"), index=False)

    top20 = imp.head(20).iloc[::-1]   # reverse for horizontal bar (largest on top)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    axes[0].barh(top20["feature"], top20["mdi_importance"], color="#4c72b0")
    axes[0].set_xlabel("Mean Decrease in Gini (MDI)")
    axes[0].set_title("Top-20 variables by Gini importance\n(replicates paper's dotplot)")
    top20p = (imp.sort_values("perm_importance", ascending=False)
              .head(20).iloc[::-1])
    axes[1].barh(top20p["feature"], top20p["perm_importance"], color="#55a868")
    axes[1].set_xlabel("Permutation importance (Δ AUC)")
    axes[1].set_title("Top-20 variables by permutation importance")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out1 = os.path.join(FIG_DIR, "fig_var_importance.png")
    fig.savefig(out1, dpi=140)
    print(f"Variable-importance figure saved -> {out1}")

    overlap = len(set(imp.head(20)["feature"]) & set(PAPER_TOP_VARS))
    print(f"Top-20 overlap with the paper's reported drivers: {overlap}/20")

    # --- (2) Partial dependence (manual, on the 9 grid variables) -------
    n_grid = 20
    # subsample rows the PDP averages over (pure speed; result is unchanged)
    pdp_sub = np.concatenate([war, rng.choice(peace, size=1500, replace=False)])
    Xpdp = X[pdp_sub]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.ravel()
    for k, (var, label) in enumerate(PDP_VARS):
        j = feats.index(var)
        vals = np.linspace(np.percentile(X[:, j], 2),
                           np.percentile(X[:, j], 98), n_grid)
        pdp = np.zeros_like(vals)
        for gi, v in enumerate(vals):
            Xg = Xpdp.copy()
            Xg[:, j] = v
            pdp[gi] = rf.predict_proba(Xg)[:, 1].mean()
        axes[k].plot(vals, pdp, color="#c44e52", linewidth=2)
        axes[k].set_xlabel(label, fontsize=9)
        axes[k].set_ylabel("P(civil war)", fontsize=9)
        axes[k].grid(alpha=0.3)
        # hide the y-axis range emphasis (paper used ylim) -- keep raw
    fig.suptitle("Partial dependence of predicted civil-war probability\n"
                 "(Amelia-imputed data; replicates paper Fig. partialPlot grid)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out2 = os.path.join(FIG_DIR, "fig_partial_dependence.png")
    fig.savefig(out2, dpi=140)
    print(f"Partial-dependence figure saved -> {out2}")


if __name__ == "__main__":
    main()
