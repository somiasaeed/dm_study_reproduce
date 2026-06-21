"""
04_plot_results.py
==================
Makes the headline bar chart: the 4 models' cross-validated AUC on Muchlinski's
imputed (leaky) data. This is the "reported" performance that looks impressive
-- the Random Forest appears to crush Logistic Regression.

For reference we overlay the corrected AUC values reported by Kapoor &
Narayanan (Table A1) once leakage is fixed, so the inflation is visible at a
glance.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Kapoor & Narayanan (2022), Table A1 -- corrected AUC after fixing leakage.
# (These are the paper's target values; our own correction is shown in the
#  real-data demo, step 3.)
PAPER_CORRECTED = {
    "Fearon & Laitin (2003)": 0.54,
    "Collier & Hoeffler (2004)": 0.57,
    "Hegre & Sambanis (2006)": 0.68,
    "Muchlinski et al. (2016)": 0.64,
}
SHORT = {
    "Fearon & Laitin (2003)": "Fearon &\nLaitin (LR)",
    "Collier & Hoeffler (2004)": "Collier &\nHoeffler (LR)",
    "Hegre & Sambanis (2006)": "Hegre &\nSambanis (LR)",
    "Muchlinski et al. (2016)": "Muchlinski\n(Random Forest)",
}


def main():
    rep = pd.read_csv(os.path.join(RESULTS_DIR, "reported_auc.csv"))

    names = list(rep["Model"])
    reported = rep["CV AUC (mean)"].values
    corrected = np.array([PAPER_CORRECTED[n] for n in names])

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w/2, reported, w, label="Reported (leaky, our reproduction)",
                color="#d62728", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, corrected, w, label="Corrected (Kapoor & Narayanan, Table A1)",
                color="#2ca02c", edgecolor="black", linewidth=0.5)

    for rect in list(b1) + list(b2):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.008,
                f"{rect.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[n] for n in names], fontsize=9)
    ax.set_ylabel("AUC (area under ROC)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Civil-war prediction: reported vs leakage-corrected performance\n"
                 "Fixing data leakage erases the Random Forest's apparent advantage")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    out = os.path.join(FIG_DIR, "fig_reported_vs_corrected.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Headline figure saved -> {out}")


if __name__ == "__main__":
    main()
