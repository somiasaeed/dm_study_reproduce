"""
02_simulation.py
================
STEP 2 - The Appendix B.2 simulation (the heart of the paper's argument).

We build a toy world where predicting the target is genuinely hard, then show
that imputing the train and test sets TOGETHER makes a model look far better
than it really is -- and the more data is missing, the worse the inflation.

Setup (exactly as in the paper, Appendix B.2):
    - onset : binary target   (1000 peace, 1000 war -> balanced)
    - gdp   : the only feature = N(0,1) + onset
              (so gdp carries a NOISY signal about onset)
    - 50/50 random train/test split
    - delete a fraction p of gdp values, impute, train a Random Forest,
      measure test accuracy

Two pipelines are compared for every missingness level p:
    LEAKY     -> impute train+test TOGETHER using the target `onset` as a
                 helper column (this is Muchlinski's mistake).
    CORRECTED -> impute using only the training data; test rows get the
                 training mean (no peeking at the test set).

Expected result (the paper's Figure A2):
    - LEAKY accuracy RISES as missingness grows (fake signal is injected)
    - CORRECTED accuracy FALLS toward the chance level (real signal is lost)

Output: figures/fig_simulation_leakage.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 666
N_PER_CLASS = 1000                 # paper: 1000 onset=0, 1000 onset=1
MISSING_LEVELS = np.arange(0, 96, 5)   # 0% ... 95% in steps of 5
N_REPS = 25                        # paper uses 100; 25 gives tight CIs and is fast


def make_data(rng):
    """onset (binary), gdp = N(0,1) + onset  -> noisy signal."""
    onset = np.array([0] * N_PER_CLASS + [1] * N_PER_CLASS)
    gdp = rng.normal(0, 1, size=len(onset)) + onset
    return onset, gdp


def impute_leaky(gdp, onset, mask, rng):
    """Muchlinski's error: impute train+test TOGETHER, using onset as a
    helper. The imputer therefore 'sees' the test rows (and their labels)."""
    gdp_masked = np.where(mask, np.nan, gdp)
    M = np.column_stack([gdp_masked, onset.astype(float)])   # [gdp, onset]
    imp = IterativeImputer(
        random_state=SEED, sample_posterior=False, max_iter=10,
        initial_strategy="mean", imputation_order="ascending",
    )
    M_filled = imp.fit_transform(M)        # fit on EVERYTHING (train+test)
    return M_filled[:, 0]                  # the imputed gdp


def impute_corrected(gdp, onset, mask, train_idx, test_idx):
    """No leakage: imputation model is learned from TRAIN only.
    Test rows' missing gdp is filled with the training mean -- the model has
    no access to test rows or their labels at imputation time."""
    gdp_masked = np.where(mask, np.nan, gdp)
    train_mean = np.nanmean(gdp_masked[train_idx])
    out = gdp_masked.copy()
    out[np.isnan(out)] = train_mean
    return out


def run_one(p, rng):
    onset, gdp = make_data(rng)
    idx = rng.permutation(len(onset))
    half = len(onset) // 2
    train_idx, test_idx = idx[:half], idx[half:]

    # delete a fraction p of gdp values (in both train and test)
    mask = rng.random(len(gdp)) < (p / 100.0)

    accs = {}
    for label, gdp_imp in [
        ("Leaky",     impute_leaky(gdp, onset, mask, rng)),
        ("Corrected", impute_corrected(gdp, onset, mask, train_idx, test_idx)),
    ]:
        X = gdp_imp.reshape(-1, 1)
        rf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED
        )
        rf.fit(X[train_idx], onset[train_idx])
        pred = rf.predict(X[test_idx])
        accs[label] = accuracy_score(onset[test_idx], pred)
    return accs


def main():
    print("=" * 64)
    print("STEP 2: Simulation - how imputing train+test together inflates")
    print("out-of-sample accuracy (Kapoor & Narayanan, Appendix B.2)")
    print("=" * 64)
    print(f"Missingness levels: 0%..95%, {N_REPS} repetitions each\n")

    results = {"Leaky": [], "Corrected": []}
    rng_master = np.random.RandomState(SEED)

    for p in MISSING_LEVELS:
        L, C = [], []
        for _ in range(N_REPS):
            rng = np.random.RandomState(rng_master.randint(1 << 31))
            a = run_one(p, rng)
            L.append(a["Leaky"]); C.append(a["Corrected"])
        results["Leaky"].append((np.mean(L), np.std(L)))
        results["Corrected"].append((np.mean(C), np.std(C)))
        print(f"  missing {p:2d}% -> Leaky {np.mean(L):.3f} | "
              f"Corrected {np.mean(C):.3f}", flush=True)

    # ---- save data -----------------------------------------------------
    rows = []
    for i, p in enumerate(MISSING_LEVELS):
        lm, ls = results["Leaky"][i]
        cm, cs = results["Corrected"][i]
        rows.append({"missing_pct": int(p),
                     "leaky_acc": round(lm, 4), "leaky_std": round(ls, 4),
                     "corrected_acc": round(cm, 4), "corrected_std": round(cs, 4)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "simulation_results.csv"),
                              index=False)

    # ---- plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, marker in [("Leaky", "#d62728", "o"),
                                 ("Corrected", "#2ca02c", "s")]:
        means = np.array([m for m, s in results[label]])
        stds = np.array([s for m, s in results[label]])
        ax.plot(MISSING_LEVELS, means, color=color, marker=marker,
                linewidth=2, label=label)
        ax.fill_between(MISSING_LEVELS, means - stds, means + stds,
                        color=color, alpha=0.15)
    ax.axhline(0.5, color="grey", linestyle=":", label="chance (0.5)")
    ax.set_xlabel("Proportion of missing values in 'gdp'")
    ax.set_ylabel("Out-of-sample accuracy")
    ax.set_title("Data leakage from joint train+test imputation\n"
                 "(replicating Kapoor & Narayanan, Appendix B.2)")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)
    out = os.path.join(FIG_DIR, "fig_simulation_leakage.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nFigure saved -> {out}")

    # ---- conclusion ----------------------------------------------------
    leaky_low = results["Leaky"][0][0]
    leaky_high = results["Leaky"][-1][0]
    print(f"\nLeaky accuracy went {leaky_low:.3f} -> {leaky_high:.3f} as "
          f"missingness rose (should INCREASE = inflated performance).")
    print(f"Corrected accuracy changed {results['Corrected'][0][0]:.3f} -> "
          f"{results['Corrected'][-1][0]:.3f} (should stay near chance).")
    print("\nThis proves the mechanism: the more data you impute jointly,")
    print("the more fake signal you inject, and the better the model LOOKS.")


if __name__ == "__main__":
    main()
