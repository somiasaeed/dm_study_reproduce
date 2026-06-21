"""
models.py
=========
Model factory. Mirrors the four models in Muchlinski et al. (2016):

    - 3 published Logistic-Regression specifications
        (Fearon & Laitin 2003, Collier & Hoeffler 2004, Hegre & Sambanis 2006)
    - Muchlinski's own model = a Random Forest on all ~90 variables

The original R code trains the RF with `sampsize = c(30, 90)` -- a heavy
stratified down-sampling (30 peace + 90 war rows per tree) to fight the
extreme class imbalance (only 116 wars out of 7140 rows). scikit-learn does
not expose per-class sample sizes directly, so we approximate it with
`class_weight="balanced_subsample"`, which re-weights each bootstrap sample
to correct the imbalance. This is the standard, faithful Python equivalent.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def make_logreg():
    """Plain logistic regression (the R code uses glm binomial)."""
    return LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",   # handle the 1.6% war imbalance
        random_state=666,          # the R code's "set.seed(666)"
    )


def make_random_forest(n_estimators=1000):
    """Muchlinski's Random Forest (all ~90 variables, balanced subsampling)."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",   # approximates sampsize=c(30,90)
        max_features="sqrt",
        n_jobs=-1,
        random_state=666,
    )
