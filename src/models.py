"""
models.py
=========
Model factory. Mirrors the four models in Muchlinski et al. (2016):

    - 3 published Logistic-Regression specifications
        (Fearon & Laitin 2003, Collier & Hoeffler 2004, Hegre & Sambanis 2006)
    - Muchlinski's own model = a Random Forest on all ~90 variables

For each logistic spec the paper actually trains TWO estimators:
    - `method="glm"`  -> ordinary logistic regression   (make_logreg)
    - `method="plr"`  -> Firth's penalized logistic reg. (make_firth_logit)
Firth's bias-reduction is the standard fix for the quasi-complete separation
that plagues class-imbalanced data (only 116 wars out of 7140 rows). The
`firthlogist` package on PyPI requires Python <3.11, so we implement Firth's
modified-score estimator directly in numpy (see `FirthLogit`).

The original R code trains the RF with `sampsize = c(30, 90)` -- a heavy
stratified down-sampling (30 peace + 90 war rows per tree) to fight the
extreme class imbalance. scikit-learn does not expose per-class sample sizes
directly, so we provide TWO faithful options:
    - make_random_forest()            -> class_weight="balanced_subsample"
                                         (the standard weighted approximation)
    - make_random_forest_downsampled() -> manual per-tree (30 peace + 90 war)
                                         bootstrap, matching sampsize=c(30,90)
"""
import numpy as np
from scipy.special import expit  # logistic sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


RANDOM_STATE = 666  # the R code's "set.seed(666)"


# ---------------------------------------------------------------------------
# Plain logistic regression (R code: glm binomial)
# ---------------------------------------------------------------------------
def make_logreg():
    """Plain logistic regression (the R code uses glm binomial)."""
    return LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",   # handle the 1.6% war imbalance
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Firth's penalized (bias-reduced) logistic regression (R code: method="plr")
# ---------------------------------------------------------------------------
class FirthLogit(BaseEstimator, ClassifierMixin):
    """Firth's bias-reduced logistic regression via modified-score IRLS.

    Firth (1993) removes the first-order bias of the MLE by penalising the
    log-likelihood with (1/2) log|I(beta)| -- the Jeffreys invariant prior.
    For binary logistic regression the modified (bias-corrected) score has
    the closed form (Heinze & Schemper 2002):

        U*(beta)_j = sum_i [ (y_i - pi_i) + (0.5 - pi_i) * h_i ] x_ij

    where h_i are the diagonal leverages of the Fisher-weighted hat matrix
    H = W^{1/2} X (X' W X)^{-1} X' W^{1/2}, W = diag(pi_i (1 - pi_i)).
    We solve U*(beta) = 0 by Fisher-scoring (a.k.a. modified IRLS).

    This is the faithful Python counterpart of caret's `method="plr"` /
    R package stepPlr / logistf's Firth correction.
    """

    def __init__(self, max_iter=100, tol=1e-6, ridge=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge

    def __sklearn_tags__(self):
        # Force the classifier tag -- sklearn >=1.6 no longer infers it from
        # ClassifierMixin for user classes, which breaks roc_auc scoring.
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel().astype(float)
        n, p = X.shape

        # design matrix with intercept
        Xi = np.column_stack([np.ones(n), X])
        beta = np.zeros(p + 1)

        for _ in range(self.max_iter):
            eta = Xi @ beta
            pi = expit(eta)
            w = pi * (1.0 - pi)                       # Fisher weights
            XtWX = Xi.T @ (Xi * w[:, None]) + self.ridge * np.eye(p + 1)
            inv = np.linalg.inv(XtWX)

            # leverages h_i = w_i * x_i' (X'WX)^{-1} x_i   (no n x n matrix)
            quad = np.einsum("ij,jk,ik->i", Xi, inv, Xi)
            h = w * quad

            # modified (Firth-corrected) score and Fisher-scoring step
            modif = (y - pi) + (0.5 - pi) * h
            Ustar = Xi.T @ modif
            step = inv @ Ustar

            beta_new = beta + step
            if np.max(np.abs(step)) < self.tol:
                beta = beta_new
                break
            beta = beta_new

        self.coef_ = beta
        self.intercept_ = beta[0]
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        Xi = np.column_stack([np.ones(X.shape[0]), X])
        return Xi @ self.coef_

    def predict_proba(self, X):
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_firth_logit():
    """Firth's penalized logistic regression (matches R method="plr")."""
    return FirthLogit(max_iter=100, tol=1e-6)


# ---------------------------------------------------------------------------
# Random Forest, class_weight variant (approximates sampsize via weighting)
# ---------------------------------------------------------------------------
def make_random_forest(n_estimators=1000):
    """Muchlinski's Random Forest (all ~90 variables, balanced subsampling)."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",   # approximates sampsize=c(30,90)
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Random Forest, explicit per-tree stratified down-sampling (sampsize=c(30,90))
# ---------------------------------------------------------------------------
class DownsampledRandomForest(BaseEstimator, ClassifierMixin):
    """Random Forest that grows each tree on a (30 peace + 90 war) bootstrap
    sample, exactly as R's `randomForest(..., sampsize=c(30,90), strata=...)`.

    This is the most faithful Python realisation of Muchlinski's class-
    imbalance handling: `class_weight="balanced_subsample"` re-weights a full
    bootstrap sample, whereas the original *physically* draws 30 peace and
    90 war rows per tree (replace=TRUE). Default mtry=sqrt(p), nodesize=1,
    matching the R randomForest classification defaults.
    """

    def __init__(self, n_estimators=1000, n_peace=30, n_war=90,
                 max_features="sqrt", random_state=RANDOM_STATE):
        self.n_estimators = n_estimators
        self.n_peace = n_peace
        self.n_war = n_war
        self.max_features = max_features
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        war_idx = np.where(y == 1)[0]
        peace_idx = np.where(y == 0)[0]

        self.trees_ = []
        self.inbag_ = []   # list of in-bag index arrays (for OOB estimates)
        for _ in range(self.n_estimators):
            # R randomForest: sampsize=c(30,90), replace=TRUE, strata=warstds
            w_draw = rng.choice(war_idx, size=self.n_war, replace=True)
            p_draw = rng.choice(peace_idx, size=self.n_peace, replace=True)
            idx = np.concatenate([p_draw, w_draw])
            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                random_state=rng.randint(0, 2**31 - 1),
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            self.inbag_.append(np.unique(idx))

        self.classes_ = np.array([0, 1])
        # Out-of-bag decision function (mean war-prob over trees that did NOT
        # see each sample) -- mirrors randomForest's OOB votes.
        oob = np.zeros((n, 2))
        cnt = np.zeros(n)
        for tree, inbag in zip(self.trees_, self.inbag_):
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[inbag] = False
            if oob_mask.any():
                oob[oob_mask] += tree.predict_proba(X[oob_mask])
                cnt[oob_mask] += 1
        safe = cnt > 0
        self.oob_decision_function_ = np.where(
            safe[:, None], oob / np.where(cnt == 0, 1, cnt)[:, None], np.nan)
        oob_pred = (self.oob_decision_function_[safe, 1] >= 0.5).astype(int)
        self.oob_score_ = float((oob_pred == y[safe]).mean())
        self.oob_n_ = int(safe.sum())
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        proba = np.zeros((X.shape[0], 2))
        for tree in self.trees_:
            proba += tree.predict_proba(X)
        proba /= len(self.trees_)
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_random_forest_downsampled(n_estimators=1000):
    """Muchlinski's Random Forest with explicit sampsize=c(30,90)
    stratified per-tree down-sampling (most faithful to the R code)."""
    return DownsampledRandomForest(n_estimators=n_estimators)
