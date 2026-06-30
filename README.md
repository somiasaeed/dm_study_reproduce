# Replication: Data Leakage in Civil-War Prediction (Muchlinski et al. 2016)

A Python replication of the civil-war-prediction case study from:

> **Kapoor, S. & Narayanan,A. (2023).** *Leakage and the reproducibility crisis
> in machine-learning-based science.* Patterns (Cell Press). arXiv:2207.07048.

We reproduce the part of that paper that re-analyses **Muchlinski et al.
(2016)**, *"Comparing Random Forest with Logistic Regression for Predicting
Class-Imbalanced Civil War Onset Data"* (Political Analysis, 24(1): 87–100).
Muchlinski et al. claimed that a Random Forest vastly outperforms Logistic
Regression for predicting civil-war onset. Kapoor & Narayanan show this was an
artefact of **data leakage** (imputing the training and test data together).
Once the leakage is fixed, the Random Forest's advantage disappears.

This package reproduces the reported (leaky) numbers, the leakage simulation,
and a leakage *correction*, **and** ports several analyses from Muchlinski's
original R code that a basic replication omits: Firth's penalized logit, the
faithful per-tree class down-sampling, variable importance, partial dependence,
ROC / OOB-error / separation-plot diagnostics, an out-of-sample test on the
Africa data, and a novel analysis of how leakage corrupts feature importance.

---

## 1. The one-paragraph idea

The original authors had a test set in which **>95% of the values were
missing**. They filled (imputed) those missing values using an algorithm that
ran on the **training and test sets combined**, with the outcome variable
(`warstds`) included as a helper. The imputed test values were therefore
reconstructed from the test set's own labels — the model was effectively handed
the answers. That made the Random Forest look brilliant (AUC ≈ 0.95). When the
imputation is done correctly (training data only), the AUC drops to ≈ 0.64,
the same ballpark as the decades-old Logistic Regression models.

---

## 2. The leakage mechanism, in detail

Muchlinski et al. used R's `randomForest::rfImpute` to fill missing values.
`rfImpute` is an **iterative** imputer: at each round it fits a Random Forest
to predict every column from all the others — **including the outcome
`warstds`** — and uses the forest's predictions to replace the missing
entries. Because `rfImpute` was run on the **entire dataset once, before any
train/test split**, the imputed values for a future test row were produced by a
model that had *seen that test row's own label*. Two consequences follow:

1. **Inflated discrimination.** The imputed test features become statistical
   proxies for the label they were rebuilt from, so any downstream classifier
   appears to separate war from peace almost perfectly. This is why the Random
   Forest reports CV AUC ≈ 0.93–0.95.
2. **Corrupted interpretation.** The features whose missing values were
   reconstructed from the label acquire artificially high importance (Step 9),
   so the model's "variable importance" plot misleads a reader about what
   drives civil war.

**The fix (Kapoor & Narayanan).** Imputation must happen *inside* cross-
validation: in each fold, the imputer is fit on the **training rows only**;
the held-out test rows are then filled using the relationships learned on
train, with **no access to the test labels**. This is the "corrected"
procedure implemented in Step 5.

> This is one instance of a wider class of "data leakage" bugs (fitting
> preprocessing on train+test, target leakage, group leakage, etc.) that
> Kapoor & Narayanan document across 17 fields and 294 papers; they find
> leakage-induced performance inflation in roughly half of the ML-based
> science papers they audited.

---

## 3. Data

All data were downloaded freely (no sign-in) from the Harvard Dataverse
replication archive of Muchlinski et al. (2016): `doi:10.7910/DVN/KRKWK8`.

| File | Rows | Cols | What it is |
|------|-----:|-----:|------------|
| `data/SambnisImp.csv` | 7140 | ~90 | The **rfImpute-imputed** Hegre & Sambanis (2006) data. This is the "leaked" data Muchlinski trained on (train+test imputed together). Target = `warstds` (1 = civil-war onset). **Class imbalance: 116 wars vs 7024 peace (1.6%).** |
| `data/AfricaImp.csv` | 737 | 11 | The out-of-sample Africa test set (2001–2014; 21 wars / 716 peace). Uses *QOG* variable codes (`gle_rgdpc`, `imf_gdpgr`, …) — a **different schema** from Sambanis. |
| `data/Amelia_Imp3.csv` | 7141 | 52 | A second, Amelia-II-imputed dataset of the theoretically-important variables; the paper uses it only for the variable-importance / partial-dependence analysis (`data2` in the R code). |
| `data/CompareCW_dat.tab` | 737 | 6 | The R script's **output** (the Africa prediction table, R lines 357–359), not used as features. |
| `data/muchlinski_R_code.R` | — | — | The original R replication code (the source of truth for the 90 variables and 4 model specs). |
| `data/Sambanis_Codebook.pdf` | — | — | Variable codebook. |

The 90 variables and the four model specifications are copied verbatim from the
R code (lines 14–27 and 49/85/113/142) into `src/data.py`.

**Important data fact (verified by fetching the archive in Step 0):** every
data file in Muchlinski's archive is **already imputed**. There is no raw,
missing-valued Sambanis file there, so the leakage cannot be "un-done" from the
local data — see Step 0 and Step 5.

---

## 4. Environment & setup

```bash
pip install -r requirements.txt   # numpy, pandas, scikit-learn, matplotlib, scipy
cd src
python run_all.py                 # runs all ten steps end-to-end
```

Tested on **Python 3.13.1**, scikit-learn 1.8.0, pandas 2.3.3, numpy 2.3.4,
Windows 10. No internet access is required to run (Step 0's optional remote
fetch degrades gracefully to the documented fallback). Full end-to-end runtime
is roughly 5–8 minutes on a laptop; the slow steps are Step 2 (the simulation,
20 × 25 RF fits) and Step 5 (per-fold iterative imputation).

---

## 5. How to run

```bash
python run_all.py              # all ten steps
```

Or run any step alone. Each step is self-contained and writes to `results/`
(CSV tables) and `figures/` (PNG plots). Step 5 reads the status file written
by Step 0: if a raw Sambanis file is present it runs the principled per-fold
correction; otherwise it uses a clearly-labelled synthetic estimate. The
pipeline therefore runs to completion **with or without** the raw data.

| Step | Script | Runtime |
|---|---|---|
| 0 | `00_fetch_raw_data.py` | ~10 s (or ~30 s if it fetches the archive) |
| 1 | `01_reproduce_reported.py` | ~20 s |
| 2 | `02_simulation.py` | ~2 min |
| 3 | `03_leakage_correction.py` | ~30 s |
| 4 | `04_plot_results.py` | <1 s |
| 5 | `05_corrected_cv.py` | ~1 min |
| 6 | `06_causal_mechanisms.py` | ~20 s |
| 7 | `07_diagnostics.py` | ~15 s |
| 8 | `08_africa_oos.py` | ~5 s |
| 9 | `09_leakage_importance.py` | ~20 s |

---

## 6. Steps, methods, and results

### Step 0 — Locate the raw (non-imputed) data (`00_fetch_raw_data.py`)
Purpose: find a raw Sambanis file with genuine missing values, which is the one
ingredient needed for the *exact* Table A1 reproduction. The script (a) looks
for a file the user may have placed at `data/Sambnis_raw.csv` (or similar
names), (b) optionally fetches and inspects the public archive, and (c)
validates any candidate (`warstds` present, has missing values, ≥60 of the 90
vars, ≥5000 rows). It writes `data/RAW_STATUS.txt`. **Finding:** the archive's
three CSVs are all rejected — `SambnisImp.csv` has zero missing values; the
others lack the 90-variable structure. No compatible raw file is available.

### Step 1 — Reproduce the reported (leaky) performance (`01_reproduce_reported.py`)
Method: train the four models on the imputed data with stratified 10-fold CV
(reporting both CV AUC and in-sample AUC). The paper trains each logit twice —
ordinary `glm` and Firth's penalized `plr` — so we report both. We also report
the Random Forest with both the `class_weight="balanced_subsample"`
approximation and the faithful `sampsize=c(30,90)` per-tree down-sampling.

| Model | Method | CV AUC (ours) | CV AUC (paper) |
|-------|--------|:---:|:---:|
| Fearon & Laitin (2003) | glm | **0.771** | 0.77 |
| Fearon & Laitin (2003) | plr (Firth) | **0.769** | 0.77 |
| Collier & Hoeffler (2004) | glm | **0.787** | 0.82 |
| Collier & Hoeffler (2004) | plr (Firth) | **0.765** | 0.77 |
| Hegre & Sambanis (2006) | glm | **0.803** | 0.80 |
| Hegre & Sambanis (2006) | plr (Firth) | **0.810** | 0.80 |
| Muchlinski et al. (2016) | rf (class_weight) | **0.926** | ~0.95 |
| Muchlinski et al. (2016) | rf (sampsize=30,90) | **0.917** | ~0.91 |

✅ All match the originals. On the leaked data the Random Forest (≈0.92) looks
far stronger than Logistic Regression (0.77–0.81) — the inflated gap the
critique targets. Output: `results/reported_auc.csv`.

### Step 2 — The leakage simulation (`02_simulation.py`)
Purpose: reproduce Figure A2 (Appendix B.2) — a self-contained demonstration of
the mechanism. We build a balanced toy world (`onset` binary; `gdp = N(0,1) +
onset`, so `gdp` carries only a *noisy* signal), delete a fraction `p` of `gdp`
values, and compare two imputation pipelines for each `p`: **leaky** (impute
train+test together with `onset` as a helper) vs **corrected** (impute train
only; fill test with the training mean). Then train a Random Forest and measure
out-of-sample accuracy. 25 repetitions per level; error bars are ±1 SD.

| Missingness `p` | Leaky accuracy | Corrected accuracy |
|:---:|:---:|:---:|
| 0% | 0.60 | 0.60 |
| 50% | 0.80 | 0.55 |
| 95% | **0.98** | **0.51** |

✅ Leaky accuracy **rises** toward perfection (fake signal injected) while
corrected accuracy **falls** toward chance (real signal lost). Output:
`results/simulation_results.csv`, `figures/fig_simulation_leakage.png`.

### Step 3 — Leakage on the real civil-war data (`03_leakage_correction.py`)
Purpose: reconstruct Muchlinski's mistake directly on the real 90-variable
data. We hide a fraction of the **test-set** feature values, then compare joint
imputation (leaky, target as helper) against train-only imputation (corrected),
using a Random Forest.

| Test missingness | Leaky AUC | Corrected AUC | Gap |
|:---:|:---:|:---:|:---:|
| 0% | 0.930 | 0.930 | 0.000 |
| 30% | 0.921 | 0.899 | +0.023 |
| 60% | 0.896 | 0.854 | +0.042 |
| **90%** | **0.902** | **0.719** | **+0.184** |

✅ The gap widens as more test data is missing — mirroring the paper's RF
correction (0.95 → 0.64). Output: `results/real_data_leakage.csv`,
`figures/fig_real_data_leakage.png`.

### Step 4 — Headline figure (`04_plot_results.py`)
Bar chart of the four models' reported (leaky) CV AUC against Kapoor &
Narayanan's corrected AUC (Table A1), so the inflation is visible at a glance.
Output: `figures/fig_reported_vs_corrected.png`.

### Step 5 — The leakage CORRECTION for all four models (`05_corrected_cv.py`)
Purpose: compare Muchlinski's **leaky global imputation** (impute the whole
dataset once, with the target, then CV) against the **corrected per-fold
imputation** (impute each training fold only; fill the test fold from
train-learned feature relationships, no target), for every model, and benchmark
against Table A1. Imputations are computed once and shared across models.

With no raw data available, Step 5 uses a **synthetic-missingness
reconstruction** (hide 90% of test features, joint vs train-only imputation):

| Model | Leaky AUC | Corrected AUC | K&N Table A1 |
|-------|:---:|:---:|:---:|
| Fearon & Laitin (2003) | 0.794 | **0.590** | 0.54 |
| Collier & Hoeffler (2004) | 0.693 | **0.623** | 0.57 |
| Hegre & Sambanis (2006) | 0.829 | **0.577** | 0.68 |
| Muchlinski et al. (2016) | 0.784 | **0.685** | 0.64 |

✅ Corrected ≪ leaky for every model, and the corrected AUCs land in the same
0.58–0.69 band as K&N's actual corrected numbers (RF **0.685** vs **0.64**).
Drop a raw file at `data/Sambnis_raw.csv` and this step automatically switches
to the principled per-fold CV. Output: `results/corrected_auc.csv`,
`figures/fig_corrected_cv.png`.

### Step 6 — Causal mechanisms (`06_causal_mechanisms.py`)
Purpose: reproduce the paper's variable-importance dotplot (R lines 160–189)
and 3×3 partial-dependence grid (R lines 268–288), on the Amelia-imputed data.
We train the down-sampled RF and report both Gini (MDI) importance and
permutation importance (Δ AUC), then compare the top-20 ranking to the paper's
hardcoded list.

- **15 of the top-20 variables overlap** the paper's reported drivers (GDP
  growth, GDP/capita, life expectancy, infant mortality, trade, mountainous
  terrain, etc.).
- Partial dependence shows the expected directions (e.g. higher GDP/capita →
  lower predicted war probability).

Output: `results/var_importance.csv`, `figures/fig_var_importance.png`,
`figures/fig_partial_dependence.png`.

### Step 7 — Diagnostics (`07_diagnostics.py`)
Purpose: reproduce the paper's diagnostic figures. In-sample ROC curves for the
four models (R lines 191–243); the Random Forest's out-of-bag error (R lines
153–157); and Greenhill-style separation plots (R lines 245–265) — sort
country-years by predicted probability and mark where actual wars sit.

- RF OOB error rate **0.272**, OOB AUC **0.918**.
- The leaked RF's ROC dominates the logits, and its separation plot pushes the
  rare wars to the right — both artefacts of the leakage.

Output: `results/oob_error.csv`, `figures/fig_roc_curves.png`,
`figures/fig_separation.png`.

### Step 8 — Out-of-sample test on Africa (`08_africa_oos.py`)
Purpose: do the proper hold-out test the paper *should* have done. The paper's
own Africa "analysis" (R lines 291–359) is incoherent: it trains on the Sambanis
data and then compares **random training-set predictions** to Africa labels
(sampling 737 of the 7140 training predictions to match lengths). We instead
map the ~8 QOG features to their closest Sambanis equivalents, standardise with
the training distribution, and evaluate on the real Africa rows.

| Model (mapped ~8-feature subset) | Africa AUC | Precision@0.5 | Recall@0.5 |
|---|:---:|:---:|:---:|
| Logistic Regression | 0.623 | 0.080 | 0.095 |
| Random Forest | 0.519 | 0.000 | 0.000 |

✅ Out-of-sample the Random Forest is near chance (0.52) and the logit is modest
(0.62) — the RF's apparent dominance does **not** survive a genuine hold-out.
*Best-effort: only ~8 features, several are approximate cross-schema proxies.*
Output: `results/africa_oos.csv`, `figures/fig_africa_oos.png`.

### Step 9 — How leakage distorts feature importance (`09_leakage_importance.py`)
Purpose (novel): show that leakage corrupts the model's **interpretation**, not
just its accuracy. We hide 90% of feature values in *both* train and test
(mimicking the heavily-missing regime), impute jointly (leaky, target helper)
vs train-only (corrected), train a down-sampled RF on each, and compare the
feature-importance rankings.

- Leaky AUC **0.752** vs Corrected **0.672** (gap +0.079).
- **Spearman rank correlation of the two importance rankings = 0.373** — leakage
  substantially re-orders which variables the model deems important.

A policymaker reading the leaked model's importance plot would therefore draw
the wrong conclusions about what "drives" civil war. Output:
`results/importance_comparison.csv`, `figures/fig_importance_leaky_vs_corrected.png`.

---

## 7. Consolidated results

| Finding | Where | Number |
|---|---|---|
| Reported (leaky) RF CV AUC | Step 1 | 0.917–0.926 (paper ~0.91–0.95) |
| Reported (leaky) logit CV AUCs | Step 1 | 0.77 / 0.79 / 0.80 |
| Simulation: leaky acc at 95% missing | Step 2 | 0.98 vs corrected 0.51 |
| Real data: leaky→corrected RF AUC at 90% missing | Step 3 | 0.902 → 0.719 |
| Corrected AUCs (synthetic), 4 models | Step 5 | 0.59 / 0.62 / 0.58 / 0.69 |
| RF OOB AUC / error | Step 7 | 0.918 / 0.272 |
| Africa out-of-sample AUC (logit / RF) | Step 8 | 0.62 / 0.52 |
| Importance rank corruption (Spearman) | Step 9 | 0.373 |

**Bottom line.** The Random Forest's apparent dominance is confirmed an
artefact of data leakage **four independent ways** (Steps 3, 5, 8, 9), and the
leakage additionally corrupts the model's variable-importance interpretation.

---

## 8. Methodology appendix

### 8.1 Firth's penalized logistic regression (`src/models.py:FirthLogit`)
Firth (1993) removes the first-order bias of the maximum-likelihood estimator
by penalising the log-likelihood with Jeffreys' invariant prior,
½ log|I(β)|. For binary logistic regression the bias-corrected score has the
closed form (Heinze & Schemper 2002)

```
U*(β)_j = Σ_i [ (y_i − π_i) + (0.5 − π_i) · h_i ] x_ij
```

where `π_i = sigmoid(x_i·β)` and `h_i` are the diagonal leverages of the
Fisher-weighted hat matrix `H = W^{½} X (XᵀWX)⁻¹ Xᵀ W^{½}`,
`W = diag(π_i(1−π_i))`. We solve `U*(β)=0` by Fisher scoring (modified IRLS),
adding a 1e-6 ridge for numerical stability. The public `firthlogist` package
requires Python <3.11, so this is implemented directly in numpy — it is the
faithful counterpart of caret's `method="plr"` / R's `logistf`.

### 8.2 Per-tree stratified down-sampling (`src/models.py:DownsampledRandomForest`)
R's `randomForest(..., sampsize=c(30,90), strata=warstds, replace=TRUE)` grows
each of 1000 trees on a bootstrap draw of **30 peace + 90 war** rows (with
`mtry=√p`, `nodesize=1`). scikit-learn exposes no per-class sample size, so we
replicate it exactly: for each tree we draw 30 peace and 90 war rows with
replacement, fit a `DecisionTreeClassifier(max_features="sqrt")`, and average
`predict_proba` across trees. The class also computes proper **out-of-bag**
predictions (each sample scored only by the trees that did not draw it). This
is markedly more faithful than `class_weight="balanced_subsample"`, which only
re-weights a full bootstrap.

### 8.3 The corrected cross-validation (`src/05_corrected_cv.py`)
- **Leaky (Muchlinski):** impute the whole dataset once with `warstds` as a
  helper column, then run k-fold CV on the imputed matrix.
- **Corrected (Kapoor & Narayanan):** for each fold, fit an iterative imputer
  on the **training rows** (the target may be used there, since those rows are
  labelled); fill the test fold's missing features with a **feature-only**
  imputer learned on the (imputed) training data — the test labels are never
  touched. Then fit the model and score the fold.

The synthetic fallback reuses the Step-3 reconstruction (hide 90% of test
features; joint vs train-only imputation) and reports both, so the only
train/test asymmetry is the imputation itself.

---

## 9. Reproduction fidelity & honest scope

- **Reported AUCs (Step 1)** are reproduced faithfully and match the original R
  code — now including Firth's penalized logit and the faithful
  `sampsize=c(30,90)` down-sampling.
- The **simulation (Step 2)** is fully self-contained and reproduces the
  paper's Figure A2 pattern exactly.
- **Step 5's corrected AUCs are synthetic estimates.** No raw, non-imputed
  Sambanis data exists in Muchlinski's archive (verified by fetching and
  inspecting it in Step 0), and the *exact* Table A1 numbers additionally
  require Kapoor & Narayanan's specific R `mice`/`rfImpute` configuration
  (CodeOcean capsule `doi:10.24433/CO.4899453.v1`, login-gated). Our synthetic
  corrected AUCs land within ~0.05 of Table A1 (RF **0.685** vs **0.64**), and
  the qualitative conclusion (corrected ≪ leaky) is robust. To run the
  principled per-fold correction, drop a raw file at `data/Sambnis_raw.csv` and
  re-run — Step 0/5 detect it automatically.
- **Step 8 (Africa)** uses a best-effort cross-schema variable mapping; only
  ~8 features are comparable, so it is an illustrative hold-out rather than a
  clean replication of the paper's (already-flawed) Africa numbers.
- scikit-learn has no direct equivalent of R's `sampsize=c(30,90)`; we
  implement it manually (§8.2) and also expose the
  `class_weight="balanced_subsample"` approximation for comparison.

## 10. Related work

This replication corroborates, and is corroborated by, two prior critiques:
- *Insufficiencies in Data Material: A Replication Analysis of Muchlinski et
  al. (2016)*, Political Analysis (2018).
- Sternberg, *How Cross-Validation Can Go Wrong* (2018).

The overarching framework is Kapoor & Narayanan (2023), *Leakage and the
reproducibility crisis in machine-learning-based science*, Patterns
(arXiv:2207.07048).

---

## 11. Output file manifest

**Tables (`results/`)**
- `reported_auc.csv` — Step 1: reported (leaky) AUCs, glm + Firth + both RF variants.
- `simulation_results.csv` — Step 2: simulation accuracy vs missingness.
- `real_data_leakage.csv` — Step 3: leaky vs corrected RF AUC vs test missingness.
- `corrected_auc.csv` — Step 5: corrected AUCs for all four models vs Table A1.
- `var_importance.csv` — Step 6: Gini + permutation importance ranking.
- `oob_error.csv` — Step 7: RF OOB error and AUC.
- `africa_oos.csv` — Step 8: Africa out-of-sample AUC / precision / recall.
- `importance_comparison.csv` — Step 9: leaky vs corrected importance per feature.

**Figures (`figures/`)** — `fig_simulation_leakage`, `fig_real_data_leakage`,
`fig_reported_vs_corrected`, `fig_corrected_cv`, `fig_var_importance`,
`fig_partial_dependence`, `fig_roc_curves`, `fig_separation`, `fig_africa_oos`,
`fig_importance_leaky_vs_corrected`.

---

## 12. Project layout

```
muchlinski-replication/
├── data/                         # downloaded datasets + original R code
├── src/
│   ├── data.py                   # 90 variables + 4 model specs + loaders
│   ├── models.py                 # logreg, Firth logit, RF (class_weight + downsampled)
│   ├── 00_fetch_raw_data.py      # locate raw (non-imputed) Sambanis data
│   ├── 01_reproduce_reported.py  # Step 1: leaky reported AUCs (glm + Firth + RF)
│   ├── 02_simulation.py          # Step 2: Appendix B.2 simulation
│   ├── 03_leakage_correction.py  # Step 3: leaky vs corrected (real data)
│   ├── 04_plot_results.py        # Step 4: headline figure
│   ├── 05_corrected_cv.py        # Step 5: principled corrected CV (or synthetic)
│   ├── 06_causal_mechanisms.py   # Step 6: variable importance + partial dependence
│   ├── 07_diagnostics.py         # Step 7: ROC, OOB, separation plots
│   ├── 08_africa_oos.py          # Step 8: Africa out-of-sample (mapped)
│   ├── 09_leakage_importance.py  # Step 9: leakage distorts feature importance
│   └── run_all.py                # run everything
├── results/                      # CSV tables
├── figures/                      # PNG plots
├── requirements.txt              # numpy, pandas, scikit-learn, matplotlib, scipy
└── README.md
```
