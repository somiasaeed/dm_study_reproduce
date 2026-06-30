"""
data.py
=======
Loads the Muchlinski et al. (2016) civil-war data and holds the exact
model specifications copied from the original R replication code
(`muchlinski_R_code.R`, lines 14-148).

Source datasets (downloaded into ../data/):
    - SambnisImp.csv  : the rfImpute-imputed Hegre & Sambanis (2006) data.
                        7140 country-year rows. This is the "leaked" data
                        Muchlinski actually trained on (train+test imputed
                        together). Target column = `warstds` (0=peace, 1=war).
    - AfricaImp.csv   : the out-of-sample Africa test set (2001-2014).
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ---------------------------------------------------------------------------
# The 90 variables Muchlinski et al. use, copied verbatim from the R code
# (lines 14-27). `warstds` is the target (civil war onset); the rest are the
# 87 predictors used by the Random Forest model.
# ---------------------------------------------------------------------------
VARS_90 = [
    "warstds", "ager", "agexp", "anoc", "army85", "autch98", "auto4",
    "autonomy", "avgnabo", "centpol3", "coldwar", "decade1", "decade2",
    "decade3", "decade4", "dem", "dem4", "demch98", "dlang", "drel",
    "durable", "ef", "ef2", "ehet", "elfo", "elfo2", "etdo4590",
    "expgdp", "exrec", "fedpol3", "fuelexp", "gdpgrowth", "geo1", "geo2",
    "geo34", "geo57", "geo69", "geo8", "illiteracy", "incumb", "infant",
    "inst", "inst3", "life", "lmtnest", "ln_gdpen", "lpopns", "major", "manuexp", "milper",
    "mirps0", "mirps1", "mirps2", "mirps3", "nat_war", "ncontig",
    "nmgdp", "nmdp4_alt", "numlang", "nwstate", "oil", "p4mchg",
    "parcomp", "parreg", "part", "partfree", "plural", "plurrel",
    "pol4", "pol4m", "pol4sq", "polch98", "polcomp", "popdense",
    "presi", "pri", "proxregc", "ptime", "reg", "regd4_alt", "relfrac", "seceduc",
    "second", "semipol3", "sip2", "sxpnew", "sxpsq", "tnatwar", "trade",
    "warhist", "xconst",
]

# ---------------------------------------------------------------------------
# The three published Logistic-Regression model specifications, copied from
# the R code (the formulas on lines 49, 85, 113). The 4th model (Muchlinski's
# own) is "all 90 variables" -> a Random Forest.
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "Fearon & Laitin (2003)": [
        "warhist", "ln_gdpen", "lpopns", "lmtnest", "ncontig", "oil",
        "nwstate", "inst3", "pol4", "ef", "relfrac",
    ],
    "Collier & Hoeffler (2004)": [
        "sxpnew", "sxpsq", "ln_gdpen", "gdpgrowth", "warhist", "lmtnest",
        "ef", "popdense", "lpopns", "coldwar", "seceduc", "ptime",
    ],
    "Hegre & Sambanis (2006)": [
        "lpopns", "ln_gdpen", "inst3", "parreg", "geo34", "proxregc",
        "gdpgrowth", "anoc", "partfree", "nat_war", "lmtnest", "decade1",
        "pol4sq", "nwstate", "regd4_alt", "etdo4590", "milper", "geo1",
        "tnatwar", "presi",
    ],
}


def load_training_data():
    """Return (X_rf, y) where X_rf has all 87 predictors (for the RF model)
    and y is the 0/1 civil-war-onset target, from the imputed SambnisImp data.
    Also returns the full imputed dataframe for the logistic sub-models."""
    path = os.path.join(DATA_DIR, "SambnisImp.csv")
    df = pd.read_csv(path)

    # Keep only rows that have a usable target and the 90 vars.
    keep = [c for c in VARS_90 if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["warstds"])
    df["warstds"] = df["warstds"].astype(int)

    y = df["warstds"].values
    return df, y


def load_africa_test():
    """Load the out-of-sample Africa test set (2001-2014).

    NOTE: this file uses *QOG* variable codes (gle_rgdpc, imf_gdpgr, ...),
    NOT the Sambanis names, so it cannot be fed straight into the 90-variable
    models. See src/08_africa_oos.py for the variable-mapping that bridges
    the two schemas.
    """
    path = os.path.join(DATA_DIR, "AfricaImp.csv")
    return pd.read_csv(path)


def load_amelia():
    """Load the Amelia-II imputed data (``data2`` in the R code).

    This is a *separate, differently-imputed* dataset of the theoretically
    important variables, which the paper uses ONLY for the causal-mechanism
    analysis (variable importance + partial dependence, R lines 160-189 &
    268-288). We drop the id/country/year/atwards columns exactly as the R
    code does (lines 163-164). Returns (df, y) where df includes `warstds`.
    """
    path = os.path.join(DATA_DIR, "Amelia_Imp3.csv")
    df = pd.read_csv(path)
    drop = [c for c in ["Unnamed: 0", "country", "year", "atwards"]
            if c in df.columns]
    df = df.drop(columns=drop)
    df = df.dropna(subset=["warstds"])
    df["warstds"] = df["warstds"].astype(int)
    y = df["warstds"].values
    return df, y


if __name__ == "__main__":
    df, y = load_training_data()
    print(f"Training data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Class balance -> peace(0): {(y==0).sum()}, war(1): {(y==1).sum()}")
    print(f"War rate: {y.mean()*100:.2f}%  (severely imbalanced)")
    print(f"\nVariables available: {len(df.columns)} of {len(VARS_90)} requested")
    missing_vars = set(VARS_90) - set(df.columns)
    if missing_vars:
        print(f"MISSING variables: {missing_vars}")
    else:
        print("All 90 variables present.")
