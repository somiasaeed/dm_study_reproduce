"""
00_fetch_raw_data.py
====================
STEP 0 - Locate the RAW (non-imputed) Sambanis data needed for a principled
Kapoor & Narayanan leakage correction (their Table A1: 0.54/0.57/0.68/0.64).

Why this is hard
----------------
Every file in Muchlinski's replication archive (doi:10.7910/DVN/KRKWK8) is
ALREADY imputed -- `SambnisImp.csv` (rfImpute), `Amelia_Imp3.csv` (Amelia),
and `AfricaImp.csv`. There is no raw, missing-valued Sambanis file there, so
the leakage cannot be "un-done" from what we already have. The exact Table A1
numbers additionally require Kapoor & Narayanan's specific R `mice`/`rfImpute`
configuration (CodeOcean capsule doi:10.24433/CO.4899453.v1, login-gated).

What this script does
---------------------
1. Looks for a raw file that the user may have placed in `data/`
   (e.g. `Sambnis_raw.csv`, `Sambanis.csv`).
2. Optionally tries a small list of public remote mirrors (speculative -- the
   canonical raw source is the original Hegre & Sambanis 2006 replication
   archive, whose schema differs from Muchlinski's processed set).
3. Validates any candidate: must contain `warstds`, must have missing values,
   and must overlap the 90 Muchlinski variables.
4. Writes `data/RAW_STATUS.txt` recording the outcome.

Step 05 reads that status: if a valid raw file exists it runs the principled
leaky-vs-corrected CV; otherwise it falls back to a clearly-labelled
synthetic-missingness estimate.

If you have the raw file, just drop it in `data/Sambnis_raw.csv` and re-run.
"""
import os
import sys
import io
import zipfile

try:
    import urllib.request as urlreq
except Exception:  # pragma: no cover
    urlreq = None

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATA_DIR, VARS_90   # noqa: E402

# Local candidate file names (case-insensitive) the user may supply.
LOCAL_CANDIDATES = [
    "Sambnis_raw.csv", "Sambnis.csv", "Sambanis.csv", "sambanis.csv",
    "Sambanis_raw.csv", "hs2006.csv",
]
# Speculative public mirrors. Treated as best-effort; failures fall back.
REMOTE_CANDIDATES = [
    # Original-format bundle of Muchlinski's archive (will NOT contain raw
    # data, but is the authoritative archive to inspect).
    "https://dataverse.harvard.edu/api/access/dataset/:persistentId/"
    "?persistentId=doi:10.7910/DVN/KRKWK8",
]

MIN_VAR_OVERLAP = 60     # require at least this many of the 90 vars present
MIN_ROWS = 5000


def validate(df):
    """Return (ok, reason). A valid raw frame has warstds, missing values,
    enough rows, and a reasonable overlap with the 90 Muchlinski variables."""
    if "warstds" not in df.columns:
        return False, "no 'warstds' column"
    overlap = len(set(df.columns) & set(VARS_90))
    if overlap < MIN_VAR_OVERLAP:
        return False, f"only {overlap} of the 90 Muchlinski vars present"
    if len(df) < MIN_ROWS:
        return False, f"only {len(df)} rows (< {MIN_ROWS})"
    n_missing = int(df[list(set(df.columns) & set(VARS_90))].isna().sum().sum())
    if n_missing == 0:
        return False, "no missing values (already imputed -- not raw)"
    return True, f"OK: {len(df)} rows, {overlap}/90 vars, {n_missing} missing cells"


def try_local():
    for name in LOCAL_CANDIDATES:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception as e:
                continue
            ok, reason = validate(df)
            if ok:
                return path, df, reason
            else:
                print(f"  [local] {name} found but rejected: {reason}")
    return None, None, None


def try_remote():
    if urlreq is None:
        return None, None, None
    for url in REMOTE_CANDIDATES:
        print(f"  [remote] attempting {url[:70]}...")
        try:
            req = urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlreq.urlopen(req, timeout=30) as resp:
                data = resp.read()
            # The Dataverse endpoint returns a ZIP of the archive.
            zf = zipfile.ZipFile(io.BytesIO(data))
            for member in zf.namelist():
                if member.lower().endswith(".csv"):
                    try:
                        df = pd.read_csv(zf.open(member))
                    except Exception:
                        continue
                    if "warstds" in df.columns:
                        ok, reason = validate(df)
                        if ok:
                            out = os.path.join(DATA_DIR, "Sambnis_raw.csv")
                            df.to_csv(out, index=False)
                            return out, df, f"{member}: {reason}"
                        else:
                            print(f"    {member}: rejected ({reason})")
        except Exception as e:
            print(f"    remote attempt failed: {e}")
    return None, None, None


def main():
    print("=" * 64)
    print("STEP 0: Locating raw (non-imputed) Sambanis data")
    print("=" * 64)

    path, df, reason = try_local()
    if path is None:
        path, df, reason = try_remote()

    status_path = os.path.join(DATA_DIR, "RAW_STATUS.txt")
    if path is not None:
        with open(status_path, "w") as f:
            f.write(f"FOUND\n{path}\n{reason}\n")
        print(f"\nSUCCESS: valid raw data found -> {path}")
        print(f"        ({reason})")
        print("Step 05 will run the principled leaky-vs-corrected CV.")
    else:
        with open(status_path, "w") as f:
            f.write("NOT_FOUND\n")
            f.write("No compatible raw Sambanis file is available.\n")
            f.write("Muchlinski's archive (doi:10.7910/DVN/KRKWK8) contains "
                    "only imputed files.\n")
            f.write("To enable the exact Table A1 correction, obtain the raw "
                    "non-imputed Sambanis data (Hegre & Sambanis 2006 "
                    "replication archive, or Kapoor & Narayanan's CodeOcean "
                    "capsule doi:10.24433/CO.4899453.v1) and save it as "
                    "data/Sambnis_raw.csv, then re-run.\n")
        print("\nNo compatible raw data found.")
        print("Step 05 will use the synthetic-missingness fallback "
              "(clearly labelled).")
        print(f"Status written -> {status_path}")
        print("\nTo enable the exact correction later, drop a raw Sambanis "
              "CSV at data/Sambnis_raw.csv and re-run.")


if __name__ == "__main__":
    main()
