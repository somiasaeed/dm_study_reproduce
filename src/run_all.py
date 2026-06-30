"""
run_all.py
==========
Run the whole replication end-to-end:

    python run_all.py

Steps:
    0. 00_fetch_raw_data.py      -> locate raw (non-imputed) Sambanis data
    1. 01_reproduce_reported.py  -> reproduce Muchlinski's leaky AUCs (glm + Firth + RF)
    2. 02_simulation.py          -> Appendix B.2 leakage simulation
    3. 03_leakage_correction.py  -> leaky vs corrected imputation on real data
    4. 04_plot_results.py        -> headline comparison figure
    5. 05_corrected_cv.py        -> principled leaky vs corrected CV (or synthetic fallback)
    6. 06_causal_mechanisms.py   -> variable importance + partial dependence
    7. 07_diagnostics.py         -> ROC curves, OOB error, separation plots
    8. 08_africa_oos.py          -> out-of-sample test on Africa (mapped)
    9. 09_leakage_importance.py  -> how leakage distorts feature importance
"""
import runpy
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEPS = [
    "00_fetch_raw_data.py",
    "01_reproduce_reported.py",
    "02_simulation.py",
    "03_leakage_correction.py",
    "04_plot_results.py",
    "05_corrected_cv.py",
    "06_causal_mechanisms.py",
    "07_diagnostics.py",
    "08_africa_oos.py",
    "09_leakage_importance.py",
]


def main():
    for step in STEPS:
        path = os.path.join(HERE, step)
        print("\n" + "#" * 70)
        print(f"# RUNNING {step}")
        print("#" * 70)
        runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()

