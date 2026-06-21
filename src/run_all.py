"""
run_all.py
==========
Run the whole replication end-to-end:

    python run_all.py

Steps:
    1. 01_reproduce_reported.py   -> reproduce Muchlinski's leaky AUCs
    2. 02_simulation.py           -> Appendix B.2 leakage simulation
    3. 03_leakage_correction.py   -> leaky vs corrected imputation on real data
    4. 04_plot_results.py         -> headline comparison figure
"""
import runpy
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEPS = [
    "01_reproduce_reported.py",
    "02_simulation.py",
    "03_leakage_correction.py",
    "04_plot_results.py",
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
