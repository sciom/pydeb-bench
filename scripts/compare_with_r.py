#!/usr/bin/env python3
"""Side-by-side comparison of Python vs R benchmark outputs.

Reads the Python ``benchmark_output/benchmark_results.csv`` produced by
``python -m debcompare`` and the original ``../benchmark_results.csv``
produced by ``benchmark.R`` in the paper's root directory, and prints a
merged table with RMSE / R^2 ratios.

Because R and Python use different RNGs, exact numerical reproduction is
impossible without porting the RNG state; the goal here is to verify that
the three-paradigm *ordering* and order-of-magnitude RMSEs agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # R writes "Coverage_95" as NA which pandas parses fine; just normalise.
    return df.set_index(["Paradigm", "Task"])


def main() -> int:
    here = Path(__file__).resolve().parent
    py_path = here.parent / "benchmark_output" / "benchmark_results.csv"
    r_path = here.parent.parent / "benchmark_results.csv"

    if not py_path.exists():
        print(f"Python benchmark output not found: {py_path}", file=sys.stderr)
        print("Run `python -m debcompare -o benchmark_output` first.", file=sys.stderr)
        return 1
    if not r_path.exists():
        print(f"R benchmark output not found: {r_path}", file=sys.stderr)
        return 1

    py = load(py_path)
    r = load(r_path)

    cols = ["RMSE", "R2", "MAE", "Coverage_95"]
    merged = pd.concat(
        {"R": r[cols], "Python": py[cols]},
        axis=1,
    )
    merged = merged.swaplevel(axis=1).sort_index(axis=1)

    print("Benchmark comparison (R vs Python):\n")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(merged.to_string())

    print("\nKey sanity checks:")
    for task in ("Interpolation (20 C)", "Extrapolation (25 C)"):
        py_deb = py.loc[("Classical DEB", task), "RMSE"]
        py_bayes = py.loc[("Bayesian DEB", task), "RMSE"]
        py_rf = py.loc[("Random Forest", task), "RMSE"]
        print(f"  {task}:")
        print(f"    Classical DEB RMSE: {py_deb:.4f}")
        print(f"    Bayesian DEB RMSE:  {py_bayes:.4f}  (close to classical? {abs(py_deb - py_bayes) < 0.01})")
        print(f"    Random Forest RMSE: {py_rf:.4f}  (worse than DEB on extrapolation? {py_rf > py_deb})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
