"""debcompare — reproducible benchmark for the Hackenberger & Djerdj (2026) review.

Run the full three-paradigm comparison with::

    python -m debcompare

or from Python::

    from debcompare import run_benchmark
    results = run_benchmark()
"""

from debcompare.benchmark import run_benchmark, BenchmarkResult
from debcompare.data import simulate_daphnia_dataset
from debcompare.metrics import rmse, r_squared, mae, coverage_95

__all__ = [
    "run_benchmark",
    "BenchmarkResult",
    "simulate_daphnia_dataset",
    "rmse",
    "r_squared",
    "mae",
    "coverage_95",
]
