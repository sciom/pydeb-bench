"""Smoke test: benchmark runs end to end and produces sensible numbers.

Uses a deliberately small number of MCMC draws so the test is fast (~30 s).
Full numerical agreement with R is validated separately.
"""

from __future__ import annotations

import pytest

from debcompare import run_benchmark


@pytest.mark.slow
def test_benchmark_runs_end_to_end(tmp_path):
    result = run_benchmark(
        draws=400,
        tune=400,
        chains=2,
        n_post_samples=200,
        random_seed=42,
        output_dir=tmp_path,
        save_figure=False,
    )

    # Results DataFrame has the expected shape.
    assert len(result.results_df) == 6
    assert set(result.results_df["Paradigm"]) == {"Classical DEB", "Bayesian DEB", "Random Forest"}

    # Classical & Bayesian DEB RMSE are small (close to true sigma ~0.12),
    # and RF is substantially worse on extrapolation.
    df = result.results_df
    classical_interp = df[(df["Paradigm"] == "Classical DEB") & (df["Task"] == "Interpolation (20 C)")]["RMSE"].iloc[0]
    rf_extrap = df[(df["Paradigm"] == "Random Forest") & (df["Task"] == "Extrapolation (25 C)")]["RMSE"].iloc[0]
    assert classical_interp < 0.2
    assert rf_extrap > 0.3

    # Coverage reported for Bayesian.
    bayes_cov = df[(df["Paradigm"] == "Bayesian DEB") & (df["Task"] == "Interpolation (20 C)")]["Coverage_95"].iloc[0]
    assert 0.8 <= bayes_cov <= 1.0

    # CSV written.
    assert (tmp_path / "benchmark_results.csv").exists()
