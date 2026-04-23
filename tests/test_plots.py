"""Smoke tests for the plotting gallery.

Each plot must (a) return a Figure, and (b) write both a PDF and a PNG
when ``save_to`` is given.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pydeb.bayes import build_growth_model
from pydeb.core.model import deb_growth
from pydeb.core.params import DEBParams
from pydeb.plots import (
    plot_arrhenius,
    plot_posterior_corner,
    plot_posterior_predictive_fan,
    plot_prior_posterior,
    plot_residuals,
    plot_trace_and_rank,
    render_gallery,
)


@pytest.fixture(scope="module")
def small_idata():
    """Cheap InferenceData from a tiny NUTS run."""
    true = DEBParams.daphnia_magna()
    times = np.tile(np.linspace(0, 21, 10), 3)
    data = pd.DataFrame({
        "time": times,
        "temp": 20.0,
        "L_obs": deb_growth(times, true, 20.0) + np.random.default_rng(0).normal(0, 0.1, times.size),
    })
    model = build_growth_model(data)
    with model:
        idata = pm.sample(
            draws=200, tune=200, chains=2,
            random_seed=1, progressbar=False,
        )
    return idata


@pytest.fixture(scope="module")
def fake_benchmark():
    """Cheap BenchmarkResult-like container for plot_residuals and gallery."""
    from debcompare import run_benchmark

    return run_benchmark(
        draws=200, tune=200, chains=2,
        n_post_samples=50, random_seed=42,
        save_figure=False, progressbar=False,
    )


def _assert_saved(tmp_path, stem):
    assert (tmp_path / f"{stem}.pdf").exists()
    assert (tmp_path / f"{stem}.png").exists()


def test_plot_arrhenius(tmp_path):
    fig = plot_arrhenius(save_to=tmp_path / "arrh")
    assert isinstance(fig, plt.Figure)
    _assert_saved(tmp_path, "arrh")
    plt.close(fig)


def test_plot_posterior_predictive_fan(small_idata, tmp_path):
    fig = plot_posterior_predictive_fan(small_idata,
                                        save_to=tmp_path / "fan",
                                        n_samples=50)
    _assert_saved(tmp_path, "fan")
    plt.close(fig)


def test_plot_prior_posterior(small_idata, tmp_path):
    fig = plot_prior_posterior(small_idata, save_to=tmp_path / "pp")
    _assert_saved(tmp_path, "pp")
    plt.close(fig)


def test_plot_posterior_corner(small_idata, tmp_path):
    fig = plot_posterior_corner(small_idata, save_to=tmp_path / "corner")
    _assert_saved(tmp_path, "corner")
    plt.close(fig)


def test_plot_trace_and_rank(small_idata, tmp_path):
    fig = plot_trace_and_rank(small_idata, save_to=tmp_path / "trace")
    _assert_saved(tmp_path, "trace")
    plt.close(fig)


@pytest.mark.slow
def test_plot_residuals(fake_benchmark, tmp_path):
    fig = plot_residuals(fake_benchmark, save_to=tmp_path / "resid")
    _assert_saved(tmp_path, "resid")
    plt.close(fig)


@pytest.mark.slow
def test_render_gallery(fake_benchmark, tmp_path):
    written = render_gallery(fake_benchmark, tmp_path / "gallery")
    assert set(written.keys()) == {
        "arrhenius", "predictive_fan", "prior_posterior",
        "posterior_corner", "residuals", "trace_rank",
    }
    for name, paths in written.items():
        for p in paths:
            assert p.exists(), f"{name}: {p} missing"
