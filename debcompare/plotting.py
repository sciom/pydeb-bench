"""Three-panel benchmark figure matching the layout of the R benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pydeb.bayes.diagnostics import credible_band
from pydeb.core.model import deb_growth


CLASSICAL_COLOR = "steelblue"
BAYES_COLOR = "darkorange"
RF_COLOR = "forestgreen"
TRUE_COLOR = "0.4"


def save_benchmark_figure(
    result,
    output_dir: Path,
    random_seed: int = 42,
    n_post_samples: int = 500,
) -> None:
    """Render and save the three-panel comparison figure.

    Parameters
    ----------
    result : BenchmarkResult
    output_dir : Path
        Directory to write ``benchmark_figure.pdf`` and ``.png``.
    random_seed, n_post_samples : int
        Passed through to posterior predictive draws for the credible band.
    """
    ds = result.dataset
    cf = result.classical
    idata = result.bayesian_idata
    rf = result.rf
    true_params = ds.true_params

    time_fine = np.linspace(0, 21, 220)

    # Point predictions
    classical_20 = deb_growth(time_fine, cf.params, 20.0)
    classical_25 = deb_growth(time_fine, cf.params, 25.0)
    true_20 = deb_growth(time_fine, true_params, 20.0)
    true_25 = deb_growth(time_fine, true_params, 25.0)

    # Bayesian credible band on the mean curve (no obs noise)
    band_20 = credible_band(
        idata, time_fine, temp_C=20.0,
        T_A=true_params.T_A, T_ref_C=true_params.T_ref_C,
        n_samples=n_post_samples, add_noise=False,
        random_seed=random_seed,
    )
    band_25 = credible_band(
        idata, time_fine, temp_C=25.0,
        T_A=true_params.T_A, T_ref_C=true_params.T_ref_C,
        n_samples=n_post_samples, add_noise=False,
        random_seed=random_seed + 1,
    )

    # Random Forest on fine grid
    rf_grid_20 = pd.DataFrame({"time": time_fine, "temp": 20.0})
    rf_grid_25 = pd.DataFrame({"time": time_fine, "temp": 25.0})
    rf_20 = rf.predict(rf_grid_20)
    rf_25 = rf.predict(rf_grid_25)

    fig = plt.figure(figsize=(9.0, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8], hspace=0.4, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[1, :])

    def _panel(ax, obs_df, time_fine, mean_band, lo_band, hi_band,
               classical_curve, rf_curve, true_curve, title):
        ax.fill_between(time_fine, lo_band, hi_band,
                        color=BAYES_COLOR, alpha=0.22, label="Bayesian 95% CI")
        ax.plot(time_fine, true_curve, linestyle="--", color=TRUE_COLOR,
                lw=1.0, label="True model")
        ax.plot(time_fine, classical_curve, color=CLASSICAL_COLOR,
                lw=1.5, label="Classical DEB")
        ax.plot(time_fine, mean_band, color=BAYES_COLOR,
                lw=1.5, label="Bayesian DEB (median)")
        ax.plot(time_fine, rf_curve, color=RF_COLOR,
                lw=1.5, label="Random Forest")
        ax.scatter(obs_df["time"], obs_df["L_obs"],
                   s=14, alpha=0.5, color="black", edgecolor="none",
                   label="Observations")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Body length (mm)")
        ax.set_title(title, fontweight="bold", fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)

    _panel(ax1, ds.train, time_fine, band_20["median"], band_20["lower"],
           band_20["upper"], classical_20, rf_20, true_20,
           "A) Interpolation (20 °C)")
    _panel(ax2, ds.test, time_fine, band_25["median"], band_25["lower"],
           band_25["upper"], classical_25, rf_25, true_25,
           "B) Extrapolation (25 °C)")
    ax1.legend(loc="lower right", fontsize=7, framealpha=0.9)

    # Panel C: RMSE bars
    plot_df = result.results_df.copy()
    tasks = ["Interpolation (20 C)", "Extrapolation (25 C)"]
    paradigms = ["Classical DEB", "Bayesian DEB", "Random Forest"]
    width = 0.35
    x = np.arange(len(paradigms))
    for i, task in enumerate(tasks):
        heights = [
            plot_df[(plot_df["Paradigm"] == p) & (plot_df["Task"] == task)]["RMSE"].iloc[0]
            for p in paradigms
        ]
        offset = (i - 0.5) * width
        ax3.bar(x + offset, heights, width=width,
                label=task, edgecolor="black", linewidth=0.5,
                color=("0.75" if i == 0 else "0.35"))
    ax3.set_xticks(x)
    ax3.set_xticklabels(paradigms)
    ax3.set_ylabel("RMSE (mm)")
    ax3.set_title("C) Performance comparison", fontweight="bold", fontsize=10, loc="left")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle("")
    fig.text(
        0.02, 0.01,
        "Blue: Classical DEB. Orange: Bayesian DEB (band = 95% credible interval). "
        "Green: Random Forest. Dashed grey: true generating model.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    output_dir = Path(output_dir)
    fig.savefig(output_dir / "benchmark_figure.pdf")
    fig.savefig(output_dir / "benchmark_figure.png", dpi=200)
    plt.close(fig)
