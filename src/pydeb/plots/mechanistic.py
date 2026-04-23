"""Mechanistic-model-oriented figures.

- ``plot_arrhenius`` — the Arrhenius temperature correction curve with
  training/test bands and annotated extrapolation zones.
- ``plot_posterior_predictive_fan`` — predicted growth curves at several
  temperatures with credible bands, showing mechanistic extrapolation
  beyond the training domain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from pydeb.core.params import DEBParams
from pydeb.core.temperature import arrhenius_correction
from pydeb.plots.style import OKABE_ITO, PARADIGM_COLORS, publication_style, save_figure


def plot_arrhenius(
    params: DEBParams | None = None,
    *,
    temp_range_C: tuple[float, float] = (5.0, 35.0),
    train_temp_C: float = 20.0,
    test_temp_C: float = 25.0,
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Visualise the Arrhenius temperature correction factor.

    The plot shows :math:`TC(T) = \\exp(T_A/T_{\\mathrm{ref}} - T_A/T)` as a
    function of temperature, with the training temperature, test
    temperature, and full extrapolation zone highlighted.

    Parameters
    ----------
    params : DEBParams
        Uses ``params.T_A`` and ``params.T_ref_C``. Defaults to AmP *Daphnia
        magna* values.
    temp_range_C : tuple
        X-axis range in degrees Celsius.
    train_temp_C, test_temp_C : float
        Temperatures to annotate.
    save_to : path or None
        If given, writes PDF + PNG via :func:`save_figure`.
    """
    if params is None:
        params = DEBParams.daphnia_magna()

    temps = np.linspace(*temp_range_C, 300)
    TC = arrhenius_correction(temps, T_A=params.T_A, T_ref_C=params.T_ref_C)

    with publication_style():
        fig, ax = plt.subplots(figsize=(7.0, 4.2))

        # Extrapolation zone (above training temperature).
        ax.axvspan(
            train_temp_C, temp_range_C[1],
            color=OKABE_ITO["vermilion"], alpha=0.08, zorder=0,
            label="Extrapolation zone",
        )

        ax.plot(temps, TC, color=OKABE_ITO["blue"], lw=2.0,
                label=r"$TC(T)=\exp(T_A/T_{\mathrm{ref}} - T_A/T)$")

        # Reference line at TC = 1.
        ax.axhline(1.0, color="0.4", ls=":", lw=0.8)

        # Training point.
        TC_train = float(arrhenius_correction(train_temp_C,
                                              T_A=params.T_A, T_ref_C=params.T_ref_C))
        ax.plot(train_temp_C, TC_train, "o", color=OKABE_ITO["blue"],
                markersize=8, zorder=5)
        ax.annotate(
            f"Training\n{train_temp_C:.0f} °C\nTC = {TC_train:.2f}",
            xy=(train_temp_C, TC_train), xytext=(train_temp_C - 6, TC_train + 0.4),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7),
        )

        # Test point.
        TC_test = float(arrhenius_correction(test_temp_C,
                                             T_A=params.T_A, T_ref_C=params.T_ref_C))
        ax.plot(test_temp_C, TC_test, "D", color=OKABE_ITO["vermilion"],
                markersize=8, zorder=5)
        ax.annotate(
            f"Test (extrap.)\n{test_temp_C:.0f} °C\nTC = {TC_test:.2f}",
            xy=(test_temp_C, TC_test), xytext=(test_temp_C + 3.5, TC_test - 0.5),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7),
        )

        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Rate correction factor  TC(T)")
        ax.set_title("Arrhenius temperature correction")
        ax.set_xlim(temp_range_C)

        # Parameter annotation.
        ax.text(
            0.02, 0.96,
            f"$T_A$ = {params.T_A:.0f} K\n"
            f"$T_{{\\mathrm{{ref}}}}$ = {params.T_ref_C:.0f} °C",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", edgecolor="0.5", alpha=0.9, pad=4),
        )

        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()

    if save_to is not None:
        save_figure(fig, save_to)
    return fig


def plot_posterior_predictive_fan(
    idata,
    *,
    params: DEBParams | None = None,
    temperatures_C: Sequence[float] = (10.0, 15.0, 20.0, 25.0, 30.0),
    t_max: float = 21.0,
    n_samples: int = 400,
    train_data=None,
    train_temp_C: float = 20.0,
    save_to: str | Path | None = None,
    random_seed: int = 0,
) -> plt.Figure:
    """Fan of posterior predictive growth curves across temperatures.

    Each temperature contributes a median curve with a 95 % credible band,
    showing how the mechanistic DEB model extrapolates beyond the training
    temperature thanks to the Arrhenius correction.

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior from :func:`pydeb.bayes.fit_growth`.
    params : DEBParams
        Provides fixed ``T_A`` and ``T_ref_C``. Defaults to AmP *D. magna*.
    temperatures_C : sequence of float
        Temperatures for which to draw credible bands.
    t_max : float
        Plot time horizon (days).
    n_samples : int
        Number of posterior draws per temperature.
    train_data : DataFrame or None
        Optional training observations to overlay. Expected columns
        ``time`` and ``L_obs``.
    train_temp_C : float
        Temperature at which training observations were collected.
    save_to : path or None
    random_seed : int
    """
    from pydeb.bayes.diagnostics import credible_band

    if params is None:
        params = DEBParams.daphnia_magna()
    t = np.linspace(0, t_max, 200)

    with publication_style():
        fig, ax = plt.subplots(figsize=(7.5, 4.8))

        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=min(temperatures_C), vmax=max(temperatures_C))

        for temp in temperatures_C:
            band = credible_band(
                idata, t, temp_C=temp,
                T_A=params.T_A, T_ref_C=params.T_ref_C,
                n_samples=n_samples, add_noise=False,
                random_seed=random_seed,
            )
            color = cmap(norm(temp))
            ax.fill_between(t, band["lower"], band["upper"],
                            color=color, alpha=0.18)
            ax.plot(t, band["median"], color=color, lw=1.8,
                    label=f"{temp:.0f} °C")

        # Overlay training data if provided.
        if train_data is not None:
            ax.scatter(
                train_data["time"], train_data["L_obs"],
                s=22, color="black", zorder=5,
                label=f"Training data ({train_temp_C:.0f} °C)",
                edgecolor="white", linewidth=0.7,
            )

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Body length (mm)")
        ax.set_title("Posterior predictive growth across temperatures")
        ax.set_xlim(0, t_max)

        leg = ax.legend(title="Temperature", loc="lower right", fontsize=8,
                        ncol=2)
        leg.get_title().set_fontsize(8)
        fig.tight_layout()

    if save_to is not None:
        save_figure(fig, save_to)
    return fig
