"""Paradigm-comparison figures: residuals by paradigm and task."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pydeb.plots.style import PARADIGM_COLORS, publication_style, save_figure


PRED_COLS = {
    "Classical DEB": "pred_classical",
    "Bayesian DEB":  "pred_bayesian",
    "Random Forest": "pred_rf",
}


def plot_residuals(
    benchmark_result,
    *,
    save_to: str | Path | None = None,
) -> plt.Figure:
    """2 × 3 grid of residual plots: rows are tasks (interp/extrap),
    columns are paradigms.

    Horizontal reference line at zero; axis limits are shared across all
    panels so that the Random-Forest drift at the extrapolation temperature
    is visually obvious relative to the two DEB paradigms.

    Parameters
    ----------
    benchmark_result : BenchmarkResult
        Output of :func:`debcompare.run_benchmark`. Must include the
        per-paradigm prediction columns in ``dataset.train`` and
        ``dataset.test``.
    save_to : path or None
    """
    ds = benchmark_result.dataset
    tasks = [
        ("Interpolation (20 °C)", ds.train),
        ("Extrapolation (25 °C)", ds.test),
    ]
    paradigms = list(PRED_COLS.keys())

    # Shared y-limits across panels (symmetric around zero) for fair visual
    # comparison.
    all_resid = []
    for _, df in tasks:
        for p, col in PRED_COLS.items():
            all_resid.append(df["L_obs"].to_numpy() - df[col].to_numpy())
    ylim = float(np.max(np.abs(np.concatenate(all_resid))) * 1.1)

    with publication_style():
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6),
                                 sharex=True, sharey=True)

        for i, (task_name, df) in enumerate(tasks):
            for j, paradigm in enumerate(paradigms):
                ax = axes[i, j]
                resid = df["L_obs"].to_numpy() - df[PRED_COLS[paradigm]].to_numpy()
                rmse = float(np.sqrt(np.mean(resid ** 2)))

                ax.axhline(0, color="0.3", lw=0.8, ls="--")
                ax.scatter(df["time"], resid,
                           color=PARADIGM_COLORS[paradigm],
                           s=28, alpha=0.7, edgecolor="white", linewidth=0.5)
                ax.text(0.03, 0.95, f"RMSE = {rmse:.3f} mm",
                        transform=ax.transAxes, fontsize=8, va="top",
                        bbox=dict(facecolor="white", edgecolor="0.5",
                                  alpha=0.85, pad=2))
                ax.set_ylim(-ylim, ylim)

                if i == 0:
                    ax.set_title(paradigm, fontsize=10, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(f"{task_name}\nresidual (mm)", fontsize=9)
                if i == 1:
                    ax.set_xlabel("Time (days)")

        fig.suptitle("Residuals by paradigm and task", y=1.00, fontsize=12,
                     fontweight="bold")
        fig.tight_layout()

    if save_to is not None:
        save_figure(fig, save_to)
    return fig
