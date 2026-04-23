"""pydeb.plots — publication-quality educational figures.

Each plot function returns a matplotlib ``Figure`` (so it can be shown
interactively with ``plt.show()`` or embedded in a notebook) and also accepts
an optional ``save_to`` argument. If given, the figure is written to
``<save_to>.pdf`` and ``<save_to>.png`` via :func:`save_figure`.

Gallery
-------
``plot_arrhenius``
    Arrhenius temperature correction curve with training/extrapolation annotations.
``plot_posterior_predictive_fan``
    Predicted growth curves across several temperatures with 95 % credible bands.
``plot_prior_posterior``
    Prior vs posterior overlay for the four DEB parameters.
``plot_posterior_corner``
    Joint posterior: histograms on the diagonal, 2D hex-bin densities below.
``plot_residuals``
    Residuals by paradigm and task (2 × 3 grid) with a shared y-axis.
``plot_trace_and_rank``
    MCMC diagnostic: trace and rank histograms.

``render_gallery``
    Convenience wrapper that runs all six plots and writes a directory of
    PDF/PNG outputs.
"""

from pydeb.plots.bayesian import (
    plot_posterior_corner,
    plot_prior_posterior,
    plot_trace_and_rank,
)
from pydeb.plots.comparison import plot_residuals
from pydeb.plots.mechanistic import plot_arrhenius, plot_posterior_predictive_fan
from pydeb.plots.style import (
    OKABE_ITO,
    PARADIGM_COLORS,
    apply_style,
    publication_style,
    save_figure,
)


__all__ = [
    "plot_arrhenius",
    "plot_posterior_predictive_fan",
    "plot_prior_posterior",
    "plot_posterior_corner",
    "plot_residuals",
    "plot_trace_and_rank",
    "render_gallery",
    # style helpers
    "OKABE_ITO",
    "PARADIGM_COLORS",
    "apply_style",
    "publication_style",
    "save_figure",
]


def render_gallery(benchmark_result, output_dir, *, random_seed: int = 0) -> dict:
    """Render all six plots in one call.

    Parameters
    ----------
    benchmark_result : debcompare.BenchmarkResult
        The full benchmark output.
    output_dir : path
        Directory to write the plot files. Created if missing.
    random_seed : int
        Seed for posterior predictive resampling in the predictive fan.

    Returns
    -------
    dict
        Mapping plot name → list of written paths.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, list] = {}

    fig = plot_arrhenius(benchmark_result.dataset.true_params,
                         save_to=output_dir / "01_arrhenius")
    written["arrhenius"] = [output_dir / "01_arrhenius.pdf",
                            output_dir / "01_arrhenius.png"]
    plt.close(fig)

    fig = plot_posterior_predictive_fan(
        benchmark_result.bayesian_idata,
        params=benchmark_result.dataset.true_params,
        train_data=benchmark_result.dataset.train,
        save_to=output_dir / "02_predictive_fan",
        random_seed=random_seed,
    )
    written["predictive_fan"] = [output_dir / "02_predictive_fan.pdf",
                                 output_dir / "02_predictive_fan.png"]
    plt.close(fig)

    fig = plot_prior_posterior(benchmark_result.bayesian_idata,
                               save_to=output_dir / "03_prior_posterior")
    written["prior_posterior"] = [output_dir / "03_prior_posterior.pdf",
                                  output_dir / "03_prior_posterior.png"]
    plt.close(fig)

    fig = plot_posterior_corner(benchmark_result.bayesian_idata,
                                save_to=output_dir / "04_posterior_corner")
    written["posterior_corner"] = [output_dir / "04_posterior_corner.pdf",
                                   output_dir / "04_posterior_corner.png"]
    plt.close(fig)

    fig = plot_residuals(benchmark_result,
                         save_to=output_dir / "05_residuals")
    written["residuals"] = [output_dir / "05_residuals.pdf",
                            output_dir / "05_residuals.png"]
    plt.close(fig)

    fig = plot_trace_and_rank(benchmark_result.bayesian_idata,
                              save_to=output_dir / "06_trace_rank")
    written["trace_rank"] = [output_dir / "06_trace_rank.pdf",
                             output_dir / "06_trace_rank.png"]
    plt.close(fig)

    return written
