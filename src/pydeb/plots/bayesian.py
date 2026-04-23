"""Bayesian-workflow figures: prior vs posterior, joint posterior, MCMC diagnostics."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pydeb.core.params import AMP_PRIORS, LogNormalPrior, PriorSpec, UniformPrior
from pydeb.plots.style import OKABE_ITO, publication_style, save_figure


PARAM_LABELS = {
    "Linf":  r"$L_\infty$ (mm)",
    "rB":    r"$r_B$ (d$^{-1}$)",
    "L0":    r"$L_0$ (mm)",
    "sigma": r"$\sigma$ (mm)",
}


def _prior_pdf(spec, x: np.ndarray) -> np.ndarray:
    """Return the density of a ``LogNormalPrior`` or ``UniformPrior`` on grid ``x``."""
    if isinstance(spec, LogNormalPrior):
        # scipy.stats.lognorm uses shape=sigma, scale=exp(mu).
        return stats.lognorm.pdf(x, s=spec.sigma_log, scale=np.exp(spec.mu_log))
    if isinstance(spec, UniformPrior):
        width = spec.upper - spec.lower
        return np.where((x >= spec.lower) & (x <= spec.upper), 1.0 / width, 0.0)
    raise TypeError(f"Unsupported prior: {type(spec).__name__}")


def _prior_range(spec, posterior_samples: np.ndarray) -> tuple[float, float]:
    """Pick a sensible x-range that covers prior mass and the posterior."""
    p_lo, p_hi = np.quantile(posterior_samples, [0.001, 0.999])
    if isinstance(spec, LogNormalPrior):
        lo = min(p_lo, float(stats.lognorm.ppf(0.02, s=spec.sigma_log,
                                               scale=np.exp(spec.mu_log))))
        hi = max(p_hi, float(stats.lognorm.ppf(0.98, s=spec.sigma_log,
                                               scale=np.exp(spec.mu_log))))
    elif isinstance(spec, UniformPrior):
        lo = min(p_lo, spec.lower)
        hi = max(p_hi, spec.upper)
    else:
        lo, hi = p_lo, p_hi
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def plot_prior_posterior(
    idata,
    *,
    priors: PriorSpec = AMP_PRIORS,
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Overlay AmP-informed priors on the marginal posteriors.

    Shows at a glance how much the likelihood has updated each parameter:
    tight, shifted posteriors imply well-identified parameters; near-prior
    posteriors imply weak data information.
    """
    param_names = list(PARAM_LABELS.keys())
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.8))
        for ax, name in zip(axes.flat, param_names):
            post = idata.posterior[name].values.reshape(-1)
            spec = getattr(priors, name)
            lo, hi = _prior_range(spec, post)
            grid = np.linspace(lo, hi, 400)

            # Posterior KDE.
            kde = stats.gaussian_kde(post)
            pdf_post = kde(grid)

            # Prior PDF.
            pdf_prior = _prior_pdf(spec, grid)

            ax.fill_between(grid, pdf_post, color=OKABE_ITO["orange"],
                            alpha=0.35, label="Posterior")
            ax.plot(grid, pdf_post, color=OKABE_ITO["orange"], lw=1.8)
            ax.plot(grid, pdf_prior, color=OKABE_ITO["blue"], lw=1.6,
                    ls="--", label="Prior")

            post_mean = float(np.mean(post))
            ax.axvline(post_mean, color="0.3", lw=0.8, ls=":")

            ax.set_title(name)
            ax.set_xlabel(PARAM_LABELS[name])
            ax.set_ylabel("Density")
            ax.set_xlim(lo, hi)

        axes[0, 0].legend(loc="upper right", fontsize=8)
        fig.suptitle("Prior vs. posterior", y=1.00, fontsize=12,
                     fontweight="bold")
        fig.tight_layout()

    if save_to is not None:
        save_figure(fig, save_to)
    return fig


def plot_posterior_corner(
    idata,
    *,
    var_names: tuple[str, ...] = ("Linf", "rB", "L0", "sigma"),
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Pairwise posterior plot: joint densities + marginal histograms.

    The diagonal holds marginal histograms; the lower triangle shows 2D
    hex-bin densities. Parameter correlations reveal identifiability
    structure.
    """
    n = len(var_names)
    samples = {name: idata.posterior[name].values.reshape(-1) for name in var_names}

    with publication_style():
        fig, axes = plt.subplots(n, n, figsize=(2.0 * n, 2.0 * n))

        for i, name_i in enumerate(var_names):
            for j, name_j in enumerate(var_names):
                ax = axes[i, j]
                if i == j:
                    ax.hist(samples[name_i], bins=40,
                            color=OKABE_ITO["orange"], alpha=0.85,
                            edgecolor="white", linewidth=0.3)
                    mean = float(np.mean(samples[name_i]))
                    ax.axvline(mean, color="0.2", lw=0.8, ls=":")
                    if i == 0:
                        ax.set_ylabel("Count", fontsize=8)
                    ax.grid(alpha=0.25)
                elif j < i:
                    ax.hexbin(samples[name_j], samples[name_i],
                              gridsize=30, cmap="viridis",
                              mincnt=1, linewidths=0.0)
                    r = float(np.corrcoef(samples[name_j], samples[name_i])[0, 1])
                    ax.text(0.05, 0.93, f"r = {r:+.2f}",
                            transform=ax.transAxes, fontsize=8,
                            bbox=dict(facecolor="white", edgecolor="0.5",
                                      alpha=0.85, pad=2))
                    ax.grid(alpha=0.2)
                else:
                    ax.axis("off")
                    continue

                if i == n - 1:
                    ax.set_xlabel(PARAM_LABELS[name_j], fontsize=9)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if j == 0 and i != 0:
                    ax.set_ylabel(PARAM_LABELS[name_i], fontsize=9)
                elif j != 0:
                    ax.tick_params(axis="y", labelleft=False)

        fig.suptitle("Posterior joint distribution", y=0.995, fontsize=12,
                     fontweight="bold")
        fig.tight_layout()

    if save_to is not None:
        save_figure(fig, save_to)
    return fig


def plot_trace_and_rank(
    idata,
    *,
    var_names: tuple[str, ...] = ("Linf", "rB", "L0", "sigma"),
    save_to: str | Path | None = None,
) -> plt.Figure:
    """MCMC diagnostics: trace plots (left) + rank histograms (right).

    Trace plots reveal chain stationarity and mixing; rank histograms
    (Vehtari et al. 2021) should look uniform for converged chains.
    """
    n_params = len(var_names)

    with publication_style():
        fig, axes = plt.subplots(n_params, 2, figsize=(9.5, 2.1 * n_params))
        for i, name in enumerate(var_names):
            trace_ax = axes[i, 0]
            rank_ax = axes[i, 1]

            # Trace: one line per chain.
            da = idata.posterior[name]  # dims: (chain, draw)
            n_chains, n_draws = da.shape
            x = np.arange(n_draws)
            for c in range(n_chains):
                trace_ax.plot(x, da.values[c],
                              lw=0.6, alpha=0.75,
                              label=f"chain {c}")
            trace_ax.set_title(f"Trace: {name}", fontsize=10,
                               fontweight="bold", loc="left")
            trace_ax.set_xlabel("Draw")
            trace_ax.set_ylabel(PARAM_LABELS[name])
            if i == 0:
                trace_ax.legend(fontsize=7, loc="upper right", ncol=n_chains)

            # Rank plot via arviz.
            az.plot_rank(idata, var_names=[name], kind="bars", ax=rank_ax)
            rank_ax.set_title(f"Rank: {name}", fontsize=10,
                              fontweight="bold", loc="left")

        fig.suptitle("MCMC diagnostics", y=1.00, fontsize=12, fontweight="bold")
        fig.tight_layout()

    if save_to is not None:
        save_figure(fig, save_to)
    return fig
