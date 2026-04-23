"""Post-sampling diagnostics and posterior predictive utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

import arviz as az

from pydeb.core.temperature import arrhenius_correction


PARAM_NAMES = ("Linf", "rB", "L0", "sigma")


def summarise(idata: az.InferenceData, var_names=PARAM_NAMES) -> pd.DataFrame:
    """Return an ArviZ summary (mean, sd, hdi, r-hat, ess) for DEB parameters."""
    return az.summary(idata, var_names=list(var_names), round_to=4)


def posterior_predictive_growth(
    idata: az.InferenceData,
    t: ArrayLike,
    temp_C: float | ArrayLike = 20.0,
    T_A: float = 8000.0,
    T_ref_C: float = 20.0,
    n_samples: int = 500,
    add_noise: bool = True,
    random_seed: int | None = 0,
) -> np.ndarray:
    """Draw posterior predictive samples of body length on a (t, temp_C) grid.

    Parameters
    ----------
    idata : InferenceData
        Output of :func:`pydeb.bayes.fit_growth`.
    t : array_like
        Time points (days), 1-D.
    temp_C : float or array_like
        Temperature(s) in C. Scalar is broadcast over ``t``; array must
        match length of ``t``.
    T_A, T_ref_C : float
        Arrhenius parameters (must match the model used to fit).
    n_samples : int
        Number of posterior draws to use.
    add_noise : bool
        If True (default) adds Gaussian observation noise with the posterior
        draws of ``sigma`` (posterior predictive for new observations). If
        False, returns the latent curve draws (useful for a credible band on
        the mean curve).
    random_seed : int or None
        Seed for the observation-noise draws.

    Returns
    -------
    ndarray of shape (n_samples, len(t))
        Predictive samples.
    """
    t_arr = np.asarray(t, dtype=float)
    TC = arrhenius_correction(temp_C, T_A=T_A, T_ref_C=T_ref_C)
    TC_arr = np.broadcast_to(np.asarray(TC, dtype=float), t_arr.shape)

    post = idata.posterior
    # Flatten chain x draw into a single sample axis.
    Linf = post["Linf"].values.reshape(-1)
    rB = post["rB"].values.reshape(-1)
    L0 = post["L0"].values.reshape(-1)
    sigma = post["sigma"].values.reshape(-1)

    total = Linf.size
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(total, size=min(n_samples, total), replace=False)

    # shape (n_samples, len(t))
    Linf_s = Linf[idx][:, None]
    rB_s = rB[idx][:, None]
    L0_s = L0[idx][:, None]
    sigma_s = sigma[idx][:, None]

    mean_curves = Linf_s - (Linf_s - L0_s) * np.exp(-rB_s * TC_arr[None, :] * t_arr[None, :])

    if add_noise:
        noise = rng.normal(loc=0.0, scale=sigma_s, size=mean_curves.shape)
        return mean_curves + noise
    return mean_curves


def credible_band(
    idata: az.InferenceData,
    t: ArrayLike,
    temp_C: float | ArrayLike = 20.0,
    T_A: float = 8000.0,
    T_ref_C: float = 20.0,
    level: float = 0.95,
    n_samples: int = 500,
    add_noise: bool = False,
    random_seed: int | None = 0,
) -> dict[str, np.ndarray]:
    """Convenience wrapper returning median and lower/upper quantiles."""
    samples = posterior_predictive_growth(
        idata,
        t,
        temp_C=temp_C,
        T_A=T_A,
        T_ref_C=T_ref_C,
        n_samples=n_samples,
        add_noise=add_noise,
        random_seed=random_seed,
    )
    alpha = (1.0 - level) / 2.0
    return {
        "median": np.quantile(samples, 0.5, axis=0),
        "lower": np.quantile(samples, alpha, axis=0),
        "upper": np.quantile(samples, 1.0 - alpha, axis=0),
    }
