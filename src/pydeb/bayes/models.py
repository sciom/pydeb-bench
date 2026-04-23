"""PyMC models for Bayesian DEB growth calibration, plus a classical MLE baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.optimize import minimize

from pydeb.core.params import (
    AMP_PRIORS,
    DEBParams,
    LogNormalPrior,
    PriorSpec,
    UniformPrior,
)
from pydeb.core.temperature import arrhenius_correction


REQUIRED_COLUMNS = ("time", "temp", "L_obs")


def _require_columns(data: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}."
        )


def _prior_node(name: str, spec, model: pm.Model):
    """Build a PyMC prior variable from a LogNormalPrior or UniformPrior spec."""
    with model:
        if isinstance(spec, LogNormalPrior):
            return pm.LogNormal(name, mu=spec.mu_log, sigma=spec.sigma_log)
        if isinstance(spec, UniformPrior):
            return pm.Uniform(name, lower=spec.lower, upper=spec.upper)
    raise TypeError(f"Unsupported prior type for {name!r}: {type(spec).__name__}")


def build_growth_model(
    data: pd.DataFrame,
    priors: PriorSpec = AMP_PRIORS,
    T_A: float = 8000.0,
    T_ref_C: float = 20.0,
) -> pm.Model:
    """Build a PyMC model for Bayesian calibration of the simplified DEB model.

    The model is

        L_obs[i] ~ Normal( deb_growth(t[i], theta, temp[i]), sigma )

    with theta = (Linf, rB, L0) given LogNormal priors from ``priors`` and
    sigma a weakly-informative Uniform prior.

    The Arrhenius parameters ``T_A`` and ``T_ref_C`` are treated as known
    (fixed at the AmP default for *Daphnia magna*) to match the benchmark
    setup. Making them stochastic is a natural extension but drastically
    worsens identifiability without multi-temperature data.

    Parameters
    ----------
    data : DataFrame
        Observed growth data with columns ``time``, ``temp``, ``L_obs``.
    priors : PriorSpec
        Prior specification. Default is AmP-informed for Daphniidae.
    T_A, T_ref_C : float
        Arrhenius parameters (fixed).

    Returns
    -------
    pymc.Model
        The model; not yet sampled.
    """
    _require_columns(data)

    t = data["time"].to_numpy(dtype=float)
    temp_C = data["temp"].to_numpy(dtype=float)
    L_obs = data["L_obs"].to_numpy(dtype=float)

    TC = arrhenius_correction(temp_C, T_A=T_A, T_ref_C=T_ref_C)

    model = pm.Model()
    Linf = _prior_node("Linf", priors.Linf, model)
    rB = _prior_node("rB", priors.rB, model)
    L0 = _prior_node("L0", priors.L0, model)
    sigma = _prior_node("sigma", priors.sigma, model)

    with model:
        L_pred = Linf - (Linf - L0) * pt.exp(-rB * TC * t)
        pm.Normal("obs", mu=L_pred, sigma=sigma, observed=L_obs)

    return model


def fit_growth(
    data: pd.DataFrame,
    priors: PriorSpec = AMP_PRIORS,
    *,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 3,
    target_accept: float = 0.9,
    random_seed: int = 42,
    T_A: float = 8000.0,
    T_ref_C: float = 20.0,
    progressbar: bool = False,
):
    """Fit the Bayesian DEB growth model and return an ArviZ InferenceData.

    This is the turnkey entry point. It accepts any DataFrame with the three
    required columns, builds the model with AmP-informed priors, runs NUTS,
    and returns posterior samples.

    Parameters
    ----------
    data : DataFrame
        Must contain ``time``, ``temp``, ``L_obs``.
    priors : PriorSpec
        Prior specification; see ``pydeb.core.params.AMP_PRIORS``.
    draws, tune, chains : int
        NUTS sampling parameters.
    target_accept : float
        NUTS target acceptance rate; increase toward 0.95 if divergences appear.
    random_seed : int
        Seed for reproducibility.
    T_A, T_ref_C : float
        Fixed Arrhenius parameters.
    progressbar : bool
        Show PyMC's progress bar.

    Returns
    -------
    arviz.InferenceData
        Posterior and sample stats.
    """
    model = build_growth_model(data, priors=priors, T_A=T_A, T_ref_C=T_ref_C)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )
    return idata


# ---------------------------------------------------------------------------
# Classical (non-Bayesian) baseline
# ---------------------------------------------------------------------------


@dataclass
class ClassicalFit:
    """Result of maximum-likelihood / least-squares fit."""

    params: DEBParams
    runtime_s: float
    success: bool
    message: str
    nfev: int

    def summary(self) -> str:
        p = self.params
        return (
            f"Linf = {p.Linf:.3f} mm, rB = {p.rB:.4f} d^-1, L0 = {p.L0:.3f} mm "
            f"(runtime {self.runtime_s:.3f} s, nfev={self.nfev})"
        )


def classical_fit(
    data: pd.DataFrame,
    *,
    initial: DEBParams | None = None,
    T_A: float = 8000.0,
    T_ref_C: float = 20.0,
    max_iter: int = 5000,
) -> ClassicalFit:
    """Nelder--Mead least-squares fit of (Linf, rB, L0) to a growth dataset.

    Same objective as the R benchmark: sum of squared residuals between the
    analytic DEB growth curve and observed lengths. Temperature is applied
    per row via Arrhenius correction.
    """
    _require_columns(data)

    t = data["time"].to_numpy(dtype=float)
    temp_C = data["temp"].to_numpy(dtype=float)
    L_obs = data["L_obs"].to_numpy(dtype=float)

    TC = arrhenius_correction(temp_C, T_A=T_A, T_ref_C=T_ref_C)

    def objective(theta: np.ndarray) -> float:
        Linf, rB, L0 = theta
        if Linf <= 0 or rB <= 0 or L0 <= 0 or L0 >= Linf:
            return 1e12
        pred = Linf - (Linf - L0) * np.exp(-rB * TC * t)
        return float(np.sum((L_obs - pred) ** 2))

    if initial is None:
        x0 = np.array([5.0, 0.12, 0.9])
    else:
        x0 = np.array([initial.Linf, initial.rB, initial.L0])

    t0 = time.perf_counter()
    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-6, "fatol": 1e-8})
    runtime = time.perf_counter() - t0

    params = DEBParams(
        Linf=float(res.x[0]),
        rB=float(res.x[1]),
        L0=float(res.x[2]),
        T_A=T_A,
        T_ref_C=T_ref_C,
    )
    return ClassicalFit(
        params=params,
        runtime_s=runtime,
        success=bool(res.success),
        message=str(res.message),
        nfev=int(res.nfev),
    )
