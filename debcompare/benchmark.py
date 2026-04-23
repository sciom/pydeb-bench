"""Orchestration: classical DEB vs Bayesian DEB vs Random Forest.

Reproduces Section 7 of Hackenberger & Djerdj (2026). Outputs a tidy results
DataFrame (compatible with ``benchmark_results.csv`` from ``benchmark.R``)
and writes a three-panel figure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pydeb.bayes import classical_fit, fit_growth
from pydeb.bayes.diagnostics import credible_band, posterior_predictive_growth, summarise
from pydeb.core.model import deb_growth
from pydeb.core.params import AMP_PRIORS, DEBParams
from pydeb.ml.baselines import fit_random_forest

from debcompare.data import DaphniaDataset, simulate_daphnia_dataset
from debcompare.metrics import coverage_95, mae, r_squared, rmse


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Everything produced by one benchmark run."""

    results_df: pd.DataFrame
    dataset: DaphniaDataset
    classical: Any
    bayesian_idata: Any
    bayesian_posterior_summary: pd.DataFrame
    rf: Any
    runtimes: dict[str, float] = field(default_factory=dict)

    def to_csv(self, path: str | Path) -> None:
        self.results_df.to_csv(path, index=False)


def _metrics_row(paradigm: str, task: str, obs, pred, *, runtime=None, coverage=None) -> dict:
    return {
        "Paradigm": paradigm,
        "Task": task,
        "RMSE": rmse(obs, pred),
        "R2": r_squared(obs, pred),
        "MAE": mae(obs, pred),
        "Coverage_95": coverage if coverage is not None else float("nan"),
        "Runtime_s": runtime if runtime is not None else float("nan"),
    }


def run_benchmark(
    *,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 3,
    target_accept: float = 0.9,
    n_post_samples: int = 500,
    random_seed: int = 42,
    output_dir: str | Path | None = None,
    save_figure: bool = True,
    progressbar: bool = False,
) -> BenchmarkResult:
    """Run the full three-paradigm benchmark end to end.

    Parameters
    ----------
    draws, tune, chains, target_accept : int / float
        NUTS sampler controls for Bayesian DEB.
    n_post_samples : int
        Number of posterior draws used to build predictive intervals.
    random_seed : int
        Seed controlling the synthetic dataset and all resampling.
    output_dir : path or None
        If given, writes ``benchmark_results.csv`` and, if ``save_figure``,
        ``benchmark_figure.pdf`` / ``benchmark_figure.png`` to this directory.
    save_figure : bool
        Whether to save the three-panel comparison figure.
    progressbar : bool
        Pass through to PyMC's sampler.

    Returns
    -------
    BenchmarkResult
    """
    logger.info("Simulating synthetic Daphnia magna dataset (seed=%d)", random_seed)
    dataset = simulate_daphnia_dataset(random_seed=random_seed)
    true_params = dataset.true_params
    train = dataset.train
    test = dataset.test

    runtimes: dict[str, float] = {}

    # ----- Classical DEB -----
    logger.info("Fitting classical DEB (Nelder-Mead)")
    cf = classical_fit(train, T_A=true_params.T_A, T_ref_C=true_params.T_ref_C)
    runtimes["classical"] = cf.runtime_s
    train = train.assign(pred_classical=deb_growth(train["time"], cf.params, train["temp"]))
    test = test.assign(pred_classical=deb_growth(test["time"], cf.params, test["temp"]))

    # ----- Bayesian DEB -----
    logger.info(
        "Sampling Bayesian DEB (NUTS, %d chains x %d draws, tune=%d)",
        chains, draws, tune,
    )
    import time as _time
    t_bayes_start = _time.perf_counter()
    idata = fit_growth(
        train,
        priors=AMP_PRIORS,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        T_A=true_params.T_A,
        T_ref_C=true_params.T_ref_C,
        progressbar=progressbar,
    )
    runtimes["bayesian"] = _time.perf_counter() - t_bayes_start
    post_summary = summarise(idata)

    # Point predictions = posterior means
    post = idata.posterior
    Linf_mean = float(post["Linf"].mean())
    rB_mean = float(post["rB"].mean())
    L0_mean = float(post["L0"].mean())
    params_bayes = DEBParams(
        Linf=Linf_mean, rB=rB_mean, L0=L0_mean,
        T_A=true_params.T_A, T_ref_C=true_params.T_ref_C,
    )
    train = train.assign(pred_bayesian=deb_growth(train["time"], params_bayes, train["temp"]))
    test = test.assign(pred_bayesian=deb_growth(test["time"], params_bayes, test["temp"]))

    # Posterior predictive (with observation noise) for coverage
    pred_train = posterior_predictive_growth(
        idata, train["time"].to_numpy(), temp_C=train["temp"].to_numpy(),
        T_A=true_params.T_A, T_ref_C=true_params.T_ref_C,
        n_samples=n_post_samples, add_noise=True, random_seed=random_seed,
    )
    pred_test = posterior_predictive_growth(
        idata, test["time"].to_numpy(), temp_C=test["temp"].to_numpy(),
        T_A=true_params.T_A, T_ref_C=true_params.T_ref_C,
        n_samples=n_post_samples, add_noise=True, random_seed=random_seed + 1,
    )
    train_lo, train_hi = np.quantile(pred_train, [0.025, 0.975], axis=0)
    test_lo, test_hi = np.quantile(pred_test, [0.025, 0.975], axis=0)
    cov_train = coverage_95(train["L_obs"], train_lo, train_hi)
    cov_test = coverage_95(test["L_obs"], test_lo, test_hi)
    train["bayes_lo"] = train_lo
    train["bayes_hi"] = train_hi
    test["bayes_lo"] = test_lo
    test["bayes_hi"] = test_hi

    # ----- Random Forest -----
    logger.info("Fitting Random Forest baseline")
    rf_res = fit_random_forest(train, random_state=random_seed)
    runtimes["rf"] = rf_res.runtime_s
    train = train.assign(pred_rf=rf_res.predict(train))
    test = test.assign(pred_rf=rf_res.predict(test))

    # ----- Assemble results -----
    rows = [
        _metrics_row("Classical DEB", "Interpolation (20 C)",
                     train["L_obs"], train["pred_classical"], runtime=runtimes["classical"]),
        _metrics_row("Classical DEB", "Extrapolation (25 C)",
                     test["L_obs"], test["pred_classical"]),
        _metrics_row("Bayesian DEB", "Interpolation (20 C)",
                     train["L_obs"], train["pred_bayesian"],
                     runtime=runtimes["bayesian"], coverage=cov_train),
        _metrics_row("Bayesian DEB", "Extrapolation (25 C)",
                     test["L_obs"], test["pred_bayesian"], coverage=cov_test),
        _metrics_row("Random Forest", "Interpolation (20 C)",
                     train["L_obs"], train["pred_rf"], runtime=runtimes["rf"]),
        _metrics_row("Random Forest", "Extrapolation (25 C)",
                     test["L_obs"], test["pred_rf"]),
    ]
    results_df = pd.DataFrame(rows)

    # Update dataset on returned container (mutate in place)
    dataset.train = train
    dataset.test = test

    result = BenchmarkResult(
        results_df=results_df,
        dataset=dataset,
        classical=cf,
        bayesian_idata=idata,
        bayesian_posterior_summary=post_summary,
        rf=rf_res,
        runtimes=runtimes,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "benchmark_results.csv")
        if save_figure:
            from debcompare.plotting import save_benchmark_figure

            save_benchmark_figure(
                result,
                output_dir=output_dir,
                random_seed=random_seed,
                n_post_samples=n_post_samples,
            )

    return result
