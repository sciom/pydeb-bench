"""Synthetic *Daphnia magna* growth data generator for the benchmark.

Mirrors the data-generating process of the R ``benchmark.R`` script: true
DEB parameters at AmP values for *D. magna*, Gaussian observation noise with
a positivity floor, training at 20 C and a held-out extrapolation set at
25 C.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pydeb.core.model import deb_growth
from pydeb.core.params import DEBParams


@dataclass
class DaphniaDataset:
    """Training + test datasets as used in the paper benchmark."""

    train: pd.DataFrame
    test: pd.DataFrame
    true_params: DEBParams

    def combined(self) -> pd.DataFrame:
        out = pd.concat([self.train.assign(split="train"),
                         self.test.assign(split="test")],
                        ignore_index=True)
        return out


def simulate_daphnia_dataset(
    true_params: DEBParams | None = None,
    times: np.ndarray | None = None,
    n_rep: int = 5,
    temp_train_C: float = 20.0,
    temp_test_C: float = 25.0,
    positivity_floor: float = 0.3,
    random_seed: int = 42,
) -> DaphniaDataset:
    """Generate a synthetic *D. magna* growth dataset.

    Parameters
    ----------
    true_params : DEBParams
        Ground-truth DEB parameters. Default: AmP-derived for *D. magna*.
    times : array_like
        Observation time points (days). Default: ``np.arange(0, 22.5, 1.5)``.
    n_rep : int
        Number of replicates per time point.
    temp_train_C : float
        Temperature for the training set.
    temp_test_C : float
        Temperature for the held-out extrapolation set.
    positivity_floor : float
        Observations below this value are clamped upward. Matches the R script.
    random_seed : int
        Seed for noise draws.

    Returns
    -------
    DaphniaDataset
        Container with ``train``, ``test``, and ``true_params``.
    """
    if true_params is None:
        true_params = DEBParams.daphnia_magna()
    if times is None:
        times = np.arange(0.0, 22.5, 1.5)  # 15 points, 0..21 days

    rng = np.random.default_rng(random_seed)

    def _simulate(temp_C: float) -> pd.DataFrame:
        tt = np.repeat(times, n_rep)
        reps = np.tile(np.arange(1, n_rep + 1), len(times))
        L_true = deb_growth(tt, true_params, temp_C=temp_C)
        noise = rng.normal(0.0, true_params.sigma, size=tt.size)
        L_obs = np.maximum(L_true + noise, positivity_floor)
        return pd.DataFrame(
            {
                "time": tt,
                "rep": reps,
                "temp": temp_C,
                "L_true": L_true,
                "L_obs": L_obs,
            }
        )

    train = _simulate(temp_train_C)
    test = _simulate(temp_test_C)
    return DaphniaDataset(train=train, test=test, true_params=true_params)
