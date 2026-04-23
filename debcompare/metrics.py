"""Performance metrics: RMSE, R^2, MAE, 95% coverage.

Identical definitions to those in ``benchmark.R``."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rmse(obs: ArrayLike, pred: ArrayLike) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def r_squared(obs: ArrayLike, pred: ArrayLike) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae(obs: ArrayLike, pred: ArrayLike) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(obs - pred)))


def coverage_95(obs: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Fraction of observations falling inside [lower, upper]."""
    obs = np.asarray(obs, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (obs >= lower) & (obs <= upper)
    return float(np.mean(inside))
