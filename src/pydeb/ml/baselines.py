"""Scikit-learn baselines for the benchmark.

Provides wrappers for Random Forest and Gradient Boosting regression of body
length on (time, temperature). Kept deliberately thin: the point of the
benchmark is that ML baselines need no mechanistic knowledge but pay for it
in extrapolation, not that the baselines are optimally tuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


FEATURES = ["time", "temp"]
TARGET = "L_obs"


@dataclass
class FitResult:
    """Container for a fitted ML baseline."""

    model: Any
    runtime_s: float
    feature_importances: dict[str, float]

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        X = _require_features(data)
        return self.model.predict(X)


def _require_features(data: pd.DataFrame) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data[FEATURES]


def fit_random_forest(
    data: pd.DataFrame,
    n_estimators: int = 500,
    random_state: int = 42,
    **kwargs,
) -> FitResult:
    """Fit a Random Forest regressor matching the R benchmark defaults.

    The R benchmark uses ``mtry = 1`` (one predictor tried per split), so the
    scikit-learn equivalent is ``max_features=1``.
    """
    import time

    X = _require_features(data)
    y = data[TARGET].to_numpy()

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=1,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )

    t0 = time.perf_counter()
    model.fit(X, y)
    runtime = time.perf_counter() - t0

    importances = dict(zip(FEATURES, (float(v) for v in model.feature_importances_)))
    return FitResult(model=model, runtime_s=runtime, feature_importances=importances)


def fit_gradient_boosting(
    data: pd.DataFrame,
    n_estimators: int = 500,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    random_state: int = 42,
    **kwargs,
) -> FitResult:
    """Fit a gradient boosting regressor. Not used in the default benchmark,
    but exposed for extensions and exploratory comparisons.
    """
    import time

    X = _require_features(data)
    y = data[TARGET].to_numpy()

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        **kwargs,
    )

    t0 = time.perf_counter()
    model.fit(X, y)
    runtime = time.perf_counter() - t0

    importances = dict(zip(FEATURES, (float(v) for v in model.feature_importances_)))
    return FitResult(model=model, runtime_s=runtime, feature_importances=importances)
