"""Metric function tests."""

from __future__ import annotations

import numpy as np
import pytest

from debcompare.metrics import coverage_95, mae, r_squared, rmse


def test_rmse_zero_on_perfect_predictions():
    obs = np.array([1.0, 2.0, 3.0])
    assert rmse(obs, obs) == 0.0


def test_r_squared_one_on_perfect_predictions():
    obs = np.array([1.0, 2.0, 3.0])
    assert r_squared(obs, obs) == pytest.approx(1.0)


def test_r_squared_below_zero_for_bad_predictions():
    obs = np.array([1.0, 2.0, 3.0])
    pred = np.array([10.0, 10.0, 10.0])
    assert r_squared(obs, pred) < 0


def test_mae_matches_hand_computation():
    obs = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 1.5, 4.0])
    assert mae(obs, pred) == pytest.approx((0.5 + 0.5 + 1.0) / 3)


def test_coverage_95_handles_all_inside():
    obs = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([10.0, 10.0, 10.0])
    assert coverage_95(obs, lo, hi) == 1.0


def test_coverage_95_handles_none_inside():
    obs = np.array([1.0, 2.0, 3.0])
    lo = np.array([5.0, 5.0, 5.0])
    hi = np.array([10.0, 10.0, 10.0])
    assert coverage_95(obs, lo, hi) == 0.0
