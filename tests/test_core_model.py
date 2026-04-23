"""Unit tests for the DEB forward model."""

from __future__ import annotations

import numpy as np
import pytest

from pydeb.core.model import deb_growth, DEBGrowthModel
from pydeb.core.params import DEBParams
from pydeb.core.temperature import arrhenius_correction


def test_arrhenius_correction_at_reference_is_one():
    assert arrhenius_correction(20.0, T_A=8000.0, T_ref_C=20.0) == pytest.approx(1.0)


def test_arrhenius_correction_monotonic_increasing():
    temps = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    TC = arrhenius_correction(temps, T_A=8000.0, T_ref_C=20.0)
    assert np.all(np.diff(TC) > 0)


def test_deb_growth_at_t0_equals_L0():
    params = DEBParams.daphnia_magna()
    assert deb_growth(0.0, params, temp_C=20.0) == pytest.approx(params.L0)


def test_deb_growth_asymptote_approaches_Linf():
    params = DEBParams.daphnia_magna()
    L_late = deb_growth(1000.0, params, temp_C=20.0)
    assert L_late == pytest.approx(params.Linf, rel=1e-6)


def test_deb_growth_monotonic_in_time():
    params = DEBParams.daphnia_magna()
    t = np.linspace(0, 21, 50)
    L = deb_growth(t, params, temp_C=20.0)
    assert np.all(np.diff(L) > 0)


def test_deb_growth_warmer_grows_faster_but_same_asymptote():
    params = DEBParams.daphnia_magna()
    t = np.linspace(0, 21, 50)
    L20 = deb_growth(t, params, temp_C=20.0)
    L25 = deb_growth(t, params, temp_C=25.0)
    # warmer grows faster
    assert np.all(L25 >= L20 - 1e-9)
    # same asymptote
    assert deb_growth(1e6, params, temp_C=25.0) == pytest.approx(params.Linf, rel=1e-6)


def test_deb_growth_model_wrapper_matches_function():
    params = DEBParams.daphnia_magna()
    model = DEBGrowthModel(params)
    t = np.array([0.0, 5.0, 10.0, 15.0])
    np.testing.assert_allclose(model.predict(t, 20.0), deb_growth(t, params, 20.0))
