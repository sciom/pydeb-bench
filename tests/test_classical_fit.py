"""Classical Nelder-Mead fit should recover true parameters on clean data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pydeb.bayes import classical_fit
from pydeb.core.model import deb_growth
from pydeb.core.params import DEBParams


def test_classical_fit_recovers_true_params_on_clean_data():
    true = DEBParams.daphnia_magna()
    times = np.linspace(0, 21, 15)
    data = pd.DataFrame({
        "time": times,
        "temp": 20.0,
        "L_obs": deb_growth(times, true, temp_C=20.0),
    })
    result = classical_fit(data, T_A=true.T_A, T_ref_C=true.T_ref_C)
    assert result.success
    assert result.params.Linf == pytest.approx(true.Linf, rel=1e-2)
    assert result.params.rB == pytest.approx(true.rB, rel=5e-2)
    assert result.params.L0 == pytest.approx(true.L0, rel=5e-2)


def test_classical_fit_handles_two_temperatures():
    true = DEBParams.daphnia_magna()
    times = np.linspace(0, 21, 15)
    frames = []
    for temp in (20.0, 25.0):
        frames.append(pd.DataFrame({
            "time": times,
            "temp": temp,
            "L_obs": deb_growth(times, true, temp_C=temp),
        }))
    data = pd.concat(frames, ignore_index=True)
    result = classical_fit(data, T_A=true.T_A, T_ref_C=true.T_ref_C)
    assert result.success
    assert result.params.Linf == pytest.approx(true.Linf, rel=1e-2)
