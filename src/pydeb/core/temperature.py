"""Arrhenius temperature correction for DEB rate parameters."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def arrhenius_correction(
    temp_C: ArrayLike,
    T_A: float = 8000.0,
    T_ref_C: float = 20.0,
) -> np.ndarray:
    """Arrhenius correction factor for a DEB rate parameter.

    The correction scales a rate measured at ``T_ref_C`` to the temperature
    ``temp_C``. The standard single-Arrhenius form is

        TC(T) = exp(T_A / T_ref - T_A / T),

    with both temperatures in kelvin.

    Parameters
    ----------
    temp_C : array_like
        Target temperature(s) in degrees Celsius.
    T_A : float
        Arrhenius temperature in kelvin. AmP default for Daphnia magna is 8000.
    T_ref_C : float
        Reference temperature in degrees Celsius (AmP default: 20 C).

    Returns
    -------
    ndarray
        Multiplicative correction factor, same shape as ``temp_C``.
    """
    T = np.asarray(temp_C, dtype=float) + 273.15
    T_ref = T_ref_C + 273.15
    return np.exp(T_A / T_ref - T_A / T)
