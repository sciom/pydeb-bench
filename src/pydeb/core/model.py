"""DEB forward model: Arrhenius-corrected von Bertalanffy growth in structural length.

This is the same simplified DEB model used in the benchmark of Hackenberger &
Djerdj (2026), Section 7. It preserves the two features that distinguish DEB
predictions from empirical growth curves: (1) a physiologically meaningful
temperature correction, and (2) parameters that can be cross-species compared
against AmP entries.

The full standard DEB model (with explicit reserve E, structural V, maturity
E_H, and reproduction buffer E_R) is intentionally not implemented here; that
is left to a future ``pydeb.standard`` submodule.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from pydeb.core.params import DEBParams
from pydeb.core.temperature import arrhenius_correction


def deb_growth(
    t: ArrayLike,
    params: Union[DEBParams, dict],
    temp_C: float | ArrayLike = 20.0,
) -> np.ndarray:
    """Structural length at time ``t`` under constant temperature ``temp_C``.

    Implements the analytic solution to

        dL/dt = rB(T) * (Linf - L),      L(0) = L0,

    giving L(t) = Linf - (Linf - L0) * exp(-rB(T) t), with rB(T) = rB * TC(T)
    where TC(T) is the Arrhenius correction.

    Parameters
    ----------
    t : array_like
        Time points in days. Scalar or 1-D array.
    params : DEBParams or dict
        DEB parameters. A dict is accepted to stay close to the R interface.
    temp_C : float or array_like
        Temperature in degrees Celsius. If array, must broadcast with ``t``.

    Returns
    -------
    ndarray
        Structural length in mm.
    """
    if isinstance(params, dict):
        params = DEBParams(**{k: params[k] for k in params if k in DEBParams.__annotations__})

    TC = arrhenius_correction(temp_C, T_A=params.T_A, T_ref_C=params.T_ref_C)
    rB_T = params.rB * TC
    t_arr = np.asarray(t, dtype=float)
    return params.Linf - (params.Linf - params.L0) * np.exp(-rB_T * t_arr)


class DEBGrowthModel:
    """Thin object wrapper around :func:`deb_growth` for pipeline use.

    Keeps parameters and offers ``predict`` with sklearn-like signature so the
    classical DEB paradigm and the ML baselines share a unified call pattern
    in :mod:`debcompare`.
    """

    def __init__(self, params: DEBParams | None = None):
        self.params = params if params is not None else DEBParams.daphnia_magna()

    def predict(self, t: ArrayLike, temp_C: float | ArrayLike = 20.0) -> np.ndarray:
        return deb_growth(t, self.params, temp_C=temp_C)

    def __repr__(self) -> str:
        p = self.params
        return (
            f"DEBGrowthModel(Linf={p.Linf:.3f}, rB={p.rB:.4f}, "
            f"L0={p.L0:.3f}, T_A={p.T_A:.0f}, T_ref_C={p.T_ref_C:.1f})"
        )
