"""pydeb.bayes — turnkey Bayesian DEB estimation built on PyMC.

Public API
----------
``fit_growth``
    End-to-end Bayesian calibration of the simplified DEB growth model from a
    time-series of body length observations. Defaults to AmP-informed priors.

``build_growth_model``
    Lower-level access: returns the PyMC ``Model`` object so users can add
    custom priors, likelihoods, or composite multi-dataset fits.

``classical_fit``
    Maximum-likelihood / least-squares calibration of the same growth model,
    used as the "classical DEB" arm of the benchmark.

``summarise``
    Thin wrapper around ``arviz.summary`` with the parameters the simplified
    DEB model uses.

``posterior_predictive_growth``
    Generate posterior predictive samples on an arbitrary ``(t, temp_C)``
    grid; used for credible bands in the benchmark figure.
"""

from pydeb.bayes.models import (
    build_growth_model,
    fit_growth,
    classical_fit,
    ClassicalFit,
)
from pydeb.bayes.diagnostics import summarise, posterior_predictive_growth

__all__ = [
    "build_growth_model",
    "fit_growth",
    "classical_fit",
    "ClassicalFit",
    "summarise",
    "posterior_predictive_growth",
]
