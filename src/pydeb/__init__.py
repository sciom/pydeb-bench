"""pydeb — Dynamic Energy Budget modelling in Python.

Companion software for Hackenberger & Djerdj (2026), Ecological Modelling.
"""

from pydeb.core.params import DEBParams
from pydeb.core.model import deb_growth, DEBGrowthModel
from pydeb.core.temperature import arrhenius_correction

__version__ = "0.1.0"

__all__ = [
    "DEBParams",
    "deb_growth",
    "DEBGrowthModel",
    "arrhenius_correction",
    "__version__",
]


def __getattr__(name):
    """Lazy access to plotting subpackage.

    ``pydeb.plots`` is only imported on demand so that headless users who
    never call a plot function do not pay the matplotlib import cost.
    """
    if name == "plots":
        from pydeb import plots as _plots
        return _plots
    raise AttributeError(f"module 'pydeb' has no attribute {name!r}")
