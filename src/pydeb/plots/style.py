"""Consistent styling for pydeb publication-quality figures.

- Okabe-Ito colour palette (colour-blind safe, 8 colours).
- Serif font family, tuned line widths, tight default margins.
- ``save_figure`` writes PDF (vector) and PNG (raster) side by side.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt


# Okabe & Ito (2008) colour-blind-safe palette.
OKABE_ITO = {
    "black":    "#000000",
    "orange":   "#E69F00",
    "sky":      "#56B4E9",
    "green":    "#009E73",
    "yellow":   "#F0E442",
    "blue":     "#0072B2",
    "vermilion":"#D55E00",
    "purple":   "#CC79A7",
}

# Paradigm colour assignments used across figures.
PARADIGM_COLORS = {
    "Classical DEB":  OKABE_ITO["blue"],
    "Bayesian DEB":   OKABE_ITO["orange"],
    "Random Forest":  OKABE_ITO["green"],
    "True model":     "0.30",
    "Observations":   "0.15",
}


DEFAULT_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.3",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


@contextmanager
def publication_style() -> Iterator[None]:
    """Temporarily apply the pydeb plot style.

    Usage::

        with publication_style():
            fig, ax = plt.subplots()
            ...
    """
    with mpl.rc_context(DEFAULT_RC):
        yield


def apply_style() -> None:
    """Apply the pydeb plot style globally (persistent until reset)."""
    mpl.rcParams.update(DEFAULT_RC)


def save_figure(fig: plt.Figure, save_to: str | Path, *, formats: tuple[str, ...] = ("pdf", "png"),
                dpi: int = 200) -> list[Path]:
    """Save ``fig`` to one or more files.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to write.
    save_to : str or Path
        Base path *without* extension. ``"fig/arrhenius"`` yields
        ``"fig/arrhenius.pdf"`` and ``"fig/arrhenius.png"``.
    formats : tuple of str
        File extensions. Default ``("pdf", "png")``.
    dpi : int
        DPI for raster formats.

    Returns
    -------
    list of Path
        Paths actually written.
    """
    base = Path(save_to)
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        target = base.with_suffix(f".{ext}")
        fig.savefig(target, dpi=dpi)
        written.append(target)
    return written
