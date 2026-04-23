# pydeb-bench

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sciom/pydeb-bench/main?labpath=notebooks/binder_demo.ipynb)
[![DOI](https://zenodo.org/badge/1219074061.svg)](https://doi.org/10.5281/zenodo.19709804)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Python companion software for:

> Hackenberger, B.K. & Djerdj, T. (2026). *Mechanistic, Bayesian, and Machine
> Learning Approaches to Metabolic Modelling: A Critical Review of Dynamic
> Energy Budget Theory and Its Alternatives.* Ecological Modelling.

> Click the **launch binder** badge above to run the interactive demo in your
> browser with no installation — it reproduces the paper's three-paradigm
> comparison and four of the six educational figures in about 30 seconds.

Two packages, one repository:

- **`pydeb`** — Dynamic Energy Budget forward model and a turnkey Bayesian
  estimation workflow built on PyMC, with AmP-database-informed priors.
- **`debcompare`** — a reproducible benchmark harness that runs classical DEB,
  Bayesian DEB, and Random Forest side-by-side on a *Daphnia magna* growth
  scenario and writes the same metrics reported in Section 7 of the paper.

## Install

```bash
cd SOFTWARE
pip install -e .
```

For Jupyter notebooks:

```bash
pip install -e ".[notebook]"
```

## Quickstart: reproduce the paper benchmark (#1)

```bash
python -m debcompare -v -o benchmark_output
```

This runs all three paradigms on a synthetic *D. magna* dataset seeded to
match `benchmark.R`, writes `benchmark_output/benchmark_results.csv`, and
saves a three-panel comparison figure. Typical runtime on a laptop is
under 30 seconds.

Add `--gallery` to also render a 6-figure educational plot suite:

```bash
python -m debcompare --gallery -v -o benchmark_output
```

Compare to the R benchmark:

```bash
python scripts/compare_with_r.py
```

## Educational plot gallery

`pydeb.plots` provides six publication-quality figures. Each function
returns a matplotlib `Figure` (show interactively) and accepts a `save_to=`
argument that writes both PDF (vector) and PNG (200 dpi) side by side.

| # | Function | Purpose |
|---|----------|---------|
| 1 | `plot_arrhenius` | Arrhenius $TC(T)$ curve with training/test points and extrapolation zone |
| 2 | `plot_posterior_predictive_fan` | Posterior growth curves across temperatures with 95 % credible bands |
| 3 | `plot_prior_posterior` | Prior vs posterior overlay for $L_\infty$, $r_B$, $L_0$, $\sigma$ |
| 4 | `plot_posterior_corner` | Joint posterior: histograms + 2D hex-bin densities with correlations |
| 5 | `plot_residuals` | 2×3 residual grid (task × paradigm) with shared axes |
| 6 | `plot_trace_and_rank` | MCMC diagnostics: chain traces + rank histograms |

Programmatic use:

```python
from pydeb.plots import plot_arrhenius, render_gallery
fig = plot_arrhenius(save_to="figures/arrhenius")   # writes .pdf + .png
render_gallery(result, "gallery/")                   # all 6 at once
```

## Quickstart: turnkey Bayesian DEB (#2)

```python
import pandas as pd
from pydeb.bayes import fit_growth, summarise

data = pd.read_csv("my_growth_data.csv")  # columns: time, L_obs, temp
idata = fit_growth(data, draws=2000, tune=1000, chains=3)
print(summarise(idata))
```

Priors default to AmP-derived values for the *Daphniidae* family. Override via
the `priors=` argument (see `pydeb.core.params.PriorSpec`). The full public API
also exposes:

- `pydeb.bayes.classical_fit` — Nelder-Mead least-squares baseline
- `pydeb.bayes.build_growth_model` — raw PyMC model for custom likelihoods
- `pydeb.bayes.posterior_predictive_growth` — predictive samples on any
  `(t, temperature)` grid for credible bands, out-of-sample evaluation, etc.

## Structure

```
SOFTWARE/
├── src/pydeb/
│   ├── core/        forward model, parameters, Arrhenius correction
│   ├── bayes/       PyMC models, AmP-informed priors, diagnostics
│   ├── ml/          sklearn baselines (RF, gradient boosting)
│   └── plots/       6 publication-quality educational figures
├── debcompare/      benchmark orchestration + CLI
├── tests/           unit + smoke tests (23 tests)
├── scripts/         standalone runnable scripts
│   ├── run_benchmark.py
│   └── compare_with_r.py
└── notebooks/       tutorial notebooks
```

## Running in the browser (Binder)

The `binder/` directory configures a zero-install environment on
[mybinder.org](https://mybinder.org):

- `binder/runtime.txt` — Python 3.10
- `binder/apt.txt` — system BLAS for PyTensor
- `binder/postBuild` — `pip install -e ".[notebook]"` to bootstrap the package

Clicking the Binder badge opens `notebooks/binder_demo.ipynb`, which fits all
three paradigms and renders four gallery figures on the Binder server. No
Python or local installation required.

## Reproducibility vs. R

The Python benchmark reproduces the R benchmark's paradigm ordering and
coverage properties, but numerical RMSEs differ because the two stacks use
different RNG streams and slightly different random-forest split semantics
(sklearn's `max_features=1` vs. R's `randomForest(mtry=1)`). Expect:

- Classical / Bayesian DEB RMSE within ±0.04 mm across interpolation and
  extrapolation; Bayesian credible intervals near nominal 95 % coverage.
- Random Forest RMSE substantially larger on the 25 C extrapolation task
  than on the 20 C training task (typical gap: 3–7 ×).

## Scope

This is *not* a full re-implementation of the standard DEB model (embryo +
juvenile + adult with maturity threshold, reserve dynamics, full covariation
method). The forward model is the DEB-consistent simplified parameterisation
used in the paper's Section 7 benchmark: Arrhenius-corrected von Bertalanffy
growth in structural length. Extension to the full standard DEB model is
planned for a subsequent release.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

16 tests covering the forward model, classical fit, metrics, and an
end-to-end benchmark smoke test (~20 s total).

## License

MIT.

## Citation

If you use this software, please cite the accompanying review:

> Hackenberger, B.K. & Djerdj, T. (2026). Mechanistic, Bayesian, and Machine
> Learning Approaches to Metabolic Modelling. *Ecological Modelling* (in press).
