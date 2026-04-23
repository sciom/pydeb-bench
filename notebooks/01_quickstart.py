"""Quickstart notebook (as a script for easy version control).

Convert to .ipynb with::

    jupytext --to notebook 01_quickstart.py

Or simply run as-is::

    python 01_quickstart.py
"""

# %% [markdown]
# # pydeb quickstart
#
# Fit the turnkey Bayesian DEB growth model to a synthetic *Daphnia magna*
# dataset and compare against classical least-squares and a Random Forest
# baseline.

# %%
import numpy as np

from pydeb.bayes import classical_fit, fit_growth
from pydeb.bayes.diagnostics import summarise
from pydeb.core.model import deb_growth
from pydeb.core.params import DEBParams
from pydeb.ml.baselines import fit_random_forest

from debcompare import simulate_daphnia_dataset
from debcompare.metrics import rmse

# %% [markdown]
# ## 1. Simulate a growth dataset

# %%
ds = simulate_daphnia_dataset(random_seed=42)
train, test = ds.train, ds.test
print(f"Train: {len(train)} rows at {train['temp'].iloc[0]} C")
print(f"Test:  {len(test)} rows at {test['temp'].iloc[0]} C")

# %% [markdown]
# ## 2. Classical DEB

# %%
cf = classical_fit(train, T_A=ds.true_params.T_A, T_ref_C=ds.true_params.T_ref_C)
print(cf.summary())

# %% [markdown]
# ## 3. Bayesian DEB (NUTS)

# %%
idata = fit_growth(train, draws=1000, tune=1000, chains=2, random_seed=42)
print(summarise(idata))

# %% [markdown]
# ## 4. Random Forest baseline

# %%
rf = fit_random_forest(train, random_state=42)
pred_rf = rf.predict(test)
print(f"RF extrapolation RMSE: {rmse(test['L_obs'], pred_rf):.3f} mm")

# %% [markdown]
# ## 5. Compare DEB extrapolation
#
# Classical and Bayesian DEB should extrapolate accurately to 25 C because
# temperature is built into the model via Arrhenius correction.

# %%
params_bayes = DEBParams(
    Linf=float(idata.posterior["Linf"].mean()),
    rB=float(idata.posterior["rB"].mean()),
    L0=float(idata.posterior["L0"].mean()),
    T_A=ds.true_params.T_A,
    T_ref_C=ds.true_params.T_ref_C,
)
pred_classical = deb_growth(test["time"], cf.params, test["temp"])
pred_bayesian = deb_growth(test["time"], params_bayes, test["temp"])

print(f"Classical DEB extrapolation RMSE: {rmse(test['L_obs'], pred_classical):.3f} mm")
print(f"Bayesian DEB extrapolation RMSE:  {rmse(test['L_obs'], pred_bayesian):.3f} mm")
print(f"Random Forest extrapolation RMSE: {rmse(test['L_obs'], pred_rf):.3f} mm")
