# %% [markdown]
# # 01: first look at the data
# Questions: do log returns look Gaussian? Are they independent? Is volatility constant?
# These are exactly the assumptions GBM makes, so this sets up the whole project.
#
# Run `python -m sfr run data` first. Runs as `# %%` cells in the VSCode Interactive Window.
# All data access goes through `sfr.data`, so this notebook does not care which source was used.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from sfr.config import load_config
from sfr.data import load_returns
from sfr.evaluation import acf, stylized_facts
from sfr.processes import GBM

cfg = load_config()
NAME = cfg["data"]["main_asset"]
TD = cfg["settings"]["trading_days"]
df = load_returns(NAME, cfg)
r = df["log_ret"]
print(df.tail())

# %% [markdown]
# ## 1. Price and returns

# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
df["price"].plot(ax=ax[0], title=f"{NAME} adjusted close")
r.plot(ax=ax[1], lw=0.5, title="Daily log returns")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Summary statistics and tests (annualised with 252 trading days)

# %%
pd.Series(stylized_facts(r, TD))

# %% [markdown]
# ## 3. Normality check: histogram + QQ plot
# GBM implies log returns ~ Normal. Fat tails show up as QQ points peeling away at both ends.

# %%
mu_d, sigma_d = r.mean(), r.std(ddof=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(r, bins=100, density=True, alpha=0.6, label="empirical")
x = np.linspace(r.min(), r.max(), 400)
ax[0].plot(x, st.norm.pdf(x, mu_d, sigma_d), label="Normal fit")
ax[0].set_yscale("log")
ax[0].set_ylim(bottom=1e-2)
ax[0].set_title("Log-return density (log y-axis)")
ax[0].legend()
st.probplot(r, dist="norm", plot=ax[1])
ax[1].set_title("QQ plot vs Normal")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Independence and volatility clustering
# GBM: returns i.i.d. ACF of r should be ~0; ACF of r^2 or |r| clearly positive means
# volatility clustering, the motivation for GARCH.

# %%
lags = 40
band = 1.96 / np.sqrt(len(r))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
for a, series, title in zip(ax, [r, r**2, r.abs()], ["returns", "squared returns", "absolute returns"]):
    a.bar(range(1, lags + 1), acf(series, lags)[1:], width=0.6)
    a.axhspan(-band, band, color="0.85")
    a.set_title(f"ACF of {title}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Is volatility constant? Rolling 21-day annualised vol

# %%
roll = r.rolling(21).std() * np.sqrt(TD)
roll.plot(figsize=(10, 3.5), title="Rolling 21-day annualised volatility")
plt.axhline(sigma_d * np.sqrt(TD), color="k", ls="--", label="full-sample sigma")
plt.legend()
plt.show()

# %% [markdown]
# ## 6. One fitted GBM path vs the real series

# %%
gbm = GBM.fit(r, dt=1 / TD)
print(gbm)
path = gbm.simulate(df["price"].iloc[0], T=len(r) / TD, n_steps=len(r) - 1, rng=cfg["settings"]["seed"])[:, 0]
fig, ax = plt.subplots(figsize=(10, 4))
df["price"].plot(ax=ax, label="actual")
ax.plot(df.index, path, label="GBM sample path", alpha=0.8)
ax.set_title("Actual vs one GBM path (same mu, sigma)")
ax.legend()
plt.show()

# %% [markdown]
# ## Observations (fill in)
# - Tails: ...
# - Autocorrelation of returns: ...
# - Autocorrelation of squared returns: ...
# - Does a GBM path look "too smooth" in calm/crisis periods? ...
