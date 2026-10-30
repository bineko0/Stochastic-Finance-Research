# %% [markdown]
# # Week 1: first look at the data
# Questions: do log returns look Gaussian? Are they independent? Is volatility constant?
# These are exactly the assumptions GBM makes, so this sets up the whole project.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.graphics.tsaplots import plot_acf

PROCESSED = Path.cwd() / "02_Data" / "Processed"  # run with workspace root as cwd
if not PROCESSED.exists():
    PROCESSED = Path(__file__).resolve().parent / "Processed"

NAME = "SP500"
df = pd.read_csv(PROCESSED / f"{NAME}_returns.csv", index_col="Date", parse_dates=True)
r = df["log_ret"]
print(df.tail())
print(f"{len(r)} observations, {r.index.min().date()} -> {r.index.max().date()}")

# %% [markdown]
# ## 1. Price and returns

# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
df["Adj Close"].plot(ax=ax[0], title=f"{NAME} adjusted close")
r.plot(ax=ax[1], lw=0.5, title="Daily log returns")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Summary statistics (annualised with 252 trading days)

# %%
mu_d, sigma_d = r.mean(), r.std(ddof=1)
stats = pd.Series(
    {
        "mean (daily)": mu_d,
        "std (daily)": sigma_d,
        "mu annual": mu_d * 252,
        "sigma annual": sigma_d * np.sqrt(252),
        "skewness": st.skew(r),
        "excess kurtosis": st.kurtosis(r),  # Fisher: normal = 0
        "min": r.min(),
        "max": r.max(),
    }
)
print(stats.round(5))

jb = st.jarque_bera(r)
print(f"Jarque-Bera stat={jb.statistic:.1f}, p={jb.pvalue:.3g}")

# %% [markdown]
# ## 3. Normality check: histogram + QQ plot
# GBM implies log returns ~ Normal. Fat tails show up as QQ points peeling away at both ends.

# %%
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
# GBM: returns independent AND identically distributed. ACF of r should be ~0 (weak form);
# ACF of r^2 or |r| being clearly positive means volatility clustering, the motivation for GARCH.

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
plot_acf(r, lags=40, ax=ax[0], title="ACF of returns")
plot_acf(r**2, lags=40, ax=ax[1], title="ACF of squared returns")
plot_acf(r.abs(), lags=40, ax=ax[2], title="ACF of absolute returns")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Is volatility constant? Rolling 21-day annualised vol

# %%
roll = r.rolling(21).std() * np.sqrt(252)
roll.plot(figsize=(10, 3.5), title="Rolling 21-day annualised volatility")
plt.axhline(sigma_d * np.sqrt(252), color="k", ls="--", label="full-sample sigma")
plt.legend()
plt.show()

# %% [markdown]
# ## 6. First simulation: GBM path with sample (mu, sigma) vs the real series
# Exact discretisation: S_{t+dt} = S_t * exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)
# Here mu, sigma are annualised estimates from log returns, so mu_log_mean = (mu - sigma^2/2).

# %%
rng = np.random.default_rng(42)
n = len(r)
dt = 1 / 252
sigma = sigma_d * np.sqrt(252)
mu = mu_d * 252 + 0.5 * sigma**2  # drift of the price process
Z = rng.standard_normal(n)
logS = np.log(df["Adj Close"].iloc[0]) + np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
sim = pd.Series(np.exp(logS), index=r.index)

fig, ax = plt.subplots(figsize=(10, 4))
df["Adj Close"].plot(ax=ax, label="actual")
sim.plot(ax=ax, label="GBM sample path", alpha=0.8)
ax.set_title("Actual vs one GBM path (same mu, sigma)")
ax.legend()
plt.show()

# %% [markdown]
# ## Observations (fill in)
# - Tails: ...
# - Autocorrelation of returns: ...
# - Autocorrelation of squared returns: ...
# - Does a GBM path look "too smooth" in calm/crisis periods? ...
