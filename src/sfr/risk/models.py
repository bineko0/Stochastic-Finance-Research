from __future__ import annotations

import numpy as np
from scipy import stats

from sfr.registry import PROCESSES, RISK_MODELS, VOL_MODELS
from sfr.risk.base import RiskModel, SampleRiskModel


@RISK_MODELS.register("historical")
class HistoricalSimulation(SampleRiskModel):
    """Empirical quantile of the window: no distributional assumption."""

    def fit(self, returns):
        self.sample = returns.to_numpy(float)
        return self


@RISK_MODELS.register("gaussian")
class GaussianVaR(RiskModel):
    """Variance-covariance VaR: returns ~ N(mu, sigma^2)."""

    def fit(self, returns):
        self.mu, self.sigma = returns.mean(), returns.std(ddof=1)
        return self

    def var(self, level):
        return float(-(self.mu + self.sigma * stats.norm.ppf(1 - level)))

    def es(self, level):
        p = 1 - level
        return float(-self.mu + self.sigma * stats.norm.pdf(stats.norm.ppf(p)) / p)


@RISK_MODELS.register("student_t")
class StudentTVaR(RiskModel):
    """Returns ~ loc + scale * t(nu), fitted by MLE: fat tails."""

    def fit(self, returns):
        self.nu, self.loc, self.scale = stats.t.fit(returns.to_numpy(float))
        self.nu = max(self.nu, 2.05)
        return self

    def var(self, level):
        return float(-(self.loc + self.scale * stats.t.ppf(1 - level, self.nu)))

    def es(self, level):
        p, nu = 1 - level, self.nu
        q = stats.t.ppf(p, nu)
        return float(-self.loc + self.scale * stats.t.pdf(q, nu) / p * (nu + q**2) / (nu - 1))


@RISK_MODELS.register("monte_carlo")
class MonteCarloVaR(SampleRiskModel):
    """Fit any registered StochasticProcess to the window, simulate next-day log returns."""

    def __init__(self, process: str = "gbm", n_sims: int = 10_000, dt: float = 1 / 252, seed: int = 42):
        self.process_name, self.n_sims, self.dt = process, n_sims, dt
        self._rng = np.random.default_rng(seed)

    def fit(self, returns):
        proc = PROCESSES.get(self.process_name).fit(returns, self.dt)
        paths = proc.simulate(1.0, self.dt, 1, self.n_sims, rng=self._rng)
        self.sample = np.log(paths[-1])
        return self


@RISK_MODELS.register("filtered_historical")
class FilteredHistoricalSimulation(SampleRiskModel):
    """FHS: standardised residuals z = r / sigma_t from any registered VolatilityModel,
    rescaled by tomorrow's sigma forecast. The volatility model is re-fitted every
    `refit_every` calls to fit() (parameters change slowly; sigma is updated daily)."""

    def __init__(self, vol_model: dict | str = "garch", refit_every: int = 20):
        self.vol_spec, self.refit_every = vol_model, refit_every
        self._model, self._calls = None, 0

    def fit(self, returns):
        if self._model is None or self._calls % self.refit_every == 0:
            self._model = VOL_MODELS.from_spec(self.vol_spec).fit(returns)
        self._calls += 1
        sigma = self._model.predict(returns)
        z = (returns / sigma).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        self.sample = self._model.forecast(returns, 1)[0] * z
        return self
