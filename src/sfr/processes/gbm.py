from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from sfr.processes.base import StochasticProcess, cumulate
from sfr.registry import PROCESSES


@PROCESSES.register("gbm")
@dataclass
class GBM(StochasticProcess):
    """Geometric Brownian motion  dS = mu S dt + sigma S dW.

    Exact scheme: S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z).
    """

    mu: float = 0.05
    sigma: float = 0.2

    def simulate(self, x0=100.0, T=1.0, n_steps=252, n_paths=1, rng=None):
        rng = np.random.default_rng(rng)
        dt = T / n_steps
        z = rng.standard_normal((n_steps, n_paths))
        log_inc = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
        return x0 * np.exp(cumulate(log_inc))

    @classmethod
    def fit(cls, log_returns, dt=1 / 252):
        """MLE: log returns ~ N((mu - sigma^2/2) dt, sigma^2 dt)."""
        r = np.asarray(log_returns, float)
        sigma = r.std(ddof=1) / np.sqrt(dt)
        return cls(mu=float(r.mean() / dt + 0.5 * sigma**2), sigma=float(sigma))

    def log_likelihood(self, log_returns, dt=1 / 252):
        r = np.asarray(log_returns, float)
        m = (self.mu - 0.5 * self.sigma**2) * dt
        return float(stats.norm.logpdf(r, m, self.sigma * np.sqrt(dt)).sum())
