from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from sfr.processes.base import StochasticProcess, cumulate
from sfr.registry import PROCESSES


@PROCESSES.register("brownian")
@dataclass
class BrownianMotion(StochasticProcess):
    """Arithmetic Brownian motion  dX = mu dt + sigma dW.  mu=0, sigma=1 is standard BM."""

    mu: float = 0.0
    sigma: float = 1.0

    def simulate(self, x0=0.0, T=1.0, n_steps=252, n_paths=1, rng=None):
        rng = np.random.default_rng(rng)
        dt = T / n_steps
        dX = self.mu * dt + self.sigma * np.sqrt(dt) * rng.standard_normal((n_steps, n_paths))
        return x0 + cumulate(dX)

    @classmethod
    def fit(cls, log_returns, dt=1 / 252):
        x = np.asarray(log_returns, float)
        return cls(mu=float(x.mean() / dt), sigma=float(x.std(ddof=1) / np.sqrt(dt)))

    def log_likelihood(self, log_returns, dt=1 / 252):
        x = np.asarray(log_returns, float)
        return float(stats.norm.logpdf(x, self.mu * dt, self.sigma * np.sqrt(dt)).sum())
