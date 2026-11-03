from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

from sfr.processes.base import StochasticProcess, cumulate
from sfr.processes.gbm import GBM
from sfr.registry import PROCESSES


@PROCESSES.register("merton")
@dataclass
class MertonJumpDiffusion(StochasticProcess):
    """Merton (1976):  dS/S = (mu - lam k) dt + sigma dW + (J - 1) dN,
    N ~ Poisson(lam), ln J ~ N(jump_mean, jump_std^2), k = E[J - 1].

    The -lam k compensator makes mu the total expected return, so
    with_drift(r - q) gives the risk-neutral process.
    """

    mu: float = 0.05
    sigma: float = 0.15
    lam: float = 1.0          # jumps per year
    jump_mean: float = -0.05
    jump_std: float = 0.10

    @property
    def kappa(self) -> float:
        return np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1

    def _diffusion_drift(self) -> float:
        return self.mu - self.lam * self.kappa - 0.5 * self.sigma**2

    def simulate(self, x0=100.0, T=1.0, n_steps=252, n_paths=1, rng=None):
        rng = np.random.default_rng(rng)
        dt = T / n_steps
        shape = (n_steps, n_paths)
        n_jumps = rng.poisson(self.lam * dt, shape)
        jumps = n_jumps * self.jump_mean + np.sqrt(n_jumps) * self.jump_std * rng.standard_normal(shape)
        log_inc = self._diffusion_drift() * dt + self.sigma * np.sqrt(dt) * rng.standard_normal(shape) + jumps
        return x0 * np.exp(cumulate(log_inc))

    def log_likelihood(self, log_returns, dt=1 / 252, n_max: int = 10):
        """Poisson mixture of normals, truncated at n_max jumps per step."""
        r = np.asarray(log_returns, float)[:, None]
        n = np.arange(n_max + 1)[None, :]
        log_w = stats.poisson.logpmf(n, self.lam * dt)
        mean = self._diffusion_drift() * dt + n * self.jump_mean
        std = np.sqrt(self.sigma**2 * dt + n * self.jump_std**2)
        return float(special.logsumexp(log_w + stats.norm.logpdf(r, mean, std), axis=1).sum())

    @classmethod
    def fit(cls, log_returns, dt=1 / 252, n_max: int = 10):
        r = np.asarray(log_returns, float)
        gbm = GBM.fit(r, dt)
        x0 = [gbm.mu, 0.8 * gbm.sigma, 10.0, 0.0, 2 * r.std()]
        bounds = [(-2, 2), (1e-3, 2), (1e-2, 252), (-0.5, 0.5), (1e-3, 0.5)]

        def nll(p):
            return -cls(*p).log_likelihood(r, dt, n_max)

        res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        return cls(*map(float, res.x))
