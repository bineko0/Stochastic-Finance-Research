"""Heston (1993) stochastic volatility.

    dS = mu S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,     corr(dW1, dW2) = rho

`Heston` is a StochasticProcess (simulation, MC pricing).
`HestonVolatility` wraps it as a VolatilityModel for the forecast comparison.
For option pricing, calibrate to option prices instead of using fit() here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sfr.processes.base import StochasticProcess
from sfr.registry import PROCESSES, VOL_MODELS
from sfr.volatility.base import VolatilityModel


@PROCESSES.register("heston")
@dataclass
class Heston(StochasticProcess):
    mu: float = 0.05
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7

    @property
    def feller(self) -> bool:
        """2 kappa theta >= xi^2: variance stays strictly positive."""
        return 2 * self.kappa * self.theta >= self.xi**2

    def simulate(self, x0=100.0, T=1.0, n_steps=252, n_paths=1, rng=None, return_variance=False):
        """Full-truncation Euler for v, log-Euler for S."""
        rng = np.random.default_rng(rng)
        dt = T / n_steps
        logS = np.empty((n_steps + 1, n_paths))
        v = np.empty((n_steps + 1, n_paths))
        logS[0], v[0] = np.log(x0), self.v0
        for t in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * rng.standard_normal(n_paths)
            vp = np.maximum(v[t], 0.0)
            logS[t + 1] = logS[t] + (self.mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
            v[t + 1] = v[t] + self.kappa * (self.theta - vp) * dt + self.xi * np.sqrt(vp * dt) * z2
        S = np.exp(logS)
        return (S, np.maximum(v, 0.0)) if return_variance else S

    @classmethod
    def fit(cls, log_returns, dt=1 / 252, rv_window: int = 21):
        """Rough moment estimate from realised variance on non-overlapping blocks.

        Discretised CIR:  dv / sqrt(v) = kappa theta h / sqrt(v) - kappa h sqrt(v) + xi sqrt(h) eps,
        h = rv_window * dt, estimated by OLS. rho = corr(block return shocks, eps).
        """
        r = np.asarray(log_returns, float)
        n_blocks = len(r) // rv_window
        blocks = r[len(r) - n_blocks * rv_window:].reshape(n_blocks, rv_window)
        v = np.maximum((blocks**2).mean(axis=1) / dt, 1e-8)   # annualised realised variance
        h = rv_window * dt
        sv = np.sqrt(v[:-1])
        y = np.diff(v) / sv
        X = np.column_stack([1 / sv, sv])
        (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
        kappa = max(-b / h, 1e-2)
        theta = a / (kappa * h) if a > 0 else v.mean()
        resid = y - X @ np.array([a, b])
        xi = resid.std(ddof=2) / np.sqrt(h)
        ret_shock = blocks[1:].sum(axis=1) - blocks.sum(axis=1).mean()
        rho = float(np.clip(np.corrcoef(ret_shock, resid)[0, 1], -0.99, 0.99))
        mu = r.mean() / dt + 0.5 * v.mean()
        return cls(*map(float, (mu, v[-1], kappa, theta, xi, rho)))

    def expected_variance(self, v_now, tau):
        """E[v_{t+tau} | v_t] = theta + (v_t - theta) exp(-kappa tau)."""
        return self.theta + (v_now - self.theta) * np.exp(-self.kappa * tau)


@VOL_MODELS.register("heston")
class HestonVolatility(VolatilityModel):
    """Variance state = annualised realised variance of the last rv_window days,
    forecast with Heston's mean-reverting conditional expectation."""

    def __init__(self, rv_window: int = 21, dt: float = 1 / 252):
        self.rv_window = rv_window
        self.dt = dt
        self.process: Heston | None = None

    def fit(self, returns):
        self.process = Heston.fit(returns.dropna(), self.dt, self.rv_window)
        return self

    def _state(self, returns):
        return (returns**2).rolling(self.rv_window).mean() / self.dt

    def predict(self, returns):
        v = self._state(returns).shift(1)
        ev = self.process.expected_variance(v, self.dt)
        return pd.Series(np.sqrt(ev * self.dt), index=returns.index, name="heston")

    def forecast(self, returns, horizon=1):
        v_now = self._state(returns).iloc[-1]
        tau = self.dt * np.arange(1, horizon + 1)
        return np.sqrt(self.process.expected_variance(v_now, tau) * self.dt)

    def params(self):
        return self.process.params() if self.process else {}
