from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special

from sfr.registry import VOL_MODELS
from sfr.volatility.base import VolatilityModel

_SCALE = 100.0  # optimise on percent returns for numerical stability


def _variance(x: np.ndarray, omega: float, alpha: float, beta: float, var0: float) -> np.ndarray:
    var = np.empty(len(x) + 1)
    var[0] = var0
    for t in range(len(x)):
        var[t + 1] = omega + alpha * x[t] ** 2 + beta * var[t]
    return var


def _nll(x, var, dist, nu=None):
    if dist == "normal":
        return 0.5 * np.sum(np.log(2 * np.pi) + np.log(var) + x**2 / var)
    # standardised Student-t with nu degrees of freedom (unit variance)
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return -np.sum(c - 0.5 * np.log(var) - (nu + 1) / 2 * np.log1p(x**2 / (var * (nu - 2))))


@VOL_MODELS.register("garch")
class GARCH11(VolatilityModel):
    """GARCH(1,1) by maximum likelihood:  sigma^2_t = omega + alpha e^2_{t-1} + beta sigma^2_{t-1},
    e_t = r_t - mu.  dist = "normal" or "t" (Student-t innovations)."""

    def __init__(self, dist: str = "normal", demean: bool = True):
        if dist not in ("normal", "t"):
            raise ValueError("dist must be 'normal' or 't'")
        self.dist = dist
        self.demean = demean
        self.mu = self.omega = self.alpha = self.beta = self.nu = None
        self._var0 = None

    def fit(self, returns):
        r = returns.dropna().to_numpy(float)
        self.mu = r.mean() if self.demean else 0.0
        x = (r - self.mu) * _SCALE
        var0 = x.var()

        def objective(p):
            omega, alpha, beta = p[:3]
            var = _variance(x, omega, alpha, beta, var0)[:-1]
            return _nll(x, var, self.dist, p[3] if self.dist == "t" else None)

        x0 = [0.05 * var0, 0.08, 0.90] + ([8.0] if self.dist == "t" else [])
        bounds = [(1e-8, 10 * var0), (1e-6, 0.5), (1e-6, 0.9999)] + ([(2.1, 100)] if self.dist == "t" else [])
        stationarity = {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}
        res = optimize.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=[stationarity])
        omega, self.alpha, self.beta = res.x[:3]
        self.nu = res.x[3] if self.dist == "t" else None
        self.omega = omega / _SCALE**2
        self._var0 = var0 / _SCALE**2
        self.converged = bool(res.success)
        return self

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        return self.omega / (1 - self.persistence)

    def _path(self, returns: pd.Series) -> np.ndarray:
        if self.omega is None:
            raise RuntimeError("call fit() first")
        x = returns.to_numpy(float) - self.mu
        return _variance(x, self.omega, self.alpha, self.beta, self._var0)

    def predict(self, returns):
        return pd.Series(np.sqrt(self._path(returns)[:-1]), index=returns.index, name="garch")

    def forecast(self, returns, horizon=1):
        nxt = self._path(returns)[-1]
        vl = self.long_run_variance
        k = np.arange(horizon)
        return np.sqrt(vl + self.persistence**k * (nxt - vl))

    def params(self):
        return {"mu": self.mu, "omega": self.omega, "alpha": self.alpha, "beta": self.beta,
                "nu": self.nu, "dist": self.dist}
