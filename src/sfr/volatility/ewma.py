from __future__ import annotations

import numpy as np
import pandas as pd

from sfr.registry import VOL_MODELS
from sfr.volatility.base import VolatilityModel


@VOL_MODELS.register("ewma")
class EWMA(VolatilityModel):
    """RiskMetrics: sigma^2_t = lam sigma^2_{t-1} + (1 - lam) r^2_{t-1}.

    Started at the mean squared return of the first `init_window` days.
    """

    def __init__(self, lam: float = 0.94, init_window: int = 30):
        self.lam = lam
        self.init_window = init_window

    def fit(self, returns):
        return self  # lam is fixed by convention; estimate it here if you want

    def _variance(self, r: np.ndarray) -> np.ndarray:
        var = np.empty(len(r) + 1)
        var[0] = np.mean(r[: self.init_window] ** 2)
        for t in range(len(r)):
            var[t + 1] = self.lam * var[t] + (1 - self.lam) * r[t] ** 2
        return var

    def predict(self, returns):
        var = self._variance(returns.to_numpy(float))
        return pd.Series(np.sqrt(var[:-1]), index=returns.index, name="ewma")

    def forecast(self, returns, horizon=1):
        return np.full(horizon, np.sqrt(self._variance(returns.to_numpy(float))[-1]))
