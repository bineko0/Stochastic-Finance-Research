from __future__ import annotations

import numpy as np

from sfr.registry import VOL_MODELS
from sfr.volatility.base import VolatilityModel


@VOL_MODELS.register("historical")
class HistoricalVolatility(VolatilityModel):
    """Rolling sample standard deviation over the last `window` days."""

    def __init__(self, window: int = 21):
        self.window = window

    def fit(self, returns):
        return self  # nothing to estimate

    def predict(self, returns):
        return returns.rolling(self.window).std().shift(1)

    def forecast(self, returns, horizon=1):
        return np.full(horizon, returns.iloc[-self.window:].std())
