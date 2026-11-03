from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class VolatilityModel(ABC):
    """Volatility of DAILY log returns, in decimal units (not annualised).

    fit(train)              estimate parameters on a training sample
    predict(returns)        sigma_t for every date, using only returns before t
                            -> a genuine one-step-ahead forecast, same index
    forecast(returns, h)    sigma for the h days after the last date
    """

    @abstractmethod
    def fit(self, returns: pd.Series) -> "VolatilityModel": ...

    @abstractmethod
    def predict(self, returns: pd.Series) -> pd.Series: ...

    @abstractmethod
    def forecast(self, returns: pd.Series, horizon: int = 1) -> np.ndarray: ...

    def params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


def annualize(daily_vol, trading_days: int = 252):
    return daily_vol * np.sqrt(trading_days)
