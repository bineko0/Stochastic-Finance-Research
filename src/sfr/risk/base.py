from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RiskModel(ABC):
    """Next-day loss distribution of a log-return series.

    fit(window)   build the distribution from the estimation window
    var(level)    Value-at-Risk at confidence `level` (e.g. 0.99), as a POSITIVE loss
    es(level)     Expected Shortfall = E[loss | loss >= VaR], positive
    """

    @abstractmethod
    def fit(self, returns: pd.Series) -> "RiskModel": ...

    @abstractmethod
    def var(self, level: float) -> float: ...

    @abstractmethod
    def es(self, level: float) -> float: ...

    def rolling(self, returns: pd.Series, window: int, levels, start=None) -> pd.DataFrame:
        """Out-of-sample VaR/ES for each date >= start, using the `window` days before it."""
        returns = returns.dropna()
        first = window if start is None else max(window, returns.index.searchsorted(pd.Timestamp(start), "right"))
        rows = []
        for i in range(first, len(returns)):
            self.fit(returns.iloc[i - window:i])
            row = {"date": returns.index[i]}
            for lv in levels:
                row[f"var_{lv:g}"] = self.var(lv)
                row[f"es_{lv:g}"] = self.es(lv)
            rows.append(row)
        return pd.DataFrame(rows).set_index("date")


class SampleRiskModel(RiskModel):
    """Helper for models that end up with a sample of next-day returns."""

    sample: np.ndarray

    def var(self, level):
        return float(-np.quantile(self.sample, 1 - level))

    def es(self, level):
        q = np.quantile(self.sample, 1 - level)
        return float(-self.sample[self.sample <= q].mean())
