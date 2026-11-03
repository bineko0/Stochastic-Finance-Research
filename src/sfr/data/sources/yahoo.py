from __future__ import annotations

import pandas as pd

from sfr.data.sources.base import DataSource
from sfr.registry import SOURCES


@SOURCES.register("yahoo")
class YahooSource(DataSource):
    """Yahoo Finance via yfinance. Keeps 'Adj Close' (auto_adjust=False)."""

    def __init__(self, auto_adjust: bool = False):
        self.auto_adjust = auto_adjust

    def fetch(self, symbol, start, end):
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("Yahoo source needs yfinance:  pip install yfinance") from exc
        df = yf.download(symbol, start=start, end=end, auto_adjust=self.auto_adjust, progress=False)
        if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        return df
