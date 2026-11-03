from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Fetch daily prices for one symbol.

    Return a DataFrame indexed by date, in whatever column names the vendor uses;
    the pipeline standardises it (see sfr.data.schema). Set `column_map` for
    vendor names that schema._ALIASES does not already know.
    """

    column_map: dict | None = None

    @abstractmethod
    def fetch(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame: ...


def clip_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    df = df.sort_index()
    return df.loc[pd.Timestamp(start) if start else None : pd.Timestamp(end) if end else None]
