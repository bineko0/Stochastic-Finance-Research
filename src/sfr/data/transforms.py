"""Price -> return transformations and sample splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """Columns: price, simple_ret = P_t/P_{t-1} - 1, log_ret = ln(P_t/P_{t-1}). First row dropped."""
    price = prices[price_col].astype(float)
    out = pd.DataFrame({"price": price, "simple_ret": price.pct_change(), "log_ret": np.log(price).diff()})
    return out.dropna()


def split_sample(series: pd.Series | pd.DataFrame, train_end: str):
    """(train, test): train is <= train_end, test is strictly after."""
    cut = pd.Timestamp(train_end)
    return series.loc[:cut], series.loc[series.index > cut]


def align(frames: dict[str, pd.Series]) -> pd.DataFrame:
    """Inner-join several series on common dates (different market holidays)."""
    return pd.concat(frames, axis=1, join="inner").dropna()
