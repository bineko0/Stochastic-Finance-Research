"""The single price format every source is converted to.

    index : DatetimeIndex named "date", tz-naive, sorted, unique
    columns: subset of PRICE_COLUMNS, float; "close" and "adj_close" always present
"""
from __future__ import annotations

import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

# Lower-cased, underscore-joined vendor names -> canonical name.
_ALIASES = {
    "adj_close": "adj_close",
    "adjclose": "adj_close",
    "adjusted_close": "adj_close",
    "close_adj": "adj_close",
    "px_last": "close",        # Bloomberg
    "last_price": "close",
    "last": "close",
    "price": "close",
    "px_open": "open",
    "px_high": "high",
    "px_low": "low",
    "px_volume": "volume",
    "vol": "volume",
}


def _canonical(column) -> str:
    name = str(column).strip().lower().replace(" ", "_").replace(".", "_")
    return _ALIASES.get(name, name)


def standardize(df: pd.DataFrame, column_map: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if column_map:
        df = df.rename(columns=column_map)
    df.columns = [_canonical(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    if "close" not in df.columns and "adj_close" not in df.columns:
        raise ValueError(f"No close price column found in {list(df.columns)}")
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    if "close" not in df.columns:
        df["close"] = df["adj_close"]

    index = pd.DatetimeIndex(pd.to_datetime(df.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    df.index = index.normalize()
    df.index.name = "date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    keep = [c for c in PRICE_COLUMNS if c in df.columns]
    return df[keep].apply(pd.to_numeric, errors="coerce").astype(float)
