from __future__ import annotations

import pandas as pd

from sfr.config import resolve_path
from sfr.data.sources.base import DataSource, clip_dates
from sfr.registry import SOURCES


@SOURCES.register("csv")
class CSVSource(DataSource):
    """Local CSV files, one per symbol: <directory>/<filename.format(symbol=...)>.

    Works for Bloomberg exports (PX_LAST, PX_OPEN... are recognised), Refinitiv,
    Kaggle, or anything else with a date column and a price column.
    """

    def __init__(self, directory="data/external", filename="{symbol}.csv", date_column="Date",
                 column_map=None, date_format=None, sep=","):
        self.directory = directory
        self.filename = filename
        self.date_column = date_column
        self.column_map = column_map or None
        self.date_format = date_format
        self.sep = sep

    def fetch(self, symbol, start, end):
        path = resolve_path(self.directory) / self.filename.format(symbol=symbol)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        df = pd.read_csv(path, sep=self.sep)
        if self.date_column not in df.columns:
            raise KeyError(f"{path.name}: no date column {self.date_column!r} in {list(df.columns)}")
        df.index = pd.to_datetime(df.pop(self.date_column), format=self.date_format)
        df.index.name = "Date"
        return clip_dates(df, start, end)
