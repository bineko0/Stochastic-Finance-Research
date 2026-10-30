"""Download raw daily price data from Yahoo Finance into 02_Data/Raw/.

Raw files are saved exactly as downloaded (Adj Close kept, no cleaning).
Run from anywhere:  python 02_Data/download_data.py
"""
from pathlib import Path
import datetime as dt

import pandas as pd
import yfinance as yf

RAW = Path(__file__).resolve().parent / "Raw"
RAW.mkdir(parents=True, exist_ok=True)

# name -> Yahoo ticker
TICKERS = {
    "SP500": "^GSPC",       # US equity index (main series for GBM / GARCH tests)
    "SPY": "SPY",           # tradable ETF (later: options snapshot)
    "AAPL": "AAPL",         # single stock
    "MSFT": "MSFT",         # single stock
    "VIX": "^VIX",          # implied vol index (compare with realised vol)
    "EUROSTOXX50": "^STOXX50E",  # European index
}

START = "2010-01-01"
END = None  # None = up to today


def download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df


def main() -> None:
    log = []
    for name, ticker in TICKERS.items():
        df = download(ticker)
        if df.empty:
            print(f"[WARN] {name} ({ticker}): no data returned")
            continue
        path = RAW / f"{name}.csv"
        df.to_csv(path)
        log.append(
            {
                "name": name,
                "ticker": ticker,
                "rows": len(df),
                "first_date": df.index.min().date(),
                "last_date": df.index.max().date(),
                "downloaded_on": dt.date.today().isoformat(),
                "file": path.name,
            }
        )
        print(f"[OK] {name:12s} {len(df):5d} rows  {df.index.min().date()} -> {df.index.max().date()}")

    pd.DataFrame(log).to_csv(RAW / "download_log.csv", index=False)


if __name__ == "__main__":
    main()
