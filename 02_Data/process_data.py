"""Turn raw price files into return series in 02_Data/Processed/.

Output columns: Adj Close, simple_ret, log_ret  (first row dropped).
Run:  python 02_Data/process_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "Raw"
PROCESSED = HERE / "Processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def build_returns(df: pd.DataFrame) -> pd.DataFrame:
    price = df["Adj Close"].astype(float)
    out = pd.DataFrame({"Adj Close": price})
    out["simple_ret"] = price.pct_change()
    out["log_ret"] = np.log(price).diff()
    return out.dropna()


def main() -> None:
    for path in sorted(RAW.glob("*.csv")):
        if path.name == "download_log.csv":
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        if df["Adj Close"].isna().any():
            print(f"[INFO] {path.stem}: {df['Adj Close'].isna().sum()} missing prices (dropped via returns)")
        if (df["Adj Close"] <= 0).any():
            print(f"[WARN] {path.stem}: non-positive prices found, check data")
        rets = build_returns(df)
        out = PROCESSED / f"{path.stem}_returns.csv"
        rets.to_csv(out)
        print(f"[OK] {out.name}: {len(rets)} rows")


if __name__ == "__main__":
    main()
