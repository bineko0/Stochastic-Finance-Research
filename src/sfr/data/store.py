"""Read/write processed data. This is the contract between the data stage and everything else.

    data/processed/prices/<ASSET>.csv    standardised prices (schema.PRICE_COLUMNS)
    data/processed/returns/<ASSET>.csv   price, simple_ret, log_ret
"""
from __future__ import annotations

import pandas as pd

from sfr.config import load_config, project_path


def _dir(cfg: dict, kind: str):
    path = project_path(cfg, "processed") / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_frame(cfg: dict, kind: str, name: str, df: pd.DataFrame) -> None:
    df.to_csv(_dir(cfg, kind) / f"{name}.csv")


def load_frame(cfg: dict | None, kind: str, name: str) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = _dir(cfg, kind) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the data stage first:  python -m sfr run data")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_prices(name: str, cfg: dict | None = None) -> pd.DataFrame:
    return load_frame(cfg, "prices", name)


def load_returns(name: str, cfg: dict | None = None) -> pd.DataFrame:
    return load_frame(cfg, "returns", name)


def load_log_returns(name: str, cfg: dict | None = None) -> pd.Series:
    return load_returns(name, cfg)["log_ret"].rename(name)


def available_assets(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    return sorted(p.stem for p in _dir(cfg, "returns").glob("*.csv"))
