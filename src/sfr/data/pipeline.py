"""fetch -> save raw -> standardise -> validate -> clean -> returns -> save processed."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from sfr.config import project_path
from sfr.data import store
from sfr.data.schema import standardize
from sfr.data.transforms import compute_returns
from sfr.data.validation import clean_prices, validate_prices
from sfr.registry import SOURCES


def make_source(cfg: dict, name: str):
    params = (cfg["data"].get("sources") or {}).get(name) or {}
    return SOURCES.build(name, **params)


def build_dataset(cfg: dict, assets: list[str] | None = None) -> pd.DataFrame:
    data_cfg = cfg["data"]
    sample = cfg["settings"]["sample"]
    checks = dict(data_cfg.get("validation") or {})
    strict = checks.pop("strict", False)
    raw_root = project_path(cfg, "raw")

    log = []
    for name in assets or list(data_cfg["assets"]):
        spec = data_cfg["assets"][name]
        source_name = spec.get("source", data_cfg["default_source"])
        row = {"asset": name, "symbol": spec["symbol"], "source": source_name,
               "built_on": dt.date.today().isoformat()}
        try:
            source = make_source(cfg, source_name)
            raw = source.fetch(spec["symbol"], sample.get("start"), sample.get("end"))
            raw_dir = raw_root / source_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw.to_csv(raw_dir / f"{name}.csv")

            prices = standardize(raw, getattr(source, "column_map", None))
            report = validate_prices(prices, name, **checks)
            row["checks"] = report.summary()
            if not report.ok:
                raise ValueError(report.summary())

            prices = clean_prices(prices)
            returns = compute_returns(prices)
            store.save_frame(cfg, "prices", name, prices)
            store.save_frame(cfg, "returns", name, returns)
            row.update(status="ok", rows=len(returns),
                       first=returns.index.min().date(), last=returns.index.max().date())
            print(f"[OK]   {name:12s} {source_name:9s} {len(returns):5d} rows  "
                  f"{row['first']} -> {row['last']}  ({report.summary()})")
        except Exception as exc:
            if strict:
                raise
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[FAIL] {name:12s} {source_name:9s} {row['error']}")
        log.append(row)

    log_df = pd.DataFrame(log)
    log_path = project_path(cfg, "processed") / "build_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    return log_df
