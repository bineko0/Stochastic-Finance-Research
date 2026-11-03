# Data

## Pipeline

```
python -m sfr run data
  source.fetch(symbol)        -> data/raw/<source>/<ASSET>.csv        as returned, never edited
  standardize + validate      -> data/processed/prices/<ASSET>.csv    schema: src/sfr/data/schema.py
  compute_returns             -> data/processed/returns/<ASSET>.csv   price, simple_ret, log_ret
  build log                   -> data/processed/build_log.csv         source, rows, dates, checks
```

Sources: `yahoo` (default), `csv` (files in `data/external/`, e.g. Bloomberg exports), `synthetic` (offline).
Switch in [configs/data.yaml](../configs/data.yaml).

## Series

| Name | Yahoo ticker | Role |
| --- | --- | --- |
| SP500 | ^GSPC | Main index series |
| SPY | SPY | ETF, candidate for an option snapshot (Module 5) |
| AAPL, MSFT | AAPL, MSFT | Single stocks, portfolio for Module 6 |
| VIX | ^VIX | Implied vol, compare against realised vol |
| EUROSTOXX50 | ^STOXX50E | European index |

## Conventions

- Returns are computed from `adj_close` (dividends/splits adjusted); sources without it use `close`.
- 252 trading days per year (`settings.trading_days`).
- Log returns for modelling (additive over time, natural for GBM); simple returns for portfolio aggregation.

## Known caveats

- Yahoo data can be revised; `build_log.csv` records the build date.
- Index series (^GSPC, ^VIX) have no dividends, so `adj_close == close`.
- Time zones/holidays differ across markets; use `sfr.data.transforms.align` before multi-asset work.
