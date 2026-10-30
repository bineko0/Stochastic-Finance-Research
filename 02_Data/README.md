# 02_Data

## Pipeline

```
download_data.py  ->  Raw/*.csv (+ download_log.csv)      never edited by hand
process_data.py   ->  Processed/*_returns.csv             regenerated any time
week1_exploration.py                                      stylised facts, first GBM path
```

## Series

| Name | Yahoo ticker | Role |
| --- | --- | --- |
| SP500 | ^GSPC | Main index series |
| SPY | SPY | ETF, candidate for an option snapshot (Module 3) |
| AAPL, MSFT | AAPL, MSFT | Single stocks, portfolio for Module 4 |
| VIX | ^VIX | Implied vol, compare against realised vol |
| EUROSTOXX50 | ^STOXX50E | European index |

## Columns

- Raw: `Open, High, Low, Close, Adj Close, Volume` (as returned by Yahoo).
- Processed: `Adj Close, simple_ret = P_t/P_{t-1} - 1, log_ret = ln(P_t/P_{t-1})`.

## Conventions

- Use `Adj Close` (dividends/splits adjusted) for returns.
- 252 trading days per year for annualisation.
- Log returns for modelling (additive over time, natural for GBM); simple returns for portfolio aggregation.

## Known caveats

- Yahoo data can be revised; `download_log.csv` records the download date.
- Index series (^GSPC, ^VIX) have no dividends, so `Adj Close == Close`.
- Time zones/holidays differ across markets; align dates before multi-asset work (Module 4).
