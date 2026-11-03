# Architecture

## Layers and contracts

```
            configs/*.yaml  (choices)
                  │
 ┌────────────────▼─────────────────┐
 │ data:  Source.fetch → standardize │   any vendor format
 │        → validate → returns       │
 └────────────────┬─────────────────┘
                  │  data/processed/{prices,returns}/<ASSET>.csv   ← CONTRACT
     ┌────────────┼──────────────┬──────────────┐
     ▼            ▼              ▼              ▼
 processes    volatility      pricing          risk
 (simulate,   (fit, predict,  (price(option,   (fit, var, es,
  fit, loglik) forecast)       market))         rolling)
     └─────┬──────┴──────┬───────┴──────┬───────┘
           ▼             ▼              ▼
      evaluation (metrics, tests)   reporting → reports/
```

| Interface | File | Must provide |
| --- | --- | --- |
| `DataSource` | [src/sfr/data/sources/base.py](../src/sfr/data/sources/base.py) | `fetch(symbol, start, end) -> DataFrame` indexed by date |
| Price schema | [src/sfr/data/schema.py](../src/sfr/data/schema.py) | `date` index; `close`, `adj_close` (+ optional open/high/low/volume) |
| `StochasticProcess` | [src/sfr/processes/base.py](../src/sfr/processes/base.py) | `simulate`; optional `fit`, `log_likelihood`; field `mu` for `with_drift` |
| `VolatilityModel` | [src/sfr/volatility/base.py](../src/sfr/volatility/base.py) | `fit`, `predict` (no look-ahead), `forecast` — daily decimal vol |
| `Pricer` | [src/sfr/pricing/base.py](../src/sfr/pricing/base.py) | `price(option, market) -> PricingResult` |
| `RiskModel` | [src/sfr/risk/base.py](../src/sfr/risk/base.py) | `fit`, `var(level)`, `es(level)` — positive losses |

Rules that keep parts independent:

1. Only the data stage touches sources. Everything else reads via `sfr.data.load_*`.
2. Components are referred to by registered name (`type:` in YAML), never imported by a stage directly.
3. Stages exchange files only. No stage imports another stage.
4. Units: time in years, parameters annualised; volatility models work on daily log returns.

## Recipes

### Change the data source

All assets: `default_source: csv` in [configs/data.yaml](../configs/data.yaml) (or `--set data.default_source=csv`).
One asset: `SP500: {symbol: SPX_Index, source: csv}`. For the CSV source, put files in `data/external/`.
Bloomberg column names (`PX_LAST`, `PX_OPEN`, ...) are recognised; other names go in `column_map`.

### Add a new data source (e.g. Bloomberg API)

```python
# src/sfr/data/sources/bloomberg.py
from sfr.data.sources.base import DataSource
from sfr.registry import SOURCES

@SOURCES.register("bloomberg")
class BloombergSource(DataSource):
    def __init__(self, field="PX_LAST"):
        self.field = field
    def fetch(self, symbol, start, end):
        ...  # return a DataFrame indexed by date
```

Then import it in [src/sfr/data/sources/__init__.py](../src/sfr/data/sources/__init__.py) and add
`bloomberg: {field: PX_LAST}` under `sources:` in `configs/data.yaml`.

### Add a model

Same pattern, e.g. a GJR-GARCH:

```python
@VOL_MODELS.register("gjr_garch")
class GJRGarch(VolatilityModel):
    def fit(self, returns): ...
    def predict(self, returns): ...
    def forecast(self, returns, horizon=1): ...
```

Import it in `src/sfr/volatility/__init__.py`, then add `gjr: {type: gjr_garch}` under
`volatility.models` in `configs/models.yaml`. It now appears in the forecast comparison,
and can be used by the risk stage (`fhs_gjr: {type: filtered_historical, vol_model: {type: gjr_garch}}`).
A new `StochasticProcess` is automatically usable by the Monte Carlo pricer and MC VaR.

### Add a stage

Create `src/sfr/stages/<name>.py` with `run(cfg)`, add it to `STAGES` in
[src/sfr/stages/__init__.py](../src/sfr/stages/__init__.py). Read inputs with `sfr.data.load_*`,
write outputs with `sfr.reporting.save_table/save_figure`.

### Check nothing broke

`python -m pytest`. [tests/test_pipeline.py](../tests/test_pipeline.py) runs every stage end to end on synthetic data.
