import numpy as np
import pandas as pd

from sfr.data import build_dataset, load_returns
from sfr.data.schema import standardize
from sfr.data.validation import validate_prices
from sfr.registry import SOURCES


def test_standardize_yahoo_and_bloomberg_names():
    idx = pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-02"])
    yahoo = pd.DataFrame({"Open": [1, 2, 3], "Close": [1, 2, 3], "Adj Close": [1, 2, 3]}, index=idx)
    bbg = pd.DataFrame({"PX_LAST": [1.0, 2.0, 3.0], "PX_VOLUME": [5, 6, 7]}, index=idx)
    for raw in (yahoo, bbg):
        out = standardize(raw)
        assert out.index.is_monotonic_increasing and out.index.is_unique
        assert {"close", "adj_close"} <= set(out.columns)
        assert out.index.name == "date"


def test_validation_flags_bad_prices():
    df = pd.DataFrame({"adj_close": [100.0, -1.0, 101.0] + [100.0] * 40},
                      index=pd.bdate_range("2020-01-01", periods=43))
    report = validate_prices(df, "x")
    assert report.ok and "non-positive" in report.summary()
    assert not validate_prices(df.iloc[:0], "empty").ok


def test_pipeline_with_synthetic_source(cfg):
    log = build_dataset(cfg)
    assert (log["status"] == "ok").all()
    r = load_returns("SP500", cfg)
    assert list(r.columns) == ["price", "simple_ret", "log_ret"]
    assert np.allclose(np.exp(r["log_ret"]) - 1, r["simple_ret"])


def test_switching_to_csv_source_needs_only_config(cfg, tmp_path):
    """Same pipeline, different source: write a vendor-style CSV and point the config at it."""
    ext = tmp_path / "external"
    ext.mkdir()
    prices = SOURCES.build("synthetic").fetch("X", "2018-01-01", "2019-12-31")["Close"]
    pd.DataFrame({"Date": prices.index.strftime("%d/%m/%Y"), "PX_LAST": prices.values}).to_csv(
        ext / "SPX_Index.csv", index=False)
    cfg["data"]["assets"] = {"SP500": {"symbol": "SPX_Index", "source": "csv"}}
    cfg["data"]["sources"]["csv"].update(directory=str(ext), date_format="%d/%m/%Y")
    log = build_dataset(cfg)
    assert log.loc[0, "status"] == "ok", log.to_dict()
    assert len(load_returns("SP500", cfg)) == len(prices) - 1
