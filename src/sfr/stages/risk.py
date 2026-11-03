from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from sfr.config import main_asset
from sfr.data import load_log_returns
from sfr.registry import RISK_MODELS
from sfr.reporting import save_figure, save_table
from sfr.risk import backtest_var


def run(cfg: dict) -> None:
    rc = cfg["risk"]
    asset = main_asset(cfg, rc.get("asset"))
    r = load_log_returns(asset, cfg)
    levels, window = rc["levels"], rc["window"]
    start = cfg["settings"]["sample"]["train_end"]

    rolled, rows, latest = {}, [], []
    for label, spec in rc["models"].items():
        model = RISK_MODELS.from_spec(spec)
        est = model.rolling(r, window, levels, start=start)
        rolled[label] = est
        for lv in levels:
            bt = backtest_var(r, est[f"var_{lv:g}"], lv, est[f"es_{lv:g}"])
            rows.append({"model": label, "level": lv, **bt})
        fresh = RISK_MODELS.from_spec(spec).fit(r.iloc[-window:])
        latest.append({"model": label, **{f"{k}_{lv:g}": getattr(fresh, k)(lv) for lv in levels for k in ("var", "es")}})
        print(f"  {label}: {len(est)} out-of-sample days")

    save_table(cfg, "risk_backtest", pd.DataFrame(rows))
    save_table(cfg, "risk_latest_estimates", pd.DataFrame(latest))

    lv = max(levels)
    test = r.loc[next(iter(rolled.values())).index]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.index, test, color="0.6", lw=0.5, label="log return")
    for label, est in rolled.items():
        ax.plot(est.index, -est[f"var_{lv:g}"], lw=1, label=f"-VaR {lv:g} {label}")
    ax.set_title(f"{asset}: rolling 1-day VaR ({window}-day window), out-of-sample")
    ax.legend(ncol=3, fontsize=8)
    save_figure(cfg, "risk_var_backtest", fig)
