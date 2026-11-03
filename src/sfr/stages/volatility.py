from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sfr.config import dt, main_asset
from sfr.data import load_log_returns
from sfr.data.store import save_frame
from sfr.data.transforms import split_sample
from sfr.evaluation import diebold_mariano, mse, qlike
from sfr.registry import VOL_MODELS
from sfr.reporting import save_figure, save_table
from sfr.volatility import annualize


def run(cfg: dict) -> None:
    vcfg = cfg["models"]["volatility"]
    asset = main_asset(cfg)
    r = load_log_returns(asset, cfg)
    train, test = split_sample(r, cfg["settings"]["sample"]["train_end"])
    if test.empty:
        raise ValueError("No out-of-sample data after settings.sample.train_end")
    proxy = test.to_numpy() ** 2   # squared return as realised-variance proxy

    preds, params = {}, []
    for label, spec in vcfg["models"].items():
        model = VOL_MODELS.from_spec(spec).fit(train)
        preds[label] = model.predict(r)           # fitted on train, filtered through all data
        params.append({"model": label, **{k: v for k, v in model.params().items() if v is not None}})
    pred = pd.DataFrame(preds)
    save_frame(cfg, "volatility", asset, pred)    # downstream stages may reuse these
    save_table(cfg, "volatility_params", pd.DataFrame(params))

    bench = vcfg.get("benchmark", next(iter(preds)))
    losses = {m: {"mse": mse(proxy, pred.loc[test.index, m] ** 2), "qlike": qlike(proxy, pred.loc[test.index, m] ** 2)}
              for m in preds}
    rows = []
    for m, loss in losses.items():
        row = {"model": m, "mse_x1e8": np.nanmean(loss["mse"]) * 1e8, "qlike": np.nanmean(loss["qlike"])}
        if m != bench:
            dm = diebold_mariano(loss["qlike"], losses[bench]["qlike"])
            row.update({f"dm_qlike_vs_{bench}": dm["dm_stat"], "dm_p": dm["dm_p"]})
        rows.append(row)
    save_table(cfg, "volatility_forecast_eval", pd.DataFrame(rows).sort_values("qlike"))

    td = round(1 / dt(cfg))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.index, annualize(test.abs(), td), color="0.8", lw=0.6, label="|r| (annualised)")
    for m in preds:
        ax.plot(test.index, annualize(pred.loc[test.index, m], td), lw=1, label=m)
    ax.set_title(f"{asset}: out-of-sample one-day-ahead volatility forecasts")
    ax.set_ylim(0, np.nanquantile(annualize(test.abs(), td), 0.995))
    ax.legend(ncol=3)
    save_figure(cfg, "volatility_forecasts", fig)
