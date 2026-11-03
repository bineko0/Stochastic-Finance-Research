from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sfr.config import dt as get_dt
from sfr.config import main_asset
from sfr.data import load_log_returns, load_prices
from sfr.evaluation import aic, bic
from sfr.registry import PROCESSES
from sfr.reporting import save_figure, save_table


def run(cfg: dict) -> None:
    pcfg = cfg["models"]["processes"]
    asset = main_asset(cfg)
    r = load_log_returns(asset, cfg)
    s0 = load_prices(asset, cfg)["adj_close"].iloc[-1]
    dt, seed = get_dt(cfg), cfg["settings"]["seed"]

    rows, fitted = [], {}
    for name in pcfg["fit"]:
        proc = PROCESSES.get(name).fit(r, dt=dt)
        fitted[name] = proc
        try:
            ll = proc.log_likelihood(r, dt)
        except NotImplementedError:
            ll = np.nan
        k = proc.n_params
        rows.append({"model": name, **{p: round(v, 6) for p, v in proc.params().items()},
                     "log_lik": ll, "aic": aic(ll, k), "bic": bic(ll, k, len(r))})
    save_table(cfg, "processes_fit", pd.DataFrame(rows))

    # Simulated daily returns vs data: do fitted models reproduce the tails?
    sim = pcfg["simulation"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(r.quantile(0.0005), r.quantile(0.9995), 120)
    ax[0].hist(r, bins=bins, density=True, histtype="step", lw=2, color="k", label=f"{asset} data")
    n_days = round(sim["horizon_years"] / dt)
    for name, proc in fitted.items():
        paths = proc.simulate(s0, sim["horizon_years"], n_days, sim["n_paths"], rng=seed)
        sim_r = np.diff(np.log(paths), axis=0).ravel()
        ax[0].hist(sim_r, bins=bins, density=True, histtype="step", label=name)
        q = np.quantile(paths, [0.05, 0.5, 0.95], axis=1)
        t = np.arange(n_days + 1)
        (line,) = ax[1].plot(t, q[1], label=f"{name} median")
        ax[1].fill_between(t, q[0], q[2], color=line.get_color(), alpha=0.15)
    ax[0].set_yscale("log")
    ax[0].set_title("Daily log-return density: data vs fitted models")
    ax[0].legend()
    ax[1].set_title(f"Simulated {sim['horizon_years']:g}y price paths (5-95% band)")
    ax[1].set_xlabel("trading days ahead")
    ax[1].legend()
    save_figure(cfg, "processes_fit", fig)
