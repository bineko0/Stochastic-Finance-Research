from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sfr.config import dt, main_asset
from sfr.data import load_log_returns, load_prices
from sfr.pricing import EuropeanOption, Market, bs_greeks
from sfr.pricing.ito import ito_lemma_gbm, ito_vs_stratonovich
from sfr.registry import PRICERS
from sfr.reporting import save_figure, save_table


def _fill_auto(spec, sigma):
    if isinstance(spec, dict):
        return {k: _fill_auto(v, sigma) for k, v in spec.items()}
    return sigma if spec == "auto" else spec


def _sigma(cfg, asset):
    s = cfg["pricing"]["sigma"]
    if s == "historical":
        td = round(1 / dt(cfg))
        return float(load_log_returns(asset, cfg).iloc[-td:].std(ddof=1) * np.sqrt(td))
    return float(s)


def run(cfg: dict) -> None:
    pc = cfg["pricing"]
    asset = main_asset(cfg, pc.get("underlying"))
    spot = pc.get("spot") or float(load_prices(asset, cfg)["adj_close"].iloc[-1])
    sigma = _sigma(cfg, asset)
    market = Market(spot, pc["rate"], pc["dividend"])
    print(f"  {asset}: spot={spot:.2f} sigma={sigma:.4f} r={market.rate} q={market.dividend}")

    pricers = {label: PRICERS.from_spec(_fill_auto(copy.deepcopy(spec), sigma))
               for label, spec in pc["pricers"].items()}
    g = pc["grid"]
    options = [EuropeanOption(round(m * spot, 4), T, kind)
               for T in g["maturities"] for m in g["moneyness"] for kind in g["kinds"]]

    rows = []
    for opt in options:
        row = {"kind": opt.kind, "T": opt.maturity, "K": opt.strike, "moneyness": round(opt.strike / spot, 4)}
        for label, pricer in pricers.items():
            res = pricer.price(opt, market)
            row[label] = res.price
            if res.std_error:
                row[f"{label}_se"] = res.std_error
        rows.append(row)
    table = pd.DataFrame(rows)
    bench = pc.get("benchmark")
    if bench in pricers:
        for label in pricers:
            if label != bench:
                table[f"{label}_minus_{bench}"] = table[label] - table[bench]
    save_table(cfg, "pricing_grid", table)

    greeks = pd.DataFrame([{"kind": o.kind, "T": o.maturity, "K": o.strike,
                            **bs_greeks(spot, o.strike, o.maturity, market.rate, sigma, market.dividend, o.kind)}
                           for o in options])
    save_table(cfg, "pricing_bs_greeks", greeks)

    # Monte Carlo convergence towards Black-Scholes
    conv = pc.get("convergence")
    if conv and bench in pricers:
        atm = EuropeanOption(spot, 1.0, "call")
        exact = pricers[bench].price(atm, market).price
        spec = _fill_auto(copy.deepcopy(pc["pricers"][conv["pricer"]]), sigma)
        crow = []
        for n in conv["n_paths"]:
            for cv in (False, True):
                res = PRICERS.from_spec({**spec, "n_paths": n, "control_variate": cv}).price(atm, market)
                crow.append({"n_paths": n, "control_variate": cv, "price": res.price,
                             "std_error": res.std_error, "error": res.price - exact})
        cdf = pd.DataFrame(crow)
        save_table(cfg, "pricing_mc_convergence", cdf)
        fig, ax = plt.subplots(figsize=(7, 4))
        for cv, grp in cdf.groupby("control_variate"):
            ax.errorbar(grp["n_paths"], grp["price"], yerr=1.96 * grp["std_error"], marker="o", capsize=3,
                        label="control variate" if cv else "plain MC")
        ax.axhline(exact, color="k", ls="--", label=f"{bench} = {exact:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("number of paths")
        ax.set_title("ATM 1y call: Monte Carlo convergence (95% CI)")
        ax.legend()
        save_figure(cfg, "pricing_mc_convergence", fig)

    seed = cfg["settings"]["seed"]
    save_table(cfg, "ito_integrals", ito_vs_stratonovich(rng=seed))
    save_table(cfg, "ito_lemma_gbm", ito_lemma_gbm(sigma=sigma, rng=seed))
