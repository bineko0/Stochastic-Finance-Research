"""Runnable stages, one per research module. Each has run(cfg) and communicates with
the others ONLY through files (data/processed/, reports/), so any stage can be
re-run, changed or replaced on its own.
"""
from __future__ import annotations

import importlib
import time

STAGES = {
    "data": "Download/load, standardise, validate, compute returns",
    "processes": "Fit and simulate Brownian / GBM / Merton / Heston",
    "volatility": "Out-of-sample volatility forecast comparison",
    "pricing": "Black-Scholes vs Monte Carlo vs Heston, Ito checks",
    "risk": "Rolling VaR / ES backtests",
    "report": "Collect tables and figures into reports/summary.md",
}


def run_stage(name: str, cfg: dict) -> None:
    if name not in STAGES:
        raise KeyError(f"Unknown stage '{name}'. Available: {', '.join(STAGES)}")
    print(f"\n=== {name}: {STAGES[name]} ===")
    t0 = time.perf_counter()
    importlib.import_module(f"sfr.stages.{name}").run(cfg)
    print(f"=== {name} done in {time.perf_counter() - t0:.1f}s ===")
