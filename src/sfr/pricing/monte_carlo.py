from __future__ import annotations

import numpy as np

from sfr.pricing.base import Pricer
from sfr.pricing.instruments import PricingResult
from sfr.processes.base import StochasticProcess
from sfr.registry import PRICERS, PROCESSES


@PRICERS.register("monte_carlo")
class MonteCarloPricer(Pricer):
    """Risk-neutral Monte Carlo for ANY registered StochasticProcess.

    The process drift is replaced by r - q (process.with_drift), paths are
    simulated, and the discounted payoff is averaged. With control_variate=True
    the discounted terminal price (known mean S0 e^{-qT}) is used as control.
    """

    def __init__(self, process: str | StochasticProcess = "gbm", process_params: dict | None = None,
                 n_paths: int = 100_000, steps_per_year: int = 252, seed: int = 42,
                 control_variate: bool = True):
        if isinstance(process, str):
            process = PROCESSES.build(process, **(process_params or {}))
        self.process = process
        self.n_paths = n_paths
        self.steps_per_year = steps_per_year
        self.seed = seed
        self.control_variate = control_variate

    def price(self, option, market):
        T = option.maturity
        q_proc = self.process.with_drift(market.rate - market.dividend)
        n_steps = max(1, int(round(T * self.steps_per_year)))
        paths = q_proc.simulate(market.spot, T, n_steps, self.n_paths, rng=self.seed)
        disc = np.exp(-market.rate * T)
        y = disc * option.payoff(paths)

        if self.control_variate:
            c = disc * paths[-1]
            c_mean = market.spot * np.exp(-market.dividend * T)
            cov = np.cov(y, c)
            b = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
            y = y - b * (c - c_mean)

        return PricingResult(float(y.mean()), float(y.std(ddof=1) / np.sqrt(len(y))),
                             extra={"n_paths": self.n_paths, "n_steps": n_steps})
