from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from sfr.data.sources.base import DataSource
from sfr.registry import SOURCES


@SOURCES.register("synthetic")
class SyntheticSource(DataSource):
    """Simulated GBM + Merton jumps on business days, for offline runs and tests.

    Deterministic: the same (symbol, seed) always gives the same series. Kept
    self-contained on purpose, so the data layer does not depend on sfr.processes.
    """

    def __init__(self, mu=0.07, sigma=0.18, s0=100.0, jump_intensity=0.0, jump_mean=-0.03,
                 jump_std=0.04, seed=0, start="2010-01-01", end="2024-12-31"):
        self.mu, self.sigma, self.s0 = mu, sigma, s0
        self.lam, self.jump_mean, self.jump_std = jump_intensity, jump_mean, jump_std
        self.seed, self.start, self.end = seed, start, end

    def fetch(self, symbol, start, end):
        dates = pd.bdate_range(start or self.start, end or self.end, name="Date")
        rng = np.random.default_rng(zlib.crc32(symbol.encode()) + self.seed)
        n, dt = len(dates), 1 / 252
        k = np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1
        n_jumps = rng.poisson(self.lam * dt, n)
        jumps = n_jumps * self.jump_mean + np.sqrt(n_jumps) * self.jump_std * rng.standard_normal(n)
        log_ret = ((self.mu - self.lam * k - 0.5 * self.sigma**2) * dt
                   + self.sigma * np.sqrt(dt) * rng.standard_normal(n) + jumps)
        log_ret[0] = 0.0
        price = self.s0 * np.exp(np.cumsum(log_ret))
        return pd.DataFrame({"Close": price, "Adj Close": price}, index=dates)
