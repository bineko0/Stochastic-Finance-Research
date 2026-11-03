from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class EuropeanOption:
    strike: float
    maturity: float                       # years
    kind: Literal["call", "put"] = "call"

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """paths: (n_steps + 1, n_paths). Path-dependent options override this using all rows."""
        s_T = paths[-1]
        if self.kind == "call":
            return np.maximum(s_T - self.strike, 0.0)
        return np.maximum(self.strike - s_T, 0.0)


@dataclass(frozen=True)
class Market:
    spot: float
    rate: float = 0.0       # continuously compounded
    dividend: float = 0.0   # continuous dividend yield


@dataclass
class PricingResult:
    price: float
    std_error: float = 0.0  # 0 for closed-form prices
    extra: dict = field(default_factory=dict)

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error
