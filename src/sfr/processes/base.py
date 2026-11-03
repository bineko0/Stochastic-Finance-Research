from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass
class StochasticProcess(ABC):
    """A continuous-time model, parameters annualised (time unit = 1 year).

    simulate(...)          -> paths of shape (n_steps + 1, n_paths), row 0 = x0
    fit(log_returns, dt)   -> estimated instance (classmethod)
    log_likelihood(r, dt)  -> log-likelihood of observed increments
    with_drift(mu)         -> copy with a new drift; the MC pricer uses it to move
                              to the risk-neutral measure (mu = r - q)
    """

    @abstractmethod
    def simulate(self, x0: float, T: float, n_steps: int, n_paths: int = 1, rng=None) -> np.ndarray: ...

    @classmethod
    def fit(cls, log_returns, dt: float = 1 / 252) -> "StochasticProcess":
        raise NotImplementedError(f"{cls.__name__} has no fit()")

    def log_likelihood(self, log_returns, dt: float = 1 / 252) -> float:
        raise NotImplementedError(f"{type(self).__name__} has no log_likelihood()")

    def with_drift(self, mu: float) -> "StochasticProcess":
        return replace(self, mu=mu)

    def params(self) -> dict:
        return asdict(self)

    @property
    def n_params(self) -> int:
        return len(self.params())


def cumulate(increments: np.ndarray) -> np.ndarray:
    """(n_steps, n_paths) increments -> (n_steps + 1, n_paths) cumulative sum starting at 0."""
    return np.vstack([np.zeros((1, increments.shape[1])), np.cumsum(increments, axis=0)])
