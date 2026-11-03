"""Stochastic processes. Importing this package registers the built-ins.

Heston lives in sfr.volatility.heston (it is a stochastic-volatility model) but
is registered here too, as process "heston", so it can be simulated and used
by the Monte Carlo pricer like any other process.
"""
from sfr.processes import brownian, gbm, merton  # noqa: F401
from sfr.processes.base import StochasticProcess
from sfr.processes.brownian import BrownianMotion
from sfr.processes.gbm import GBM
from sfr.processes.merton import MertonJumpDiffusion

__all__ = ["StochasticProcess", "BrownianMotion", "GBM", "MertonJumpDiffusion"]
