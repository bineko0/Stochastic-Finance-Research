"""Volatility models. Importing this package registers the built-ins."""
from sfr.volatility import ewma, garch, heston, historical  # noqa: F401
from sfr.volatility.base import VolatilityModel, annualize
from sfr.volatility.ewma import EWMA
from sfr.volatility.garch import GARCH11
from sfr.volatility.heston import Heston, HestonVolatility
from sfr.volatility.historical import HistoricalVolatility

__all__ = ["VolatilityModel", "annualize", "EWMA", "GARCH11", "Heston", "HestonVolatility",
           "HistoricalVolatility"]
