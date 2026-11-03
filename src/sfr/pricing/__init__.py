"""Derivative pricing. Importing this package registers the built-in pricers.

Every pricer has the same call:  pricer.price(option, market) -> PricingResult
"""
from sfr.pricing import black_scholes, heston_fourier, monte_carlo  # noqa: F401
from sfr.pricing.base import Pricer
from sfr.pricing.black_scholes import BlackScholesPricer, bs_greeks, bs_price, implied_volatility
from sfr.pricing.heston_fourier import HestonFourierPricer, heston_price
from sfr.pricing.instruments import EuropeanOption, Market, PricingResult
from sfr.pricing.monte_carlo import MonteCarloPricer

__all__ = ["Pricer", "EuropeanOption", "Market", "PricingResult", "BlackScholesPricer", "MonteCarloPricer",
           "HestonFourierPricer", "bs_price", "bs_greeks", "implied_volatility", "heston_price"]
