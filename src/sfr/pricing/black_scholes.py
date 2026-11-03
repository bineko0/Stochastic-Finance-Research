from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from sfr.pricing.base import Pricer
from sfr.pricing.instruments import PricingResult
from sfr.registry import PRICERS

N, n = stats.norm.cdf, stats.norm.pdf


def _d1_d2(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)
    return K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict:
    """Theta per year, vega and rho per unit (1.00 = 100%) change."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    gamma = disc_q * n(d1) / (S * sigma * np.sqrt(T))
    vega = S * disc_q * n(d1) * np.sqrt(T)
    common = -S * disc_q * n(d1) * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta = disc_q * N(d1)
        theta = common - r * K * disc_r * N(d2) + q * S * disc_q * N(d1)
        rho = K * T * disc_r * N(d2)
    else:
        delta = -disc_q * N(-d1)
        theta = common + r * K * disc_r * N(-d2) - q * S * disc_q * N(-d1)
        rho = -K * T * disc_r * N(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_volatility(price, S, K, T, r, q=0.0, kind="call", lo=1e-6, hi=5.0) -> float:
    f = lambda s: bs_price(S, K, T, r, s, q, kind) - price  # noqa: E731
    if f(lo) * f(hi) > 0:
        return float("nan")  # price outside no-arbitrage bounds
    return optimize.brentq(f, lo, hi)


@PRICERS.register("black_scholes")
class BlackScholesPricer(Pricer):
    def __init__(self, sigma: float = 0.2):
        self.sigma = sigma

    def price(self, option, market):
        args = (market.spot, option.strike, option.maturity, market.rate, self.sigma, market.dividend, option.kind)
        return PricingResult(float(bs_price(*args)), extra=bs_greeks(*args))
