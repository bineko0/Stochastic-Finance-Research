from __future__ import annotations

import numpy as np
from scipy import integrate

from sfr.pricing.base import Pricer
from sfr.pricing.instruments import PricingResult
from sfr.registry import PRICERS


def heston_price(S, K, T, r, q, v0, kappa, theta, xi, rho, kind="call", u_max=200.0):
    """Semi-closed-form Heston price (characteristic function, 'little trap' form)."""
    x = np.log(S)

    def cf(u):
        iu = 1j * u
        b = kappa - rho * xi * iu
        d = np.sqrt(b**2 + xi**2 * (iu + u**2))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = (r - q) * iu * T + kappa * theta / xi**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0 + iu * x)

    lnK = np.log(K)
    forward = cf(-1j)  # = S e^{(r-q)T}
    p1 = 0.5 + integrate.quad(lambda u: (np.exp(-1j * u * lnK) * cf(u - 1j) / (1j * u * forward)).real,
                              1e-8, u_max, limit=200)[0] / np.pi
    p2 = 0.5 + integrate.quad(lambda u: (np.exp(-1j * u * lnK) * cf(u) / (1j * u)).real,
                              1e-8, u_max, limit=200)[0] / np.pi
    call = S * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2
    if kind == "call":
        return float(call)
    return float(call - S * np.exp(-q * T) + K * np.exp(-r * T))


@PRICERS.register("heston_fourier")
class HestonFourierPricer(Pricer):
    def __init__(self, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7):
        self.v0, self.kappa, self.theta, self.xi, self.rho = v0, kappa, theta, xi, rho

    def price(self, option, market):
        p = heston_price(market.spot, option.strike, option.maturity, market.rate, market.dividend,
                         self.v0, self.kappa, self.theta, self.xi, self.rho, option.kind)
        return PricingResult(p)
