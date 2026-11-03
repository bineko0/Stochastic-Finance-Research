from __future__ import annotations

from abc import ABC, abstractmethod

from sfr.pricing.instruments import EuropeanOption, Market, PricingResult


class Pricer(ABC):
    @abstractmethod
    def price(self, option: EuropeanOption, market: Market) -> PricingResult: ...
