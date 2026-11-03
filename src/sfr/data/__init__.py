"""Data layer.

Downstream code reads data ONLY through the store functions below, never from a
source directly. That is what makes the source swappable.
"""
from sfr.data.pipeline import build_dataset
from sfr.data.store import available_assets, load_log_returns, load_prices, load_returns

__all__ = ["build_dataset", "available_assets", "load_prices", "load_returns", "load_log_returns"]
