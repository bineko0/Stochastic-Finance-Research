"""Quality checks on standardised prices, and the minimal cleaning applied after them."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    name: str
    errors: list[str] = field(default_factory=list)    # data unusable
    warnings: list[str] = field(default_factory=list)  # usable, but look at it

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return "; ".join(self.errors + self.warnings) or "ok"


def validate_prices(
    df: pd.DataFrame,
    name: str = "",
    max_nan_frac: float = 0.05,
    max_abs_log_return: float = 0.25,
    max_gap_days: int = 7,
) -> ValidationReport:
    report = ValidationReport(name)
    if df.empty:
        report.errors.append("no rows")
        return report

    price = df["adj_close"]
    nan_frac = price.isna().mean()
    if nan_frac == 1:
        report.errors.append("all prices missing")
        return report
    if nan_frac > max_nan_frac:
        report.warnings.append(f"{nan_frac:.1%} prices missing")
    if (price <= 0).any():
        report.warnings.append(f"{int((price <= 0).sum())} non-positive prices (dropped)")

    clean = price[price > 0].dropna()
    big = (np.log(clean).diff().abs() > max_abs_log_return).sum()
    if big:
        report.warnings.append(f"{int(big)} daily moves > {max_abs_log_return:.0%}")
    gaps = clean.index.to_series().diff().dt.days
    if (gaps > max_gap_days).any():
        report.warnings.append(f"{int((gaps > max_gap_days).sum())} gaps > {max_gap_days} days")
    if len(clean) < 30:
        report.errors.append(f"only {len(clean)} valid prices")
    return report


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    price = df["adj_close"]
    return df[price.notna() & (price > 0)]
