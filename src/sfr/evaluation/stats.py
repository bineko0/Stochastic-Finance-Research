"""Descriptive statistics and the stylised facts that motivate each model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def acf(x, lags: int = 40) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = x @ x
    return np.array([1.0] + [(x[k:] @ x[:-k]) / denom for k in range(1, lags + 1)])


def ljung_box(x, lags: int = 10) -> dict:
    n = len(x)
    rho = acf(x, lags)[1:]
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, lags + 1)))
    return {"lb_stat": float(q), "lb_p": float(stats.chi2.sf(q, lags))}


def stylized_facts(r: pd.Series, trading_days: int = 252) -> dict:
    """Tests the GBM assumptions: normal, independent, constant variance."""
    r = r.dropna()
    jb = stats.jarque_bera(r)
    return {
        "n": len(r),
        "start": r.index.min().date(),
        "end": r.index.max().date(),
        "mu_annual": r.mean() * trading_days,
        "sigma_annual": r.std(ddof=1) * np.sqrt(trading_days),
        "skew": stats.skew(r),
        "excess_kurtosis": stats.kurtosis(r),
        "min": r.min(),
        "max": r.max(),
        "jarque_bera_p": float(jb.pvalue),
        "ljung_box_r_p": ljung_box(r, 10)["lb_p"],      # serial correlation in returns
        "ljung_box_r2_p": ljung_box(r**2, 10)["lb_p"],  # volatility clustering
    }
