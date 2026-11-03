"""Forecast losses and model-comparison criteria."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mse(proxy_var, forecast_var) -> np.ndarray:
    """Per-observation squared error on variances."""
    return (np.asarray(proxy_var) - np.asarray(forecast_var)) ** 2


def qlike(proxy_var, forecast_var) -> np.ndarray:
    """Patton (2011) QLIKE: RV/h - ln(RV/h) - 1, robust to a noisy variance proxy."""
    ratio = np.maximum(np.asarray(proxy_var), 1e-12) / np.asarray(forecast_var)
    return ratio - np.log(ratio) - 1


def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict:
    """H0: equal expected loss. stat < 0 means model A is better. Newey-West variance, h-1 lags."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, h):
        lrv += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    return {"dm_stat": float(stat), "dm_p": float(2 * stats.norm.sf(abs(stat)))}


def aic(log_lik: float, k: int) -> float:
    return 2 * k - 2 * log_lik


def bic(log_lik: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * log_lik
