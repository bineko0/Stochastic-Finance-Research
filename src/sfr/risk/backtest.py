"""VaR backtests. A hit is a day whose loss exceeds VaR (return < -VaR)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(hits, level: float) -> dict:
    """Unconditional coverage: is the hit rate equal to 1 - level?"""
    hits = np.asarray(hits, bool)
    n, x, p = len(hits), int(hits.sum()), 1 - level
    pi = x / n
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - pi) - _xlogy(x, pi))
    return {"n": n, "hits": x, "hit_rate": pi, "expected_rate": p, "kupiec_lr": lr,
            "kupiec_p": float(stats.chi2.sf(lr, 1))}


def christoffersen_independence(hits) -> dict:
    """Are hits independent over time (no clustering of VaR breaches)?"""
    h = np.asarray(hits, int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum()); n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum()); n11 = int(((prev == 1) & (curr == 1)).sum())
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(len(curr), 1)
    lr = -2 * (_xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
               - _xlogy(n00, 1 - pi0) - _xlogy(n01, pi0) - _xlogy(n10, 1 - pi1) - _xlogy(n11, pi1))
    return {"christoffersen_lr": lr, "christoffersen_p": float(stats.chi2.sf(lr, 1))}


def backtest_var(returns: pd.Series, var: pd.Series, level: float, es: pd.Series | None = None) -> dict:
    r, v = returns.align(var, join="inner")
    hits = (r < -v).to_numpy()
    out = {**kupiec_pof(hits, level), **christoffersen_independence(hits)}
    if es is not None and hits.any():
        # average realised loss on hit days vs predicted ES (should be close)
        e = es.reindex(r.index)
        out["mean_loss_on_hits"] = float(-r[hits].mean())
        out["mean_es_on_hits"] = float(e[hits].mean())
    return out
