"""Numerical Itô calculus: the facts behind Black-Scholes, checked by simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def brownian_increments(T=1.0, n_steps=1000, n_paths=1, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    return np.sqrt(T / n_steps) * rng.standard_normal((n_steps, n_paths))


def ito_integral(integrand_left: np.ndarray, dW: np.ndarray) -> np.ndarray:
    """sum f(t_i) dW_i, integrand evaluated at the LEFT point (non-anticipating)."""
    return np.sum(integrand_left * dW, axis=0)


def stratonovich_integral(integrand: np.ndarray, dW: np.ndarray) -> np.ndarray:
    """sum (f(t_i) + f(t_{i+1}))/2 dW_i;  integrand has n_steps + 1 rows."""
    return np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dW, axis=0)


def quadratic_variation(path: np.ndarray) -> np.ndarray:
    return np.sum(np.diff(path, axis=0) ** 2, axis=0)


def ito_vs_stratonovich(T=1.0, n_steps=1000, n_paths=2000, rng=None) -> pd.DataFrame:
    """int W dW: Itô = (W_T^2 - T)/2, Stratonovich = W_T^2 / 2. The gap is half the QV."""
    dW = brownian_increments(T, n_steps, n_paths, rng)
    W = np.vstack([np.zeros((1, n_paths)), np.cumsum(dW, axis=0)])
    WT = W[-1]
    ito = ito_integral(W[:-1], dW)
    strat = stratonovich_integral(W, dW)
    return pd.DataFrame({
        "quantity": ["Ito int W dW", "Stratonovich int W dW", "quadratic variation [W]_T"],
        "simulated_mean": [ito.mean(), strat.mean(), quadratic_variation(W).mean()],
        "theory_mean": [0.0, T / 2, T],
        "mean_abs_error_vs_closed_form": [np.abs(ito - (WT**2 - T) / 2).mean(),
                                          np.abs(strat - WT**2 / 2).mean(), np.nan],
    })


def ito_lemma_gbm(mu=0.08, sigma=0.3, T=1.0, n_steps=252, n_paths=20000, rng=None) -> pd.DataFrame:
    """Itô's lemma on f(S) = ln S gives E[ln S_T/S_0] = (mu - sigma^2/2) T, not mu T."""
    rng = np.random.default_rng(rng)
    dt = T / n_steps
    dW = np.sqrt(dt) * rng.standard_normal((n_steps, n_paths))
    S = np.ones(n_paths)
    for k in range(n_steps):                 # Euler on dS = mu S dt + sigma S dW, no exp()
        S = S * (1 + mu * dt + sigma * dW[k])
    return pd.DataFrame({
        "quantity": ["E[ln S_T / S_0]"],
        "simulated (Euler on S)": [np.log(S).mean()],
        "Ito: (mu - sigma^2/2) T": [(mu - 0.5 * sigma**2) * T],
        "naive: mu T": [mu * T],
    })
