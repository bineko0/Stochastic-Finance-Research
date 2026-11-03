import numpy as np
import pandas as pd
import pytest

from sfr.pricing import (BlackScholesPricer, EuropeanOption, HestonFourierPricer, Market, MonteCarloPricer,
                         bs_price, implied_volatility)
from sfr.pricing.ito import ito_lemma_gbm, ito_vs_stratonovich
from sfr.registry import RISK_MODELS
from sfr.risk import kupiec_pof

MKT = Market(100.0, 0.03, 0.01)


def test_put_call_parity():
    c = bs_price(100, 95, 0.7, 0.03, 0.25, 0.01, "call")
    p = bs_price(100, 95, 0.7, 0.03, 0.25, 0.01, "put")
    assert c - p == pytest.approx(100 * np.exp(-0.01 * 0.7) - 95 * np.exp(-0.03 * 0.7))


def test_implied_vol_round_trip():
    p = bs_price(100, 110, 0.5, 0.02, 0.31)
    assert implied_volatility(p, 100, 110, 0.5, 0.02) == pytest.approx(0.31, abs=1e-8)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_monte_carlo_matches_black_scholes(kind):
    opt = EuropeanOption(105, 1.0, kind)
    bs = BlackScholesPricer(0.2).price(opt, MKT).price
    mc = MonteCarloPricer("gbm", {"sigma": 0.2}, n_paths=200_000, steps_per_year=1).price(opt, MKT)
    assert abs(mc.price - bs) < 4 * mc.std_error


def test_heston_fourier_reduces_to_bs_and_matches_mc():
    opt = EuropeanOption(100, 1.0, "call")
    tiny_xi = HestonFourierPricer(v0=0.04, theta=0.04, xi=1e-3, rho=0.0).price(opt, MKT).price
    assert tiny_xi == pytest.approx(BlackScholesPricer(0.2).price(opt, MKT).price, rel=1e-3)

    params = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7}
    exact = HestonFourierPricer(**params).price(opt, MKT).price
    mc = MonteCarloPricer("heston", params, n_paths=50_000, steps_per_year=200).price(opt, MKT)
    assert abs(mc.price - exact) < 4 * mc.std_error + 0.05  # + Euler bias


def test_ito_checks():
    t = ito_vs_stratonovich(n_paths=500, rng=0)
    assert t.loc[0, "mean_abs_error_vs_closed_form"] < 0.05
    g = ito_lemma_gbm(n_paths=20000, rng=0).iloc[0]
    assert abs(g["simulated (Euler on S)"] - g["Ito: (mu - sigma^2/2) T"]) < 0.01


@pytest.mark.parametrize("name", ["historical", "gaussian", "student_t", "monte_carlo", "filtered_historical"])
def test_risk_models_on_normal_data(name):
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 2000), index=pd.bdate_range("2010-01-01", periods=2000))
    m = RISK_MODELS.build(name).fit(r)
    assert m.var(0.99) == pytest.approx(0.0233, rel=0.15)
    assert m.es(0.99) > m.var(0.99) > m.var(0.95) > 0


def test_kupiec_accepts_correct_rate_rejects_wrong():
    rng = np.random.default_rng(1)
    assert kupiec_pof(rng.random(2000) < 0.01, 0.99)["kupiec_p"] > 0.05
    assert kupiec_pof(rng.random(2000) < 0.05, 0.99)["kupiec_p"] < 0.01
