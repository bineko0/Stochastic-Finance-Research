import numpy as np
import pandas as pd
import pytest

from sfr.processes import GBM, MertonJumpDiffusion
from sfr.registry import PROCESSES, VOL_MODELS
from sfr.volatility import GARCH11, Heston

DT = 1 / 252


@pytest.mark.parametrize("name", ["brownian", "gbm", "merton", "heston"])
def test_simulate_shape_and_start(name):
    paths = PROCESSES.build(name).simulate(100.0, 1.0, 50, 7, rng=0)
    assert paths.shape == (51, 7)
    assert np.allclose(paths[0], 100.0)


@pytest.mark.parametrize("proc", [GBM(0.08, 0.25), MertonJumpDiffusion(mu=0.08), Heston(mu=0.08)])
def test_expected_terminal_price_is_exp_mu_t(proc):
    s = proc.simulate(1.0, 1.0, 50, 100_000, rng=1)[-1]
    assert s.mean() == pytest.approx(np.exp(0.08), rel=0.01)


def test_gbm_fit_recovers_parameters():
    s = GBM(0.1, 0.3).simulate(1.0, 40.0, 40 * 252, 1, rng=2)[:, 0]
    fit = GBM.fit(np.diff(np.log(s)), DT)
    assert fit.sigma == pytest.approx(0.3, rel=0.03)


def test_merton_beats_gbm_on_jump_data():
    s = MertonJumpDiffusion(0.05, 0.12, 5.0, -0.04, 0.05).simulate(1.0, 15.0, 15 * 252, 1, rng=3)[:, 0]
    r = np.diff(np.log(s))
    assert MertonJumpDiffusion.fit(r, DT).log_likelihood(r, DT) > GBM.fit(r, DT).log_likelihood(r, DT)


def _garch_sample(n=4000, omega=2e-6, alpha=0.08, beta=0.9, seed=4):
    rng = np.random.default_rng(seed)
    r, var = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    return pd.Series(r, index=pd.bdate_range("2000-01-01", periods=n))


def test_garch_recovers_persistence():
    m = GARCH11().fit(_garch_sample())
    assert m.persistence == pytest.approx(0.98, abs=0.02)
    assert m.alpha == pytest.approx(0.08, abs=0.04)


@pytest.mark.parametrize("name", ["historical", "ewma", "garch", "heston"])
def test_volatility_predict_has_no_lookahead(name):
    r = _garch_sample(1500)
    model = VOL_MODELS.build(name).fit(r.iloc[:1000])
    base = model.predict(r)
    shocked = r.copy()
    shocked.iloc[1200:] *= 10
    after = model.predict(shocked)
    pd.testing.assert_series_equal(base.iloc[:1201], after.iloc[:1201])
    assert len(model.forecast(r, 5)) == 5
