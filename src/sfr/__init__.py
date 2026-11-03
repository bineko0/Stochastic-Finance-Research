"""sfr: Stochastic Finance Research.

Layers (each only talks to the one below through a small, fixed interface):

    data          -> standardised prices / returns in data/processed/
    processes     -> StochasticProcess   (simulate, fit, log_likelihood)
    volatility    -> VolatilityModel     (fit, predict, forecast)
    pricing       -> Pricer              (price(option, market))
    risk          -> RiskModel           (fit, var, es)
    evaluation    -> metrics and statistical tests, model-agnostic
    stages        -> one runnable experiment per research module
"""

__version__ = "0.1.0"
