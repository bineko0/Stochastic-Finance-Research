# 04 Volatility models

Code: `src/sfr/volatility/` · Stage: `python -m sfr run volatility` · Output: `reports/tables/volatility_forecast_eval.csv`

## Historical volatility
## EWMA (RiskMetrics, lambda = 0.94)
## GARCH(1,1)
- Recursion, stationarity (alpha + beta < 1), long-run variance, forecast term structure.
- MLE with normal and Student-t innovations.

## Heston stochastic volatility
- CIR variance, Feller condition, correlation rho (leverage effect).
- Here: rough moment estimate from realised variance; for pricing, calibrate to options.

## Evaluation
- Proxy: squared return. Losses: MSE, QLIKE. Diebold–Mariano test.

## Observations
- ...
