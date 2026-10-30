# Evaluation Design (decide now, not in Week 8)

Fixing this early prevents tuning the evaluation to the results.

## Sample split

- Full sample: 2010-01-01 to latest.
- Train: ______ to ______ ; Test (never touched during fitting): ______ to ______.
- Window scheme: rolling / expanding (choose one): ______ ; re-estimation frequency: ______.

## Volatility forecasting

- Horizon: 1 day / 5 days / 21 days (choose): ______
- Realised-vol proxy (ground truth): squared return / 5-min realised variance (not available from Yahoo) / 21-day forward realised vol: ______
- Models compared: rolling vol, EWMA (lambda = 0.94), GARCH(1,1) [, Heston if implemented].
- Loss functions: MSE and QLIKE (QLIKE is robust to the noisy proxy): `QLIKE = RV/h - ln(RV/h) - 1`.
- Significance: Diebold–Mariano test (optional).

## Model fit (in-sample)

- GBM: Jarque–Bera, QQ plot, kurtosis, Ljung–Box on r and r^2.
- GARCH: standardised residuals i.i.d.? Ljung–Box on z and z^2, ARCH-LM.
- Information criteria: AIC / BIC.

## Pricing checks (Module 3)

- MC vs Black–Scholes: price difference vs standard error, convergence in N.
- Market comparison: implied vol from snapshot vs model vol.

## Risk backtests (Module 4)

- 1-day 95% and 99% VaR, rolling.
- Kupiec unconditional coverage; Christoffersen independence.

## Reproducibility

- `numpy.random.default_rng(seed)`, seed recorded in each notebook.
- All figures saved from code, not manually.
