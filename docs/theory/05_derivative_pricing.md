# 05 Derivative pricing

Code: `src/sfr/pricing/` · Stage: `python -m sfr run pricing` · Outputs: `pricing_grid.csv`, `pricing_mc_convergence.csv`, `ito_*.csv`

## Itô calculus
- Itô integral (left point) vs Stratonovich; quadratic variation; Itô's lemma.
- Numerical checks: `sfr.pricing.ito`.

## Black–Scholes
- Replication / risk-neutral derivation, closed form, Greeks, implied volatility.

## Monte Carlo
- Risk-neutral simulation for any process (GBM, Merton, Heston), standard error, convergence O(1/sqrt N).
- Variance reduction: control variate (implemented); antithetic, importance sampling (extensions).

## Heston semi-closed form
- Characteristic function pricing; used to validate Heston Monte Carlo.

## Observations
- ...
