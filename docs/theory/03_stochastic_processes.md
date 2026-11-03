# 03 Stochastic processes

Code: `src/sfr/processes/` · Stage: `python -m sfr run processes` · Output: `reports/tables/processes_fit.csv`

## Brownian motion
- Definition, properties (independent Gaussian increments, continuity, quadratic variation = t).
- Simulation: cumulative sum of N(0, dt) increments.

## Geometric Brownian motion
- SDE dS = mu S dt + sigma S dW; solution via Itô's lemma on ln S.
- Exact scheme; MLE of (mu, sigma) from log returns.
- Assumptions tested on data: normality, independence, constant variance.

## Merton jump diffusion
- Compound Poisson jumps with lognormal sizes; compensator lam k.
- Likelihood as a Poisson mixture of normals; what the extra parameters buy (AIC/BIC).

## Observations
- ...
