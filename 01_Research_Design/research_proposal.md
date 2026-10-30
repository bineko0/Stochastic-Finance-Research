# Research Proposal: Stochastic Modeling in Financial Markets

**Author:** Le Duc Long  **Duration:** 14 weeks  **Split:** 50% stochastic mathematics / 50% empirical + computational

## 1. Main question

How well can different stochastic models capture the dynamics of financial markets, and how can they be applied to derivatives pricing and risk analysis?

## 2. Sub-questions

1. Asset prices: what are the limits of GBM on historical data? Does Merton jump-diffusion describe extremes better?
2. Volatility: constant, time-varying, or stochastic? How do GARCH (and, if time permits, Heston) differ, and which forecasts better out-of-sample?
3. Pricing: how do Itô's lemma, martingales and risk-neutral measure lead to Black–Scholes, and do Monte Carlo and closed form agree?
4. Risk: how do model assumptions change VaR / ES?

## 3. Scope

**Core:** Brownian motion, GBM, SDEs; historical vol, EWMA, GARCH(1,1); Itô's lemma, martingale basics; Black–Scholes + Monte Carlo; model fitting and out-of-sample evaluation.

**Extension (if time):** Poisson / Merton jumps, OU, CIR, Heston, Girsanov, variance reduction, portfolio optimisation.

## 4. Data

See `02_Data/README.md`. Primary: S&P 500 daily, 2010 to present. Option data: snapshot or simulated if no good history is available.

## 5. Methods

- Estimation: MLE / moment-based for GBM and GARCH.
- Simulation: Euler and exact schemes, fixed seeds.
- Evaluation: see `evaluation_design.md`.

## 6. Deliverables per module

| Module | Deliverable |
| --- | --- |
| 1 | Simulator for BM / GBM (/ jumps) vs real data |
| 2 | Notebooks comparing volatility models |
| 3 | European option pricer (analytic + MC) with sensitivities |
| 4 | MC risk simulator: VaR, ES across models |

## 7. Timeline

| Week | Topic | Output |
| --- | --- | --- |
| 1 | Research design + data | Proposal, dataset |
| 2 | Random walk + BM | Simulation notebook |
| 3 | GBM + SDE | GBM implementation |
| 4 | Poisson + jump-diffusion (ext.) | Jump simulation |
| 5 | Historical vol + EWMA | Vol analysis |
| 6 | ARCH + GARCH | GARCH estimation |
| 7 | OU + CIR + Heston (ext.) | SV simulation |
| 8 | Fitting + forecast evaluation | Model comparison |
| 9 | Itô integral + lemma | Math notes |
| 10 | Martingales + risk-neutral | Pricing framework |
| 11 | Black–Scholes + MC pricing | Pricing engine |
| 12 | MC + variance reduction | Experiments |
| 13 | VaR + ES | Risk analysis |
| 14 | Final analysis | Report |

## 8. Risks

- Weeks 7–8 are overloaded; treat Heston as simulation-only unless ahead.
- Option data quality; fallback is a snapshot or simulated data.
- Theory/implementation drift: write notes and code side by side each week.

## 9. Week 1 checklist

- [ ] Finalise this proposal
- [ ] Fix evaluation design (`evaluation_design.md`)
- [ ] Download and process data
- [ ] Run `week1_exploration.py`, fill in observations
- [ ] Start literature map (`literature_map.md`)
