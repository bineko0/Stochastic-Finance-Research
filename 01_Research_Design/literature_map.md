# Literature Map

Fill in as you read. One line per source: what it gives you, which module, status.

## Textbooks (anchor sources)

| Source | Module | Use | Status |
| --- | --- | --- | --- |
| Shreve, *Stochastic Calculus for Finance II* | 1, 3 | BM, Itô, martingales, risk-neutral pricing, Black–Scholes | |
| Hull, *Options, Futures, and Other Derivatives* | 3, 4 | Practitioner view, option data, VaR | |
| Glasserman, *Monte Carlo Methods in Financial Engineering* | 3, 4 | MC, variance reduction | |
| Tsay, *Analysis of Financial Time Series* | 2 | GARCH, volatility models | |
| Gatheral, *The Volatility Surface* | 2, 3 | Heston, stochastic vol | |

## Papers

| Paper | Topic | Module | Notes |
| --- | --- | --- | --- |
| Black & Scholes (1973) | Option pricing | 3 | |
| Merton (1976) | Jump-diffusion | 1 | |
| Engle (1982) | ARCH | 2 | |
| Bollerslev (1986) | GARCH | 2 | |
| Heston (1993) | Stochastic volatility | 2 | |
| Hansen & Lunde (2005) | Does anything beat GARCH(1,1)? | 2 | |
| Patton (2011) | Volatility forecast comparison, QLIKE | 2 | |
| Cont (2001) | Stylised facts of returns | 1, 2 | |

## Relationships between models

GBM (constant vol) -> Merton (adds jumps) -> GARCH (time-varying vol, discrete) -> Heston (stochastic vol, continuous).
Black–Scholes assumes GBM; MC pricing generalises it to Merton / Heston.
