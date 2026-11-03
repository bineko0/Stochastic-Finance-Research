# Stochastic Modeling in Financial Markets

From Brownian Motion to Stochastic Volatility and Derivatives Pricing.

**Main question:** How well can different stochastic models capture the dynamics of financial markets, and how can they be applied to derivatives pricing and risk analysis?

## Project map

| # | Research module | Code | Config | Stage | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | Research design | – | – | – | [docs/research_design/](docs/research_design/) |
| 2 | Data | [src/sfr/data/](src/sfr/data/) | [configs/data.yaml](configs/data.yaml) | `data` | |
| 3 | Stochastic processes: Brownian, GBM, jump diffusion | [src/sfr/processes/](src/sfr/processes/) | [configs/models.yaml](configs/models.yaml) | `processes` | [theory](docs/theory/03_stochastic_processes.md) |
| 4 | Volatility: historical, EWMA, GARCH, Heston | [src/sfr/volatility/](src/sfr/volatility/) | [configs/models.yaml](configs/models.yaml) | `volatility` | [theory](docs/theory/04_volatility_models.md) |
| 5 | Pricing: Black–Scholes, Itô calculus, Monte Carlo | [src/sfr/pricing/](src/sfr/pricing/) | [configs/pricing.yaml](configs/pricing.yaml) | `pricing` | [theory](docs/theory/05_derivative_pricing.md) |
| 6 | Risk analysis: VaR, ES, backtests | [src/sfr/risk/](src/sfr/risk/) | [configs/risk.yaml](configs/risk.yaml) | `risk` | [theory](docs/theory/06_risk_analysis.md) |
| 7 | Report | [src/sfr/stages/report.py](src/sfr/stages/report.py) | – | `report` | [reports/final/](reports/final/) |

```
configs/            every choice lives here: sources, assets, model parameters
data/raw/<source>/  as downloaded (never edited)
data/processed/     standardised prices/, returns/, volatility/  <- the only thing models read
data/external/      files you add by hand (Bloomberg exports, ...)
src/sfr/            the package: data, processes, volatility, pricing, risk, evaluation, stages
notebooks/          exploration (# %% cell scripts), import from sfr
reports/            figures/, tables/ (generated), summary.md (generated), final/ (written by you)
tests/              unit tests + end-to-end pipeline test on synthetic data
docs/               research design, theory notes, ARCHITECTURE.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                  # makes `import sfr` work everywhere (notebooks too)
```

In VSCode: `Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv`.

## Run

```bash
python -m sfr list                        # stages and every registered component
python -m sfr run all                     # data -> processes -> volatility -> pricing -> risk -> report
python -m sfr run volatility risk         # just some stages (data must exist)
python -m sfr run all --set data.default_source=synthetic   # offline, no internet
python -m pytest                          # tests, including a full synthetic pipeline run
```

Results: `reports/tables/*.csv`, `reports/figures/*.png`, collected in `reports/summary.md`.

## Why it is modular

Each part only depends on a small interface, never on another part's internals:

- **Data**: any source → the same standardised format in `data/processed/`. Models read through
  `sfr.data.load_returns(...)` only, so changing Yahoo → CSV/Bloomberg is a config change.
- **Models**: every process, volatility model, pricer and risk model registers under a name and is
  picked in YAML. Adding one = one new class; nothing else changes.
- **Stages**: communicate only through files, so each can be re-run or rewritten alone.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for recipes (new source, new model, new stage).

## Rules

- `data/raw/` is never edited by hand. Everything downstream is reproducible from code.
- Every random number comes from a seeded `numpy.random.default_rng`.
- Every number and figure in the report is produced by a stage, not by hand.
- Run `python -m pytest` before committing.
