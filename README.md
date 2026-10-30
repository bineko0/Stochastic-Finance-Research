# Stochastic Modeling in Financial Markets

From Brownian Motion to Stochastic Volatility and Derivatives Pricing.

**Main question:** How well can different stochastic models capture the dynamics of financial markets, and how can they be applied to derivatives pricing and risk analysis?

## Structure

| Folder | Content |
| --- | --- |
| `01_Research_Design/` | Proposal, evaluation design, literature map |
| `02_Data/` | `Raw/` (never edited), `Processed/` (generated), data scripts |
| `03_Stochastic_Processes/` | Brownian motion, GBM, jump-diffusion |
| `04_Volatility_Models/` | Historical vol, EWMA, GARCH, Heston |
| `05_Derivatives_Pricing/` | Itô calculus, Black–Scholes, MC pricing |
| `06_Risk_Analysis/` | VaR / ES simulator |
| `07_Final_Report/` | Final write-up |

Each module has `theory/` (notes, derivations) and `implementation/` (code, experiments).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

In VSCode: `Ctrl/Cmd+Shift+P` → "Python: Select Interpreter" → pick `.venv`.

## Week 1 workflow

```bash
python 02_Data/download_data.py     # -> 02_Data/Raw/*.csv + download_log.csv
python 02_Data/process_data.py      # -> 02_Data/Processed/*_returns.csv
```

Then open `02_Data/week1_exploration.py` (runs as `# %%` cells in VSCode Interactive Window).

## Rules

- `Raw/` is never edited by hand. Everything downstream is reproducible from scripts.
- Fix random seeds in every simulation.
- Theory and implementation of a module live side by side.
