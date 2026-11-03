from __future__ import annotations

import datetime as dt

import pandas as pd

from sfr.config import project_path
from sfr.reporting import to_markdown

SECTIONS = [  # (title, table prefix, figure prefix)
    ("Data", "data_", None),
    ("Stochastic processes", "processes_", "processes_"),
    ("Volatility models", "volatility_", "volatility_"),
    ("Derivative pricing", ("pricing_", "ito_"), "pricing_"),
    ("Risk analysis", "risk_", "risk_"),
]


def run(cfg: dict) -> None:
    reports = project_path(cfg, "reports")
    tables, figures = reports / "tables", reports / "figures"
    log_path = project_path(cfg, "processed") / "build_log.csv"

    lines = ["# Results summary", "",
             f"_Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `python -m sfr run report`. "
             "Do not edit by hand; write the narrative in `reports/final/`._", ""]
    if log_path.exists():
        lines += ["## Data sources used", "", to_markdown(pd.read_csv(log_path)), ""]

    for title, tprefix, fprefix in SECTIONS:
        lines += [f"## {title}", ""]
        for t in sorted(p for p in tables.glob("*.csv") if p.name.startswith(tprefix)):
            lines += [f"### {t.stem}", "", to_markdown(pd.read_csv(t)), ""]
        if fprefix:
            for f in sorted(figures.glob(f"{fprefix}*.png")):
                lines += [f"![{f.stem}](figures/{f.name})", ""]

    out = reports / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  summary -> {out}")
