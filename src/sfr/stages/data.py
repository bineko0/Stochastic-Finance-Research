from __future__ import annotations

import pandas as pd

from sfr.config import dt
from sfr.data import build_dataset, load_log_returns
from sfr.evaluation import stylized_facts
from sfr.reporting import save_table


def run(cfg: dict) -> None:
    log = build_dataset(cfg)
    ok = log.loc[log["status"] == "ok", "asset"].tolist()
    if not ok:
        raise RuntimeError("No asset could be built; see data/processed/build_log.csv")
    if cfg["data"]["main_asset"] not in ok:
        print(f"[WARN] main asset {cfg['data']['main_asset']} failed; later stages will not run")
    rows = []
    for a in ok:
        returns = load_log_returns(a, cfg)
        trading_days = round(1 / dt(cfg))      # 252
        stats = stylized_facts(returns, trading_days)
        row = {"asset": a}
        row.update(stats)                      # gộp thống kê vào
        rows.append(row)
    facts = pd.DataFrame(rows)

    save_table(cfg, "data_stylized_facts", facts)
