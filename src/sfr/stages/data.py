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

    facts = pd.DataFrame([{"asset": a, **stylized_facts(load_log_returns(a, cfg), round(1 / dt(cfg)))}
                          for a in ok])
    save_table(cfg, "data_stylized_facts", facts)
