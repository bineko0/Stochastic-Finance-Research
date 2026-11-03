"""End-to-end: every stage runs on synthetic data. If this passes after a change,
the change did not break any other part of the project."""
import matplotlib

from sfr.config import project_path
from sfr.stages import STAGES, run_stage

matplotlib.use("Agg")


def test_all_stages_run(cfg):
    cfg["pricing"]["pricers"]["mc_gbm"]["n_paths"] = 5000
    cfg["pricing"]["pricers"]["mc_merton"]["n_paths"] = 5000
    cfg["pricing"]["pricers"]["mc_heston"]["n_paths"] = 2000
    cfg["pricing"]["convergence"]["n_paths"] = [500, 2000]
    cfg["risk"]["window"] = 250
    for name in STAGES:
        run_stage(name, cfg)

    reports = project_path(cfg, "reports")
    for table in ["processes_fit", "volatility_forecast_eval", "pricing_grid", "risk_backtest"]:
        assert (reports / "tables" / f"{table}.csv").exists(), table
    assert (reports / "summary.md").exists()
