"""Command line.

    python -m sfr run all
    python -m sfr run data volatility
    python -m sfr run all --set data.default_source=synthetic
    python -m sfr list
"""
from __future__ import annotations

import argparse
import sys

from sfr.config import load_config
from sfr.registry import ALL
from sfr.stages import STAGES, run_stage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sfr", description="Stochastic Finance Research pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run one or more stages")
    run.add_argument("stages", nargs="+", help=f"'all' or any of: {', '.join(STAGES)}")
    run.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                     help="override a config value, e.g. data.default_source=csv (repeatable)")
    run.add_argument("--config-dir", default=None)

    sub.add_parser("list", help="list stages and registered components")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        print("Stages:")
        for name, desc in STAGES.items():
            print(f"  {name:11s} {desc}")
        for kind, reg in ALL.items():
            print(f"{kind}: {', '.join(reg.names())}")
        return 0

    import matplotlib

    matplotlib.use("Agg")  # write figures to files, never open windows
    cfg = load_config(args.config_dir, args.overrides)
    stages = list(STAGES) if args.stages == ["all"] else args.stages
    for name in stages:
        run_stage(name, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
