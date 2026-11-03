"""Load configs/*.yaml into one nested dict keyed by file name.

    cfg = load_config()
    cfg["data"]["default_source"]          # from configs/data.yaml
    cfg = load_config(overrides=["data.default_source=synthetic"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def load_config(config_dir: str | Path | None = None, overrides: Iterable[str] = ()) -> dict:
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    cfg: dict[str, Any] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            cfg[path.stem] = yaml.safe_load(f) or {}
    for item in overrides:
        apply_override(cfg, item)
    return cfg


def apply_override(cfg: dict, item: str) -> None:
    """Apply 'a.b.c=value'; the value is parsed as YAML (numbers, lists, null...)."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"Override must look like 'file.key=value', got {item!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for k in parents:
        node = node.setdefault(k, {})
    node[leaf] = yaml.safe_load(raw)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def project_path(cfg: dict, key: str) -> Path:
    """Path from settings.paths, e.g. project_path(cfg, "processed")."""
    return resolve_path(cfg["settings"]["paths"][key])


def main_asset(cfg: dict, override: str | None = None) -> str:
    return override or cfg["data"]["main_asset"]


def dt(cfg: dict) -> float:
    return 1.0 / cfg["settings"]["trading_days"]
