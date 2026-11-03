"""Name -> class registries.

Every swappable component registers itself under a name, and configs/*.yaml
refer to components only by that name. Adding a new model therefore means
writing one class with a decorator; no other file changes.

    @VOL_MODELS.register("my_model")
    class MyModel(VolatilityModel): ...

    # configs/models.yaml
    volatility:
      models:
        mine: {type: my_model, some_param: 3}
"""
from __future__ import annotations

import copy
import importlib
from typing import Any

# Packages whose modules register built-in components on import.
_BUILTIN_PACKAGES = (
    "sfr.data.sources",
    "sfr.processes",
    "sfr.volatility",
    "sfr.pricing",
    "sfr.risk",
)
_loaded = False


def load_builtins() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        for pkg in _BUILTIN_PACKAGES:
            importlib.import_module(pkg)
    except Exception:
        _loaded = False
        raise


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        def decorator(obj):
            existing = self._items.get(name)
            if existing is not None and existing.__qualname__ != obj.__qualname__:
                raise KeyError(f"{self.kind} '{name}' is already registered by {existing}")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str):
        load_builtins()
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names())}") from None

    def build(self, name: str, **params: Any):
        return self.get(name)(**params)

    def from_spec(self, spec: dict | str):
        """Build from a config entry: {type: name, **params} or just "name"."""
        if isinstance(spec, str):
            return self.build(spec)
        spec = copy.deepcopy(spec)
        return self.build(spec.pop("type"), **spec)

    def names(self) -> list[str]:
        load_builtins()
        return sorted(self._items)


SOURCES = Registry("data source")
PROCESSES = Registry("stochastic process")
VOL_MODELS = Registry("volatility model")
PRICERS = Registry("pricer")
RISK_MODELS = Registry("risk model")

ALL = {
    "sources": SOURCES,
    "processes": PROCESSES,
    "volatility": VOL_MODELS,
    "pricers": PRICERS,
    "risk": RISK_MODELS,
}
