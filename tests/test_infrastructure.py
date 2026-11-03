import pytest

from sfr.config import apply_override
from sfr.registry import ALL, PROCESSES, VOL_MODELS


def test_override_parses_yaml_values():
    cfg = {"a": {"b": 1}}
    apply_override(cfg, "a.b=2.5")
    apply_override(cfg, "a.c=[1, 2]")
    apply_override(cfg, "x.y=null")
    assert cfg == {"a": {"b": 2.5, "c": [1, 2]}, "x": {"y": None}}


def test_unknown_component_lists_available():
    with pytest.raises(KeyError, match="Available: .*gbm"):
        PROCESSES.get("does_not_exist")


def test_from_spec_does_not_mutate_config():
    spec = {"type": "historical", "window": 10}
    model = VOL_MODELS.from_spec(spec)
    assert model.window == 10 and spec == {"type": "historical", "window": 10}


def test_every_registry_has_builtins():
    for kind, reg in ALL.items():
        assert reg.names(), kind
