"""Registry for experiment specs used by unified entrypoints."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from src.core.spec import ExperimentSpec


# if new experiment spec is added, add it to this registry 
# and implement a build_spec() function in its module script.
SPEC_BUILDERS: dict[str, str] = {
    "tunnel": "experiment_specs.tunnel:build_spec",
    "tunnel_min_risk_reduction": "experiment_specs.tunnel_min_risk_reduction:build_spec",
    "tunnel_lagrangian": "experiment_specs.tunnel_lagrangian:build_spec",
    "safety_shield": "experiment_specs.safety_shield:build_spec",
    "tunnel_intent": "experiment_specs.tunnel_intent:build_spec",
}


def _get_cfg_value(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    if isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, dotted_key, default=default)

    current = cfg
    for part in dotted_key.split("."):
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
        if current is default:
            return default
    return current


def get_spec_name_from_cfg(cfg: Any) -> str:
    """Resolve the configured runtime spec name."""

    spec_name = _get_cfg_value(cfg, "runtime.spec")
    if not spec_name:
        raise ValueError(
            "Missing required config field runtime.spec. "
            "Set it in configs/experiment/<name>.yaml or via runtime.spec=<name>."
        )
    return str(spec_name)


def get_spec_builder(spec_name: str) -> Callable[..., ExperimentSpec]:
    """Load one spec builder without importing every experiment module."""

    target = SPEC_BUILDERS.get(spec_name)
    if target is None:
        choices = ", ".join(sorted(SPEC_BUILDERS))
        raise ValueError(f"Unknown runtime.spec '{spec_name}'. Available specs: {choices}")

    module_name, function_name = target.split(":", 1)
    module = import_module(module_name)
    builder = getattr(module, function_name)
    return builder


def build_spec_from_cfg(cfg: Any) -> ExperimentSpec:
    """Build the spec selected by cfg.runtime.spec."""

    spec_name = get_spec_name_from_cfg(cfg)
    builder = get_spec_builder(spec_name)
    return builder(cfg)
