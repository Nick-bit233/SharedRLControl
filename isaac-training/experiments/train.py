"""Unified training entrypoint for Isaac shared-control experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _configure_cuda_env() -> None:
    import torch

    if torch.cuda.device_count() > 1:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print("[Multi GPU Detected] CUDA_VISIBLE_DEVICES not set, defaulting to GPU 0")
        else:
            print(f"[Multi GPU Detected] Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print("[Single GPU] Single GPU detected, no need to set CUDA_VISIBLE_DEVICES")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _require_config(cfg: Any) -> None:
    required = {
        "runtime.spec": "experiment spec selection",
        "env": "environment config",
        "algo": "algorithm config",
    }
    missing = [
        f"{key} ({description})"
        for key, description in required.items()
        if OmegaConf.select(cfg, key) is None
    ]
    if missing:
        raise ValueError("Missing required training config fields: " + ", ".join(missing))


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: Any) -> None:
    _configure_cuda_env()
    _require_config(cfg)

    from experiment_specs.registry import build_spec_from_cfg
    from omni_drones import init_simulation_app
    from src.core.runner import run_training

    sim_app = init_simulation_app(cfg)
    try:
        spec = build_spec_from_cfg(cfg)
        run_training(cfg, spec)
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
