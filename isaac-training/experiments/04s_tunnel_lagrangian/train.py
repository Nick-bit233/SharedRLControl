"""Compatibility shim for the historical Lagrangian tunnel training entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import hydra

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _validate_entrypoint_config(cfg: Any) -> None:
    required_keys = ("runtime", "env", "algo", "user_model")
    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise ValueError(
            "04s_tunnel_lagrangian/train.py now delegates to the unified training stack. "
            f"Missing config keys: {', '.join(missing)}. "
            "Run with experiment=tunnel_lagrangian or set runtime.spec=tunnel_lagrangian."
        )


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg: Any) -> None:
    _configure_cuda_env()
    _validate_entrypoint_config(cfg)

    from experiment_specs.registry import build_spec_from_cfg
    from omni_drones import init_simulation_app
    from src.core.runner import run_training

    sim_app = init_simulation_app(cfg)
    try:
        run_training(cfg, build_spec_from_cfg(cfg))
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()

