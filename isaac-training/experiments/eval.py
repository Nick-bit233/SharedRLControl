"""Unified evaluation entrypoint for Isaac shared-control experiments."""

from __future__ import annotations

import json
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

    if torch.cuda.device_count() > 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[Multi GPU Detected] CUDA_VISIBLE_DEVICES not set, defaulting to GPU 0")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _resolve_runtime_path(path: str | None, hydra_cfg: Any) -> str | None:
    if path is None or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(hydra_cfg.runtime.cwd, path))


def _prepare_eval_cfg(cfg: Any) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.headless = True
    cfg.wandb.mode = "disabled"
    cfg.record_video = bool(cfg.eval.get("record_video", False))
    cfg.global_view = bool(cfg.eval.get("global_view", False))
    cfg.max_iterations = 1
    if not cfg.eval.get("keep_num_envs", False):
        cfg.env.num_envs = int(cfg.eval.get("num_envs", 4))
    OmegaConf.set_struct(cfg, True)


def _serializable_info(info: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in info.items():
        if hasattr(value, "item"):
            value = value.item()
        try:
            json.dumps({key: value})
            clean[key] = value
        except TypeError:
            pass
    return clean


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: Any) -> None:
    _configure_cuda_env()
    _prepare_eval_cfg(cfg)

    from hydra.core.hydra_config import HydraConfig
    from experiment_specs.registry import build_spec_from_cfg
    from omni_drones import init_simulation_app
    from src.core.checkpointing import load_policy_for_eval
    from src.core.evaluation import evaluate_policy_to_disk
    from src.core.spec import DefaultCheckpointAdapter, RuntimeResources

    hydra_cfg = HydraConfig.get()
    checkpoint_path = _resolve_runtime_path(cfg.eval.get("checkpoint", None), hydra_cfg)
    if checkpoint_path is None:
        raise ValueError("experiments/eval.py requires eval.checkpoint=/path/to/checkpoint.pt")

    output_dir = _resolve_runtime_path(cfg.eval.get("output_dir", None), hydra_cfg) or hydra_cfg.runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    sim_app = init_simulation_app(cfg)
    try:
        spec = build_spec_from_cfg(cfg)
        dataset = spec.dataset_loader(cfg, hydra_cfg) if spec.dataset_loader is not None else None
        resources = RuntimeResources(hydra_cfg=hydra_cfg, dataset=dataset)
        env = spec.env_factory(cfg, resources)
        policy = spec.policy_factory(cfg, env)
        adapter = spec.checkpoint_adapter or DefaultCheckpointAdapter()
        load_policy_for_eval(checkpoint_path, policy, cfg, adapter=adapter, hydra_cfg=hydra_cfg)

        info = evaluate_policy_to_disk(
            cfg,
            env,
            policy,
            output_dir=output_dir,
            seed=int(cfg.eval.get("seed", 42)),
        )
        info_path = os.path.join(output_dir, "eval_info.json")
        with open(info_path, "w") as file:
            json.dump(_serializable_info(info), file, indent=2, sort_keys=True)
        print(f"[Eval] eval_info.json written: {info_path}")
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
