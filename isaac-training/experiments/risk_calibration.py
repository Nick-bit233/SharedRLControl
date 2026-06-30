"""Post-hoc dynamic-risk calibration entrypoint."""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
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
    eval_seed = int(cfg.eval.get("seed", cfg.get("seed", 42)))
    cfg.seed = eval_seed
    cfg.headless = True
    cfg.wandb.mode = "disabled"
    cfg.record_video = bool(cfg.eval.get("record_video", False))
    cfg.global_view = bool(cfg.eval.get("global_view", False))
    cfg.max_iterations = 1
    if not cfg.eval.get("keep_num_envs", False):
        cfg.env.num_envs = int(cfg.eval.get("num_envs", 4))
    OmegaConf.set_struct(cfg, True)


def _set_global_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _risk_cfg(cfg: Any) -> dict[str, Any]:
    raw = cfg.get("risk_calibration", {})
    if hasattr(raw, "items"):
        return {str(key): value for key, value in raw.items()}
    return {}


def _extract_trace(
    *,
    cfg: Any,
    trajs: Any,
    horizon_sec: float,
    near_miss_distance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.core.risk_calibration import build_calibration_trace, lidar_min_distance

    sim_dt = float(cfg.sim.dt) * float(cfg.sim.get("substeps", 1))
    horizon_steps = max(1, int(math.ceil(float(horizon_sec) / sim_dt)))
    lidar_range = float(cfg.sensor.lidar_range)

    min_dist = lidar_min_distance(
        trajs.get(("next", "agents", "observation", "lidar")),
        lidar_range=lidar_range,
    )
    extra_scores = {
        "pilot_risk_dyn_post": trajs.get(("next", "agents", "pilot_risk_dyn_post")),
        "assist_risk_dyn_post": trajs.get(("next", "agents", "assist_risk_dyn_post")),
    }
    trace = build_calibration_trace(
        score=trajs.get(("next", "agents", "assist_risk_dyn_full")),
        collision=trajs.get(("next", "stats", "collision")),
        min_lidar_distance=min_dist,
        done=trajs.get(("next", "done")),
        horizon_steps=horizon_steps,
        near_miss_distance=near_miss_distance,
        extra_scores=extra_scores,
    )
    metadata = {
        "horizon_sec": float(horizon_sec),
        "horizon_steps": int(horizon_steps),
        "sim_dt": float(sim_dt),
        "near_miss_distance": float(near_miss_distance),
        "lidar_range": float(lidar_range),
    }
    return trace, metadata


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: Any) -> None:
    _configure_cuda_env()
    _prepare_eval_cfg(cfg)
    eval_seed = int(cfg.eval.get("seed", cfg.get("seed", 42)))
    _set_global_seed(eval_seed)

    from hydra.core.hydra_config import HydraConfig
    from experiment_specs.registry import build_spec_from_cfg
    from omni_drones import init_simulation_app
    from src.core.checkpointing import load_policy_for_eval
    from src.core.evaluation import (
        flatten_first_episode_stats,
        prepare_eval_rendering,
        restore_eval_rendering,
        run_eval_rollout,
    )
    from src.core.risk_calibration import (
        bin_calibration,
        save_trace_npz,
        summarize_trace,
        threshold_exposure,
        write_csv,
    )
    from src.core.spec import DefaultCheckpointAdapter, RuntimeResources

    hydra_cfg = HydraConfig.get()
    checkpoint_path = _resolve_runtime_path(cfg.eval.get("checkpoint", None), hydra_cfg)
    if checkpoint_path is None:
        raise ValueError("risk_calibration.py requires eval.checkpoint=/path/to/checkpoint.pt")

    output_dir = _resolve_runtime_path(cfg.eval.get("output_dir", None), hydra_cfg) or hydra_cfg.runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    risk_cfg = _risk_cfg(cfg)
    horizon_sec = float(risk_cfg.get("horizon_sec", 1.5))
    near_miss_distance = float(risk_cfg.get("near_miss_distance", 0.5))

    sim_app = init_simulation_app(cfg)
    try:
        spec = build_spec_from_cfg(cfg)
        dataset = spec.dataset_loader(cfg, hydra_cfg) if spec.dataset_loader is not None else None
        resources = RuntimeResources(hydra_cfg=hydra_cfg, dataset=dataset)
        env = spec.env_factory(cfg, resources)
        policy = spec.policy_factory(cfg, env)
        adapter = spec.checkpoint_adapter or DefaultCheckpointAdapter()
        load_policy_for_eval(checkpoint_path, policy, cfg, adapter=adapter, hydra_cfg=hydra_cfg)

        record_video = bool(cfg.get("record_video", False))
        env.eval()
        try:
            prepare_eval_rendering(cfg, env, record_video=record_video)
            env.set_seed(eval_seed)
            import torch

            with torch.no_grad():
                _, trajs = run_eval_rollout(
                    cfg,
                    env,
                    policy,
                    record_video=record_video,
                    eval_max_steps=int(env.max_episode_length),
                )
        finally:
            restore_eval_rendering(env, record_video=record_video)
            env.train()

        trace, metadata = _extract_trace(
            cfg=cfg,
            trajs=trajs,
            horizon_sec=horizon_sec,
            near_miss_distance=near_miss_distance,
        )
        metadata.update(
            {
                "checkpoint": str(checkpoint_path),
                "seed": int(eval_seed),
                "method": str(risk_cfg.get("method", "")),
                "obs": int(risk_cfg.get("obs", cfg.env.get("num_obstacles", -1))),
                "num_envs": int(cfg.env.num_envs),
                "max_episode_length": int(env.max_episode_length),
                "score": "assist_risk_dyn_full",
                "risk_estimator": str(cfg.env.get("dynamic_risk", {}).get("estimator", "legacy_rollout")),
            }
        )

        info = flatten_first_episode_stats(trajs)
        info.update({f"risk_calibration/{key}": value for key, value in summarize_trace(trace).items()})
        info.update(
            {
                "risk_calibration/horizon_sec": metadata["horizon_sec"],
                "risk_calibration/horizon_steps": metadata["horizon_steps"],
                "risk_calibration/near_miss_distance": metadata["near_miss_distance"],
                "risk_calibration/sample_count": int(trace["score"].size),
            }
        )

        output = Path(output_dir)
        save_trace_npz(output / "risk_trace.npz", trace, metadata)
        write_csv(output / "risk_bins.csv", bin_calibration(trace))
        write_csv(output / "risk_threshold_exposure.csv", threshold_exposure(trace))

        info_path = output / "eval_info.json"
        info_path.write_text(json.dumps(_serializable_info(info), indent=2, sort_keys=True))
        (output / "risk_calibration_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        print(f"[RiskCalibration] risk_trace.npz written: {output / 'risk_trace.npz'}")
        print(f"[RiskCalibration] risk_bins.csv written: {output / 'risk_bins.csv'}")
        print(f"[RiskCalibration] eval_info.json written: {info_path}")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
