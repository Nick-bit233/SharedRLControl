"""Unified multi-stage campaign orchestration entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DRY_RUN = "--dry-run" in sys.argv
if DRY_RUN:
    sys.argv = [arg for arg in sys.argv if arg != "--dry-run"]


@dataclass
class StageResult:
    name: str
    output_dir: Path
    checkpoint_path: Path | None = None


def _resolve_path(path: str | None, hydra_cfg: Any) -> Path | None:
    if path is None:
        return None
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return Path(hydra_cfg.runtime.cwd) / expanded


def _timestamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _stage_overrides(stage_cfg: Any) -> list[str]:
    overrides = stage_cfg.get("overrides", [])
    return [str(item) for item in overrides]


def _read_marker(path: Path) -> Path | None:
    if not path.exists():
        return None
    checkpoint = Path(path.read_text().strip()).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = path.parent / checkpoint
    return checkpoint if checkpoint.exists() else None


def _latest_periodic_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, float, Path]] = []
    for path in output_dir.glob("**/checkpoint_*.pt"):
        stem = path.stem
        if not stem.startswith("checkpoint_"):
            continue
        suffix = stem[len("checkpoint_") :]
        if not suffix.isdigit():
            continue
        candidates.append((int(suffix), path.stat().st_mtime, path))
    if not candidates:
        return None
    return max(candidates)[2]


def _select_checkpoint(output_dir: Path, policy: str) -> Path:
    marker_names = {
        "best": "best_checkpoint_path.txt",
        "final": "final_checkpoint_path.txt",
        "latest": "latest_checkpoint_path.txt",
    }
    marker = marker_names.get(policy)
    if marker is None:
        raise ValueError(f"Unknown campaign.checkpoint_policy: {policy}")

    checkpoint = _read_marker(output_dir / marker)
    if checkpoint is not None:
        return checkpoint
    if policy == "latest":
        checkpoint = _latest_periodic_checkpoint(output_dir)
        if checkpoint is not None:
            return checkpoint

    fallback = output_dir / "checkpoint_final.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No {policy} checkpoint marker or fallback checkpoint found in {output_dir}")


def _build_train_command(
    *,
    stage_cfg: Any,
    output_dir: Path,
    previous_checkpoint: Path | None,
    checkpoint_override: str,
) -> list[str]:
    overrides = _stage_overrides(stage_cfg)
    if stage_cfg.get("init_from_previous", False) and previous_checkpoint is not None:
        overrides.append(f"{checkpoint_override}={previous_checkpoint}")
    overrides.append(f"hydra.run.dir={output_dir}")
    return [sys.executable, "experiments/train.py", *overrides]


def _run_command(command: list[str], *, dry_run: bool) -> int:
    print("[Campaign] Command:", " ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command, cwd=str(REPO_ROOT)).returncode


def _run_train_stages(cfg: DictConfig, hydra_cfg: Any, *, dry_run: bool) -> list[StageResult]:
    campaign = cfg.campaign
    stages = campaign.get("stages", [])
    if not stages:
        raise ValueError("campaign.stages must contain at least one stage")

    output_root = _resolve_path(campaign.get("output_root", None), hydra_cfg)
    if output_root is None:
        output_root = Path(hydra_cfg.runtime.cwd) / "outputs" / "campaign" / str(campaign.name)
    run_id = _timestamp()
    checkpoint_policy = str(campaign.get("checkpoint_policy", "best"))
    checkpoint_override = str(campaign.get("checkpoint_override", "init_checkpoint"))

    previous_checkpoint: Path | None = None
    results: list[StageResult] = []
    for index, stage_cfg in enumerate(stages, start=1):
        stage_name = str(stage_cfg.get("name", f"stage{index}"))
        output_dir = output_root / run_id / stage_name
        command = _build_train_command(
            stage_cfg=stage_cfg,
            output_dir=output_dir,
            previous_checkpoint=previous_checkpoint,
            checkpoint_override=checkpoint_override,
        )
        print(f"[Campaign] Stage {index}/{len(stages)}: {stage_name}")
        rc = _run_command(command, dry_run=dry_run)
        if rc != 0:
            raise SystemExit(rc)

        checkpoint: Path | None = None
        if not dry_run:
            checkpoint = _select_checkpoint(output_dir, checkpoint_policy)
            print(f"[Campaign] Selected {checkpoint_policy} checkpoint: {checkpoint}")
            previous_checkpoint = checkpoint
        results.append(StageResult(stage_name, output_dir, checkpoint))
    return results


def _validate_campaign_cfg(cfg: DictConfig) -> None:
    if "campaign" not in cfg or cfg.campaign is None:
        raise ValueError("Missing campaign config. Run with campaign=<name>.")
    if not cfg.campaign.get("name", None):
        raise ValueError("campaign.name must be set.")
    if not cfg.campaign.get("mode", None):
        raise ValueError("campaign.mode must be set.")


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig

    _validate_campaign_cfg(cfg)
    hydra_cfg = HydraConfig.get()
    mode = str(cfg.campaign.mode)
    print("[Campaign] Config:")
    print(OmegaConf.to_yaml(cfg.campaign))
    if mode == "train_stages":
        _run_train_stages(cfg, hydra_cfg, dry_run=DRY_RUN)
        return
    raise ValueError(f"Unsupported campaign.mode: {mode}")


if __name__ == "__main__":
    main()
