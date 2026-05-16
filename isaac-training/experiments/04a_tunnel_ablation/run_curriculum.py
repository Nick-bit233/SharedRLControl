#!/usr/bin/env python3
"""Run a three-stage tunnel ablation curriculum."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


VARIANT_STAGE_CONFIGS = {
    "no_residual": [
        "tunnel_ablation_no_residual_stage1",
        "tunnel_ablation_no_residual_stage2",
        "tunnel_ablation_no_residual_stage3",
    ],
    "follow_only": [
        "tunnel_ablation_follow_only_stage1",
        "tunnel_ablation_follow_only_stage2",
        "tunnel_ablation_follow_only_stage3",
    ],
    "safety_reg": [
        "tunnel_ablation_safety_reg_stage1",
        "tunnel_ablation_safety_reg_stage2",
        "tunnel_ablation_safety_reg_stage3",
    ],
}


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def find_checkpoint(output_dir: Path) -> str:
    for marker_name in ("best_checkpoint_path.txt", "final_checkpoint_path.txt"):
        marker = output_dir / marker_name
        if marker.exists():
            path = marker.read_text().strip()
            if os.path.exists(path):
                return path
    fallback = output_dir / "checkpoint_final.pt"
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError(f"No checkpoint marker found under {output_dir}")


def latest_output_dir(base: Path) -> Path:
    if not base.is_dir():
        raise FileNotFoundError(f"Output base does not exist: {base}")
    subdirs = sorted((p for p in base.iterdir() if p.is_dir()), reverse=True)
    if not subdirs:
        raise FileNotFoundError(f"No run directories found in {base}")
    return subdirs[0]


def run_stage(
    variant: str,
    stage_idx: int,
    config_name: str,
    init_checkpoint: str | None,
    resume_checkpoint: str | None,
    seed: int,
    tag: str,
    dry_run: bool,
    extra: list[str],
) -> tuple[str | None, Path, list[str]]:
    stage_num = stage_idx + 1
    output_base = Path(f"./outputs/tunnel_ablation/{variant}/stage{stage_num}/{tag}_seed{seed}")
    cmd = [
        sys.executable,
        "experiments/04a_tunnel_ablation/train.py",
        f"experiment={config_name}",
        f"seed={seed}",
        f"wandb.group={tag}",
        f"hydra.run.dir={output_base}/${{now:%Y-%m-%d_%H-%M-%S}}",
    ]
    if resume_checkpoint:
        cmd.append(f"resume_checkpoint={os.path.abspath(resume_checkpoint)}")
    elif init_checkpoint:
        cmd.append(f"init_checkpoint={os.path.abspath(init_checkpoint)}")
    cmd.extend(extra)

    print(f"[AblationCurriculum] Stage {stage_num}: {' '.join(cmd)}")
    if dry_run:
        return None, output_base, cmd

    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        raise RuntimeError(f"{variant} stage {stage_num} failed with exit code {result.returncode}")

    run_dir = latest_output_dir(output_base)
    next_checkpoint = find_checkpoint(run_dir)
    print(f"[AblationCurriculum] Stage {stage_num} output: {run_dir}")
    print(f"[AblationCurriculum] Stage {stage_num} checkpoint: {next_checkpoint}")
    return next_checkpoint, run_dir, cmd


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[AblationCurriculum] Manifest written: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged ablation curriculum")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_STAGE_CONFIGS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=3)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Warm-start the first requested stage from policy weights only.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume the first requested stage from an interrupted rich checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.start_stage < 1 or args.end_stage > 3 or args.start_stage > args.end_stage:
        parser.error("--start-stage/--end-stage must describe a non-empty range within 1..3")
    if args.checkpoint and args.resume_checkpoint:
        parser.error("--checkpoint and --resume-checkpoint are mutually exclusive")

    tag = args.tag or f"tunnel_ablation_{args.variant}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    checkpoint = args.checkpoint
    run_dirs: list[str] = []
    commands: list[list[str]] = []
    configs = VARIANT_STAGE_CONFIGS[args.variant]

    for stage_idx in range(args.start_stage - 1, args.end_stage):
        resume_checkpoint = args.resume_checkpoint if stage_idx == args.start_stage - 1 else None
        init_checkpoint = None if resume_checkpoint else checkpoint
        checkpoint, run_dir, cmd = run_stage(
            args.variant,
            stage_idx,
            configs[stage_idx],
            init_checkpoint,
            resume_checkpoint,
            args.seed,
            tag,
            args.dry_run,
            extra,
        )
        run_dirs.append(str(run_dir))
        commands.append(cmd)

    manifest = {
        "variant": args.variant,
        "seed": args.seed,
        "tag": tag,
        "git_sha": git_sha(),
        "configs": configs[args.start_stage - 1:args.end_stage],
        "run_dirs": run_dirs,
        "checkpoint": checkpoint,
        "input_checkpoint": args.checkpoint,
        "resume_checkpoint": args.resume_checkpoint,
        "dry_run": args.dry_run,
        "commands": commands,
        "extra_overrides": extra,
    }
    write_manifest(Path(f"./outputs/tunnel_ablation/manifests/{tag}_{args.variant}_seed{args.seed}.json"), manifest)


if __name__ == "__main__":
    main()
