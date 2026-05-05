#!/usr/bin/env python3
"""Run held-out evaluation grid for 04s tunnel Lagrangian checkpoints.

Run from `isaac-training/`:

    python experiments/04s_tunnel_lagrangian/run_heldout_eval.py \
        --checkpoint /path/to/checkpoint_best.pt

Pass extra Hydra overrides after `--`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from run_curriculum import DEFAULT_DATASET, DEFAULT_GEN_CONFIG, maybe_generate_dataset


DEFAULT_GRID = "40:42,50:43,60:44,65:45"


def parse_grid(spec: str) -> list[tuple[int, int]]:
    grid: list[tuple[int, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            obstacles, seed = item.split(":", 1)
            grid.append((int(obstacles), int(seed)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid grid item {item!r}; expected OBSTACLES:SEED"
            ) from exc
    if not grid:
        raise argparse.ArgumentTypeError("grid must contain at least one OBSTACLES:SEED item")
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 04s held-out eval grid")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--experiment", default="tunnel_lagrangian_stage1",
        help="Hydra experiment config to use for eval.",
    )
    parser.add_argument(
        "--grid", type=parse_grid, default=parse_grid(DEFAULT_GRID),
        help=f"Comma-separated OBSTACLES:SEED pairs (default: {DEFAULT_GRID}).",
    )
    parser.add_argument("--num-envs", type=int, default=256, help="Parallel eval env count.")
    parser.add_argument(
        "--output-dir", default="./eval_videos/lagrangian_heldout",
        help="Directory where per-grid eval outputs are written.",
    )
    parser.add_argument(
        "--dataset-path", default=DEFAULT_DATASET,
        help=f"M2 offline tunnel trajectory dataset path (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--gen-config", default=DEFAULT_GEN_CONFIG,
        help=f"Hydra config for trajectory generation (default: {DEFAULT_GEN_CONFIG}).",
    )
    parser.add_argument(
        "--regenerate-dataset", action="store_true",
        help="Delete and regenerate the offline tunnel trajectory dataset before eval.",
    )
    parser.add_argument(
        "--skip-dataset", action="store_true",
        help="Skip dataset generation and require --dataset-path to already exist.",
    )
    parser.add_argument(
        "--global-view", action="store_true",
        help="Also record global-view videos during eval.",
    )
    args, extra = parser.parse_known_args()

    checkpoint = os.path.abspath(args.checkpoint)
    dataset_path = os.path.abspath(args.dataset_path)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    maybe_generate_dataset(
        dataset_path,
        args.gen_config,
        args.regenerate_dataset,
        args.skip_dataset,
    )

    os.makedirs(output_dir, exist_ok=True)
    for obstacles, seed in args.grid:
        video_dir = os.path.join(output_dir, f"obst{obstacles}_seed{seed}")
        cmd = [
            sys.executable,
            "experiments/04s_tunnel_lagrangian/eval_video.py",
            f"experiment={args.experiment}",
            f"resume_checkpoint={checkpoint}",
            f"env.num_envs={args.num_envs}",
            "+keep_num_envs=true",
            f"env.num_obstacles={obstacles}",
            f"+eval_seed={seed}",
            f"+video_dir={video_dir}",
            f"global_view={'true' if args.global_view else 'false'}",
            f"user_model.dataset_path={dataset_path}",
        ]
        cmd.extend(extra)
        print(f"[HeldoutEval] Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.getcwd())
        if result.returncode != 0:
            raise RuntimeError(
                f"held-out eval failed for obstacles={obstacles}, seed={seed} "
                f"with exit code {result.returncode}"
            )


if __name__ == "__main__":
    main()
