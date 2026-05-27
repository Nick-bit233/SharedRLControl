#!/usr/bin/env python3
"""
M3 runner: recover-then-lock fine-tune from M2 Phase A `checkpoint_14000.pt`.

Strategy (see ana_docs/experiments/m2_diverse_pilot_resume_analysis.md
and the M3 plan in plan.md):
  - Resume from M2 Phase A iter-14000 ckpt with the new RICH resume
    mechanism (model + optimizer + curriculum + frame counter).
  - Halved LR / entropy_coef, dense save interval (250 iter) so the
    next 100%/0% peak is captured on disk.
  - Total absolute iter budget = 22000.

Usage:
    python experiments/04_tunnel_task/run_m3_finetune.py

    # Resume from a different ckpt:
    python experiments/04_tunnel_task/run_m3_finetune.py \
        --resume-from /abs/path/to/checkpoint_14000.pt

    # Custom tag / extra hydra overrides after `--`:
    python experiments/04_tunnel_task/run_m3_finetune.py --tag tunnel_m3_v2 -- \
        algo.actor.learning_rate=5e-5
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys

DEFAULT_DATASET = "./data/trajectories_tunnel.h5"
TRAIN_EXPERIMENT = "tunnel_m3_finetune"
DEFAULT_RESUME = (
    "outputs/tunnel_m2_diverse_pilot/tunnel_m2_20260423_070951/"
    "2026-04-23_07-09-54/wandb/run-20260423_071004-kdtyh522/"
    "files/checkpoint_14000.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3 recover-then-lock fine-tune")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=DEFAULT_RESUME,
        help="Checkpoint to resume from (M2 Phase A iter 14000 by default).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Path to the tunnel offline dataset (default {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="WandB run name (default: tunnel_m3_<timestamp>).",
    )
    args, extra = parser.parse_known_args()

    resume_abs = os.path.abspath(args.resume_from)
    if not os.path.exists(resume_abs):
        print(f"[M3] ERROR: resume checkpoint not found: {resume_abs}")
        sys.exit(2)

    if not os.path.exists(args.dataset_path):
        print(f"[M3] ERROR: trajectory dataset not found: {args.dataset_path}")
        print("[M3] Generate it first via run_m2_diverse_pilot.py or "
              "src/datasets/trajectory_generator.py --config-name trajectory_gen_tunnel")
        sys.exit(2)

    tag = args.tag or f"tunnel_m3_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    output_dir = f"./outputs/tunnel_m3_finetune/{tag}"

    cmd = [
        sys.executable,
        "experiments/train.py",
        f"experiment={TRAIN_EXPERIMENT}",
        f"resume_checkpoint={resume_abs}",
        f"wandb.name={tag}",
        f"wandb.group=tunnel_m3_finetune",
        f"hydra.run.dir={output_dir}/${{now:%Y-%m-%d_%H-%M-%S}}",
    ]
    if args.dataset_path != DEFAULT_DATASET:
        cmd.append(f"user_model.dataset_path={os.path.abspath(args.dataset_path)}")
    cmd.extend(extra)

    print(f"\n[M3] Launching M3 fine-tune")
    print(f"[M3] Resume from: {resume_abs}")
    print(f"[M3] Output dir : {output_dir}")
    print(f"[M3] Tag        : {tag}")
    print(f"[M3] Command    : {' '.join(cmd)}\n")

    rc = subprocess.run(cmd, cwd=os.getcwd()).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
