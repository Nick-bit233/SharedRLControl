#!/usr/bin/env python3
"""
M2 runner: feasible-but-diverse pilot distribution training.

Steps:
  1. Generate the tunnel-targeted offline trajectory dataset
     (`data/trajectories_tunnel.h5`) via the modified
     `src/datasets/trajectory_generator.py` with directional bias.
     Skipped automatically if the dataset file already exists.
  2. Launch `experiments/04_tunnel_task/train.py` with
     `experiment=tunnel_m2_diverse_pilot`.

Usage:
    # Recommended M2-A (main line): train from scratch on the diverse
    # pilot distribution. This is the version that supports the
    # "trained on diverse pilot" generalization claim in the paper.
    python experiments/04_tunnel_task/run_m2_diverse_pilot.py

    # Inherit a non-zero reg_coeff selected by M1
    python experiments/04_tunnel_task/run_m2_diverse_pilot.py --reg-coeff 0.005

    # Optional M2-B (adaptation comparison only, NOT the main claim):
    # warm-start from the paper-best checkpoint to measure how fast a
    # narrow-pilot policy can adapt. Treat result as a side bar, since
    # the feature extractor still carries narrow-distribution priors.
    python experiments/04_tunnel_task/run_m2_diverse_pilot.py \\
        --resume-from /home/ubuntu/wht/SharedRLControl/ros1/navigation_runner/cfg/tunnel/checkpoint_best.pt \\
        --tag tunnel_m2_warmstart

    # Force-regenerate the dataset
    python experiments/04_tunnel_task/run_m2_diverse_pilot.py --regenerate-dataset

    # Skip dataset generation entirely (dataset must already exist)
    python experiments/04_tunnel_task/run_m2_diverse_pilot.py --skip-dataset

Pass extra Hydra overrides after `--`:
    python experiments/04_tunnel_task/run_m2_diverse_pilot.py -- max_iterations=8010
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys

DEFAULT_DATASET = "./data/trajectories_tunnel.h5"
DEFAULT_GEN_CONFIG = "trajectory_gen_tunnel"
TRAIN_EXPERIMENT = "tunnel_m2_diverse_pilot"


def maybe_generate_dataset(
    dataset_path: str,
    gen_config: str,
    regenerate: bool,
    skip: bool,
) -> int:
    if skip:
        if not os.path.exists(dataset_path):
            print(f"[M2] ERROR: --skip-dataset given but {dataset_path} is missing.")
            return 2
        print(f"[M2] Reusing existing dataset: {dataset_path}")
        return 0

    if os.path.exists(dataset_path) and not regenerate:
        print(f"[M2] Dataset already exists, skipping generation: {dataset_path}")
        return 0

    if regenerate and os.path.exists(dataset_path):
        print(f"[M2] --regenerate-dataset set; removing existing {dataset_path}")
        os.remove(dataset_path)

    cmd = [
        sys.executable,
        "src/datasets/trajectory_generator.py",
        f"--config-name={gen_config}",
        f"output_path={dataset_path}",
    ]
    print(f"\n[M2] Generating tunnel-targeted offline trajectory dataset")
    print(f"[M2] Command: {' '.join(cmd)}\n")
    rc = subprocess.run(cmd, cwd=os.getcwd()).returncode
    if rc != 0:
        print(f"[M2] ERROR: trajectory generation exited with code {rc}")
    return rc


def run_training(
    reg_coeff: float,
    resume_from: str | None,
    tag: str,
    extra_overrides: list[str],
) -> int:
    output_dir = f"./outputs/tunnel_m2_diverse_pilot/{tag}"
    cmd = [
        sys.executable,
        "experiments/04_tunnel_task/train.py",
        f"experiment={TRAIN_EXPERIMENT}",
        f"algo.reg_coeff={reg_coeff}",
        f"wandb.name={tag}",
        f"wandb.group=tunnel_m2_diverse_pilot",
        f"hydra.run.dir={output_dir}/${{now:%Y-%m-%d_%H-%M-%S}}",
    ]
    if resume_from:
        if not os.path.exists(resume_from):
            print(f"[M2] ERROR: resume checkpoint not found: {resume_from}")
            return 2
        cmd.append(f"resume_checkpoint={resume_from}")
    cmd.extend(extra_overrides)

    print(f"\n[M2] Launching M2 training: reg_coeff={reg_coeff}, tag={tag}")
    print(f"[M2] Command: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=os.getcwd()).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M2 diverse-pilot training")
    parser.add_argument("--reg-coeff", type=float, default=0.0,
                        help="Fixed reg_coeff for M2 (default 0.0; set to M1's pick).")
    parser.add_argument("--resume-from", type=str, default=None,
                        help=("Optional checkpoint to warm-start training from. "
                              "Default (None) = train from scratch on the diverse "
                              "pilot distribution -- this is the recommended setup "
                              "for the M2 generalization claim. Warm-starting from "
                              "the paper-best checkpoint should only be used as a "
                              "secondary 'adaptation speed' comparison, since that "
                              "checkpoint was specialised on the narrow online pilot."))
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET,
                        help=f"Path to the tunnel offline dataset (default {DEFAULT_DATASET}).")
    parser.add_argument("--gen-config", type=str, default=DEFAULT_GEN_CONFIG,
                        help="Hydra config name passed to trajectory_generator.py.")
    parser.add_argument("--regenerate-dataset", action="store_true",
                        help="Delete the existing dataset (if any) and regenerate.")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Skip the dataset generation step entirely.")
    parser.add_argument("--tag", type=str, default=None,
                        help="WandB run name (default: tunnel_m2_<timestamp>).")
    args, extra = parser.parse_known_args()

    tag = args.tag or f"tunnel_m2_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    rc = maybe_generate_dataset(
        args.dataset_path, args.gen_config, args.regenerate_dataset, args.skip_dataset,
    )
    if rc != 0:
        sys.exit(rc)

    # Make the training pick up the same dataset path even if the user
    # passed a non-default location.
    if args.dataset_path != DEFAULT_DATASET:
        extra.append(f"user_model.dataset_path={args.dataset_path}")

    rc = run_training(args.reg_coeff, args.resume_from, tag, extra)
    sys.exit(rc)


if __name__ == "__main__":
    main()
