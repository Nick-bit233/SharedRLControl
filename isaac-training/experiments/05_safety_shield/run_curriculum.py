#!/usr/bin/env python3
"""
Multi-stage curriculum training pipeline for Safety Shield experiment.

Sequentially launches training runs with increasing obstacle density,
passing the best checkpoint from each stage to the next.

Usage:
    # Run all 3 stages from scratch
    python run_curriculum.py

    # Resume from stage 2 with a checkpoint from stage 1
    python run_curriculum.py --start-stage 2 --checkpoint /path/to/stage1/checkpoint_best.pt

    # Run only stage 1
    python run_curriculum.py --end-stage 1

    # Custom WandB group name
    python run_curriculum.py --group safety_shield_exp01
"""
import argparse
import datetime
import os
import subprocess
import sys


STAGE_CONFIGS = [
    "shield_stage1",
    "shield_stage2",
    "shield_stage3",
]

NUM_STAGES = len(STAGE_CONFIGS)


def find_final_checkpoint(output_dir: str) -> str:
    """Find the best (or final) checkpoint path from a completed training run.

    Prefers best checkpoint (highest eval survival rate) over final checkpoint.
    """
    best_marker = os.path.join(output_dir, "best_checkpoint_path.txt")
    if os.path.exists(best_marker):
        with open(best_marker, "r") as f:
            path = f.read().strip()
        if os.path.exists(path):
            print(f"[Pipeline] Using BEST checkpoint: {path}")
            return path

    marker = os.path.join(output_dir, "final_checkpoint_path.txt")
    if os.path.exists(marker):
        with open(marker, "r") as f:
            path = f.read().strip()
        if os.path.exists(path):
            print(f"[Pipeline] Using FINAL checkpoint (no best available): {path}")
            return path

    fallback = os.path.join(output_dir, "checkpoint_final.pt")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"No checkpoint found in {output_dir}. "
        f"Checked best_checkpoint_path.txt, final_checkpoint_path.txt, and {fallback}"
    )


def find_latest_output_dir(stage_name: str) -> str:
    """Find the most recent output directory for a given stage."""
    base = os.path.join("outputs", stage_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No output directory found for {stage_name} at {base}")
    subdirs = sorted(
        [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
        reverse=True,
    )
    if not subdirs:
        raise FileNotFoundError(f"No run directories found in {base}")
    return os.path.join(base, subdirs[0])


def run_stage(
    stage_idx: int,
    checkpoint: str | None,
    group: str,
    extra_overrides: list[str],
) -> str:
    """Run a single training stage. Returns the path to the output directory."""
    config_name = STAGE_CONFIGS[stage_idx]
    stage_num = stage_idx + 1
    print(f"\n{'='*60}")
    print(f"  STAGE {stage_num}/{NUM_STAGES}: {config_name}")
    if checkpoint:
        print(f"  Checkpoint: {checkpoint}")
    else:
        print(f"  Checkpoint: None (training from scratch)")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        "experiments/05_safety_shield/train.py",
        f"experiment={config_name}",
        f"+wandb.group={group}",
    ]
    if checkpoint:
        cmd.append(f"resume_checkpoint={checkpoint}")
    cmd.extend(extra_overrides)

    print(f"[Pipeline] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"\n[Pipeline] ERROR: Stage {stage_num} exited with code {result.returncode}")
        sys.exit(result.returncode)

    output_base = f"shield_stage{stage_num}"
    output_dir = find_latest_output_dir(output_base)
    print(f"[Pipeline] Stage {stage_num} complete. Output: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Safety Shield curriculum training pipeline")
    parser.add_argument(
        "--start-stage", type=int, default=1,
        help=f"Stage to start from (1-{NUM_STAGES}, default: 1)",
    )
    parser.add_argument(
        "--end-stage", type=int, default=NUM_STAGES,
        help=f"Stage to end at (1-{NUM_STAGES}, default: {NUM_STAGES})",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint for resuming (used for start-stage)",
    )
    parser.add_argument(
        "--group", type=str, default=None,
        help="WandB group name (default: auto-generated timestamp)",
    )
    args, extra = parser.parse_known_args()

    if args.start_stage < 1 or args.start_stage > NUM_STAGES:
        parser.error(f"--start-stage must be between 1 and {NUM_STAGES}")
    if args.end_stage < args.start_stage or args.end_stage > NUM_STAGES:
        parser.error(f"--end-stage must be >= start-stage and <= {NUM_STAGES}")

    group = args.group or f"shield_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    checkpoint = args.checkpoint

    print(f"[Pipeline] Safety Shield Curriculum Training Pipeline")
    print(f"[Pipeline] Stages: {args.start_stage} -> {args.end_stage}")
    print(f"[Pipeline] WandB group: {group}")
    print(f"[Pipeline] Extra overrides: {extra}")

    for stage_idx in range(args.start_stage - 1, args.end_stage):
        output_dir = run_stage(stage_idx, checkpoint, group, extra)
        try:
            checkpoint = find_final_checkpoint(output_dir)
            print(f"[Pipeline] Next stage will load: {checkpoint}")
        except FileNotFoundError as e:
            if stage_idx < args.end_stage - 1:
                print(f"[Pipeline] WARNING: {e}")
                print("[Pipeline] Next stage will start from scratch.")
                checkpoint = None

    print(f"\n{'='*60}")
    print(f"  SAFETY SHIELD CURRICULUM COMPLETE")
    print(f"  Stages {args.start_stage}-{args.end_stage} finished.")
    if checkpoint:
        print(f"  Final checkpoint: {checkpoint}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
