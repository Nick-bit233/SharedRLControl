#!/usr/bin/env python3
"""
Multi-stage curriculum training pipeline.

Sequentially launches training runs with increasing obstacle difficulty,
passing the final checkpoint from each stage to the next.

Usage:
    # Run all 3 stages from scratch
    python run_curriculum.py

    # Resume from stage 3 with a checkpoint from stage 2
    python run_curriculum.py --start-stage 3 --checkpoint /path/to/stage2/checkpoint_final.pt

    # Run only stages 1-3
    python run_curriculum.py --end-stage 3

    # Custom WandB group name
    python run_curriculum.py --group my_experiment_01
"""
import argparse
import datetime
import os
import subprocess
import sys


STAGE_CONFIGS = [
    "tunnel_intent_stage1",
    "tunnel_intent_stage2",
    "tunnel_intent_stage3",
]


def find_final_checkpoint(output_dir: str) -> str:
    """Find the best (or final) checkpoint path from a completed training run.
    
    Prefers best checkpoint (highest eval success) over final checkpoint.
    """
    # Prefer best checkpoint (highest eval success rate)
    best_marker = os.path.join(output_dir, "best_checkpoint_path.txt")
    if os.path.exists(best_marker):
        with open(best_marker, "r") as f:
            path = f.read().strip()
        if os.path.exists(path):
            print(f"[Pipeline] Using BEST checkpoint: {path}")
            return path

    # Fall back to final checkpoint
    marker = os.path.join(output_dir, "final_checkpoint_path.txt")
    if os.path.exists(marker):
        with open(marker, "r") as f:
            path = f.read().strip()
        if os.path.exists(path):
            print(f"[Pipeline] Using FINAL checkpoint (no best available): {path}")
            return path
    # Fallback: look for checkpoint_final.pt in output_dir
    fallback = os.path.join(output_dir, "checkpoint_final.pt")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"No checkpoint found in {output_dir}. "
        f"Checked best_checkpoint_path.txt, final_checkpoint_path.txt, and {fallback}"
    )


def find_latest_output_dir(stage_name: str) -> str:
    """Find the most recent output directory for a given stage."""
    # Hydra outputs go to ./outputs/<stage_name>/<timestamp>/
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
    """
    Run a single training stage. Returns the path to the output directory.
    """
    config_name = STAGE_CONFIGS[stage_idx]
    stage_num = stage_idx + 1
    print(f"\n{'='*60}")
    print(f"  STAGE {stage_num}/3: {config_name}")
    if checkpoint:
        print(f"  Checkpoint: {checkpoint}")
    else:
        print(f"  Checkpoint: None (training from scratch)")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        "experiments/06_tunnel_intent_task/train.py",
        f"experiment={config_name}",
        f"+wandb.group={group}",
    ]
    if checkpoint:
        cmd.append(f"init_checkpoint={checkpoint}")
    cmd.extend(extra_overrides)

    print(f"[Pipeline] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"\n[Pipeline] ERROR: Stage {stage_num} exited with code {result.returncode}")
        sys.exit(result.returncode)

    # Find the output directory that was just created
    output_base = f"tunnel_intent_stage{stage_num}"
    output_dir = find_latest_output_dir(output_base)
    print(f"[Pipeline] Stage {stage_num} complete. Output: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Multi-stage curriculum training pipeline")
    parser.add_argument(
        "--start-stage", type=int, default=1,
        help="Stage to start from (1-3, default: 1)",
    )
    parser.add_argument(
        "--end-stage", type=int, default=3,
        help="Stage to end at (1-3, default: 3)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint used to warm-start the first requested stage.",
    )
    parser.add_argument(
        "--group", type=str, default=None,
        help="WandB group name (default: auto-generated timestamp)",
    )
    args, extra = parser.parse_known_args()

    if args.start_stage < 1 or args.start_stage > 3:
        parser.error("--start-stage must be between 1 and 3")
    if args.end_stage < args.start_stage or args.end_stage > 3:
        parser.error("--end-stage must be >= start-stage and <= 3")

    group = args.group or f"tunnel_intent_curriculum_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    checkpoint = args.checkpoint

    print(f"[Pipeline] Curriculum Training Pipeline")
    print(f"[Pipeline] Stages: {args.start_stage} -> {args.end_stage}")
    print(f"[Pipeline] WandB group: {group}")
    print(f"[Pipeline] Extra overrides: {extra}")

    for stage_idx in range(args.start_stage - 1, args.end_stage):
        output_dir = run_stage(stage_idx, checkpoint, group, extra)
        # Pass checkpoint to next stage
        try:
            checkpoint = find_final_checkpoint(output_dir)
            print(f"[Pipeline] Next stage will load: {checkpoint}")
        except FileNotFoundError as e:
            if stage_idx < args.end_stage - 1:
                print(f"[Pipeline] WARNING: {e}")
                print("[Pipeline] Next stage will start from scratch.")
                checkpoint = None

    print(f"\n{'='*60}")
    print(f"  CURRICULUM COMPLETE")
    print(f"  Stages {args.start_stage}-{args.end_stage} finished.")
    if checkpoint:
        print(f"  Final checkpoint: {checkpoint}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
