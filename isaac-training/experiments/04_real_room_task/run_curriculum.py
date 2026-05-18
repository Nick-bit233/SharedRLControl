#!/usr/bin/env python3
"""Single-stage long training runner for the fixed real-room map."""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys


def find_checkpoint(output_dir: str) -> str:
    for marker_name in ("best_checkpoint_path.txt", "final_checkpoint_path.txt"):
        marker = os.path.join(output_dir, marker_name)
        if os.path.exists(marker):
            with open(marker, "r") as f:
                path = f.read().strip()
            if os.path.exists(path):
                return path
    fallback = os.path.join(output_dir, "checkpoint_final.pt")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"No checkpoint found in {output_dir}")


def find_latest_output_dir() -> str:
    base = os.path.join("outputs", "real_room")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No output directory found: {base}")
    subdirs = sorted(
        [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
        reverse=True,
    )
    if not subdirs:
        raise FileNotFoundError(f"No run directories found in {base}")
    return os.path.join(base, subdirs[0])


def run_training(init_checkpoint: str | None, group: str, extra: list[str]) -> str:
    cmd = [
        sys.executable,
        "experiments/04_real_room_task/train.py",
        "experiment=real_room",
        f"+wandb.group={group}",
    ]
    if init_checkpoint:
        cmd.append(f"init_checkpoint={init_checkpoint}")
    cmd.extend(extra)
    print(f"[RealRoomRunner] Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        sys.exit(result.returncode)
    return find_latest_output_dir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-stage real-room SRLC training")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    args, extra = parser.parse_known_args()

    group = args.group or f"real_room_long_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    output_dir = run_training(args.checkpoint, group, extra)
    checkpoint = find_checkpoint(output_dir)
    print(f"[RealRoomRunner] Finished. Best/final checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
