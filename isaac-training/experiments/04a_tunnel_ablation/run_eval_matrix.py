#!/usr/bin/env python3
"""Evaluate ablation checkpoints with the shared experiment-04 eval script."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_manifests(paths: list[str]) -> list[dict]:
    manifests: list[dict] = []
    for spec in paths:
        path = Path(spec)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file in files:
            manifests.append(json.loads(file.read_text()))
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation eval matrix")
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=["./outputs/tunnel_ablation/manifests"],
        help="Manifest JSON files or directories.",
    )
    parser.add_argument("--output-dir", default="./outputs/tunnel_ablation/eval")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--global-view", action="store_true")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    manifests = load_manifests(args.manifests)
    if not manifests:
        raise FileNotFoundError("No manifests found for evaluation")

    failures: list[tuple[str, int]] = []
    for manifest in manifests:
        checkpoint = manifest.get("checkpoint")
        if not checkpoint or not os.path.exists(checkpoint):
            print(f"[AblationEval] Skipping manifest without checkpoint: {manifest}")
            failures.append((manifest.get("variant", "unknown"), 2))
            continue
        variant = manifest.get("variant", "unknown")
        seed = manifest.get("seed", "frozen")
        config = manifest.get("config") or (manifest.get("configs") or ["tunnel_ablation_no_curriculum"])[-1]
        video_dir = Path(args.output_dir) / f"{variant}_seed{seed}"
        cmd = [
            sys.executable,
            "experiments/04_tunnel_task/eval_video.py",
            f"experiment={config}",
            f"resume_checkpoint={os.path.abspath(checkpoint)}",
            f"env.num_envs={args.num_envs}",
            "+keep_num_envs=true",
            f"+video_dir={video_dir}",
            f"global_view={'true' if args.global_view else 'false'}",
        ]
        cmd.extend(extra)
        print(f"[AblationEval] Command: {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=os.getcwd()).returncode
        if rc != 0:
            failures.append((variant, rc))

    if failures:
        print("[AblationEval] Failures:")
        for variant, rc in failures:
            print(f"  {variant}: exit {rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
