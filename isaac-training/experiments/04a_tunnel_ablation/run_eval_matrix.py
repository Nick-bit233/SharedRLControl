#!/usr/bin/env python3
"""Evaluate ablation checkpoints with the shared experiment-04 eval script."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def normalize_checkpoint_path(path_text: str) -> str:
    path = Path(path_text).expanduser()
    if path.exists():
        return str(path.resolve())
    marker = "SharedRLControl/"
    if marker in path_text:
        suffix = path_text.split(marker, 1)[1]
        candidate = Path.cwd().parent / suffix
        if candidate.exists():
            return str(candidate.resolve())
    return str(path)


def load_manifests(paths: list[str]) -> list[dict]:
    manifests: list[dict] = []
    for spec in paths:
        path = Path(spec)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file in files:
            manifests.append(json.loads(file.read_text()))
    return manifests


def checkpoint_policy_mode(checkpoint: str) -> str:
    import torch

    loaded = torch.load(checkpoint, map_location="cpu")
    state_dict = loaded["policy"] if isinstance(loaded, dict) and "policy" in loaded else loaded
    keys = set(state_dict.keys())
    residual_keys = {
        "action_parameter_module.residual_scale",
        "residual_action_module.residual_scale",
    }
    return "residual" if keys & residual_keys else "direct"


def manifest_variant(manifest: dict, manifest_idx: int) -> str:
    variant = manifest.get("variant")
    if variant:
        return str(variant)
    return f"checkpoint_{manifest_idx:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation eval matrix")
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=["./outputs/tunnel_ablation/manifests"],
        help="Manifest JSON files or directories.",
    )
    parser.add_argument("--output-dir", default="./outputs/tunnel_ablation/eval")
    parser.add_argument("--eval-config", default="tunnel_ablation_eval")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--policy-mode", choices=("auto", "residual", "direct"), default="auto")
    parser.add_argument("--global-view", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    manifests = load_manifests(args.manifests)
    if not manifests:
        raise FileNotFoundError("No manifests found for evaluation")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, int | str, int]] = []
    for manifest_idx, manifest in enumerate(manifests):
        checkpoint = manifest.get("checkpoint")
        checkpoint = normalize_checkpoint_path(checkpoint) if checkpoint else None
        if not checkpoint or not os.path.exists(checkpoint):
            print(f"[AblationEval] Skipping manifest without checkpoint: {manifest}")
            failures.append((manifest_variant(manifest, manifest_idx), "all", 2))
            continue
        variant = manifest_variant(manifest, manifest_idx)
        policy_mode = (
            checkpoint_policy_mode(checkpoint)
            if args.policy_mode == "auto"
            else args.policy_mode
        )
        for eval_seed in args.eval_seeds:
            video_dir = output_dir / variant / f"eval_seed_{eval_seed}"
            cmd = [
                sys.executable,
                "experiments/04_tunnel_task/eval_video.py",
                f"experiment={args.eval_config}",
                f"resume_checkpoint={os.path.abspath(checkpoint)}",
                f"algo.policy_mode={policy_mode}",
                f"env.num_envs={args.num_envs}",
                "+keep_num_envs=true",
                f"+eval_seed={eval_seed}",
                f"+video_dir={video_dir}",
                f"global_view={'true' if args.global_view else 'false'}",
            ]
            cmd.extend(extra)
            print(f"[AblationEval] Command: {' '.join(cmd)}")
            if args.dry_run:
                continue
            rc = subprocess.run(cmd, cwd=os.getcwd()).returncode
            if rc != 0:
                failures.append((variant, eval_seed, rc))
                continue
            video_dir.mkdir(parents=True, exist_ok=True)
            (video_dir / "eval_manifest.json").write_text(
                json.dumps(
                    {
                        "variant": variant,
                        "eval_seed": eval_seed,
                        "checkpoint": os.path.abspath(checkpoint),
                        "policy_mode": policy_mode,
                        "eval_config": args.eval_config,
                        "num_envs": args.num_envs,
                        "source_manifest": manifest,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

    if failures:
        print("[AblationEval] Failures:")
        for variant, eval_seed, rc in failures:
            print(f"  {variant} eval_seed={eval_seed}: exit {rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
