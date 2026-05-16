#!/usr/bin/env python3
"""Launch or register the paper ablation matrix."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


TRAINED_CURRICULUM_VARIANTS = {"ours_retrain", "no_residual", "follow_only", "safety_reg"}
ALL_VARIANTS = ["ours", "ours_retrain", "no_residual", "no_curriculum", "follow_only", "safety_reg"]


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


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[AblationMatrix] Manifest written: {path}")


def latest_output_dir(base: Path) -> Path:
    subdirs = sorted((p for p in base.iterdir() if p.is_dir()), reverse=True)
    if not subdirs:
        raise FileNotFoundError(f"No run directories found in {base}")
    return subdirs[0]


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


def register_ours(args: argparse.Namespace, tag: str) -> None:
    ours_checkpoint = args.ours_checkpoint
    if not ours_checkpoint:
        candidates = [
            "../ros1/navigation_runner/cfg/tunnel/checkpoint_best.pt",
            "ros1/navigation_runner/cfg/tunnel/checkpoint_best.pt",
        ]
        ours_checkpoint = next((p for p in candidates if os.path.exists(p)), None)
    if not ours_checkpoint:
        raise ValueError(
            "--ours-checkpoint is required when registering variant=ours "
            "and no default tunnel checkpoint was found"
        )
    checkpoint = os.path.abspath(ours_checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Ours checkpoint not found: {checkpoint}")
    manifest = {
        "variant": "ours",
        "checkpoint": checkpoint,
        "config": args.ours_config,
        "tag": tag,
        "git_sha": git_sha(),
        "source": "frozen_current_best",
        "trained_by_matrix": False,
    }
    write_manifest(Path(f"./outputs/tunnel_ablation/manifests/{tag}_ours.json"), manifest)


def run_no_curriculum(
    seed: int,
    tag: str,
    dry_run: bool,
    skip_existing: bool,
    resume_checkpoint: str | None,
    extra: list[str],
) -> int:
    manifest_path = Path(f"./outputs/tunnel_ablation/manifests/{tag}_no_curriculum_seed{seed}.json")
    if skip_existing and manifest_path.exists():
        print(f"[AblationMatrix] Skipping existing manifest: {manifest_path}")
        return 0
    output_base = Path(f"./outputs/tunnel_ablation/no_curriculum/{tag}_seed{seed}")
    cmd = [
        sys.executable,
        "experiments/04a_tunnel_ablation/train.py",
        "experiment=tunnel_ablation_no_curriculum",
        f"seed={seed}",
        f"wandb.group={tag}",
        f"hydra.run.dir={output_base}/${{now:%Y-%m-%d_%H-%M-%S}}",
    ]
    if resume_checkpoint:
        cmd.append(f"resume_checkpoint={os.path.abspath(resume_checkpoint)}")
    cmd.extend(extra)
    print(f"[AblationMatrix] NoCurriculum command: {' '.join(cmd)}")
    if dry_run:
        return 0
    rc = subprocess.run(cmd, cwd=os.getcwd()).returncode
    if rc != 0:
        return rc
    run_dir = latest_output_dir(output_base)
    checkpoint = find_checkpoint(run_dir)
    write_manifest(
        manifest_path,
        {
            "variant": "no_curriculum",
            "seed": seed,
            "tag": tag,
            "git_sha": git_sha(),
            "config": "tunnel_ablation_no_curriculum",
            "run_dirs": [str(run_dir)],
            "checkpoint": checkpoint,
            "resume_checkpoint": os.path.abspath(resume_checkpoint) if resume_checkpoint else None,
            "trained_by_matrix": True,
            "commands": [cmd],
            "extra_overrides": extra,
        },
    )
    return 0


def run_curriculum_variant(
    variant: str,
    seed: int,
    tag: str,
    dry_run: bool,
    skip_existing: bool,
    resume_checkpoint: str | None,
    extra: list[str],
) -> int:
    manifest_path = Path(f"./outputs/tunnel_ablation/manifests/{tag}_{variant}_seed{seed}.json")
    if skip_existing and manifest_path.exists():
        print(f"[AblationMatrix] Skipping existing manifest: {manifest_path}")
        return 0
    cmd = [
        sys.executable,
        "experiments/04a_tunnel_ablation/run_curriculum.py",
        f"--variant={variant}",
        f"--seed={seed}",
        f"--tag={tag}",
    ]
    if dry_run:
        cmd.append("--dry-run")
    if resume_checkpoint:
        cmd.append(f"--resume-checkpoint={os.path.abspath(resume_checkpoint)}")
    cmd.extend(extra)
    print(f"[AblationMatrix] Curriculum command: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=os.getcwd()).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tunnel ablation matrix")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--ours-checkpoint", default=None)
    parser.add_argument("--ours-config", default="tunnel_m3_finetune")
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume a single selected training variant/seed from an interrupted checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    tag = args.tag or f"tunnel_ablation_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    train_variants = [variant for variant in args.variants if variant != "ours"]
    if args.resume_checkpoint and (len(train_variants) != 1 or len(args.seeds) != 1):
        parser.error("--resume-checkpoint requires exactly one trainable variant and one seed")
    failures: list[tuple[str, int, int]] = []

    if "ours" in args.variants:
        register_ours(args, tag)

    for variant in args.variants:
        if variant == "ours":
            continue
        for seed in args.seeds:
            if variant in TRAINED_CURRICULUM_VARIANTS:
                rc = run_curriculum_variant(
                    variant,
                    seed,
                    tag,
                    args.dry_run,
                    args.skip_existing,
                    args.resume_checkpoint,
                    extra,
                )
            elif variant == "no_curriculum":
                rc = run_no_curriculum(
                    seed,
                    tag,
                    args.dry_run,
                    args.skip_existing,
                    args.resume_checkpoint,
                    extra,
                )
            else:
                raise ValueError(f"Unhandled variant: {variant}")
            if rc != 0:
                failures.append((variant, seed, rc))

    if failures:
        print("[AblationMatrix] Failures:")
        for variant, seed, rc in failures:
            print(f"  {variant} seed={seed}: exit {rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
