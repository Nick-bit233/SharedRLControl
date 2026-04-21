#!/usr/bin/env python3
"""
M1 sweep: tunnel reg_coeff ablation (no-reg / tiny-reg mainline check).

Each run uses the same tunnel environment, reward weights, learning
rates, and pilot distribution. Only `algo.reg_coeff` varies. Runs are
launched serially.

Default sweep values (mirrors plan.md M1):
    reg_coeff in {0.0, 1e-3, 5e-3, 1e-2}

Usage:
    # Full sweep, train each run from scratch
    python experiments/04_tunnel_task/run_m1_noreg_sweep.py

    # Warm-start every run from the current paper-best checkpoint
    python experiments/04_tunnel_task/run_m1_noreg_sweep.py \\
        --resume-from /home/ubuntu/wht/SharedRLControl/ros1/navigation_runner/cfg/tunnel/checkpoint_best.pt

    # Custom reg list
    python experiments/04_tunnel_task/run_m1_noreg_sweep.py \\
        --reg-values 0.0 0.005

    # Pass extra Hydra overrides (after `--`)
    python experiments/04_tunnel_task/run_m1_noreg_sweep.py -- max_iterations=8010
"""
import argparse
import datetime
import os
import subprocess
import sys

DEFAULT_REG_VALUES = [0.0, 1e-3, 5e-3, 1e-2]


def run_single(
    reg_coeff: float,
    sweep_tag: str,
    resume_from: str | None,
    extra_overrides: list[str],
) -> int:
    run_name = f"{sweep_tag}_reg_{reg_coeff:.4f}".rstrip("0").rstrip(".")
    output_dir = f"./outputs/tunnel_m1_noreg/reg_{reg_coeff}"

    print(f"\n{'=' * 60}")
    print(f"  M1 SWEEP: reg_coeff = {reg_coeff}")
    print(f"  Output: {output_dir}")
    if resume_from:
        print(f"  Warm-start: {resume_from}")
    print(f"{'=' * 60}\n")

    cmd = [
        sys.executable,
        "experiments/04_tunnel_task/train.py",
        "experiment=tunnel_m1_noreg",
        f"algo.reg_coeff={reg_coeff}",
        f"wandb.name={run_name}",
        f"wandb.group={sweep_tag}",
        f"hydra.run.dir={output_dir}/${{now:%Y-%m-%d_%H-%M-%S}}",
    ]
    if resume_from:
        cmd.append(f"resume_checkpoint={resume_from}")
    cmd.extend(extra_overrides)

    print(f"[M1Sweep] Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 reg_coeff sweep for tunnel task")
    parser.add_argument(
        "--reg-values", type=float, nargs="+", default=None,
        help=f"reg_coeff values to sweep (default: {DEFAULT_REG_VALUES})",
    )
    parser.add_argument(
        "--start-from", type=float, default=None,
        help="Skip reg_coeff values strictly less than this (for resuming a sweep)",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to a checkpoint to warm-start every run from.",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="WandB group / run name prefix (default: tunnel_m1_<timestamp>)",
    )
    args, extra = parser.parse_known_args()

    reg_values = args.reg_values or DEFAULT_REG_VALUES
    sweep_tag = args.tag or f"tunnel_m1_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    if args.start_from is not None:
        reg_values = [r for r in reg_values if r >= args.start_from]

    if args.resume_from and not os.path.exists(args.resume_from):
        print(f"[M1Sweep] ERROR: resume checkpoint not found: {args.resume_from}")
        sys.exit(2)

    print(f"[M1Sweep] Tunnel M1 No-Reg / Tiny-Reg Sweep")
    print(f"[M1Sweep] reg_coeff values: {reg_values}")
    print(f"[M1Sweep] WandB group/prefix: {sweep_tag}")
    print(f"[M1Sweep] Resume from: {args.resume_from}")
    print(f"[M1Sweep] Extra Hydra overrides: {extra}")
    print(f"[M1Sweep] Total runs: {len(reg_values)}")

    results: dict[float, int] = {}
    for i, reg in enumerate(reg_values):
        print(f"\n[M1Sweep] === Run {i + 1}/{len(reg_values)}: reg_coeff={reg} ===")
        rc = run_single(reg, sweep_tag, args.resume_from, extra)
        results[reg] = rc
        if rc != 0:
            print(f"[M1Sweep] WARNING: reg_coeff={reg} exited with code {rc}; continuing.")

    print(f"\n{'=' * 60}\n  M1 SWEEP COMPLETE\n{'=' * 60}")
    print(f"{'reg_coeff':>12}  {'status':>10}")
    print(f"{'-' * 24}")
    for reg in reg_values:
        rc = results[reg]
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"{reg:>12.4f}  {status:>10}")

    failed = sum(1 for rc in results.values() if rc != 0)
    if failed:
        print(f"\n[M1Sweep] {failed}/{len(reg_values)} runs failed.")
    else:
        print(f"\n[M1Sweep] All {len(reg_values)} runs succeeded.")
    print(f"[M1Sweep] Compare runs in WandB group: {sweep_tag}")


if __name__ == "__main__":
    main()
