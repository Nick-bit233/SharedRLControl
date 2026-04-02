#!/usr/bin/env python3
"""
Pareto ablation sweep: run independent training with different reg_coeff values.

Each run uses the same environment and reward weights (pareto_ablation config),
differing only in algo.reg_coeff. Runs are serial — no checkpoint chaining.

Usage:
    # Run full sweep (6 reg_coeff values)
    python run_pareto_sweep.py

    # Resume from a specific reg_coeff (skip completed ones)
    python run_pareto_sweep.py --start-from 0.1

    # Custom reg_coeff values
    python run_pareto_sweep.py --reg-values 0.0 0.01 0.1 0.5

    # With extra Hydra overrides
    python run_pareto_sweep.py max_iterations=8010 env.num_obstacles=30
"""
import argparse
import datetime
import os
import subprocess
import sys

DEFAULT_REG_VALUES = [0.0, 0.01, 0.05, 0.1, 0.3, 1.0]


def run_single(
    reg_coeff: float,
    sweep_tag: str,
    extra_overrides: list[str],
) -> int:
    """Run a single training with the given reg_coeff. Returns exit code."""
    run_name = f"{sweep_tag}_reg_{reg_coeff:.4f}".rstrip("0").rstrip(".")
    output_dir = f"./outputs/pareto_sweep/reg_{reg_coeff}"

    print(f"\n{'='*60}")
    print(f"  PARETO SWEEP: reg_coeff = {reg_coeff}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        "experiments/05_safety_shield/train.py",
        "experiment=pareto_ablation",
        f"algo.reg_coeff={reg_coeff}",
        f"wandb.name={run_name}",
        f"hydra.run.dir={output_dir}/${{now:%Y-%m-%d_%H-%M-%S}}",
    ]
    cmd.extend(extra_overrides)

    print(f"[Sweep] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Pareto ablation: sweep reg_coeff")
    parser.add_argument(
        "--reg-values", type=float, nargs="+", default=None,
        help=f"reg_coeff values to sweep (default: {DEFAULT_REG_VALUES})",
    )
    parser.add_argument(
        "--start-from", type=float, default=None,
        help="Skip reg_coeff values before this (for resuming)",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="WandB run name prefix (default: auto-generated, e.g. pareto_20260402)",
    )
    args, extra = parser.parse_known_args()

    reg_values = args.reg_values or DEFAULT_REG_VALUES
    sweep_tag = args.tag or f"pareto_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    if args.start_from is not None:
        reg_values = [r for r in reg_values if r >= args.start_from]

    print(f"[Sweep] Pareto Ablation Sweep")
    print(f"[Sweep] reg_coeff values: {reg_values}")
    print(f"[Sweep] WandB name prefix: {sweep_tag}")
    print(f"[Sweep] Extra overrides: {extra}")
    print(f"[Sweep] Total runs: {len(reg_values)}")

    results = {}
    for i, reg in enumerate(reg_values):
        print(f"\n[Sweep] === Run {i+1}/{len(reg_values)}: reg_coeff={reg} ===")
        rc = run_single(reg, sweep_tag, extra)
        results[reg] = rc
        if rc != 0:
            print(f"[Sweep] WARNING: reg_coeff={reg} exited with code {rc}")
            print(f"[Sweep] Continuing to next run...")

    print(f"\n{'='*60}")
    print(f"  PARETO SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"{'reg_coeff':>12}  {'status':>10}")
    print(f"{'-'*24}")
    for reg in reg_values:
        rc = results[reg]
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"{reg:>12.4f}  {status:>10}")

    failed = sum(1 for rc in results.values() if rc != 0)
    if failed:
        print(f"\n[Sweep] {failed}/{len(reg_values)} runs failed.")
    else:
        print(f"\n[Sweep] All {len(reg_values)} runs succeeded!")
    print(f"[Sweep] Run analyze_pareto.py to extract results.")


if __name__ == "__main__":
    main()
