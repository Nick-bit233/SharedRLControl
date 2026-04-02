#!/usr/bin/env python3
"""
Analyze Pareto ablation sweep results.

Extracts final eval metrics from each reg_coeff run and produces:
- Summary table (printed)
- CSV file for external plotting
- ASCII Pareto plot (RMSE vs Survival)

Usage:
    python analyze_pareto.py
    python analyze_pareto.py --sweep-dir ./outputs/pareto_sweep
    python analyze_pareto.py --last-n 5   # average last 5 evals instead of 3
"""
import argparse
import csv
import os
import re
import sys


def find_latest_run(reg_dir: str) -> str:
    """Find the most recent run directory inside a reg_X.XX folder."""
    if not os.path.isdir(reg_dir):
        return None
    subdirs = sorted(
        [d for d in os.listdir(reg_dir) if os.path.isdir(os.path.join(reg_dir, d))],
        reverse=True,
    )
    return os.path.join(reg_dir, subdirs[0]) if subdirs else None


def extract_evals(log_path: str) -> list[dict]:
    """Parse all eval info dicts from a train.log file."""
    evals = []
    pattern = re.compile(r"\[Eval\] eval info: ({.*})")
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    d = eval(m.group(1))
                    evals.append(d)
                except Exception:
                    pass
    return evals


def main():
    parser = argparse.ArgumentParser(description="Analyze Pareto ablation results")
    parser.add_argument(
        "--sweep-dir", type=str, default="./outputs/pareto_sweep",
        help="Root directory of pareto sweep outputs",
    )
    parser.add_argument(
        "--last-n", type=int, default=3,
        help="Number of final evals to average (default: 3)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Output CSV path (default: <sweep-dir>/pareto_results.csv)",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    last_n = args.last_n
    csv_path = args.csv or os.path.join(sweep_dir, "pareto_results.csv")

    if not os.path.isdir(sweep_dir):
        print(f"ERROR: sweep directory not found: {sweep_dir}")
        sys.exit(1)

    # Discover all reg_* directories
    reg_dirs = sorted(
        [d for d in os.listdir(sweep_dir) if d.startswith("reg_")],
        key=lambda x: float(x.split("_", 1)[1]),
    )

    if not reg_dirs:
        print(f"ERROR: no reg_* directories found in {sweep_dir}")
        sys.exit(1)

    # Key metrics to extract
    metrics = [
        "survival_rate", "collision_rate", "tracking_rmse", "intervention_mean",
        "episode_len", "return", "diag_reward_tracking", "diag_reward_safety",
        "diag_penalty_height", "diag_danger_level",
    ]

    results = []

    print(f"Analyzing {len(reg_dirs)} runs from {sweep_dir} (last {last_n} evals avg)")
    print()

    for reg_name in reg_dirs:
        reg_val = float(reg_name.split("_", 1)[1])
        reg_path = os.path.join(sweep_dir, reg_name)
        run_dir = find_latest_run(reg_path)

        if run_dir is None:
            print(f"  SKIP {reg_name}: no run directory found")
            continue

        log_path = os.path.join(run_dir, "train.log")
        if not os.path.exists(log_path):
            print(f"  SKIP {reg_name}: no train.log found")
            continue

        evals = extract_evals(log_path)
        if len(evals) < last_n:
            print(f"  WARN {reg_name}: only {len(evals)} evals (need {last_n})")
            if len(evals) == 0:
                continue

        last_evals = evals[-last_n:]
        row = {"reg_coeff": reg_val, "num_evals": len(evals)}
        for m in metrics:
            key = f"eval/{m}"
            vals = [e.get(key, 0.0) for e in last_evals]
            row[m] = sum(vals) / len(vals)
        results.append(row)

    if not results:
        print("ERROR: no valid results found")
        sys.exit(1)

    # Print summary table
    print(f"{'reg_coeff':>10} {'Survival':>10} {'Collision':>10} {'RMSE':>10} "
          f"{'Interv':>10} {'EpLen':>8} {'Return':>10} {'DangerLvl':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['reg_coeff']:>10.4f} {r['survival_rate']:>10.3f} {r['collision_rate']:>10.3f} "
              f"{r['tracking_rmse']:>10.3f} {r['intervention_mean']:>10.3f} "
              f"{r['episode_len']:>8.1f} {r['return']:>10.1f} {r['diag_danger_level']:>10.3f}")

    # Pareto ASCII plot
    print(f"\n{'='*60}")
    print("PARETO PLOT: Tracking RMSE (X) vs Survival Rate (Y)")
    print(f"{'='*60}")

    # Normalize to grid
    rows, cols = 20, 50
    rmse_vals = [r["tracking_rmse"] for r in results]
    surv_vals = [r["survival_rate"] for r in results]
    rmse_min, rmse_max = min(rmse_vals) * 0.9, max(rmse_vals) * 1.1
    surv_min, surv_max = 0.0, max(surv_vals) * 1.2

    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    for r in results:
        x = int((r["tracking_rmse"] - rmse_min) / (rmse_max - rmse_min) * (cols - 1))
        y = int((r["survival_rate"] - surv_min) / (surv_max - surv_min) * (rows - 1))
        x = max(0, min(cols - 1, x))
        y = max(0, min(rows - 1, y))
        # Label with first char of reg_coeff
        label = f"{r['reg_coeff']:.2f}"[-3:]
        grid[rows - 1 - y][x] = "*"

    for i, row in enumerate(grid):
        surv_label = surv_max - i * (surv_max - surv_min) / rows
        print(f"{surv_label:>5.2f} |{''.join(row)}|")
    print(f"      +{'-'*cols}+")
    print(f"       {rmse_min:.2f}{' '*(cols-8)}{rmse_max:.2f}")
    print(f"       {'Tracking RMSE -->':^{cols}}")

    # Print point labels
    print("\nPoints:")
    for r in results:
        print(f"  reg={r['reg_coeff']:.4f}: RMSE={r['tracking_rmse']:.3f}, "
              f"Survival={r['survival_rate']:.3f}, Interv={r['intervention_mean']:.3f}")

    # Write CSV
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    fieldnames = ["reg_coeff", "num_evals"] + metrics
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
