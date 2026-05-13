#!/usr/bin/env python3
"""Compare held-out eval_info.json files across multiple runs.

Example:
    python experiments/04s_tunnel_lagrangian/compare_eval_results.py \
        --run 04_m2=eval_videos/native04_m2_heldout_multiseed \
        --run 04s_A6=eval_videos/04s_m2_A6_stage2_heldout_multiseed
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics


METRICS = (
    "success",
    "collision",
    "score",
    "safety_cost",
    "min_dist",
    "task_reward",
    "return",
    "episode_len",
    "above",
    "below",
)


def parse_run(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=DIR")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return name, pathlib.Path(path).expanduser()


def load_eval_dir(path: pathlib.Path) -> dict[tuple[int, int], dict[str, float]]:
    rows: dict[tuple[int, int], dict[str, float]] = {}
    for info_path in sorted(path.glob("obst*_seed*/eval_info.json")):
        match = re.search(r"obst(\d+)_seed(\d+)", str(info_path))
        if match is None:
            continue
        data = json.loads(info_path.read_text())
        success = float(data["eval/success"])
        collision = float(data["eval/collision"])
        above = float(data.get("eval/above_bound", 0.0) or 0.0)
        below = float(data.get("eval/below_bound", 0.0) or 0.0)
        rows[(int(match.group(1)), int(match.group(2)))] = {
            "success": success,
            "collision": collision,
            "score": success - 0.5 * collision - 0.2 * above - 0.2 * below,
            "safety_cost": float(data.get("eval/diag_safety_cost", float("nan"))),
            "min_dist": float(data.get("eval/diag_min_dist_to_obs", float("nan"))),
            "task_reward": float(data.get("eval/diag_reward_task", float("nan"))),
            "return": float(data.get("eval/return", float("nan"))),
            "episode_len": float(data.get("eval/episode_len", float("nan"))),
            "above": above,
            "below": below,
        }
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def print_summary(runs: dict[str, dict[tuple[int, int], dict[str, float]]], keys: list[tuple[int, int]]) -> None:
    header = ["run", "n", *[f"{metric}_mean" for metric in METRICS], "worst_collision", "min_success"]
    print(",".join(header))
    for name, rows_by_key in runs.items():
        rows = [rows_by_key[key] for key in keys]
        values = []
        for metric in METRICS:
            metric_values = [row[metric] for row in rows]
            values.append(f"{mean(metric_values):.6f}")
        print(",".join([
            name,
            str(len(rows)),
            *values,
            f"{max(row['collision'] for row in rows):.6f}",
            f"{min(row['success'] for row in rows):.6f}",
        ]))


def print_pairwise(
    runs: dict[str, dict[tuple[int, int], dict[str, float]]],
    keys: list[tuple[int, int]],
    baseline: str,
) -> None:
    if baseline not in runs:
        return
    print(f"\n# paired deltas relative to {baseline}: run - {baseline}")
    print("run,metric,mean_delta,min_delta,max_delta,std_delta")
    base_rows = runs[baseline]
    for name, rows_by_key in runs.items():
        if name == baseline:
            continue
        for metric in METRICS:
            deltas = [rows_by_key[key][metric] - base_rows[key][metric] for key in keys]
            std = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            print(",".join([
                name,
                metric,
                f"{mean(deltas):.6f}",
                f"{min(deltas):.6f}",
                f"{max(deltas):.6f}",
                f"{std:.6f}",
            ]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare held-out eval result directories.")
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="Run mapping as NAME=DIR")
    parser.add_argument("--baseline", help="Run name used for paired deltas.")
    args = parser.parse_args()

    runs = {name: load_eval_dir(path) for name, path in args.run}
    missing = [name for name, rows in runs.items() if not rows]
    if missing:
        raise FileNotFoundError(f"no eval_info.json files found for: {', '.join(missing)}")
    common_keys = sorted(set.intersection(*(set(rows) for rows in runs.values())))
    if not common_keys:
        raise ValueError("runs do not share any common (obstacles, seed) cases")

    print("# common cases:", " ".join(f"{obst}:{seed}" for obst, seed in common_keys))
    print_summary(runs, common_keys)
    if args.baseline:
        print_pairwise(runs, common_keys, args.baseline)


if __name__ == "__main__":
    main()
