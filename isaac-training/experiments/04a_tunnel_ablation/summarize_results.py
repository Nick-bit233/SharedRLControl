#!/usr/bin/env python3
"""Summarize ablation eval_info.json files into CSV and a simple TeX table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    "eval/success",
    "eval/collision",
    "eval/truncated",
    "eval/episode_len",
    "eval/return",
    "eval/diag_reward_task",
]


def find_eval_files(eval_dir: Path) -> list[Path]:
    return sorted(eval_dir.glob("**/eval_info.json"))


def row_from_eval(path: Path) -> dict:
    data = json.loads(path.read_text())
    row = {
        "run": path.parent.name,
        "eval_dir": str(path.parent),
    }
    for metric in METRICS:
        row[metric] = data.get(metric, data.get(metric.replace("eval/", "eval/stats_"), ""))
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", *METRICS, "eval_dir"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[AblationSummary] CSV written: {path}")


def write_tex(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Variant & Success & Collision & Timeout & Return \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['run']} & {row.get('eval/success', '')} & "
            f"{row.get('eval/collision', '')} & {row.get('eval/truncated', '')} & "
            f"{row.get('eval/return', '')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))
    print(f"[AblationSummary] TeX written: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation eval outputs")
    parser.add_argument("--eval-dir", default="./outputs/tunnel_ablation/eval")
    parser.add_argument("--csv", default="./outputs/tunnel_ablation/ablation_summary.csv")
    parser.add_argument("--tex", default=None)
    args = parser.parse_args()

    rows = [row_from_eval(path) for path in find_eval_files(Path(args.eval_dir))]
    if not rows:
        raise FileNotFoundError(f"No eval_info.json files found under {args.eval_dir}")
    write_csv(rows, Path(args.csv))
    if args.tex:
        write_tex(rows, Path(args.tex))


if __name__ == "__main__":
    main()
