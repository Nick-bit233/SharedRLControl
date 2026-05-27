#!/usr/bin/env python3
"""Summarize ablation evaluation files into raw and aggregate tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = [
    "eval/success",
    "eval/collision",
    "eval/timeout",
    "eval/terminated",
    "eval/episode_len",
    "eval/return",
    "eval/diag_reward_task",
    "eval/tcr_at_1",
    "eval/tcr_at_2",
    "eval/tcr_at_5",
    "eval/cte",
    "eval/crr_0.5m",
    "eval/crr_1.0m",
    "eval/dmin_min",
    "eval/dmin_mean",
]
EVAL_INFO_FILENAME = "eval" + "_info.json"


def find_eval_files(eval_dir: Path) -> list[Path]:
    return sorted(eval_dir.glob(f"**/{EVAL_INFO_FILENAME}"))


def load_metadata(eval_info_path: Path) -> dict:
    meta_path = eval_info_path.parent / "eval_manifest.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    parent = eval_info_path.parent
    return {
        "variant": parent.parent.name if parent.name.startswith("eval_seed_") else parent.name,
        "eval_seed": parent.name.removeprefix("eval_seed_") if parent.name.startswith("eval_seed_") else "",
    }


def metric_value(data: dict, metric: str):
    value = data.get(metric, data.get(metric.replace("eval/", "eval/stats_"), ""))
    if value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def row_from_eval(path: Path) -> dict:
    data = json.loads(path.read_text())
    meta = load_metadata(path)
    row = {
        "variant": meta.get("variant", path.parent.name),
        "eval_seed": meta.get("eval_seed", ""),
        "policy_mode": meta.get("policy_mode", ""),
        "eval_config": meta.get("eval_config", ""),
        "checkpoint": meta.get("checkpoint", ""),
        "eval_dir": str(path.parent),
    }
    for metric in METRICS:
        row[metric] = metric_value(data, metric)
    return row


def finite_numbers(rows: list[dict], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(metric, "")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def std(values: list[float]) -> float | str:
    if len(values) < 2:
        return 0.0 if values else ""
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def aggregate_rows(raw_rows: list[dict]) -> list[dict]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        by_variant[str(row["variant"])].append(row)

    aggregate = []
    for variant, rows in sorted(by_variant.items()):
        item = {"variant": variant, "n": len(rows)}
        policy_modes = sorted({str(row.get("policy_mode", "")) for row in rows if row.get("policy_mode", "")})
        item["policy_mode"] = ",".join(policy_modes)
        for metric in METRICS:
            values = finite_numbers(rows, metric)
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = std(values)
        aggregate.append(item)
    return aggregate


def write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[AblationSummary] CSV written: {path}")


def fmt(value) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def write_tex(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Variant & Success & Collision & Timeout & TCR@1 & CTE & CRR@0.5 \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']} & "
            f"{fmt(row.get('eval/success_mean', ''))} & "
            f"{fmt(row.get('eval/collision_mean', ''))} & "
            f"{fmt(row.get('eval/timeout_mean', ''))} & "
            f"{fmt(row.get('eval/tcr_at_1_mean', ''))} & "
            f"{fmt(row.get('eval/cte_mean', ''))} & "
            f"{fmt(row.get('eval/crr_0.5m_mean', ''))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))
    print(f"[AblationSummary] TeX written: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation eval outputs")
    parser.add_argument("--eval-dir", default="./outputs/tunnel_ablation/eval")
    parser.add_argument("--csv", default="./outputs/tunnel_ablation/ablation_summary.csv")
    parser.add_argument("--raw-csv", default=None)
    parser.add_argument("--tex", default=None)
    args = parser.parse_args()

    raw_rows = [row_from_eval(path) for path in find_eval_files(Path(args.eval_dir))]
    if not raw_rows:
        raise FileNotFoundError(f"No evaluation info files found under {args.eval_dir}")

    raw_fields = ["variant", "eval_seed", "policy_mode", "eval_config", *METRICS, "checkpoint", "eval_dir"]
    raw_csv = Path(args.raw_csv) if args.raw_csv else Path(args.csv).with_name(Path(args.csv).stem + "_raw.csv")
    write_csv(raw_rows, raw_fields, raw_csv)

    aggregate = aggregate_rows(raw_rows)
    aggregate_fields = ["variant", "n", "policy_mode"]
    for metric in METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(aggregate, aggregate_fields, Path(args.csv))
    if args.tex:
        write_tex(aggregate, Path(args.tex))


if __name__ == "__main__":
    main()
