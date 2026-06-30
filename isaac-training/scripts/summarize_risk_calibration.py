#!/usr/bin/env python3
"""Summarize dynamic-risk calibration traces and generate plots."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.core.risk_calibration import (
    DEFAULT_RISK_BINS,
    bin_calibration,
    concat_traces,
    format_float,
    load_trace_npz,
    summarize_trace,
    threshold_exposure,
    write_csv,
)


RUN_RE = re.compile(r"obs(?P<obs>\d+)_(?P<method>baseline|full)_seed(?P<seed>\d+)")


def _run_identity(path: Path, metadata: dict[str, Any]) -> tuple[int, str, int]:
    match = RUN_RE.fullmatch(path.parent.name)
    if match:
        return int(match.group("obs")), match.group("method"), int(match.group("seed"))
    return (
        int(metadata.get("obs", -1)),
        str(metadata.get("method", "unknown")),
        int(metadata.get("seed", -1)),
    )


def load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("obs*_*/risk_trace.npz")):
        trace, metadata = load_trace_npz(path)
        obs, method, seed = _run_identity(path, metadata)
        records.append(
            {
                "path": path,
                "obs": obs,
                "method": method,
                "seed": seed,
                "trace": trace,
                "metadata": metadata,
            }
        )
    return records


def _prefix_row(prefix: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**prefix, **row} for row in rows]


def _write_group_outputs(root: Path, records: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    run_bin_rows: list[dict[str, Any]] = []
    run_threshold_rows: list[dict[str, Any]] = []
    run_summary_rows: list[dict[str, Any]] = []
    group_bin_rows: list[dict[str, Any]] = []
    group_threshold_rows: list[dict[str, Any]] = []
    group_summary_rows: list[dict[str, Any]] = []
    group_traces: dict[tuple[int, str], dict[str, np.ndarray]] = {}

    for record in records:
        ident = {"obs": record["obs"], "method": record["method"], "seed": record["seed"]}
        trace = record["trace"]
        run_bin_rows.extend(_prefix_row(ident, bin_calibration(trace)))
        run_threshold_rows.extend(_prefix_row(ident, threshold_exposure(trace)))
        run_summary_rows.append({**ident, **summarize_trace(trace), "path": str(record["path"])})

    for obs in sorted({int(record["obs"]) for record in records}):
        for method in ["baseline", "full"]:
            traces = [
                record["trace"]
                for record in records
                if int(record["obs"]) == obs and str(record["method"]) == method
            ]
            if not traces:
                continue
            group_trace = concat_traces(traces)
            group_traces[(obs, method)] = group_trace
            ident = {"obs": obs, "method": method, "n_runs": len(traces)}
            group_bin_rows.extend(_prefix_row(ident, bin_calibration(group_trace)))
            group_threshold_rows.extend(_prefix_row(ident, threshold_exposure(group_trace)))
            group_summary_rows.append({**ident, **summarize_trace(group_trace)})

    write_csv(root / "risk_bins_by_run.csv", run_bin_rows)
    write_csv(root / "risk_threshold_exposure_by_run.csv", run_threshold_rows)
    write_csv(root / "risk_summary_by_run.csv", run_summary_rows)
    write_csv(root / "risk_bins_by_group.csv", group_bin_rows)
    write_csv(root / "risk_threshold_exposure_by_group.csv", group_threshold_rows)
    write_csv(root / "risk_summary_by_group.csv", group_summary_rows)
    return group_traces


def _rate_by_bin(trace: dict[str, np.ndarray], key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = bin_calibration(trace)
    x = np.asarray([(row["risk_low"] + row["risk_high"]) / 2.0 for row in rows], dtype=np.float32)
    y = np.asarray([row[key] for row in rows], dtype=np.float32)
    return x, y


def _sample_fraction_by_bin(trace: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rows = bin_calibration(trace)
    x = np.asarray([(row["risk_low"] + row["risk_high"]) / 2.0 for row in rows], dtype=np.float32)
    y = np.asarray([row["sample_fraction"] for row in rows], dtype=np.float32)
    return x, y


def _make_plots(root: Path, group_traces: dict[tuple[int, str], dict[str, np.ndarray]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    colors = {"baseline": "#2f6fbb", "full": "#c43d3d"}
    for obs in sorted({obs for obs, _ in group_traces}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        metric_specs = [
            ("future_collision_rate", "Future collision rate"),
            ("future_near_miss_rate", "Future near-miss rate"),
            ("future_event_rate", "Future collision or near-miss rate"),
        ]
        for ax, (metric, title) in zip(axes, metric_specs):
            for method in ["baseline", "full"]:
                trace = group_traces.get((obs, method))
                if trace is None:
                    continue
                x, y = _rate_by_bin(trace, metric)
                ax.plot(x, y, marker="o", label=method, color=colors[method])
            ax.set_title(title)
            ax.set_xlabel("risk score bin center")
            ax.set_ylabel("empirical rate")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend()
        path = root / f"risk_calibration_obs{obs}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        width = 0.035
        for offset, method in [(-width / 2, "baseline"), (width / 2, "full")]:
            trace = group_traces.get((obs, method))
            if trace is None:
                continue
            x, y = _sample_fraction_by_bin(trace)
            ax.bar(x + offset, y, width=width, label=method, color=colors[method], alpha=0.75)
        ax.set_title(f"Risk exposure distribution, obs={obs}")
        ax.set_xlabel("risk score bin center")
        ax.set_ylabel("sample fraction")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        path = root / f"risk_exposure_obs{obs}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _event_rate_above(trace: dict[str, np.ndarray], threshold: float) -> float:
    score = trace["score"]
    mask = score >= threshold
    if not mask.any():
        return float("nan")
    return float(trace["future_event"][mask].mean())


def _fraction_above(trace: dict[str, np.ndarray], threshold: float) -> float:
    score = trace["score"]
    if score.size == 0:
        return float("nan")
    return float((score >= threshold).mean())


def _corr_from_bins(trace: dict[str, np.ndarray]) -> float:
    rows = [row for row in bin_calibration(trace) if int(row["count"]) > 0]
    if len(rows) < 2:
        return float("nan")
    x = np.asarray([row["score_mean"] for row in rows], dtype=np.float64)
    y = np.asarray([row["future_event_rate"] for row in rows], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.std(x[mask]) == 0.0 or np.std(y[mask]) == 0.0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _write_summary(root: Path, records: list[dict[str, Any]], group_traces: dict[tuple[int, str], dict[str, np.ndarray]], plot_paths: list[Path]) -> None:
    lines: list[str] = []
    lines.append("# Risk Calibration Summary")
    lines.append("")
    lines.append(f"- traces: {len(records)}")
    lines.append(f"- output: `{root}`")
    if records:
        metadata = records[0]["metadata"]
        lines.append(f"- horizon_sec: {metadata.get('horizon_sec', 'unknown')}")
        lines.append(f"- near_miss_distance: {metadata.get('near_miss_distance', 'unknown')}")
        lines.append(f"- risk_estimator: {metadata.get('risk_estimator', 'unknown')}")
        lines.append(f"- score: {metadata.get('score', 'unknown')}")
    lines.append("")
    lines.append("## Group Metrics")
    lines.append("")
    lines.append(
        "| obs | method | samples | score_mean | event_rate | collision_rate | near_miss_rate | frac_score_ge_0.5 | event_ge_0.5 | bin_corr |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for obs in sorted({obs for obs, _ in group_traces}):
        for method in ["baseline", "full"]:
            trace = group_traces.get((obs, method))
            if trace is None:
                continue
            summary = summarize_trace(trace)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(obs),
                        method,
                        str(summary["count"]),
                        format_float(summary["score_mean"]),
                        format_float(summary["future_event_rate"]),
                        format_float(summary["future_collision_rate"]),
                        format_float(summary["future_near_miss_rate"]),
                        format_float(_fraction_above(trace, 0.5)),
                        format_float(_event_rate_above(trace, 0.5)),
                        format_float(_corr_from_bins(trace)),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Interpretation Heuristic")
    lines.append("")
    for obs in sorted({obs for obs, _ in group_traces}):
        baseline = group_traces.get((obs, "baseline"))
        full = group_traces.get((obs, "full"))
        if baseline is None or full is None:
            continue
        base_summary = summarize_trace(baseline)
        full_summary = summarize_trace(full)
        base_corr = _corr_from_bins(baseline)
        full_corr = _corr_from_bins(full)
        base_exposure = _fraction_above(baseline, 0.5)
        full_exposure = _fraction_above(full, 0.5)
        base_high_event = _event_rate_above(baseline, 0.5)
        full_high_event = _event_rate_above(full, 0.5)
        lines.append(
            f"- obs={obs}: bin-event corr baseline={format_float(base_corr)}, "
            f"full={format_float(full_corr)}; frac(score>=0.5) baseline={format_float(base_exposure)}, "
            f"full={format_float(full_exposure)}; event(score>=0.5) baseline={format_float(base_high_event)}, "
            f"full={format_float(full_high_event)}."
        )
        if all(math.isfinite(v) for v in [base_corr, full_corr]) and max(base_corr, full_corr) < 0.2:
            lines.append("  Calibration warning: empirical event rate is weakly related to risk bins.")
        elif math.isfinite(full_exposure) and math.isfinite(base_exposure) and full_exposure > base_exposure * 1.1:
            lines.append("  Exposure warning: full visits more high-risk states despite a usable risk signal.")
        if (
            math.isfinite(float(base_summary["future_event_rate"]))
            and math.isfinite(float(full_summary["future_event_rate"]))
            and math.isfinite(full_exposure)
            and math.isfinite(base_exposure)
            and float(full_summary["future_event_rate"]) > float(base_summary["future_event_rate"])
            and full_exposure <= base_exposure
        ):
            lines.append(
                "  Decision check: full has higher empirical event rate without higher score>=0.5 exposure; "
                "this points to risk-score miscalibration or missing risk factors rather than simply visiting "
                "more high-score states."
            )
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    for filename in [
        "risk_bins_by_group.csv",
        "risk_threshold_exposure_by_group.csv",
        "risk_summary_by_group.csv",
        "risk_bins_by_run.csv",
        "risk_threshold_exposure_by_run.csv",
        "risk_summary_by_run.csv",
    ]:
        lines.append(f"- `{filename}`")
    for path in plot_paths:
        lines.append(f"- `{path.name}`")
    lines.append("")
    (root / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    records = load_records(root)
    if not records:
        raise SystemExit(f"no risk_trace.npz records found under {root}")

    group_traces = _write_group_outputs(root, records)
    plot_paths = _make_plots(root, group_traces)
    _write_summary(root, records, group_traces, plot_paths)

    expected = sorted({(record["obs"], record["method"], record["seed"]) for record in records})
    print(f"records={len(records)} unique_runs={len(expected)}")
    print(root / "risk_bins_by_group.csv")
    print(root / "risk_threshold_exposure_by_group.csv")
    print(root / "risk_summary_by_group.csv")
    print(root / "summary.md")
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
