#!/usr/bin/env python3
"""Create and optionally start a dynamic-risk calibration seed sweep."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def parse_csv_ints(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_csv_text(text: str) -> list[str]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def newest_checkpoint(base: Path, patterns: list[str]) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(base.glob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint_best.pt found under {base}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def q(value: os.PathLike[str] | str) -> str:
    return shlex.quote(str(value))


def build_script(
    *,
    repo: Path,
    out_root: Path,
    venv: Path,
    baseline_ckpt: Path,
    full_ckpt: Path,
    seeds: list[int],
    obs_values: list[int],
    methods: list[str],
    horizon_sec: float,
    near_miss_distance: float,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        "",
        f"source {q(venv / 'bin' / 'activate')}",
        f"cd {q(repo)}",
        "",
        f"OUT={q(out_root)}",
        'mkdir -p "$OUT/logs"',
        "",
        f"BASELINE_CKPT={q(baseline_ckpt)}",
        f"FULL_CKPT={q(full_ckpt)}",
        "",
    ]

    for obs in obs_values:
        for method in methods:
            if method not in {"baseline", "full"}:
                raise ValueError(f"unsupported method: {method}")
            ckpt_expr = '"$BASELINE_CKPT"' if method == "baseline" else '"$FULL_CKPT"'
            for seed in seeds:
                run_name = f"obs{obs}_{method}_seed{seed}"
                lines.extend(
                    [
                        f"run_name={q(run_name)}",
                        'out_dir="$OUT/$run_name"',
                        'hydra_dir="$OUT/hydra_$run_name"',
                        'log_file="$OUT/logs/$run_name.log"',
                        'echo "[$(date)] START $run_name" | tee -a "$OUT/queue.log"',
                        "HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 python experiments/risk_calibration.py \\",
                        "  experiment=tunnel_min_risk_reduction \\",
                        f"  eval.checkpoint={ckpt_expr} \\",
                        '  eval.output_dir="$out_dir" \\',
                        "  eval.keep_num_envs=true \\",
                        "  eval.record_video=false \\",
                        "  record_video=false \\",
                        "  eval_visualization=false \\",
                        "  global_view=false \\",
                        f"  seed={seed} \\",
                        f"  eval.seed={seed} \\",
                        f"  env.num_obstacles={obs} \\",
                        "  env.dynamic_risk.mode=logging_only \\",
                        f"  +risk_calibration.horizon_sec={horizon_sec:.10g} \\",
                        f"  +risk_calibration.near_miss_distance={near_miss_distance:.10g} \\",
                        f"  +risk_calibration.method={method} \\",
                        f"  +risk_calibration.obs={obs} \\",
                        f"  +risk_calibration.seed={seed} \\",
                        '  hydra.run.dir="$hydra_dir" \\',
                        '  > "$log_file" 2>&1',
                        "rc=$?",
                        'if [ "$rc" -eq 0 ] && [ -f "$out_dir/risk_trace.npz" ]; then',
                        '  echo "[$(date)] OK $run_name" | tee -a "$OUT/queue.log"',
                        "else",
                        '  echo "[$(date)] FAILED $run_name rc=$rc" | tee -a "$OUT/queue.log"',
                        "fi",
                        "",
                    ]
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/cpfs/user/wanghaotian/SharedRLControl/isaac-training"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_csv_ints, required=True)
    parser.add_argument("--obs", type=parse_csv_ints, default=parse_csv_ints("50,30"))
    parser.add_argument("--methods", type=parse_csv_text, default=parse_csv_text("baseline,full"))
    parser.add_argument("--out-name", default="risk_calibration_H1p5_nm0p5")
    parser.add_argument("--session", default=None)
    parser.add_argument("--venv", type=Path, default=Path("/cpfs/user/wanghaotian/env_isaaclab"))
    parser.add_argument("--baseline-ckpt", type=Path, default=None)
    parser.add_argument("--full-ckpt", type=Path, default=None)
    parser.add_argument("--horizon-sec", type=float, default=1.5)
    parser.add_argument("--near-miss-distance", type=float, default=0.5)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    base = args.base.resolve()
    out_root = base / "posthoc_risk_eval" / args.out_name
    if out_root.exists() and not args.allow_existing:
        raise SystemExit(f"output exists: {out_root} (use --allow-existing to reuse)")

    baseline_ckpt = args.baseline_ckpt or newest_checkpoint(base, ["baseline*/**/checkpoint_best.pt"])
    full_ckpt = args.full_ckpt or newest_checkpoint(base, ["full*/**/checkpoint_best.pt"])

    out_root.mkdir(parents=True, exist_ok=True)
    script_path = out_root / "run_queue.sh"
    script_path.write_text(
        build_script(
            repo=repo,
            out_root=out_root,
            venv=args.venv.resolve(),
            baseline_ckpt=baseline_ckpt.resolve(),
            full_ckpt=full_ckpt.resolve(),
            seeds=args.seeds,
            obs_values=args.obs,
            methods=args.methods,
            horizon_sec=args.horizon_sec,
            near_miss_distance=args.near_miss_distance,
        )
        + "\n"
    )
    script_path.chmod(0o755)

    session = args.session or f"risk_calib_{args.out_name}"
    print(f"script={script_path}")
    print(f"out_root={out_root}")
    print(f"queue_log={out_root / 'queue.log'}")
    print(f"baseline_ckpt={baseline_ckpt}")
    print(f"full_ckpt={full_ckpt}")
    print(f"session={session}")
    print(f"num_jobs={len(args.seeds) * len(args.obs) * len(args.methods)}")

    if args.start:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, str(script_path)], check=True)
        subprocess.run(["tmux", "ls"], check=False)


if __name__ == "__main__":
    main()
