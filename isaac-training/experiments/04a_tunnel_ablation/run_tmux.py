#!/usr/bin/env python3
"""Launch tunnel ablation jobs inside tmux sessions."""
from __future__ import annotations

import argparse
import datetime
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ISAAC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ISAAC_ROOT.parent
DEFAULT_LOG_DIR = ISAAC_ROOT / "outputs" / "tunnel_ablation" / "tmux_logs"
CURRICULUM_VARIANTS = ("ours_retrain", "no_residual", "follow_only", "safety_reg")


def strip_separator(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def sanitize_session_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return safe[:80] or "ablation"


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_checkpoint_path(path_text: str) -> Path:
    path = Path(path_text.strip()).expanduser()
    if path.exists():
        return path.resolve()

    marker = "SharedRLControl/"
    if marker in path_text:
        suffix = path_text.split(marker, 1)[1].strip()
        candidate = REPO_ROOT / suffix
        if candidate.exists():
            return candidate.resolve()

    return path


def read_marker(marker_path: Path) -> Path | None:
    if not marker_path.exists():
        return None
    checkpoint = normalize_checkpoint_path(marker_path.read_text())
    return checkpoint if checkpoint.exists() else None


def latest_output_dir(base: Path) -> Path:
    if not base.is_dir():
        raise FileNotFoundError(f"Output base does not exist: {base}")
    subdirs = [path for path in base.iterdir() if path.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run directories found in {base}")
    return max(subdirs, key=lambda path: path.stat().st_mtime)


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def latest_periodic_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in run_dir.glob("**/checkpoint_*.pt")
        if checkpoint_step(path) >= 0
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: (checkpoint_step(path), path.stat().st_mtime)).resolve()


def find_checkpoint(run_dir: Path, kind: str) -> Path:
    marker_names = {
        "latest": "latest_checkpoint_path.txt",
        "final": "final_checkpoint_path.txt",
        "best": "best_checkpoint_path.txt",
    }
    if kind in marker_names:
        checkpoint = read_marker(run_dir / marker_names[kind])
        if checkpoint is not None:
            return checkpoint

    if kind == "latest":
        checkpoint = latest_periodic_checkpoint(run_dir)
        if checkpoint is not None:
            return checkpoint

    fallback_names = {
        "final": "checkpoint_final.pt",
        "best": "checkpoint_best.pt",
    }
    fallback_name = fallback_names.get(kind)
    if fallback_name:
        candidates = list(run_dir.glob(f"**/{fallback_name}"))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime).resolve()

    raise FileNotFoundError(f"No {kind} checkpoint found under {run_dir}")


def resolve_no_curriculum_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        checkpoint = normalize_checkpoint_path(args.checkpoint)
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = ISAAC_ROOT / run_dir
    else:
        output_base = ISAAC_ROOT / "outputs" / "tunnel_ablation" / "no_curriculum" / f"{args.tag}_seed{args.seed}"
        run_dir = latest_output_dir(output_base)
    return find_checkpoint(run_dir, args.checkpoint_kind)


def resolve_curriculum_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.checkpoint:
        checkpoint = normalize_checkpoint_path(args.checkpoint)
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    if args.start_stage <= 1:
        return None

    previous_stage = args.start_stage - 1
    output_base = (
        ISAAC_ROOT
        / "outputs"
        / "tunnel_ablation"
        / args.variant
        / f"stage{previous_stage}"
        / f"{args.source_tag}_seed{args.seed}"
    )
    run_dir = latest_output_dir(output_base)
    return find_checkpoint(run_dir, args.checkpoint_kind)


def build_no_curriculum_resume(args: argparse.Namespace) -> tuple[str, list[str]]:
    checkpoint = resolve_no_curriculum_checkpoint(args)
    extra = strip_separator(args.extra)
    command = [
        sys.executable,
        "experiments/04a_tunnel_ablation/run_matrix.py",
        "--variants",
        "no_curriculum",
        "--seeds",
        str(args.seed),
        "--tag",
        args.tag,
        "--resume-checkpoint",
        str(checkpoint),
        *extra,
    ]
    default_session = f"ablate-no-curriculum-resume-s{args.seed}-{args.tag}"
    return default_session, command


def build_curriculum_rerun(args: argparse.Namespace) -> tuple[str, list[str]]:
    checkpoint = resolve_curriculum_checkpoint(args)
    extra = strip_separator(args.extra)
    command = [
        sys.executable,
        "experiments/04a_tunnel_ablation/run_curriculum.py",
        "--variant",
        args.variant,
        "--seed",
        str(args.seed),
        "--tag",
        args.tag,
        "--start-stage",
        str(args.start_stage),
        "--end-stage",
        str(args.end_stage),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    command.extend(extra)
    default_session = f"ablate-{args.variant}-s{args.seed}-stage{args.start_stage}-{args.end_stage}-{args.tag}"
    return default_session, command


def build_matrix(args: argparse.Namespace) -> tuple[str, list[str]]:
    extra = strip_separator(args.matrix_args)
    if not extra:
        raise ValueError("matrix requires arguments after --, e.g. matrix -- --variants follow_only --seeds 42")
    command = [
        sys.executable,
        "experiments/04a_tunnel_ablation/run_matrix.py",
        *extra,
    ]
    return f"ablate-matrix-{timestamp()}", command


def build_custom(args: argparse.Namespace) -> tuple[str, list[str]]:
    command = strip_separator(args.command)
    if not command:
        raise ValueError("custom requires a command after --")
    return args.name, command


def write_tmux_script(command: list[str], script_file: Path, log_file: Path) -> None:
    script_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(command)
    script_file.write_text(
        "#!/usr/bin/env bash\n"
        "set -o pipefail\n"
        f"cd {shlex.quote(str(ISAAC_ROOT))}\n"
        f"LOG_FILE={shlex.quote(str(log_file))}\n"
        "echo \"[tmux-ablation] start: $(date -Is)\" | tee -a \"$LOG_FILE\"\n"
        f"echo {shlex.quote('[tmux-ablation] cwd: ' + str(ISAAC_ROOT))} | tee -a \"$LOG_FILE\"\n"
        f"echo {shlex.quote('[tmux-ablation] command: ' + command_text)} | tee -a \"$LOG_FILE\"\n"
        f"({command_text}) 2>&1 | tee -a \"$LOG_FILE\"\n"
        "status=${PIPESTATUS[0]}\n"
        "echo \"[tmux-ablation] exit=${status}: $(date -Is)\" | tee -a \"$LOG_FILE\"\n"
        "exit \"$status\"\n"
    )
    script_file.chmod(0o755)


def tmux_session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def launch_tmux(args: argparse.Namespace, default_session: str, command: list[str]) -> None:
    session = sanitize_session_name(args.session or default_session)
    log_dir = Path(args.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = ISAAC_ROOT / log_dir
    run_id = timestamp()
    log_file = log_dir / f"{session}_{run_id}.log"
    script_file = log_dir / "scripts" / f"{session}_{run_id}.sh"

    print(f"[tmux-ablation] session: {session}")
    print(f"[tmux-ablation] log: {log_file}")
    print(f"[tmux-ablation] command: {shlex.join(command)}")

    if args.dry_run:
        print(f"[tmux-ablation] dry-run script: {script_file}")
        return

    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is not installed or not on PATH")

    if tmux_session_exists(session):
        if not args.replace:
            raise RuntimeError(
                f"tmux session already exists: {session}. "
                f"Attach with `tmux attach -t {session}` or rerun with --replace."
        )
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)

    write_tmux_script(command, script_file, log_file)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", str(script_file)], check=True)
    print(f"[tmux-ablation] started. Attach with: tmux attach -t {session}")
    if args.attach:
        subprocess.run(["tmux", "attach", "-t", session], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tunnel ablation commands inside tmux")
    parser.add_argument("--session", default=None, help="tmux session name")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory for tmux logs")
    parser.add_argument("--attach", action="store_true", help="Attach after starting the tmux session")
    parser.add_argument("--replace", action="store_true", help="Replace an existing tmux session with the same name")
    parser.add_argument("--dry-run", action="store_true", help="Print command and write wrapper without starting tmux")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    no_curr = subparsers.add_parser("no-curriculum-resume", help="Resume interrupted NoCurriculum training")
    no_curr.add_argument("--seed", type=int, default=42)
    no_curr.add_argument("--tag", default="paper_ablation_v1")
    no_curr.add_argument("--checkpoint", default=None, help="Explicit checkpoint to resume")
    no_curr.add_argument("--run-dir", default=None, help="Interrupted run dir; defaults to latest tag/seed run")
    no_curr.add_argument("--checkpoint-kind", choices=("latest", "final", "best"), default="latest")
    no_curr.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to run_matrix.py after --")
    no_curr.set_defaults(builder=build_no_curriculum_resume)

    curriculum = subparsers.add_parser("curriculum-rerun", help="Rerun curriculum stages from a warm-start checkpoint")
    curriculum.add_argument("--variant", choices=CURRICULUM_VARIANTS, required=True)
    curriculum.add_argument("--seed", type=int, default=42)
    curriculum.add_argument("--source-tag", default="paper_ablation_v1", help="Tag used to find the previous-stage checkpoint")
    curriculum.add_argument("--tag", default="paper_ablation_v1_fixed", help="Tag for the rerun")
    curriculum.add_argument("--start-stage", type=int, default=2)
    curriculum.add_argument("--end-stage", type=int, default=3)
    curriculum.add_argument("--checkpoint", default=None, help="Explicit warm-start checkpoint")
    curriculum.add_argument("--checkpoint-kind", choices=("final", "best", "latest"), default="final")
    curriculum.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to run_curriculum.py after --")
    curriculum.set_defaults(builder=build_curriculum_rerun)

    matrix = subparsers.add_parser("matrix", help="Run any run_matrix.py invocation in tmux")
    matrix.add_argument("matrix_args", nargs=argparse.REMAINDER, help="Arguments for run_matrix.py after --")
    matrix.set_defaults(builder=build_matrix)

    custom = subparsers.add_parser("custom", help="Run an arbitrary command in tmux from isaac-training/")
    custom.add_argument("--name", required=True, help="Default tmux session name")
    custom.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --")
    custom.set_defaults(builder=build_custom)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        default_session, command = args.builder(args)
        launch_tmux(args, default_session, command)
    except Exception as exc:
        parser.exit(1, f"[tmux-ablation] ERROR: {exc}\n")


if __name__ == "__main__":
    main()
