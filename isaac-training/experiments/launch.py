"""Unified foreground/tmux launcher for train, eval, and campaign commands."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.launching import (
    build_entrypoint_command,
    default_session_name,
    launch_paths,
    launch_tmux,
    run_foreground,
    strip_separator,
    write_wrapper_script,
)


def _add_mode_parser(subparsers: argparse._SubParsersAction, mode: str) -> None:
    parser = subparsers.add_parser(mode, help=f"Launch experiments/{mode}.py")
    parser.add_argument("--tmux", action="store_true", help="Run in a detached tmux session")
    parser.add_argument("--foreground", action="store_true", help="Run in the foreground")
    parser.add_argument("--dry-run", action="store_true", help="Print command and write wrapper only")
    parser.add_argument("--attach", action="store_true", help="Attach after starting tmux")
    parser.add_argument("--replace", action="store_true", help="Replace an existing tmux session")
    parser.add_argument("--session", default=None, help="tmux session name")
    parser.add_argument("--log-dir", default="outputs/launch_logs", help="Directory for logs and wrappers")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides after --")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch unified experiment entrypoints")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("train", "eval", "campaign"):
        _add_mode_parser(subparsers, mode)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = strip_separator(args.overrides)
    command = build_entrypoint_command(
        args.mode,
        overrides,
        python_executable=sys.executable,
    )
    session = args.session or default_session_name(args.mode, overrides)
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    script_file, log_file = launch_paths(log_dir, session)
    write_wrapper_script(command, cwd=REPO_ROOT, script_file=script_file, log_file=log_file)

    print(f"[launch] mode: {args.mode}")
    print(f"[launch] session: {session}")
    print(f"[launch] log: {log_file}")
    print(f"[launch] wrapper: {script_file}")
    print(f"[launch] command: {shlex.join(command)}")

    if args.dry_run:
        return
    if args.tmux:
        launch_tmux(
            session=session,
            script_file=script_file,
            attach=args.attach,
            replace=args.replace,
        )
        print(f"[launch] started. Attach with: tmux attach -t {session}")
        return

    raise SystemExit(run_foreground(command, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
