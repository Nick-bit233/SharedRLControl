"""Process launch helpers for experiment entrypoints."""

from __future__ import annotations

import datetime
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


ENTRYPOINTS = {
    "train": "experiments/train.py",
    "eval": "experiments/eval.py",
    "campaign": "experiments/campaign.py",
}


def strip_separator(args: list[str]) -> list[str]:
    """Drop argparse's ``--`` separator from trailing overrides."""

    return args[1:] if args and args[0] == "--" else args


def timestamp() -> str:
    """Return a compact launch timestamp."""

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_session_name(name: str) -> str:
    """Return a tmux-safe session name."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return safe[:80] or "launch"


def override_value(overrides: list[str], key: str) -> str | None:
    """Return the value for a simple Hydra override key."""

    prefix = f"{key}="
    for item in overrides:
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def default_session_name(mode: str, overrides: list[str]) -> str:
    """Build a readable session name from mode and common overrides."""

    label = override_value(overrides, "experiment") or override_value(overrides, "campaign")
    label = label or override_value(overrides, "runtime.spec") or "default"
    return sanitize_session_name(f"{mode}-{label}-{timestamp()}")


def build_entrypoint_command(
    mode: str,
    overrides: list[str],
    *,
    python_executable: str,
) -> list[str]:
    """Build the python command for one unified entrypoint."""

    script = ENTRYPOINTS.get(mode)
    if script is None:
        choices = ", ".join(sorted(ENTRYPOINTS))
        raise ValueError(f"Unknown launch mode '{mode}'. Available modes: {choices}")
    return [python_executable, script, *overrides]


def write_wrapper_script(
    command: list[str],
    *,
    cwd: Path,
    script_file: Path,
    log_file: Path,
    label: str = "launch",
) -> Path:
    """Write a shell wrapper that logs command output and exit status."""

    script_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(command)
    script_file.write_text(
        "#!/usr/bin/env bash\n"
        "set -o pipefail\n"
        f"cd {shlex.quote(str(cwd))}\n"
        f"LOG_FILE={shlex.quote(str(log_file))}\n"
        f"echo \"[{label}] start: $(date -Is)\" | tee -a \"$LOG_FILE\"\n"
        f"echo {shlex.quote(f'[{label}] cwd: {cwd}')} | tee -a \"$LOG_FILE\"\n"
        f"echo {shlex.quote(f'[{label}] command: {command_text}')} | tee -a \"$LOG_FILE\"\n"
        f"({command_text}) 2>&1 | tee -a \"$LOG_FILE\"\n"
        "status=${PIPESTATUS[0]}\n"
        f"echo \"[{label}] exit=${{status}}: $(date -Is)\" | tee -a \"$LOG_FILE\"\n"
        "exit \"$status\"\n"
    )
    script_file.chmod(0o755)
    return script_file


def tmux_session_exists(session: str) -> bool:
    """Return True when a tmux session already exists."""

    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def launch_tmux(
    *,
    session: str,
    script_file: Path,
    attach: bool = False,
    replace: bool = False,
) -> None:
    """Start a wrapper script in a detached tmux session."""

    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is not installed or not on PATH")
    if tmux_session_exists(session):
        if not replace:
            raise RuntimeError(
                f"tmux session already exists: {session}. "
                f"Attach with `tmux attach -t {session}` or rerun with --replace."
            )
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", str(script_file)], check=True)
    if attach:
        subprocess.run(["tmux", "attach", "-t", session], check=True)


def run_foreground(command: list[str], *, cwd: Path) -> int:
    """Run a launch command in the foreground."""

    return subprocess.run(command, cwd=str(cwd)).returncode


def launch_paths(log_dir: Path, session: str) -> tuple[Path, Path]:
    """Return wrapper and log paths for a session."""

    return log_dir / f"{session}.sh", log_dir / f"{session}.log"


__all__ = [
    "ENTRYPOINTS",
    "build_entrypoint_command",
    "default_session_name",
    "launch_paths",
    "launch_tmux",
    "override_value",
    "run_foreground",
    "sanitize_session_name",
    "strip_separator",
    "timestamp",
    "tmux_session_exists",
    "write_wrapper_script",
]

