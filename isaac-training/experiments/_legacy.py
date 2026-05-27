"""Compatibility helpers for deprecated experiment-directory scripts."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _has_override(args: list[str], key: str) -> bool:
    prefixes = (f"{key}=", f"+{key}=")
    return any(arg.startswith(prefixes) for arg in args)


def _exec_entrypoint(script_name: str, args: list[str]) -> None:
    script = REPO_ROOT / "experiments" / script_name
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")


def _with_default_experiment(args: list[str], default_experiment: str | None) -> list[str]:
    if default_experiment is None or _has_override(args, "experiment"):
        return args
    return [f"experiment={default_experiment}", *args]


def run_unified_training(default_experiment: str | None = None) -> None:
    args = _with_default_experiment(sys.argv[1:], default_experiment)
    _exec_entrypoint("train.py", args)


def run_unified_evaluation(default_experiment: str | None = None) -> None:
    args = _with_default_experiment(sys.argv[1:], default_experiment)
    translated: list[str] = []
    for arg in args:
        key, sep, value = arg.lstrip("+").partition("=")
        if not sep:
            translated.append(arg)
        elif key == "resume_checkpoint":
            translated.append(f"eval.checkpoint={value}")
        elif key == "video_dir":
            translated.append(f"eval.output_dir={value}")
        elif key == "eval_seed":
            translated.append(f"eval.seed={value}")
        elif key in {"global_view", "keep_num_envs"}:
            translated.append(f"eval.{key}={value}")
        else:
            translated.append(f"{key}={value}" if arg.startswith("+") else arg)
    _exec_entrypoint("eval.py", translated)


def run_unified_campaign(default_campaign: str = "tunnel_curriculum") -> None:
    args = sys.argv[1:]
    if not _has_override(args, "campaign"):
        args = [f"campaign={default_campaign}", *args]
    _exec_entrypoint("campaign.py", args)


def run_unified_launch(mode: str = "train") -> None:
    _exec_entrypoint("launch.py", [mode, *sys.argv[1:]])
