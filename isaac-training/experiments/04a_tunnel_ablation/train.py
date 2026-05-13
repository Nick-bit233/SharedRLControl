#!/usr/bin/env python3
"""Ablation training entrypoint.

This wrapper intentionally delegates to experiment 04's canonical training
implementation so ablations share the same collector, checkpointing, and eval
logic. Variant behavior is controlled by Hydra configs.
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "experiments/04_tunnel_task/train.py",
        *sys.argv[1:],
    ]
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
