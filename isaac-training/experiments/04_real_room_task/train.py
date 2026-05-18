#!/usr/bin/env python3
"""Real-room SRLC training entrypoint.

Delegates to experiment 04's canonical trainer while selecting the real-room
Hydra config by default.
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    args = sys.argv[1:]
    if not any(arg.startswith("experiment=") for arg in args):
        args = ["experiment=real_room", *args]
    cmd = [sys.executable, "experiments/04_tunnel_task/train.py", *args]
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
