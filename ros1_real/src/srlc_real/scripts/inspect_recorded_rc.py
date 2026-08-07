#!/usr/bin/env python3
"""Print the exact interval and start gate selected from an RC recording."""

import argparse
import json
import math
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from srlc_real_deployment.recorded_rc_replay import (  # noqa: E402
    RecordingFormatError,
    load_recorded_rc_timeline,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect an SRLC JSON/NPZ file using the replay loader."
    )
    parser.add_argument("recording_file")
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--channel-base", type=int, default=1)
    parser.add_argument("--forward-channel", type=int, default=2)
    parser.add_argument("--lateral-channel", type=int, default=1)
    parser.add_argument("--vertical-channel", type=int, default=3)
    args = parser.parse_args()
    motion_indices = (
        args.forward_channel - args.channel_base,
        args.lateral_channel - args.channel_base,
        args.vertical_channel - args.channel_base,
    )

    try:
        timeline = load_recorded_rc_timeline(
            args.recording_file,
            replay_start_time=args.start_time,
            replay_end_time=args.end_time,
            motion_indices=motion_indices,
        )
    except RecordingFormatError as exc:
        parser.error(str(exc))

    first = timeline.samples[0]
    output = {
        "source_file": timeline.source_path,
        "source_run_id": timeline.source_run_id,
        "source_stream": timeline.source_stream,
        "source_start_time": timeline.source_start_time,
        "source_end_time": timeline.source_end_time,
        "duration": timeline.duration,
        "samples": len(timeline.samples),
        "reference_position": timeline.reference_position,
        "reference_yaw_rad": timeline.reference_yaw,
        "reference_yaw_deg": (
            math.degrees(timeline.reference_yaw)
            if timeline.reference_yaw is not None
            else None
        ),
        "control_modes": timeline.control_modes,
        "effective_modes": timeline.effective_modes,
        "first_motion_pwm": {
            "forward": first.channels[motion_indices[0]],
            "lateral": first.channels[motion_indices[1]],
            "vertical": first.channels[motion_indices[2]],
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
