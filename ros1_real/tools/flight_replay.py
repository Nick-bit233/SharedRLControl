#!/usr/bin/env python3
"""Preview or render the ROS1 real-flight RViz replay in an isolated container."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "docker-compose.real.yml"
DEFAULT_IMAGE = "srlc_ros1_real:noetic"
DEFAULT_PCD_NAME = (
    "room601/0717_section_resampled_0p05_ascii_aligned_floor_level_z0.pcd"
)


def _default_pcd():
    configured = os.environ.get("SRLC_MAP_HOST_DIR")
    if configured:
        root = Path(configured)
        if not root.is_absolute():
            root = REPO_ROOT / root
    else:
        root = REPO_ROOT.parent / "ros1" / "real_maps"
    return root / DEFAULT_PCD_NAME


def _common_arguments(parser):
    parser.add_argument("--recording", required=True, help="Recorder JSON or trusted NPZ")
    parser.add_argument(
        "--pcd-file",
        default=str(_default_pcd()),
        help="Obstacle PCD; defaults to the current 0717 room map",
    )
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--map-yaw-deg", type=float, default=0.0)
    parser.add_argument("--map-origin-x", type=float, default=0.0)
    parser.add_argument("--map-origin-y", type=float, default=0.0)
    parser.add_argument("--map-origin-z", type=float, default=0.0)
    parser.add_argument("--max-sample-gap", type=float, default=0.2)
    parser.add_argument(
        "--ceiling-z",
        type=float,
        default=2.8,
        help="Points at or above this map Z use the 18%% ceiling layer",
    )
    parser.add_argument("--input-prediction-max-xy-speed", type=float, default=0.5)
    parser.add_argument("--prediction-collision-radius", type=float, default=0.25)
    parser.add_argument("--prediction-collision-min-z", type=float, default=0.30)
    parser.add_argument("--no-input-prediction", action="store_true")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the existing image instead of rebuilding current sources",
    )


def _parser():
    parser = argparse.ArgumentParser(
        description="RViz preview and deterministic MP4 rendering for real flights."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="Open interactive RViz through X11")
    _common_arguments(preview)

    render = subparsers.add_parser("render", help="Render headlessly to H.264 MP4")
    _common_arguments(render)
    render.add_argument("--output")
    render.add_argument("--resolution", default="1920x1080")
    render.add_argument("--fps", type=int, default=30)
    render.add_argument("--lead-seconds", type=float, default=1.0)
    render.add_argument("--tail-seconds", type=float, default=2.0)
    render.add_argument("--rviz-warmup", type=float, default=3.0)
    render.add_argument("--stick-trail-seconds", type=float, default=1.5)
    render.add_argument("--no-human-input-overlay", action="store_true")
    return parser


def _validate(args):
    if (args.start_time is None) != (args.end_time is None):
        raise ValueError("--start-time and --end-time must be set together")
    recording = Path(args.recording).expanduser().resolve()
    pcd_file = Path(args.pcd_file).expanduser().resolve()
    if not recording.is_file():
        raise ValueError("recording not found: %s" % recording)
    if recording.suffix.lower() not in (".json", ".npz"):
        raise ValueError("--recording must end in .json or .npz")
    if not pcd_file.is_file():
        raise ValueError("PCD file not found: %s" % pcd_file)
    if pcd_file.suffix.lower() != ".pcd":
        raise ValueError("--pcd-file must end in .pcd")
    return recording, pcd_file


def _build_image(args):
    if args.skip_build:
        return
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "build",
            "real_runtime",
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )


def _container_common(args, recording, pcd_file):
    recording_target = "/replay/input/recording" + recording.suffix.lower()
    pcd_target = "/replay/map/obstacles.pcd"
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--network",
        "none",
        "--shm-size",
        "2g",
        "-e",
        "ROS_MASTER_URI=http://127.0.0.1:11311",
        "-e",
        "ROS_IP=127.0.0.1",
        "-e",
        "QT_X11_NO_MITSHM=1",
        "-e",
        "DISABLE_ROS1_EOL_WARNINGS=1",
        "-v",
        "%s:%s:ro" % (recording, recording_target),
        "-v",
        "%s:%s:ro" % (pcd_file, pcd_target),
    ]
    return command, recording_target, pcd_target


def _replay_options(args):
    options = [
        "--speed",
        "%.17g" % args.speed,
        "--map-yaw-deg",
        "%.17g" % args.map_yaw_deg,
        "--map-origin-x",
        "%.17g" % args.map_origin_x,
        "--map-origin-y",
        "%.17g" % args.map_origin_y,
        "--map-origin-z",
        "%.17g" % args.map_origin_z,
        "--max-sample-gap",
        "%.17g" % args.max_sample_gap,
        "--ceiling-z",
        "%.17g" % args.ceiling_z,
        "--input-prediction-max-xy-speed",
        "%.17g" % args.input_prediction_max_xy_speed,
        "--prediction-collision-radius",
        "%.17g" % args.prediction_collision_radius,
        "--prediction-collision-min-z",
        "%.17g" % args.prediction_collision_min_z,
    ]
    if args.no_input_prediction:
        options.append("--no-input-prediction")
    if args.start_time is not None:
        options.extend(
            [
                "--start-time",
                "%.17g" % args.start_time,
                "--end-time",
                "%.17g" % args.end_time,
            ]
        )
    return options


def _launch_assignments(args):
    values = [
        "speed:=%.17g" % args.speed,
        "map_yaw_deg:=%.17g" % args.map_yaw_deg,
        "map_origin_x:=%.17g" % args.map_origin_x,
        "map_origin_y:=%.17g" % args.map_origin_y,
        "map_origin_z:=%.17g" % args.map_origin_z,
        "max_sample_gap:=%.17g" % args.max_sample_gap,
        "ceiling_z:=%.17g" % args.ceiling_z,
        "input_prediction_enabled:=%s"
        % str(not args.no_input_prediction).lower(),
        "input_prediction_max_xy_speed:=%.17g"
        % args.input_prediction_max_xy_speed,
        "prediction_collision_radius:=%.17g"
        % args.prediction_collision_radius,
        "prediction_collision_min_z:=%.17g"
        % args.prediction_collision_min_z,
    ]
    if args.start_time is not None:
        values.extend(
            [
                "start_time:=%.17g" % args.start_time,
                "end_time:=%.17g" % args.end_time,
            ]
        )
    return values


def _preview(args, recording, pcd_file):
    display = os.environ.get("DISPLAY", "")
    if not display:
        raise ValueError("preview requires DISPLAY and an accessible X11 server")
    command, recording_target, pcd_target = _container_common(
        args, recording, pcd_file
    )
    command.extend(["-e", "DISPLAY=%s" % display, "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw"])
    xauthority = os.environ.get("XAUTHORITY", "")
    if xauthority and Path(xauthority).is_file():
        command.extend(
            [
                "-e",
                "XAUTHORITY=/tmp/.docker.xauth",
                "-v",
                "%s:/tmp/.docker.xauth:ro" % Path(xauthority).resolve(),
            ]
        )
    command.extend(
        [
            args.image,
            "roslaunch",
            "srlc_real",
            "flight_replay.launch",
            "recording_file:=%s" % recording_target,
            "pcd_file:=%s" % pcd_target,
            "autostart:=true",
            "rviz:=true",
            "fullscreen:=false",
        ]
        + _launch_assignments(args)
    )
    return subprocess.run(command, check=False).returncode


def _render(args, recording, pcd_file):
    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        output = (
            REPO_ROOT
            / "artifacts"
            / "replay_video"
            / (recording.stem + ".mp4")
        )
    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    command, recording_target, pcd_target = _container_common(
        args, recording, pcd_file
    )
    output_target = "/replay/output/" + output.name
    command.extend(["-v", "%s:/replay/output:rw" % output.parent])
    command.extend(
        [
            args.image,
            "rosrun",
            "srlc_real",
            "render_flight_replay.py",
            "--recording",
            recording_target,
            "--pcd-file",
            pcd_target,
            "--output",
            output_target,
            "--resolution",
            args.resolution,
            "--fps",
            str(args.fps),
            "--lead-seconds",
            "%.17g" % args.lead_seconds,
            "--tail-seconds",
            "%.17g" % args.tail_seconds,
            "--rviz-warmup",
            "%.17g" % args.rviz_warmup,
            "--stick-trail-seconds",
            "%.17g" % args.stick_trail_seconds,
            "--output-uid",
            str(os.getuid()),
            "--output-gid",
            str(os.getgid()),
        ]
        + _replay_options(args)
    )
    if args.no_human_input_overlay:
        command.append("--no-human-input-overlay")
    return subprocess.run(command, check=False).returncode


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        recording, pcd_file = _validate(args)
        _build_image(args)
        if args.command == "preview":
            return _preview(args, recording, pcd_file)
        return _render(args, recording, pcd_file)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print("flight replay failed: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
