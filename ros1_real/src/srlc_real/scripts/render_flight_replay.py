#!/usr/bin/env python3
"""Render an isolated RViz flight replay to a validated H.264 MP4."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import rosgraph
import rosnode
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from srlc_real_deployment.flight_replay import (  # noqa: E402
    FlightRecordingError,
    load_flight_timeline,
)
from srlc_real_deployment.replay_video_overlay import (  # noqa: E402
    StickHudRenderer,
    iter_stick_hud_frames,
)


def _parser():
    parser = argparse.ArgumentParser(
        description="Run RViz under Xvfb and capture a real-flight replay."
    )
    parser.add_argument("--recording", required=True)
    parser.add_argument("--pcd-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--map-yaw-deg", type=float, default=0.0)
    parser.add_argument("--map-origin-x", type=float, default=0.0)
    parser.add_argument("--map-origin-y", type=float, default=0.0)
    parser.add_argument("--map-origin-z", type=float, default=0.0)
    parser.add_argument("--max-sample-gap", type=float, default=0.2)
    parser.add_argument("--ceiling-z", type=float, default=2.8)
    parser.add_argument("--input-prediction-max-xy-speed", type=float, default=0.5)
    parser.add_argument("--prediction-collision-radius", type=float, default=0.25)
    parser.add_argument("--prediction-collision-min-z", type=float, default=0.30)
    parser.add_argument("--no-input-prediction", action="store_true")
    parser.add_argument("--no-human-input-overlay", action="store_true")
    parser.add_argument("--stick-trail-seconds", type=float, default=1.5)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--lead-seconds", type=float, default=1.0)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--rviz-warmup", type=float, default=3.0)
    parser.add_argument("--display", default=":99")
    parser.add_argument("--output-uid", type=int, default=-1)
    parser.add_argument("--output-gid", type=int, default=-1)
    return parser


def _resolution(value):
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", value)
    if not match:
        raise ValueError("resolution must use WIDTHxHEIGHT")
    width, height = int(match.group(1)), int(match.group(2))
    if width % 2 or height % 2:
        raise ValueError("H.264 yuv420p resolution dimensions must be even")
    return width, height


def _validate_args(args):
    if (args.start_time is None) != (args.end_time is None):
        raise ValueError("--start-time and --end-time must be set together")
    if not math.isfinite(args.speed) or args.speed <= 0.0:
        raise ValueError("--speed must be finite and positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if (args.output_uid < 0) != (args.output_gid < 0):
        raise ValueError("--output-uid and --output-gid must be set together")
    for name in ("lead_seconds", "tail_seconds", "rviz_warmup"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("--%s must be finite and non-negative" % name.replace("_", "-"))
    for name in (
        "input_prediction_max_xy_speed",
        "prediction_collision_radius",
        "stick_trail_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("--%s must be finite and positive" % name.replace("_", "-"))
    if (
        not math.isfinite(args.prediction_collision_min_z)
        or not math.isfinite(args.ceiling_z)
        or args.prediction_collision_min_z >= args.ceiling_z
    ):
        raise ValueError(
            "--prediction-collision-min-z must be lower than --ceiling-z"
        )
    recording = Path(args.recording).resolve()
    pcd_file = Path(args.pcd_file).resolve()
    output = Path(args.output).resolve()
    if not recording.is_file():
        raise ValueError("recording not found: %s" % recording)
    if not pcd_file.is_file():
        raise ValueError("PCD file not found: %s" % pcd_file)
    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    return recording, pcd_file, output


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _start_process(command, env, stdout=None):
    return subprocess.Popen(
        command,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
        start_new_session=True,
    )


def _stop_process(process, timeout=15.0):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.0)


def _wait_for_display(display, env, xvfb, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if xvfb.poll() is not None:
            raise RuntimeError("Xvfb exited before its display became ready")
        check = subprocess.run(
            ["xdpyinfo", "-display", display],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if check.returncode == 0:
            return
        time.sleep(0.1)
    raise RuntimeError("timed out waiting for Xvfb display %s" % display)


def _wait_for_master(launch, timeout=30.0):
    master = rosgraph.Master("/flight_replay_renderer")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if launch.poll() is not None:
            raise RuntimeError("roslaunch exited before ROS master became ready")
        try:
            master.getPid()
            return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("timed out waiting for ROS master")


def _wait_for_rviz(launch, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not rospy.is_shutdown():
        if launch.poll() is not None:
            raise RuntimeError("roslaunch exited before RViz became ready")
        try:
            if any("srlc_flight_replay_rviz" in name for name in rosnode.get_node_names()):
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("timed out waiting for RViz")


def _probe_video(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,pix_fmt",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("rendered video does not contain exactly one video stream")
    return payload, streams[0], float(payload["format"]["duration"])


def _validate_video(path, width, height, fps, expected_duration):
    payload, stream, duration = _probe_video(path)
    actual_fps = float(Fraction(stream["r_frame_rate"]))
    problems = []
    if stream.get("codec_name") != "h264":
        problems.append("codec=%s" % stream.get("codec_name"))
    if int(stream.get("width", 0)) != width or int(stream.get("height", 0)) != height:
        problems.append(
            "resolution=%sx%s" % (stream.get("width"), stream.get("height"))
        )
    if abs(actual_fps - fps) > 0.05:
        problems.append("fps=%.3f" % actual_fps)
    if stream.get("pix_fmt") != "yuv420p":
        problems.append("pix_fmt=%s" % stream.get("pix_fmt"))
    if duration < max(0.1, expected_duration - 2.0) or duration > expected_duration + 5.0:
        problems.append(
            "duration=%.3fs expected_about=%.3fs" % (duration, expected_duration)
        )
    if problems:
        raise RuntimeError("invalid rendered video: " + ", ".join(problems))
    return payload, duration


def _compose_human_input_hud(
    capture_path,
    output_path,
    timeline,
    args,
    video_duration,
    video_width,
    video_height,
    prediction_info,
):
    hud = StickHudRenderer()
    margin = 36
    panel_x = video_width - hud.width - margin
    panel_y = video_height - hud.height - margin
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    filters = [
        "[1:v]format=rgba[stick]",
        "[0:v][stick]overlay=x=%d:y=%d:eof_action=repeat:shortest=1[with_stick]"
        % (panel_x, panel_y),
        (
            "[with_stick]drawtext=fontfile=%s:text='RIGHT STICK  FWD UP':"
            "fontcolor=white@0.94:fontsize=21:x=%d:y=%d[with_title]"
        )
        % (font, panel_x + 18, panel_y + 13),
        (
            "[with_title]drawbox=x=32:y=28:w=430:h=132:"
            "color=black@0.48:t=fill[legend_bg]"
        ),
        (
            "[legend_bg]drawbox=x=52:y=57:w=30:h=6:"
            "color=0x00EBFF@1.0:t=fill[measured_swatch]"
        ),
        (
            "[measured_swatch]drawtext=fontfile=%s:text='MEASURED TRAJECTORY':"
            "fontcolor=white@0.95:fontsize=21:x=96:y=43[measured_label]"
        )
        % font,
        (
            "[measured_label]drawbox=x=52:y=96:w=30:h=7:"
            "color=0xFF50E1@0.75:t=fill[prediction_swatch]"
        ),
        (
            "[prediction_swatch]drawtext=fontfile=%s:"
            "text='RAW-INPUT ESTIMATE':fontcolor=white@0.95:"
            "fontsize=21:x=96:y=82[prediction_label]"
        )
        % font,
    ]
    if prediction_info.get("contact_predicted"):
        contact_elapsed = float(
            prediction_info.get("first_contact", {}).get("elapsed", 0.0)
        )
        filters.extend(
            [
                (
                    "[prediction_label]drawbox=x=52:y=135:w=30:h=7:"
                    "color=0xFF2020@0.95:t=fill[contact_swatch]"
                ),
                (
                    "[contact_swatch]drawtext=fontfile=%s:"
                    "text='CONTACT @ %.2f s':"
                    "fontcolor=0xFF7068@1.0:"
                    "fontsize=21:x=96:y=121[out]"
                )
                % (font, contact_elapsed),
            ]
        )
    elif prediction_info.get("collision_check") == "complete":
        filters.append(
            (
                "[prediction_label]drawtext=fontfile=%s:"
                "text='NO CONTACT PREDICTED':fontcolor=0x76F2A0@1.0:"
                "fontsize=20:x=52:y=121[out]"
            )
            % font
        )
    else:
        filters.append(
            (
                "[prediction_label]drawtext=fontfile=%s:"
                "text='CONTACT CHECK UNAVAILABLE':fontcolor=0xFFD166@1.0:"
                "fontsize=19:x=52:y=121[out]"
            )
            % font
        )
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(capture_path),
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        "%dx%d" % (hud.width, hud.height),
        "-framerate",
        str(args.fps),
        "-i",
        "pipe:0",
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-an",
        "-r",
        str(args.fps),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    write_error = None
    try:
        for frame in iter_stick_hud_frames(
            timeline,
            fps=args.fps,
            video_duration=video_duration,
            lead_seconds=args.lead_seconds,
            speed=args.speed,
            trail_seconds=args.stick_trail_seconds,
            renderer=hud,
        ):
            process.stdin.write(frame)
    except (BrokenPipeError, OSError) as exc:
        write_error = exc
    finally:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
    return_code = process.wait()
    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    if return_code != 0 or write_error is not None:
        details = stderr or str(write_error) or "unknown ffmpeg error"
        raise RuntimeError("failed to compose human-input HUD: %s" % details)


def _launch_command(args, recording, pcd_file):
    start_value = -1.0 if args.start_time is None else args.start_time
    end_value = -1.0 if args.end_time is None else args.end_time
    return [
        "roslaunch",
        "srlc_real",
        "flight_replay.launch",
        "recording_file:=%s" % recording,
        "pcd_file:=%s" % pcd_file,
        "start_time:=%.17g" % start_value,
        "end_time:=%.17g" % end_value,
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
        "autostart:=false",
        "rviz:=true",
        "fullscreen:=true",
    ]


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        width, height = _resolution(args.resolution)
        recording, pcd_file, output = _validate_args(args)
        timeline = load_flight_timeline(
            str(recording),
            start_time=args.start_time,
            end_time=args.end_time,
            map_yaw_deg=args.map_yaw_deg,
            map_origin_xyz=(
                args.map_origin_x,
                args.map_origin_y,
                args.map_origin_z,
            ),
            max_sample_gap=args.max_sample_gap,
        )
    except (ValueError, FlightRecordingError) as exc:
        print("render configuration error: %s" % exc, file=sys.stderr)
        return 2

    capture = output.with_name(output.stem + ".capture.mp4")
    partial = output.with_name(output.stem + ".partial.mp4")
    metadata_path = output.with_suffix(".json")
    runtime_dir = tempfile.mkdtemp(prefix="srlc-rviz-runtime-")
    os.chmod(runtime_dir, 0o700)
    env = os.environ.copy()
    env.update(
        {
            "DISPLAY": args.display,
            "ROS_MASTER_URI": "http://127.0.0.1:11311",
            "ROS_IP": "127.0.0.1",
            "QT_X11_NO_MITSHM": "1",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "DISABLE_ROS1_EOL_WARNINGS": "1",
            "XDG_RUNTIME_DIR": runtime_dir,
        }
    )
    os.environ.update(env)

    xvfb = None
    launch = None
    ffmpeg = None
    completion = threading.Event()
    for temporary in (capture, partial):
        if temporary.exists():
            temporary.unlink()

    try:
        print(
            "Rendering source %.3f..%.3fs (%.3fs) at %.3fx"
            % (
                timeline.source_start_time,
                timeline.source_end_time,
                timeline.duration,
                args.speed,
            ),
            flush=True,
        )
        xvfb = _start_process(
            [
                "Xvfb",
                args.display,
                "-screen",
                "0",
                "%dx%dx24" % (width, height),
                "-nolisten",
                "tcp",
                "+extension",
                "GLX",
                "+render",
                "-noreset",
            ],
            env,
        )
        _wait_for_display(args.display, env, xvfb)

        launch = _start_process(_launch_command(args, recording, pcd_file), env)
        _wait_for_master(launch)
        rospy.init_node(
            "flight_replay_renderer", anonymous=True, disable_signals=True
        )
        state = rospy.wait_for_message(
            "/srlc/replay/state", String, timeout=30.0
        )
        if state.data != "READY":
            raise RuntimeError("unexpected initial replay state: %s" % state.data)
        cloud = rospy.wait_for_message(
            "/real_map/cloud", PointCloud2, timeout=30.0
        )
        if cloud.width <= 0:
            raise RuntimeError("obstacle point cloud is empty")
        ceiling = rospy.wait_for_message(
            "/real_map/ceiling", PointCloud2, timeout=30.0
        )
        prediction_info = json.loads(
            rospy.wait_for_message(
                "/srlc/replay/input_prediction_info", String, timeout=30.0
            ).data
        )
        _wait_for_rviz(launch)
        print(
            "RViz ready with %d obstacle and %d ceiling points; "
            "warming up for %.1fs"
            % (cloud.width, ceiling.width, args.rviz_warmup),
            flush=True,
        )
        time.sleep(args.rviz_warmup)

        def complete_cb(message):
            if message.data:
                completion.set()

        complete_sub = rospy.Subscriber(
            "/srlc/replay/complete", Bool, complete_cb, queue_size=1
        )
        ffmpeg = _start_process(
            [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-loglevel",
                "warning",
                "-f",
                "x11grab",
                "-draw_mouse",
                "0",
                "-framerate",
                str(args.fps),
                "-video_size",
                "%dx%d" % (width, height),
                "-i",
                "%s.0+0,0" % args.display,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(capture),
            ],
            env,
        )
        time.sleep(args.lead_seconds)
        rospy.wait_for_service("/srlc/replay/play", timeout=10.0)
        play = rospy.ServiceProxy("/srlc/replay/play", SetBool)
        response = play(True)
        if not response.success:
            raise RuntimeError("failed to start replay: %s" % response.message)

        timeout = timeline.duration / args.speed + 30.0
        deadline = time.monotonic() + timeout
        while not completion.wait(0.2):
            if launch.poll() is not None:
                raise RuntimeError("roslaunch exited during replay")
            if ffmpeg.poll() is not None:
                raise RuntimeError("ffmpeg exited during replay")
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for replay completion")
        complete_sub.unregister()
        print("Replay complete; holding final trajectory for %.1fs" % args.tail_seconds)
        time.sleep(args.tail_seconds)

        _stop_process(ffmpeg)
        ffmpeg = None
        expected_duration = (
            args.lead_seconds + timeline.duration / args.speed + args.tail_seconds
        )
        _, capture_duration = _validate_video(
            capture, width, height, args.fps, expected_duration
        )
        hud_enabled = (
            not args.no_human_input_overlay and timeline.has_human_input
        )
        if hud_enabled:
            print(
                "Composing right-stick HUD and trajectory legend",
                flush=True,
            )
            _compose_human_input_hud(
                capture,
                partial,
                timeline,
                args,
                capture_duration,
                width,
                height,
                prediction_info,
            )
        else:
            os.replace(str(capture), str(partial))
        probe, actual_duration = _validate_video(
            partial, width, height, args.fps, expected_duration
        )
        if capture.exists():
            capture.unlink()
        os.replace(str(partial), str(output))
        metadata = {
            "recording_file": recording.name,
            "recording_sha256": _sha256(recording),
            "pcd_file": pcd_file.name,
            "pcd_sha256": _sha256(pcd_file),
            "obstacle_point_count": int(cloud.width),
            "ceiling_point_count": int(ceiling.width),
            "ceiling_z": args.ceiling_z,
            "ceiling_alpha": 0.18,
            "source_run_id": timeline.source_run_id,
            "source_start_time": timeline.source_start_time,
            "source_end_time": timeline.source_end_time,
            "source_duration": timeline.duration,
            "window_reason": timeline.window_reason,
            "speed": args.speed,
            "map_yaw_deg": args.map_yaw_deg,
            "map_origin_xyz": [
                args.map_origin_x,
                args.map_origin_y,
                args.map_origin_z,
            ],
            "lead_seconds": args.lead_seconds,
            "tail_seconds": args.tail_seconds,
            "video_duration": actual_duration,
            "resolution": "%dx%d" % (width, height),
            "fps": args.fps,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "camera_mode": "fixed_world_azimuth_position_follow",
            "human_input_overlay": {
                "enabled": hud_enabled,
                "available": timeline.has_human_input,
                "source": "samples[].human_action",
                "mapping": "screen_up=forward, screen_right=-body_lateral",
                "trail_seconds": args.stick_trail_seconds,
            },
            "trajectory_colors": {
                "measured": "cyan_opaque",
                "raw_input_estimate": "magenta_alpha_0.62",
                "estimated_contact": "red",
            },
            "input_prediction": prediction_info,
            "ffprobe": probe,
            "output_file": output.name,
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        if args.output_uid >= 0:
            os.chown(output, args.output_uid, args.output_gid)
            os.chown(metadata_path, args.output_uid, args.output_gid)
        print("VIDEO_OUTPUT=%s" % output, flush=True)
        print("VIDEO_METADATA=%s" % metadata_path, flush=True)
        return 0
    except Exception as exc:
        print("flight replay render failed: %s" % exc, file=sys.stderr, flush=True)
        for temporary in (capture, partial):
            if temporary.exists():
                temporary.unlink()
        return 1
    finally:
        _stop_process(ffmpeg)
        _stop_process(launch)
        _stop_process(xvfb)
        shutil.rmtree(runtime_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
