#!/usr/bin/env python3
"""Run and plot a recorded-RC dry run without connecting to flight hardware.

The default mode runs ``dry_run_px4.launch`` in a network-isolated Docker
container, records the simulated response, and renders three XY trajectories:

* the historical ACTIVE trajectory from the source recording;
* the simulated ASSIST/DIRECT response from the dry run;
* an obstacle- and policy-free zero-order-hold integration of the recorded RC
  motion channels.

Use ``--plot-only`` to regenerate the plot from an existing dry-run JSON.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRLC_SCRIPT_DIR = REPO_ROOT / "src/srlc_real/scripts"
if str(SRLC_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SRLC_SCRIPT_DIR))

from srlc_real_deployment.recorded_rc_replay import (  # noqa: E402
    RecordingFormatError,
    RecordedRcTimeline,
    load_recorded_rc_timeline,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT.parent / "ros1/ckpts/checkpoint_minrisk_0610.pt"
)
DEFAULT_PCD = (
    REPO_ROOT.parent
    / "ros1/real_maps/room601/"
    "0717_section_resampled_0p05_ascii_aligned_floor_level_z0.pcd"
)
RC_CONFIG = REPO_ROOT / "src/srlc_real/cfg/tunnel/rc_input_real_px4.yaml"


class ToolError(RuntimeError):
    """Raised for an actionable dry-run or plotting failure."""


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def _optional_time_pair(
    start_time: Optional[float], end_time: Optional[float]
) -> None:
    if (start_time is None) != (end_time is None):
        raise ToolError(
            "--replay-start-time and --replay-end-time must be supplied together"
        )


def _resolve_existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ToolError("%s does not exist: %s" % (label, resolved))
    return resolved


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolError("failed to read %s JSON %s: %s" % (label, path, exc)) from exc
    if not isinstance(payload, dict):
        raise ToolError("%s JSON root must be an object: %s" % (label, path))
    return payload


def _simple_yaml_scalars(path: Path) -> Dict[str, Any]:
    """Read the flat scalar fields used from the RC YAML without PyYAML."""

    values: Dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            continue
        lowered = raw_value.lower()
        if lowered in ("true", "false"):
            value: Any = lowered == "true"
        else:
            try:
                value = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError):
                value = raw_value.strip("\"'")
        values[key] = value
    return values


def _rc_parameters(
    *, forward_reverse: bool, lateral_reverse: bool
) -> Dict[str, Any]:
    config = _simple_yaml_scalars(RC_CONFIG)
    required = (
        "channel_base",
        "forward_channel",
        "lateral_channel",
        "vertical_channel",
        "pwm_min",
        "pwm_mid",
        "pwm_max",
        "deadband",
        "max_forward_speed",
        "max_lateral_speed",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ToolError(
            "RC configuration is missing required fields: %s" % ", ".join(missing)
        )

    channel_base = int(config["channel_base"])
    parameters = {
        "forward_index": int(config["forward_channel"]) - channel_base,
        "lateral_index": int(config["lateral_channel"]) - channel_base,
        "vertical_index": int(config["vertical_channel"]) - channel_base,
        "pwm_min": float(config["pwm_min"]),
        "pwm_mid": float(config["pwm_mid"]),
        "pwm_max": float(config["pwm_max"]),
        "deadband": float(config["deadband"]),
        "max_forward_speed": float(config["max_forward_speed"]),
        "max_lateral_speed": float(config["max_lateral_speed"]),
        "forward_reverse": bool(forward_reverse),
        "lateral_reverse": bool(lateral_reverse),
    }
    motion_indices = (
        parameters["forward_index"],
        parameters["lateral_index"],
        parameters["vertical_index"],
    )
    if min(motion_indices) < 0 or len(set(motion_indices)) != 3:
        raise ToolError("RC motion-channel indices are invalid: %r" % (motion_indices,))
    return parameters


def axis_from_pwm(
    pwm: int,
    *,
    pwm_min: float,
    pwm_mid: float,
    pwm_max: float,
    deadband: float,
    reverse: bool,
) -> float:
    """Match ``rc_input_node._axis`` for one recorded PWM value."""

    value_f = float(pwm)
    if value_f >= pwm_mid:
        denominator = max(1.0, pwm_max - pwm_mid)
    else:
        denominator = max(1.0, pwm_mid - pwm_min)
    value = max(-1.0, min(1.0, (value_f - pwm_mid) / denominator))
    if abs(value) < deadband:
        value = 0.0
    return -value if reverse else value


def _unique_strings(values: Iterable[str]) -> List[str]:
    output: List[str] = []
    for value in values:
        normalized = str(value).strip().upper()
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _select_mode(timeline: RecordedRcTimeline, requested: str) -> str:
    if requested != "auto":
        return requested
    modes = _unique_strings(timeline.control_modes)
    supported = [mode for mode in modes if mode in ("ASSIST", "DIRECT")]
    if len(supported) == 1:
        return supported[0].lower()
    raise ToolError(
        "cannot infer one downstream mode from recording modes %r; "
        "set --post-takeoff-mode explicitly" % (modes,)
    )


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return cleaned or "recorded_rc"


def _container_launch_command(
    *,
    recording_name: str,
    checkpoint_name: str,
    pcd_name: str,
    run_id: str,
    reference_position: Sequence[float],
    reference_yaw: float,
    mode: str,
    max_xy_speed: float,
    forward_reverse: bool,
    lateral_reverse: bool,
    replay_start_time: Optional[float],
    replay_end_time: Optional[float],
    ros_timeout: float,
) -> str:
    launch_args = [
        "use_recorded_rc_replay:=true",
        "rc_replay_file:=/input/%s" % recording_name,
        "checkpoint:=/assets/checkpoint/%s" % checkpoint_name,
        "pcd_file:=/assets/map/%s" % pcd_name,
        "fake_initial_x:=%.17g" % float(reference_position[0]),
        "fake_initial_y:=%.17g" % float(reference_position[1]),
        "fake_initial_yaw_deg:=%.17g" % math.degrees(float(reference_yaw)),
        "post_takeoff_mode:=%s" % mode,
        "fake_forward_speed:=0.0",
        "fake_lateral_speed:=0.0",
        "fake_vertical_speed:=0.0",
        "forward_reverse:=%s" % str(bool(forward_reverse)).lower(),
        "lateral_reverse:=%s" % str(bool(lateral_reverse)).lower(),
        "max_xy_speed_real:=%.17g" % max_xy_speed,
        "enable_proximity_hold:=false",
        "enable_collision_detection:=false",
        "rviz:=false",
        "record:=true",
        "output_dir:=/output",
        "run_id:=%s" % run_id,
    ]
    if replay_start_time is not None and replay_end_time is not None:
        launch_args.extend(
            [
                "replay_start_time:=%.17g" % replay_start_time,
                "replay_end_time:=%.17g" % replay_end_time,
            ]
        )
    roslaunch = " ".join(
        shlex.quote(value)
        for value in (
            ["roslaunch", "srlc_real", "dry_run_px4.launch"] + launch_args
        )
    )
    return " && ".join(
        [
            "set -e",
            "mkdir -p /tmp/replay_ws/src",
            "cp -a /workspace/src/. /tmp/replay_ws/src/",
            "cd /tmp/replay_ws",
            "catkin_make -DCATKIN_ENABLE_TESTING=OFF",
            "source devel/setup.bash",
            (
                "timeout --signal=INT --kill-after=8s %.3fs %s"
                % (ros_timeout, roslaunch)
            ),
        ]
    )


def _stream_process(command: Sequence[str], log_path: Path) -> Tuple[int, str]:
    lines: List[str] = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()
                lines.append(line)
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
            raise
        return process.wait(), "".join(lines)


def run_dry_run(
    *,
    recording: Path,
    checkpoint: Path,
    pcd_file: Path,
    output_dir: Path,
    image: str,
    timeline: RecordedRcTimeline,
    mode: str,
    max_xy_speed: float,
    forward_reverse: bool,
    lateral_reverse: bool,
    replay_start_time: Optional[float],
    replay_end_time: Optional[float],
    ros_timeout: Optional[float],
    run_id: Optional[str],
) -> Tuple[Path, Path]:
    """Run the isolated Docker dry run and return its JSON and log paths."""

    if shutil.which("docker") is None:
        raise ToolError("docker executable is not available")
    if timeline.reference_position is None or timeline.reference_yaw is None:
        raise ToolError("recording does not provide a finite replay start pose/yaw")

    timeout_value = (
        float(ros_timeout)
        if ros_timeout is not None
        else max(38.0, float(timeline.duration) + 26.0)
    )
    effective_run_id = run_id or "dry_run_replay_%s_%s" % (
        _safe_name(recording.stem),
        mode,
    )
    before = set(output_dir.glob("%s_*.json" % effective_run_id))
    log_path = output_dir / "dry_run.log"
    container_name = "srlc-recorded-rc-%d-%s" % (
        os.getpid(),
        datetime.now().strftime("%H%M%S"),
    )
    inner_command = _container_launch_command(
        recording_name=recording.name,
        checkpoint_name=checkpoint.name,
        pcd_name=pcd_file.name,
        run_id=effective_run_id,
        reference_position=timeline.reference_position,
        reference_yaw=timeline.reference_yaw,
        mode=mode,
        max_xy_speed=max_xy_speed,
        forward_reverse=forward_reverse,
        lateral_reverse=lateral_reverse,
        replay_start_time=replay_start_time,
        replay_end_time=replay_end_time,
        ros_timeout=timeout_value,
    )
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "-v",
        "%s:/workspace:ro" % REPO_ROOT,
        "-v",
        "%s:/input:ro" % recording.parent,
        "-v",
        "%s:/assets/checkpoint:ro" % checkpoint.parent,
        "-v",
        "%s:/assets/map:ro" % pcd_file.parent,
        "-v",
        "%s:/output:rw" % output_dir,
        image,
        "bash",
        "-lc",
        inner_command,
    ]
    print(
        "[recorded-rc-dryrun] source=%s mode=%s duration=%.3fs output=%s"
        % (recording, mode.upper(), timeline.duration, output_dir)
    )
    return_code, output = _stream_process(docker_command, log_path)

    replay_complete = "[RecordedRC] Replay complete" in output
    recorder_saved = "[SRLC Recorder] Saved" in output
    if not replay_complete or not recorder_saved:
        raise ToolError(
            "dry run did not complete and flush its recorder (exit=%d); see %s"
            % (return_code, log_path)
        )
    if return_code not in (0, 124):
        raise ToolError(
            "dry-run container exited with %d after recording; see %s"
            % (return_code, log_path)
        )

    created = sorted(
        set(output_dir.glob("%s_*.json" % effective_run_id)) - before,
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not created:
        candidates = sorted(
            output_dir.glob("%s_*.json" % effective_run_id),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not candidates:
            raise ToolError("recorder reported success but no output JSON was found")
        created = [candidates[-1]]
    dry_json = created[-1]
    print("[recorded-rc-dryrun] recorder=%s" % dry_json)
    return dry_json, log_path


def _sample_records(
    payload: Dict[str, Any], label: str
) -> List[Dict[str, Any]]:
    raw = payload.get("samples", [])
    if not isinstance(raw, list) or not raw:
        raise ToolError("%s has no samples array" % label)
    records: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ToolError("%s samples[%d] is not an object" % (label, index))
        records.append(item)
    return records


def _finite_xy(record: Dict[str, Any], label: str) -> Tuple[float, float]:
    value = record.get("position")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ToolError("%s has no position[0:2]" % label)
    try:
        xy = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ToolError("%s position is not numeric" % label) from exc
    if not all(math.isfinite(component) for component in xy):
        raise ToolError("%s position is not finite" % label)
    return xy


def _finite_time(record: Dict[str, Any], label: str) -> float:
    try:
        value = float(record["t"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError("%s has no numeric t" % label) from exc
    if not math.isfinite(value):
        raise ToolError("%s t is not finite" % label)
    return value


def _compress(values: Sequence[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    return [value for value, _ in itertools.groupby(values)]


def _motion_tuple(
    channels: Sequence[Any], motion_indices: Sequence[int]
) -> Tuple[int, ...]:
    if len(channels) <= max(motion_indices):
        raise ToolError("RC event does not contain all configured motion channels")
    return tuple(int(channels[index]) for index in motion_indices)


def find_replay_event_window(
    dry_payload: Dict[str, Any],
    expected_transitions: Sequence[Tuple[int, ...]],
    motion_indices: Sequence[int],
    *,
    duration: float,
    start_delay: float,
) -> Tuple[float, float, List[Dict[str, Any]], bool]:
    """Find the replay output block in a recorder stream.

    Exact transition-subsequence matching is preferred.  The ACTIVE timestamp
    plus the configured start delay is a plotting fallback for recordings whose
    first replayed motion tuple is indistinguishable from the preceding neutral
    stream.
    """

    raw_events = dry_payload.get("rc_events", [])
    channel_field = "channels"
    if not isinstance(raw_events, list) or not raw_events:
        raw_events = dry_payload.get("samples", [])
        channel_field = "rc"
    if not isinstance(raw_events, list) or not raw_events:
        raise ToolError("dry-run recording has neither rc_events nor samples")

    active: List[Dict[str, Any]] = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        if str(item.get("lifecycle_state", "")) != "ACTIVE":
            continue
        channels = item.get(channel_field)
        if not isinstance(channels, (list, tuple)):
            continue
        event = dict(item)
        event["_channels"] = channels
        event["_t"] = _finite_time(item, "dry RC event")
        event["_motion"] = _motion_tuple(channels, motion_indices)
        active.append(event)
    if len(active) < 2:
        raise ToolError("dry-run recording has too few ACTIVE RC events")
    if any(right["_t"] <= left["_t"] for left, right in zip(active, active[1:])):
        raise ToolError("dry-run ACTIVE RC timestamps are not strictly increasing")

    groups: List[Dict[str, Any]] = []
    for index, event in enumerate(active):
        if not groups or event["_motion"] != groups[-1]["motion"]:
            groups.append(
                {
                    "motion": event["_motion"],
                    "first": index,
                    "last": index,
                    "start_t": event["_t"],
                    "end_t": event["_t"],
                }
            )
        else:
            groups[-1]["last"] = index
            groups[-1]["end_t"] = event["_t"]

    expected = list(expected_transitions)
    candidate_groups: List[int] = []
    for group_index in range(0, len(groups) - len(expected) + 1):
        observed = [
            group["motion"]
            for group in groups[group_index : group_index + len(expected)]
        ]
        if observed == expected:
            candidate_groups.append(group_index)

    active_start = active[0]["_t"]
    earliest_expected = active_start + max(0.0, start_delay - 0.25)
    candidate_groups = [
        index
        for index in candidate_groups
        if groups[index]["start_t"] >= earliest_expected
    ] or candidate_groups

    if candidate_groups:
        def candidate_score(index: int) -> float:
            after = index + len(expected)
            observed_end = (
                groups[after]["start_t"]
                if after < len(groups)
                else groups[index]["start_t"] + duration
            )
            return abs((observed_end - groups[index]["start_t"]) - duration)

        first_group = min(candidate_groups, key=candidate_score)
        start_t = float(groups[first_group]["start_t"])
        after_group = first_group + len(expected)
        end_t = (
            float(groups[after_group]["start_t"])
            if after_group < len(groups)
            else start_t + duration
        )
        selected = [
            event for event in active if start_t <= event["_t"] < end_t
        ]
        transition_match = _compress(
            [event["_motion"] for event in selected]
        ) == expected
        return start_t, end_t, selected, transition_match

    start_t = active_start + start_delay
    end_t = start_t + duration
    selected = [event for event in active if start_t <= event["_t"] < end_t]
    observed = _compress([event["_motion"] for event in selected])
    transition_match = observed == expected
    if not selected:
        raise ToolError("could not locate the replay interval in dry-run RC events")
    return start_t, end_t, selected, transition_match


def _historical_path(
    source_payload: Dict[str, Any], timeline: RecordedRcTimeline
) -> Tuple[List[float], List[Tuple[float, float]], List[float]]:
    samples = _sample_records(source_payload, "source recording")
    selected: List[Dict[str, Any]] = []
    for sample in samples:
        sample_t = _finite_time(sample, "source sample")
        if timeline.source_start_time <= sample_t < timeline.source_end_time:
            selected.append(sample)
    if len(selected) < 2:
        raise ToolError("source replay interval has fewer than two position samples")

    times = [_finite_time(sample, "source sample") for sample in selected]
    positions = [
        _finite_xy(sample, "source sample") for sample in selected
    ]
    yaws: List[float] = []
    for sample in selected:
        try:
            yaw = float(sample["yaw"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError("source replay samples must contain numeric yaw") from exc
        if not math.isfinite(yaw):
            raise ToolError("source replay sample yaw is not finite")
        yaws.append(yaw)
    times.append(float(timeline.source_end_time))
    positions.append(positions[-1])
    yaws.append(yaws[-1])
    return times, positions, yaws


def _path_length(xy: Any, np: Any) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def plot_coordinates(xy: Any, axis_order: str, np: Any) -> Any:
    """Return coordinates ordered as the plot's horizontal/vertical axes."""

    coordinates = np.asarray(xy, dtype=float)
    if coordinates.shape[-1] != 2:
        raise ToolError("plot coordinates must end with exactly two XY values")
    if axis_order == "xy":
        return coordinates.copy()
    if axis_order == "yx":
        return coordinates[..., [1, 0]]
    raise ToolError("plot axis order must be 'yx' or 'xy'")


def plot_axis_limits(
    minimum: float,
    maximum: float,
    *,
    plot_axis: str,
    mirror: str,
) -> Tuple[float, float]:
    """Return axis limits, reversing the selected display direction."""

    if plot_axis not in ("horizontal", "vertical"):
        raise ToolError("plot_axis must be 'horizontal' or 'vertical'")
    if mirror not in ("none", "horizontal", "vertical", "both"):
        raise ToolError("plot mirror must be none, horizontal, vertical, or both")
    reverse = mirror == "both" or mirror == plot_axis
    return (
        (float(maximum), float(minimum))
        if reverse
        else (float(minimum), float(maximum))
    )


def integrate_recorded_rc(
    timeline: RecordedRcTimeline,
    *,
    source_sample_times: Sequence[float],
    source_yaws: Sequence[float],
    start_xy: Sequence[float],
    rc_parameters: Dict[str, Any],
    max_xy_speed: float,
    np: Any,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Integrate replayed RC motion using the real node's calibration.

    RC samples are held until the next source timestamp.  Body-frame XY velocity
    is rotated by interpolated recorded yaw and clamped to ``max_xy_speed``.
    No policy, LiDAR, proximity hold, collision response, or vehicle dynamics
    are applied.
    """

    sample_times = np.asarray(source_sample_times, dtype=float)
    yaw_values = np.unwrap(np.asarray(source_yaws, dtype=float))
    replay_source_times = np.asarray(
        [sample.source_time for sample in timeline.samples], dtype=float
    )
    replay_yaws = np.interp(replay_source_times, sample_times, yaw_values)
    elapsed = np.asarray(
        [sample.elapsed for sample in timeline.samples] + [timeline.duration],
        dtype=float,
    )
    interval_dt = np.diff(elapsed)
    if np.any(interval_dt <= 0.0):
        raise ToolError("replay integration intervals must be positive")

    positions = [np.asarray(start_xy, dtype=float)]
    body_velocities = []
    world_velocities = []
    forward_index = int(rc_parameters["forward_index"])
    lateral_index = int(rc_parameters["lateral_index"])
    for sample, yaw, dt in zip(timeline.samples, replay_yaws, interval_dt):
        forward = axis_from_pwm(
            sample.channels[forward_index],
            pwm_min=rc_parameters["pwm_min"],
            pwm_mid=rc_parameters["pwm_mid"],
            pwm_max=rc_parameters["pwm_max"],
            deadband=rc_parameters["deadband"],
            reverse=rc_parameters["forward_reverse"],
        ) * rc_parameters["max_forward_speed"]
        lateral = axis_from_pwm(
            sample.channels[lateral_index],
            pwm_min=rc_parameters["pwm_min"],
            pwm_mid=rc_parameters["pwm_mid"],
            pwm_max=rc_parameters["pwm_max"],
            deadband=rc_parameters["deadband"],
            reverse=rc_parameters["lateral_reverse"],
        ) * rc_parameters["max_lateral_speed"]
        body = np.asarray([forward, lateral], dtype=float)
        cosine = math.cos(float(yaw))
        sine = math.sin(float(yaw))
        world = np.asarray(
            [
                cosine * body[0] - sine * body[1],
                sine * body[0] + cosine * body[1],
            ],
            dtype=float,
        )
        speed = float(np.linalg.norm(world))
        if speed > max_xy_speed:
            world *= max_xy_speed / speed
        body_velocities.append(body)
        world_velocities.append(world)
        positions.append(positions[-1] + world * float(dt))
    return (
        elapsed,
        np.asarray(positions, dtype=float),
        np.asarray(body_velocities, dtype=float),
        np.asarray(world_velocities, dtype=float),
        replay_yaws,
    )


def _interpolated_dry_path(
    dry_payload: Dict[str, Any],
    start_t: float,
    end_t: float,
    np: Any,
) -> Tuple[Any, Any]:
    samples = _sample_records(dry_payload, "dry-run recording")
    all_times = np.asarray(
        [_finite_time(sample, "dry sample") for sample in samples], dtype=float
    )
    all_xy = np.asarray(
        [_finite_xy(sample, "dry sample") for sample in samples], dtype=float
    )
    if np.any(np.diff(all_times) <= 0.0):
        raise ToolError("dry-run sample timestamps are not strictly increasing")
    if start_t < all_times[0] or end_t > all_times[-1]:
        raise ToolError("dry-run position samples do not span the replay interval")
    inside = (all_times > start_t) & (all_times < end_t)
    plot_times = np.concatenate(([start_t], all_times[inside], [end_t]))
    plot_xy = np.column_stack(
        [
            np.interp(plot_times, all_times, all_xy[:, axis])
            for axis in (0, 1)
        ]
    )
    return plot_times, plot_xy


def _load_pcd_ascii_xyz(path: Path, np: Any) -> Any:
    data_line = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip().upper().startswith("DATA "):
                if line.strip().lower() != "data ascii":
                    raise ToolError("plot map must be an ASCII PCD: %s" % path)
                data_line = line_number
                break
    if data_line is None:
        raise ToolError("PCD has no DATA header: %s" % path)
    try:
        return np.loadtxt(path, skiprows=data_line, usecols=(0, 1, 2))
    except (OSError, ValueError) as exc:
        raise ToolError("failed to load PCD %s: %s" % (path, exc)) from exc


def _write_integrated_csv(
    path: Path,
    *,
    timeline: RecordedRcTimeline,
    elapsed: Any,
    positions: Any,
    body_velocities: Any,
    world_velocities: Any,
    replay_yaws: Any,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "elapsed_s",
                "source_t_s",
                "x_m",
                "y_m",
                "body_vx_mps",
                "body_vy_mps",
                "world_vx_mps",
                "world_vy_mps",
                "yaw_deg",
                "interval_dt_s",
            ]
        )
        for index, sample in enumerate(timeline.samples):
            writer.writerow(
                [
                    "%.9f" % float(elapsed[index]),
                    "%.9f" % float(sample.source_time),
                    "%.9f" % float(positions[index, 0]),
                    "%.9f" % float(positions[index, 1]),
                    "%.9f" % float(body_velocities[index, 0]),
                    "%.9f" % float(body_velocities[index, 1]),
                    "%.9f" % float(world_velocities[index, 0]),
                    "%.9f" % float(world_velocities[index, 1]),
                    "%.9f" % math.degrees(float(replay_yaws[index])),
                    "%.9f" % float(elapsed[index + 1] - elapsed[index]),
                ]
            )
        writer.writerow(
            [
                "%.9f" % float(elapsed[-1]),
                "%.9f" % float(timeline.source_end_time),
                "%.9f" % float(positions[-1, 0]),
                "%.9f" % float(positions[-1, 1]),
                "",
                "",
                "",
                "",
                "",
                "0.000000000",
            ]
        )


def render_comparison(
    *,
    source_path: Path,
    dry_path: Path,
    pcd_file: Path,
    output_dir: Path,
    timeline: RecordedRcTimeline,
    mode: str,
    max_xy_speed: float,
    rc_parameters: Dict[str, Any],
    start_delay: float,
    plot_axis_order: str,
    plot_mirror: str,
) -> Dict[str, Any]:
    """Render the three-trajectory comparison and return its metrics."""

    try:
        import numpy as np
    except ImportError as exc:
        raise ToolError("plotting requires host package numpy") from exc

    mpl_config = Path("/tmp/srlc-recorded-rc-mpl")
    cache_home = Path("/tmp/srlc-recorded-rc-cache")
    mpl_config.mkdir(parents=True, exist_ok=True)
    cache_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_home))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ToolError("plotting requires host package matplotlib") from exc

    source_payload = _load_json(source_path, "source recording")
    dry_payload = _load_json(dry_path, "dry-run recording")
    historical_times, historical_xy_list, historical_yaws = _historical_path(
        source_payload, timeline
    )
    historical_xy = np.asarray(historical_xy_list, dtype=float)

    motion_indices = (
        int(rc_parameters["forward_index"]),
        int(rc_parameters["lateral_index"]),
        int(rc_parameters["vertical_index"]),
    )
    expected_motion = [
        _motion_tuple(sample.channels, motion_indices)
        for sample in timeline.samples
    ]
    expected_transitions = _compress(expected_motion)
    replay_start_t, replay_end_t, replay_events, transition_match = (
        find_replay_event_window(
            dry_payload,
            expected_transitions,
            motion_indices,
            duration=timeline.duration,
            start_delay=start_delay,
        )
    )
    dry_times, dry_xy = _interpolated_dry_path(
        dry_payload, replay_start_t, replay_end_t, np
    )

    (
        integrated_elapsed,
        integrated_xy,
        body_velocities,
        world_velocities,
        replay_yaws,
    ) = integrate_recorded_rc(
        timeline,
        source_sample_times=historical_times,
        source_yaws=historical_yaws,
        start_xy=historical_xy[0],
        rc_parameters=rc_parameters,
        max_xy_speed=max_xy_speed,
        np=np,
    )
    csv_path = output_dir / "rc_integrated_trajectory.csv"
    _write_integrated_csv(
        csv_path,
        timeline=timeline,
        elapsed=integrated_elapsed,
        positions=integrated_xy,
        body_velocities=body_velocities,
        world_velocities=world_velocities,
        replay_yaws=replay_yaws,
    )

    historical_length = _path_length(historical_xy, np)
    dry_length = _path_length(dry_xy, np)
    integrated_length = _path_length(integrated_xy, np)
    historical_dry_endpoint = float(
        np.linalg.norm(historical_xy[-1] - dry_xy[-1])
    )
    historical_integrated_endpoint = float(
        np.linalg.norm(historical_xy[-1] - integrated_xy[-1])
    )

    progress = np.linspace(0.0, 1.0, 501)
    historical_elapsed = np.asarray(historical_times) - timeline.source_start_time
    dry_elapsed = dry_times - replay_start_t
    historical_resampled = np.column_stack(
        [
            np.interp(
                progress * timeline.duration,
                historical_elapsed,
                historical_xy[:, axis],
            )
            for axis in (0, 1)
        ]
    )
    dry_duration = replay_end_t - replay_start_t
    dry_resampled = np.column_stack(
        [
            np.interp(progress * dry_duration, dry_elapsed, dry_xy[:, axis])
            for axis in (0, 1)
        ]
    )
    integrated_resampled = np.column_stack(
        [
            np.interp(
                progress * timeline.duration,
                integrated_elapsed,
                integrated_xy[:, axis],
            )
            for axis in (0, 1)
        ]
    )
    dry_separation = np.linalg.norm(
        historical_resampled - dry_resampled, axis=1
    )
    integrated_separation = np.linalg.norm(
        historical_resampled - integrated_resampled, axis=1
    )

    dry_window_samples = [
        sample
        for sample in _sample_records(dry_payload, "dry-run recording")
        if replay_start_t <= _finite_time(sample, "dry sample") <= replay_end_t
    ]
    effective_mode_counts: Dict[str, int] = {}
    for sample in dry_window_samples:
        key = str(sample.get("effective_mode", "") or "UNKNOWN")
        effective_mode_counts[key] = effective_mode_counts.get(key, 0) + 1

    metrics = {
        "status": "completed",
        "source_recording": str(source_path),
        "dry_run_recording": str(dry_path),
        "control_mode": mode.upper(),
        "max_xy_speed_mps": max_xy_speed,
        "collision_and_proximity_guards_enabled": False,
        "plot": {
            "axis_order": plot_axis_order,
            "mirror": plot_mirror,
            "horizontal_axis": (
                "map_y_m" if plot_axis_order == "yx" else "map_x_m"
            ),
            "vertical_axis": (
                "map_x_m" if plot_axis_order == "yx" else "map_y_m"
            ),
            "trajectory_metrics_and_csv_remain_xy_ordered": True,
        },
        "start_gate": {
            "reference_xy_m": [float(value) for value in historical_xy[0]],
            "reference_yaw_deg": math.degrees(float(timeline.reference_yaw)),
            "delay_s": start_delay,
        },
        "command_replay": {
            "source_stream": timeline.source_stream,
            "source_start_t_s": timeline.source_start_time,
            "source_end_t_s": timeline.source_end_time,
            "source_duration_s": timeline.duration,
            "source_samples": len(timeline.samples),
            "source_command_transitions": len(expected_transitions),
            "dry_output_events_during_replay": len(replay_events),
            "dry_command_transitions": len(
                _compress([event["_motion"] for event in replay_events])
            ),
            "transition_sequence_exact_match": bool(transition_match),
            "dry_observed_start_t_s": replay_start_t,
            "dry_observed_end_t_s": replay_end_t,
            "dry_observed_duration_s": dry_duration,
        },
        "rc_only_integration": {
            "method": (
                "zero-order hold; calibrated PWM to body XY velocity; "
                "recorded-yaw body-to-world rotation; XY speed clamp; "
                "no policy, LiDAR, safety hold, obstacles, or dynamics"
            ),
            "forward_channel_index_zero_based": rc_parameters["forward_index"],
            "lateral_channel_index_zero_based": rc_parameters["lateral_index"],
            "forward_reverse": rc_parameters["forward_reverse"],
            "lateral_reverse": rc_parameters["lateral_reverse"],
            "deadband": rc_parameters["deadband"],
            "start_xy_m": [float(value) for value in integrated_xy[0]],
            "end_xy_m": [float(value) for value in integrated_xy[-1]],
            "path_length_m": integrated_length,
            "historical_endpoint_separation_m": historical_integrated_endpoint,
            "historical_time_normalized_rmse_m": float(
                math.sqrt(np.mean(integrated_separation ** 2))
            ),
            "historical_time_normalized_max_separation_m": float(
                integrated_separation.max()
            ),
            "csv": str(csv_path),
        },
        "trajectory": {
            "historical_start_xy_m": [
                float(value) for value in historical_xy[0]
            ],
            "historical_end_xy_m": [
                float(value) for value in historical_xy[-1]
            ],
            "historical_path_length_m": historical_length,
            "dry_run_start_xy_m": [float(value) for value in dry_xy[0]],
            "dry_run_end_xy_m": [float(value) for value in dry_xy[-1]],
            "dry_run_path_length_m": dry_length,
            "historical_dry_endpoint_separation_m": historical_dry_endpoint,
            "historical_dry_time_normalized_rmse_m": float(
                math.sqrt(np.mean(dry_separation ** 2))
            ),
            "historical_dry_time_normalized_max_separation_m": float(
                dry_separation.max()
            ),
            "dry_effective_mode_sample_counts": effective_mode_counts,
        },
        "dry_recorder": dry_payload.get("summary", {}),
    }
    metrics_path = output_dir / "replay_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    historical_plot = plot_coordinates(historical_xy, plot_axis_order, np)
    dry_plot = plot_coordinates(dry_xy, plot_axis_order, np)
    integrated_plot = plot_coordinates(integrated_xy, plot_axis_order, np)

    all_xy = np.vstack([historical_xy, dry_xy, integrated_xy])
    padding = 0.75
    xmin, ymin = all_xy.min(axis=0) - padding
    xmax, ymax = all_xy.max(axis=0) + padding
    map_xyz = _load_pcd_ascii_xyz(pcd_file, np)
    map_mask = (
        (map_xyz[:, 0] >= xmin)
        & (map_xyz[:, 0] <= xmax)
        & (map_xyz[:, 1] >= ymin)
        & (map_xyz[:, 1] <= ymax)
        & (map_xyz[:, 2] >= 0.7)
        & (map_xyz[:, 2] <= 1.3)
    )
    map_slice = map_xyz[map_mask]
    map_plot = plot_coordinates(map_slice[:, :2], plot_axis_order, np)
    all_plot = np.vstack([historical_plot, dry_plot, integrated_plot])
    horizontal_min, vertical_min = all_plot.min(axis=0) - padding
    horizontal_max, vertical_max = all_plot.max(axis=0) + padding
    horizontal_label = "Map Y (m)" if plot_axis_order == "yx" else "Map X (m)"
    vertical_label = "Map X (m)" if plot_axis_order == "yx" else "Map Y (m)"
    horizontal_limits = plot_axis_limits(
        horizontal_min,
        horizontal_max,
        plot_axis="horizontal",
        mirror=plot_mirror,
    )
    vertical_limits = plot_axis_limits(
        vertical_min,
        vertical_max,
        plot_axis="vertical",
        mirror=plot_mirror,
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "legend.fontsize": 9.3,
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#ffffff",
        }
    )
    figure_size = (11.2, 8.2) if plot_axis_order == "yx" else (8.8, 10.0)
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    if len(map_slice):
        axis.scatter(
            map_plot[:, 0],
            map_plot[:, 1],
            s=3.2,
            c="#737b86",
            alpha=0.25,
            linewidths=0,
            rasterized=True,
            label="Map slice (0.7–1.3 m; context only)",
        )

    historical_color = "#0072B2"
    dry_color = "#D55E00"
    integrated_color = "#7B2CBF"
    axis.plot(
        historical_plot[:, 0],
        historical_plot[:, 1],
        color=historical_color,
        linewidth=2.3,
        linestyle=(0, (5, 3)),
        label="Historical flight (ACTIVE, %.2f s)" % timeline.duration,
        zorder=4,
    )
    axis.plot(
        dry_plot[:, 0],
        dry_plot[:, 1],
        color=dry_color,
        linewidth=2.8,
        label="Dry-run %s replay (%.2f s)" % (mode.upper(), dry_duration),
        zorder=5,
    )
    axis.plot(
        integrated_plot[:, 0],
        integrated_plot[:, 1],
        color=integrated_color,
        linewidth=2.6,
        linestyle=(0, (2, 1.4)),
        label="RC-only integration (no policy/obstacles)",
        zorder=6,
    )

    def add_direction_arrows(xy: Any, color: str) -> None:
        for fraction in (0.25, 0.5, 0.75):
            index = max(
                1,
                min(len(xy) - 1, int(round(fraction * (len(xy) - 1)))),
            )
            previous = max(0, index - 2)
            if np.linalg.norm(xy[index] - xy[previous]) < 1e-5:
                continue
            axis.annotate(
                "",
                xy=xy[index],
                xytext=xy[previous],
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 1.7,
                    "mutation_scale": 10,
                },
                zorder=7,
            )

    add_direction_arrows(historical_plot, historical_color)
    add_direction_arrows(dry_plot, dry_color)
    add_direction_arrows(integrated_plot, integrated_color)

    start = historical_plot[0]
    axis.scatter(
        [start[0]],
        [start[1]],
        s=105,
        marker="o",
        c="#009E73",
        edgecolors="white",
        linewidths=1.5,
        zorder=9,
        label="Replay start",
    )
    for endpoint, color in (
        (historical_plot[-1], historical_color),
        (dry_plot[-1], dry_color),
        (integrated_plot[-1], integrated_color),
    ):
        axis.scatter(
            [endpoint[0]],
            [endpoint[1]],
            s=105,
            marker="X",
            c=color,
            edgecolors="white",
            linewidths=1.0,
            zorder=9,
        )

    label_box = {
        "boxstyle": "round,pad=0.28",
        "facecolor": "white",
        "alpha": 0.9,
        "edgecolor": "#c9ced6",
    }
    axis.annotate(
        "Start\nXY=(%.2f, %.2f)"
        % (historical_xy[0, 0], historical_xy[0, 1]),
        xy=start,
        xytext=(9, 9),
        textcoords="offset points",
        fontsize=9,
        color="#176b53",
        bbox=label_box,
        zorder=10,
    )
    axis.annotate(
        "Historical end\nXY=(%.2f, %.2f)"
        % (historical_xy[-1, 0], historical_xy[-1, 1]),
        xy=historical_plot[-1],
        xytext=(-118, -8),
        textcoords="offset points",
        fontsize=9,
        color=historical_color,
        bbox=label_box,
        zorder=10,
    )
    axis.annotate(
        "RC-only end\nXY=(%.2f, %.2f)"
        % (integrated_xy[-1, 0], integrated_xy[-1, 1]),
        xy=integrated_plot[-1],
        xytext=(9, 10),
        textcoords="offset points",
        fontsize=9,
        color=integrated_color,
        bbox=label_box,
        zorder=10,
    )
    axis.annotate(
        "Dry-run end\nXY=(%.2f, %.2f)"
        % (dry_xy[-1, 0], dry_xy[-1, 1]),
        xy=dry_plot[-1],
        xytext=(
            (-135, 14)
            if plot_mirror in ("horizontal", "both")
            else (9, 14)
        ),
        textcoords="offset points",
        fontsize=9,
        color=dry_color,
        bbox=label_box,
        zorder=10,
    )

    summary_text = (
        "RC transitions: %d/%d exact match\n"
        "Path length — historical %.2f m | dry %.2f m | RC-only %.2f m\n"
        "Historical endpoint error — dry %.2f m | RC-only %.2f m"
        % (
            len(_compress([event["_motion"] for event in replay_events])),
            len(expected_transitions),
            historical_length,
            dry_length,
            integrated_length,
            historical_dry_endpoint,
            historical_integrated_endpoint,
        )
    )
    axis.text(
        0.025,
        0.025,
        summary_text,
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=9.2,
        color="#29313d",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "alpha": 0.94,
            "edgecolor": "#bfc5ce",
        },
        zorder=12,
    )

    axis.set_title(
        "Recorded RC Replay — 2D Trajectory Comparison",
        pad=14,
        weight="semibold",
    )
    axis.text(
        0.5,
        1.008,
        (
            "%s mode · %.2f m/s XY limit · RC-only line ignores policy, "
            "obstacles, and dynamics · %s"
        )
        % (
            mode.upper(),
            max_xy_speed,
            {
                "none": "not mirrored",
                "horizontal": "mirrored left-right",
                "vertical": "mirrored top-bottom",
                "both": "mirrored both axes",
            }[plot_mirror],
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#586271",
    )
    axis.set_xlabel(horizontal_label)
    axis.set_ylabel(vertical_label)
    axis.set_xlim(*horizontal_limits)
    axis.set_ylim(*vertical_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#d9dde3", linewidth=0.75, alpha=0.75)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#aeb5bf")
    axis.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#c5cad2",
    )

    png_path = output_dir / "trajectory_2d.png"
    svg_path = output_dir / "trajectory_2d.svg"
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    print("[recorded-rc-dryrun] plot=%s" % png_path)
    print("[recorded-rc-dryrun] metrics=%s" % metrics_path)
    print("[recorded-rc-dryrun] rc_integration=%s" % csv_path)
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an isolated recorded-RC ROS dry run and plot historical, "
            "simulated, and obstacle-free RC-integrated XY trajectories."
        )
    )
    parser.add_argument("recording", type=Path, help="source recorder JSON")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="policy checkpoint used by the dry run",
    )
    parser.add_argument(
        "--pcd",
        type=Path,
        default=DEFAULT_PCD,
        help="ASCII PCD used by the dry run and plot background",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "artifact directory; defaults to "
            "artifacts/recorded_rc_dryrun/<recording>_<timestamp>"
        ),
    )
    parser.add_argument(
        "--plot-only",
        type=Path,
        metavar="DRY_RUN_JSON",
        help="skip Docker and plot an existing dry-run recorder JSON",
    )
    parser.add_argument(
        "--image",
        default="srlc_ros1_real:noetic",
        help="Docker image containing ROS Noetic and runtime dependencies",
    )
    parser.add_argument(
        "--post-takeoff-mode",
        choices=("auto", "assist", "direct"),
        default="auto",
        help="downstream mode; auto requires one mode in source metadata",
    )
    parser.add_argument(
        "--max-xy-speed",
        type=_positive_float,
        default=0.5,
        help="dry-run and RC-only horizontal speed clamp in m/s (default: 0.5)",
    )
    parser.add_argument(
        "--plot-axis-order",
        choices=("yx", "xy"),
        default="yx",
        help=(
            "plot horizontal/vertical map-axis order; yx matches the real "
            "layout and is the default"
        ),
    )
    parser.add_argument(
        "--plot-mirror",
        choices=("none", "horizontal", "vertical", "both"),
        default="horizontal",
        help=(
            "mirror the rendered view; horizontal (left-right) matches the "
            "real scene and is the default"
        ),
    )
    parser.add_argument(
        "--forward-reverse",
        choices=("true", "false"),
        default="false",
        help="forward RC calibration override (default: false)",
    )
    parser.add_argument(
        "--lateral-reverse",
        choices=("true", "false"),
        default="true",
        help="lateral RC calibration override (default: true)",
    )
    parser.add_argument("--replay-start-time", type=float)
    parser.add_argument("--replay-end-time", type=float)
    parser.add_argument(
        "--ros-timeout",
        type=_positive_float,
        help="roslaunch runtime before graceful SIGINT; auto-sized by default",
    )
    parser.add_argument("--run-id", help="recorder run_id prefix")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _optional_time_pair(args.replay_start_time, args.replay_end_time)
        recording = _resolve_existing_file(args.recording, "source recording")
        if recording.suffix.lower() != ".json":
            raise ToolError(
                "this combined dry-run/plot tool currently requires source JSON"
            )
        pcd_file = _resolve_existing_file(args.pcd, "PCD map")
        forward_reverse = args.forward_reverse == "true"
        lateral_reverse = args.lateral_reverse == "true"
        rc_parameters = _rc_parameters(
            forward_reverse=forward_reverse,
            lateral_reverse=lateral_reverse,
        )
        replay_config = _simple_yaml_scalars(
            REPO_ROOT
            / "src/srlc_real/cfg/tunnel/recorded_rc_replay.yaml"
        )
        start_delay = float(replay_config.get("start_delay", 1.0))
        if not math.isfinite(start_delay) or start_delay < 0.0:
            raise ToolError("recorded replay start_delay must be finite and non-negative")
        motion_indices = (
            rc_parameters["forward_index"],
            rc_parameters["lateral_index"],
            rc_parameters["vertical_index"],
        )
        timeline = load_recorded_rc_timeline(
            str(recording),
            replay_start_time=args.replay_start_time,
            replay_end_time=args.replay_end_time,
            motion_indices=motion_indices,
        )
        mode = _select_mode(timeline, args.post_takeoff_mode)

        if args.output_dir is not None:
            output_dir = args.output_dir.expanduser().resolve()
        elif args.plot_only is not None:
            output_dir = args.plot_only.expanduser().resolve().parent
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = (
                REPO_ROOT
                / "artifacts/recorded_rc_dryrun"
                / ("%s_%s" % (_safe_name(recording.stem), stamp))
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.plot_only is not None:
            dry_json = _resolve_existing_file(
                args.plot_only, "dry-run recording"
            )
        else:
            checkpoint = _resolve_existing_file(
                args.checkpoint, "policy checkpoint"
            )
            dry_json, _ = run_dry_run(
                recording=recording,
                checkpoint=checkpoint,
                pcd_file=pcd_file,
                output_dir=output_dir,
                image=args.image,
                timeline=timeline,
                mode=mode,
                max_xy_speed=args.max_xy_speed,
                forward_reverse=forward_reverse,
                lateral_reverse=lateral_reverse,
                replay_start_time=args.replay_start_time,
                replay_end_time=args.replay_end_time,
                ros_timeout=args.ros_timeout,
                run_id=args.run_id,
            )

        metrics = render_comparison(
            source_path=recording,
            dry_path=dry_json,
            pcd_file=pcd_file,
            output_dir=output_dir,
            timeline=timeline,
            mode=mode,
            max_xy_speed=args.max_xy_speed,
            rc_parameters=rc_parameters,
            start_delay=start_delay,
            plot_axis_order=args.plot_axis_order,
            plot_mirror=args.plot_mirror,
        )
        command_metrics = metrics["command_replay"]
        integration_metrics = metrics["rc_only_integration"]
        print(
            "[recorded-rc-dryrun] transitions=%d/%d match=%s"
            % (
                command_metrics["dry_command_transitions"],
                command_metrics["source_command_transitions"],
                command_metrics["transition_sequence_exact_match"],
            )
        )
        print(
            "[recorded-rc-dryrun] rc-only end=(%.3f, %.3f) path=%.3fm"
            % (
                integration_metrics["end_xy_m"][0],
                integration_metrics["end_xy_m"][1],
                integration_metrics["path_length_m"],
            )
        )
        return 0
    except (ToolError, RecordingFormatError, OSError, ValueError) as exc:
        parser.exit(2, "error: %s\n" % exc)


if __name__ == "__main__":
    sys.exit(main())
