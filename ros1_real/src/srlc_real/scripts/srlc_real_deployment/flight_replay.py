"""Load and interpolate recorded real-flight trajectories for visualization.

The real-flight recorder writes JSON files and NumPy object-array NPZ files.
JSON is the preferred interchange format.  NPZ loading intentionally enables
pickle and must therefore only be used with trusted local recordings.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


class FlightRecordingError(ValueError):
    """Raised when a recording cannot be replayed without inventing data."""


@dataclass(frozen=True)
class FlightSample:
    """One validated, map-aligned source sample."""

    elapsed: float
    source_time: float
    position: Tuple[float, float, float]
    yaw: float
    human_input: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class InterpolatedFlightPose:
    """A pose sampled on the continuous replay timeline."""

    elapsed: float
    source_time: float
    position: Tuple[float, float, float]
    yaw: float
    linear_velocity: Tuple[float, float, float]
    yaw_rate: float
    left_index: int
    right_index: int


@dataclass(frozen=True)
class FlightTimeline:
    """Validated replay window and its map-aligned trajectory."""

    samples: Tuple[FlightSample, ...]
    duration: float
    source_path: str
    source_run_id: str
    source_start_time: float
    source_end_time: float
    window_reason: str
    map_yaw_deg: float
    map_origin_xyz: Tuple[float, float, float]
    _elapsed: Tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.samples) < 2:
            raise FlightRecordingError("replay timeline must contain at least two samples")
        if not math.isfinite(self.duration) or self.duration <= 0.0:
            raise FlightRecordingError("replay duration must be finite and positive")
        elapsed = tuple(float(sample.elapsed) for sample in self.samples)
        if abs(elapsed[0]) > 1.0e-9:
            raise FlightRecordingError("replay timeline must start at elapsed=0")
        if any(right <= left for left, right in zip(elapsed, elapsed[1:])):
            raise FlightRecordingError("replay timestamps must be strictly increasing")
        if abs(elapsed[-1] - self.duration) > 1.0e-7:
            raise FlightRecordingError("last replay sample must match replay duration")
        object.__setattr__(self, "_elapsed", elapsed)

    def sample_at(self, elapsed: float) -> InterpolatedFlightPose:
        """Interpolate XYZ and yaw at a replay-relative time."""

        query = min(max(float(elapsed), 0.0), self.duration)
        if query <= 0.0:
            left_index, right_index = 0, 1
            alpha = 0.0
        elif query >= self.duration:
            left_index, right_index = len(self.samples) - 2, len(self.samples) - 1
            alpha = 1.0
        else:
            right_index = bisect.bisect_right(self._elapsed, query)
            right_index = min(max(right_index, 1), len(self.samples) - 1)
            left_index = right_index - 1
            left_elapsed = self.samples[left_index].elapsed
            right_elapsed = self.samples[right_index].elapsed
            alpha = (query - left_elapsed) / (right_elapsed - left_elapsed)

        left = self.samples[left_index]
        right = self.samples[right_index]
        segment_duration = right.elapsed - left.elapsed
        position = tuple(
            left.position[axis] + alpha * (right.position[axis] - left.position[axis])
            for axis in range(3)
        )
        yaw_delta = _wrapped_delta(right.yaw - left.yaw)
        yaw = _wrap_angle(left.yaw + alpha * yaw_delta)
        velocity = tuple(
            (right.position[axis] - left.position[axis]) / segment_duration
            for axis in range(3)
        )
        return InterpolatedFlightPose(
            elapsed=query,
            source_time=self.source_start_time + query,
            position=position,
            yaw=yaw,
            linear_velocity=velocity,
            yaw_rate=yaw_delta / segment_duration,
            left_index=left_index,
            right_index=right_index,
        )

    @property
    def has_human_input(self) -> bool:
        """Whether at least one selected sample contains recorded stick input."""

        return any(sample.human_input is not None for sample in self.samples)

    def human_input_at(self, elapsed: float) -> Tuple[float, float, float]:
        """Return zero-order-held body-frame human input at replay time."""

        query = min(max(float(elapsed), 0.0), self.duration)
        index = bisect.bisect_right(self._elapsed, query) - 1
        index = min(max(index, 0), len(self.samples) - 1)
        for candidate in range(index, -1, -1):
            value = self.samples[candidate].human_input
            if value is not None:
                return value
        return (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class _RawSample:
    source_time: float
    position: Tuple[float, float, float]
    yaw: float
    lifecycle_state: str
    armed: Optional[bool]
    landed_state: Optional[int]
    human_input: Optional[Tuple[float, float, float]]


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _wrapped_delta(angle: float) -> float:
    return _wrap_angle(angle)


def _as_builtin(value: Any) -> Any:
    if hasattr(value, "shape") and getattr(value, "shape", None) == ():
        return _as_builtin(value.item())
    if hasattr(value, "tolist"):
        return _as_builtin(value.tolist())
    if isinstance(value, list):
        return [_as_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_as_builtin(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _as_builtin(item) for key, item in value.items()}
    return value


def _load_payload(path: str) -> Dict[str, Any]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise FlightRecordingError("failed to read JSON recording: %s" % exc) from exc
    elif suffix == ".npz":
        try:
            import numpy as np

            with np.load(path, allow_pickle=True) as archive:
                payload = {
                    str(name): _as_builtin(archive[name]) for name in archive.files
                }
        except (OSError, ValueError, ImportError) as exc:
            raise FlightRecordingError("failed to read NPZ recording: %s" % exc) from exc
    else:
        raise FlightRecordingError("recording_file must end in .json or .npz")

    payload = _as_builtin(payload)
    if not isinstance(payload, dict):
        raise FlightRecordingError("recording root must be an object")
    return payload


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FlightRecordingError("%s must be numeric" % label) from exc
    if not math.isfinite(result):
        raise FlightRecordingError("%s must be finite" % label)
    return result


def _position(record: Dict[str, Any], index: int) -> Tuple[float, float, float]:
    raw = record.get("position")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise FlightRecordingError("samples[%d].position must contain XYZ" % index)
    return tuple(
        _finite_float(raw[axis], "samples[%d].position[%d]" % (index, axis))
        for axis in range(3)
    )


def _human_input(
    record: Dict[str, Any], index: int
) -> Optional[Tuple[float, float, float]]:
    raw = record.get("human_action")
    if raw is None or raw == []:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise FlightRecordingError(
            "samples[%d].human_action must contain forward and lateral values"
            % index
        )
    values = [
        _finite_float(raw[axis], "samples[%d].human_action[%d]" % (index, axis))
        for axis in range(min(3, len(raw)))
    ]
    while len(values) < 3:
        values.append(0.0)
    return (values[0], values[1], values[2])


def _optional_bool(value: Any, label: str) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise FlightRecordingError("%s must be boolean when present" % label)


def _optional_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FlightRecordingError("%s must be an integer when present" % label) from exc
    return result


def _raw_samples(payload: Dict[str, Any]) -> List[_RawSample]:
    records = payload.get("samples")
    if not isinstance(records, (list, tuple)):
        raise FlightRecordingError("recording samples must be an array")
    if len(records) < 2:
        raise FlightRecordingError("recording must contain at least two samples")

    samples: List[_RawSample] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise FlightRecordingError("samples[%d] must be an object" % index)
        if "t" not in record:
            raise FlightRecordingError("samples[%d].t is required" % index)
        if "yaw" not in record:
            raise FlightRecordingError("samples[%d].yaw is required" % index)
        samples.append(
            _RawSample(
                source_time=_finite_float(record["t"], "samples[%d].t" % index),
                position=_position(record, index),
                yaw=_finite_float(record["yaw"], "samples[%d].yaw" % index),
                lifecycle_state=str(record.get("lifecycle_state", "")).strip().upper(),
                armed=_optional_bool(record.get("armed"), "samples[%d].armed" % index),
                landed_state=_optional_int(
                    record.get("landed_state"), "samples[%d].landed_state" % index
                ),
                human_input=_human_input(record, index),
            )
        )

    times = [sample.source_time for sample in samples]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise FlightRecordingError("sample timestamps must be strictly increasing")
    return samples


def _auto_window(samples: Sequence[_RawSample]) -> Tuple[float, float, str]:
    start_index = next(
        (
            index
            for index, sample in enumerate(samples)
            if sample.lifecycle_state == "TAKEOFF"
        ),
        None,
    )
    start_reason = "takeoff"
    if start_index is None:
        start_index = next(
            (
                index
                for index, sample in enumerate(samples)
                if sample.landed_state == 2
                and (index == 0 or samples[index - 1].landed_state != 2)
            ),
            None,
        )
        start_reason = "airborne"
    if start_index is None:
        raise FlightRecordingError(
            "cannot infer a flight window; set both start_time and end_time"
        )

    seen_airborne = samples[start_index].landed_state == 2
    seen_armed = samples[start_index].armed is True
    for sample in samples[start_index + 1 :]:
        if sample.landed_state == 2:
            seen_airborne = True
        if sample.armed is True:
            seen_armed = True
        landed = seen_airborne and sample.landed_state == 1
        disarmed = seen_armed and sample.armed is False
        if landed or disarmed:
            ending = "landed" if landed else "disarmed"
            return (
                samples[start_index].source_time,
                sample.source_time,
                "auto_%s_to_%s" % (start_reason, ending),
            )

    return (
        samples[start_index].source_time,
        samples[-1].source_time,
        "auto_%s_to_recording_end" % start_reason,
    )


def _interpolate_raw(
    samples: Sequence[_RawSample], times: Sequence[float], source_time: float
) -> _RawSample:
    right_index = bisect.bisect_left(times, source_time)
    if right_index < len(samples) and abs(times[right_index] - source_time) < 1.0e-9:
        return samples[right_index]
    if right_index <= 0 or right_index >= len(samples):
        raise FlightRecordingError("requested replay boundary lies outside the recording")

    left = samples[right_index - 1]
    right = samples[right_index]
    alpha = (source_time - left.source_time) / (right.source_time - left.source_time)
    position = tuple(
        left.position[axis] + alpha * (right.position[axis] - left.position[axis])
        for axis in range(3)
    )
    yaw = _wrap_angle(left.yaw + alpha * _wrapped_delta(right.yaw - left.yaw))
    return _RawSample(
        source_time=source_time,
        position=position,
        yaw=yaw,
        lifecycle_state=left.lifecycle_state,
        armed=left.armed,
        landed_state=left.landed_state,
        human_input=left.human_input,
    )


def _select_samples(
    samples: Sequence[_RawSample], start_time: float, end_time: float
) -> List[_RawSample]:
    times = [sample.source_time for sample in samples]
    selected = [_interpolate_raw(samples, times, start_time)]
    selected.extend(
        sample for sample in samples if start_time < sample.source_time < end_time
    )
    selected.append(_interpolate_raw(samples, times, end_time))
    return selected


def load_flight_timeline(
    path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    map_yaw_deg: float = 0.0,
    map_origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    max_sample_gap: float = 0.2,
) -> FlightTimeline:
    """Load, crop, validate, transform, and normalize a real-flight recording."""

    if (start_time is None) != (end_time is None):
        raise FlightRecordingError("start_time and end_time must be set together")
    if len(map_origin_xyz) != 3:
        raise FlightRecordingError("map_origin_xyz must contain exactly three values")

    origin = tuple(
        _finite_float(value, "map_origin_xyz[%d]" % index)
        for index, value in enumerate(map_origin_xyz)
    )
    yaw_deg = _finite_float(map_yaw_deg, "map_yaw_deg")
    gap_limit = _finite_float(max_sample_gap, "max_sample_gap")
    if gap_limit <= 0.0:
        raise FlightRecordingError("max_sample_gap must be positive")

    payload = _load_payload(path)
    raw = _raw_samples(payload)
    first_time = raw[0].source_time
    last_time = raw[-1].source_time
    if start_time is None:
        selected_start, selected_end, window_reason = _auto_window(raw)
    else:
        selected_start = _finite_float(start_time, "start_time")
        selected_end = _finite_float(end_time, "end_time")
        window_reason = "explicit"
        if selected_start < first_time or selected_end > last_time:
            raise FlightRecordingError(
                "explicit replay window must lie within [%.6f, %.6f]"
                % (first_time, last_time)
            )
    if selected_end <= selected_start:
        raise FlightRecordingError("replay window must satisfy start_time < end_time")

    selected = _select_samples(raw, selected_start, selected_end)
    for left, right in zip(selected, selected[1:]):
        gap = right.source_time - left.source_time
        if gap > gap_limit + 1.0e-9:
            raise FlightRecordingError(
                "sample gap %.6fs exceeds max_sample_gap %.6fs near t=%.6f"
                % (gap, gap_limit, left.source_time)
            )

    yaw_radians = math.radians(yaw_deg)
    cos_yaw = math.cos(yaw_radians)
    sin_yaw = math.sin(yaw_radians)
    mapped: List[FlightSample] = []
    for sample in selected:
        x, y, z = sample.position
        mapped.append(
            FlightSample(
                elapsed=sample.source_time - selected_start,
                source_time=sample.source_time,
                position=(
                    cos_yaw * x - sin_yaw * y + origin[0],
                    sin_yaw * x + cos_yaw * y + origin[1],
                    z + origin[2],
                ),
                yaw=_wrap_angle(sample.yaw + yaw_radians),
                human_input=sample.human_input,
            )
        )

    summary = payload.get("summary", {})
    run_id = str(summary.get("run_id", "")) if isinstance(summary, dict) else ""
    return FlightTimeline(
        samples=tuple(mapped),
        duration=selected_end - selected_start,
        source_path=os.path.abspath(path),
        source_run_id=run_id,
        source_start_time=selected_start,
        source_end_time=selected_end,
        window_reason=window_reason,
        map_yaw_deg=yaw_deg,
        map_origin_xyz=origin,
    )


def integrate_human_input_prediction(
    timeline: FlightTimeline,
    max_xy_speed: float = 0.5,
) -> Tuple[Tuple[float, float, float], ...]:
    """Integrate recorded right-stick input without policy or obstacle response.

    Forward/lateral values are interpreted as body-frame velocity commands,
    limited to ``max_xy_speed``, rotated using the recorded yaw, and held until
    the next source sample.  Recorded altitude is retained because the right
    stick only predicts horizontal motion in this flight configuration.
    """

    speed_limit = _finite_float(max_xy_speed, "max_xy_speed")
    if speed_limit <= 0.0:
        raise FlightRecordingError("max_xy_speed must be positive")
    if not timeline.has_human_input:
        raise FlightRecordingError(
            "recording has no human_action values for input prediction"
        )

    positions: List[Tuple[float, float, float]] = [
        tuple(timeline.samples[0].position)
    ]
    last_input = (0.0, 0.0, 0.0)
    for left, right in zip(timeline.samples, timeline.samples[1:]):
        if left.human_input is not None:
            last_input = left.human_input
        forward = float(last_input[0])
        lateral = float(last_input[1])
        norm = math.hypot(forward, lateral)
        if norm > speed_limit:
            scale = speed_limit / norm
            forward *= scale
            lateral *= scale

        cosine = math.cos(left.yaw)
        sine = math.sin(left.yaw)
        world_x = cosine * forward - sine * lateral
        world_y = sine * forward + cosine * lateral
        dt = right.elapsed - left.elapsed
        previous = positions[-1]
        positions.append(
            (
                previous[0] + world_x * dt,
                previous[1] + world_y * dt,
                right.position[2],
            )
        )
    return tuple(positions)
