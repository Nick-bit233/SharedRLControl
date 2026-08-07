"""Load and sample recorded MAVROS RC input timelines.

The recorder has two supported schemas:

* current recordings contain a top-level ``rc_events`` stream captured in the
  RC subscriber callback;
* legacy recordings contain only the 20 Hz ``samples[].rc`` snapshots.

Both JSON and the recorder's object-array NPZ files are supported.  NPZ input
must only be loaded from a trusted local recording because NumPy object arrays
require pickle support.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class RecordingFormatError(ValueError):
    """Raised when a recording cannot be replayed safely."""


@dataclass(frozen=True)
class RcReplaySample:
    """One zero-order-held RC sample on a normalized replay timeline."""

    elapsed: float
    source_time: float
    channels: Tuple[int, ...]


@dataclass(frozen=True)
class RecordedRcTimeline:
    """Validated one-shot RC replay data and source-flight metadata."""

    samples: Tuple[RcReplaySample, ...]
    duration: float
    source_path: str
    source_stream: str
    source_run_id: str
    source_start_time: float
    source_end_time: float
    reference_position: Optional[Tuple[float, float, float]]
    reference_yaw: Optional[float]
    control_modes: Tuple[str, ...] = ()
    effective_modes: Tuple[str, ...] = ()
    _elapsed: Tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.samples) < 2:
            raise RecordingFormatError("replay timeline must contain at least two samples")
        if not math.isfinite(self.duration) or self.duration <= 0.0:
            raise RecordingFormatError("replay duration must be finite and positive")
        elapsed = tuple(float(sample.elapsed) for sample in self.samples)
        if elapsed[0] != 0.0:
            raise RecordingFormatError("replay timeline must start at elapsed=0")
        if any(right <= left for left, right in zip(elapsed, elapsed[1:])):
            raise RecordingFormatError("replay timestamps must be strictly increasing")
        if elapsed[-1] >= self.duration:
            raise RecordingFormatError(
                "replay duration must extend beyond the last recorded sample"
            )
        object.__setattr__(self, "_elapsed", elapsed)

    def sample_at(self, elapsed: float) -> Tuple[int, RcReplaySample]:
        """Return the sample held at ``elapsed`` and its source index."""

        query = max(0.0, float(elapsed))
        index = bisect.bisect_right(self._elapsed, query) - 1
        index = max(0, min(index, len(self.samples) - 1))
        return index, self.samples[index]


@dataclass(frozen=True)
class RcChannelOverlay:
    """Overlay recorded motion axes on a fresh physical RC stream."""

    channel_base: int = 1
    forward_channel: int = 2
    lateral_channel: int = 1
    vertical_channel: int = 3
    pwm_mid: int = 1500

    def __post_init__(self) -> None:
        indices = self.motion_indices
        if any(index < 0 for index in indices):
            raise ValueError("motion channels must not precede channel_base")
        if len(set(indices)) != len(indices):
            raise ValueError("forward, lateral, and vertical channels must be distinct")
        if not (0 <= int(self.pwm_mid) <= 65535):
            raise ValueError("pwm_mid must fit mavros_msgs/RCIn")

    @property
    def motion_indices(self) -> Tuple[int, int, int]:
        return (
            int(self.forward_channel) - int(self.channel_base),
            int(self.lateral_channel) - int(self.channel_base),
            int(self.vertical_channel) - int(self.channel_base),
        )

    @property
    def required_channel_count(self) -> int:
        return max(self.motion_indices) + 1

    def has_motion_channels(self, channels: Sequence[int]) -> bool:
        return len(channels) >= self.required_channel_count

    def overlay(
        self,
        live_channels: Iterable[int],
        recorded_channels: Sequence[int],
    ) -> List[int]:
        """Copy live auxiliary channels and recorded motion-stick channels."""

        live = [int(value) for value in live_channels]
        if not self.has_motion_channels(live):
            raise ValueError("live RC input does not contain all motion channels")
        if not self.has_motion_channels(recorded_channels):
            raise ValueError("recorded RC input does not contain all motion channels")
        for index in self.motion_indices:
            live[index] = int(recorded_channels[index])
        return live

    def neutral(self, live_channels: Iterable[int]) -> List[int]:
        """Copy live auxiliary channels and center every replayed motion axis."""

        live = [int(value) for value in live_channels]
        if not self.has_motion_channels(live):
            raise ValueError("live RC input does not contain all motion channels")
        for index in self.motion_indices:
            live[index] = int(self.pwm_mid)
        return live


def _as_builtin(value: Any) -> Any:
    """Convert NumPy object containers to ordinary Python containers."""

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
            raise RecordingFormatError("failed to read JSON recording: %s" % exc) from exc
    elif suffix == ".npz":
        try:
            import numpy as np

            with np.load(path, allow_pickle=True) as archive:
                payload = {
                    str(name): _as_builtin(archive[name]) for name in archive.files
                }
        except (OSError, ValueError, ImportError) as exc:
            raise RecordingFormatError("failed to read NPZ recording: %s" % exc) from exc
    else:
        raise RecordingFormatError("recording_file must end in .json or .npz")

    payload = _as_builtin(payload)
    if not isinstance(payload, dict):
        raise RecordingFormatError("recording root must be an object")
    return payload


def _records(payload: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    value = payload.get(name, [])
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise RecordingFormatError("%s must be an array" % name)
    output = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise RecordingFormatError("%s[%d] must be an object" % (name, index))
        output.append(item)
    return output


def _finite_time(record: Dict[str, Any], label: str) -> float:
    try:
        value = float(record["t"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordingFormatError("%s has no valid t field" % label) from exc
    if not math.isfinite(value):
        raise RecordingFormatError("%s.t must be finite" % label)
    return value


def _ordered_times(records: Sequence[Dict[str, Any]], name: str) -> List[float]:
    times = [
        _finite_time(record, "%s[%d]" % (name, index))
        for index, record in enumerate(records)
    ]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise RecordingFormatError("%s timestamps must be strictly increasing" % name)
    return times


def _active_runs(
    records: Sequence[Dict[str, Any]], activation_value: str
) -> List[Tuple[int, int]]:
    runs = []
    start = None
    for index, record in enumerate(records):
        active = str(record.get("lifecycle_state", "")) == activation_value
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(records)))
    return runs


def _median_interval(times: Sequence[float], fallback: float = 0.05) -> float:
    intervals = [
        right - left
        for left, right in zip(times, times[1:])
        if math.isfinite(right - left) and right > left
    ]
    return float(statistics.median(intervals)) if intervals else float(fallback)


def _select_window(
    records: Sequence[Dict[str, Any]],
    times: Sequence[float],
    activation_value: str,
    start_time: Optional[float],
    end_time: Optional[float],
) -> Tuple[int, int, float]:
    explicit_start = start_time is not None
    explicit_end = end_time is not None
    if explicit_start != explicit_end:
        raise RecordingFormatError(
            "replay_start_time and replay_end_time must be set together"
        )

    if explicit_start:
        start_value = float(start_time)
        end_value = float(end_time)
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or start_value < 0.0
            or end_value <= start_value
        ):
            raise RecordingFormatError(
                "explicit replay window must satisfy 0 <= start < end"
            )
        first = bisect.bisect_left(times, start_value)
        last = bisect.bisect_left(times, end_value)
        if first >= len(records) or last - first < 2:
            raise RecordingFormatError(
                "explicit replay window contains fewer than two RC samples"
            )
        return first, last, end_value

    runs = _active_runs(records, activation_value)
    if not runs:
        raise RecordingFormatError(
            "recording has no lifecycle_state=%s interval; set both "
            "replay_start_time and replay_end_time for a legacy recording"
            % activation_value
        )
    if len(runs) != 1:
        raise RecordingFormatError(
            "recording has %d disjoint lifecycle_state=%s intervals; select one "
            "with replay_start_time and replay_end_time" % (len(runs), activation_value)
        )

    first, last = runs[0]
    if last - first < 2:
        raise RecordingFormatError(
            "lifecycle_state=%s interval contains fewer than two RC samples"
            % activation_value
        )
    if last < len(records):
        window_end = times[last]
    else:
        window_end = times[last - 1] + _median_interval(times[first:last])
    return first, last, window_end


def _channels(
    record: Dict[str, Any],
    field: str,
    label: str,
    motion_indices: Sequence[int],
    pwm_lower_bound: int,
    pwm_upper_bound: int,
) -> Tuple[int, ...]:
    raw = record.get(field)
    if not isinstance(raw, (list, tuple)):
        raise RecordingFormatError("%s.%s must be an array" % (label, field))
    try:
        values = tuple(int(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecordingFormatError(
            "%s.%s contains a non-integer channel" % (label, field)
        ) from exc
    required = max(motion_indices) + 1
    if len(values) < required:
        raise RecordingFormatError(
            "%s.%s has %d channels; at least %d are required"
            % (label, field, len(values), required)
        )
    for index in motion_indices:
        value = values[index]
        if value < pwm_lower_bound or value > pwm_upper_bound:
            raise RecordingFormatError(
                "%s motion channel index %d is outside [%d, %d]: %d"
                % (label, index, pwm_lower_bound, pwm_upper_bound, value)
            )
    return values


def _finite_vector(record: Dict[str, Any], field: str) -> Optional[Tuple[float, float, float]]:
    raw = record.get(field)
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    try:
        values = (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _finite_scalar(record: Dict[str, Any], field: str) -> Optional[float]:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _unique_strings(records: Sequence[Dict[str, Any]], field: str) -> Tuple[str, ...]:
    values = []
    for record in records:
        value = str(record.get(field, "")).strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _reference_records(
    samples: Sequence[Dict[str, Any]],
    sample_times: Sequence[float],
    start_time: float,
    end_time: float,
) -> List[Dict[str, Any]]:
    first = bisect.bisect_left(sample_times, start_time)
    last = bisect.bisect_left(sample_times, end_time)
    return list(samples[first:last])


def _nearby_reference_records(
    samples: Sequence[Dict[str, Any]],
    sample_times: Sequence[float],
    start_time: float,
    end_time: float,
    max_gap: float,
) -> List[Dict[str, Any]]:
    """Return in-window samples ordered by proximity to the replay start."""

    first = bisect.bisect_left(sample_times, start_time)
    last = bisect.bisect_left(sample_times, end_time)
    candidate_indices = list(range(first, last))
    if first > 0 and start_time - sample_times[first - 1] <= max_gap:
        candidate_indices.append(first - 1)
    candidate_indices.sort(
        key=lambda index: (abs(sample_times[index] - start_time), index)
    )
    return [samples[index] for index in candidate_indices]


def load_recorded_rc_timeline(
    recording_file: str,
    *,
    activation_value: str = "ACTIVE",
    replay_start_time: Optional[float] = None,
    replay_end_time: Optional[float] = None,
    motion_indices: Sequence[int] = (1, 0, 2),
    pwm_lower_bound: int = 800,
    pwm_upper_bound: int = 2200,
    max_sample_gap: float = 0.25,
    max_replay_duration: float = 300.0,
) -> RecordedRcTimeline:
    """Load one safe, deterministic replay interval from a recorder output."""

    path_text = str(recording_file).strip()
    if not path_text:
        raise RecordingFormatError("recording_file is required")
    path = os.path.abspath(os.path.expanduser(path_text))
    if not os.path.isfile(path):
        raise RecordingFormatError("recording file does not exist: %s" % path)
    if not activation_value:
        raise RecordingFormatError("activation_value must not be empty")
    indices = tuple(int(index) for index in motion_indices)
    if not indices or any(index < 0 for index in indices):
        raise RecordingFormatError("motion channel indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise RecordingFormatError("motion channel indices must be distinct")
    if not (0 <= int(pwm_lower_bound) < int(pwm_upper_bound) <= 65535):
        raise RecordingFormatError("invalid PWM validation bounds")
    if not math.isfinite(max_sample_gap) or max_sample_gap <= 0.0:
        raise RecordingFormatError("max_sample_gap must be finite and positive")
    if not math.isfinite(max_replay_duration) or max_replay_duration <= 0.0:
        raise RecordingFormatError("max_replay_duration must be finite and positive")

    payload = _load_payload(path)
    samples = _records(payload, "samples")
    events = _records(payload, "rc_events")
    if events:
        stream = events
        stream_name = "rc_events"
        channel_field = "channels"
    elif samples:
        stream = samples
        stream_name = "sample_snapshots"
        channel_field = "rc"
    else:
        raise RecordingFormatError("recording contains neither rc_events nor samples")

    times = _ordered_times(stream, stream_name)
    first, last, source_end_time = _select_window(
        stream,
        times,
        str(activation_value),
        replay_start_time,
        replay_end_time,
    )
    selected_records = stream[first:last]
    selected_times = times[first:last]
    source_start_time = selected_times[0]
    duration = source_end_time - source_start_time
    if duration > max_replay_duration:
        raise RecordingFormatError(
            "replay duration %.3fs exceeds max_replay_duration %.3fs"
            % (duration, max_replay_duration)
        )

    intervals = [
        right - left for left, right in zip(selected_times, selected_times[1:])
    ]
    if replay_start_time is not None:
        intervals.append(source_start_time - float(replay_start_time))
    intervals.append(source_end_time - selected_times[-1])
    largest_gap = max(intervals)
    if largest_gap > max_sample_gap:
        raise RecordingFormatError(
            "recorded RC gap %.3fs exceeds max_sample_gap %.3fs"
            % (largest_gap, max_sample_gap)
        )

    replay_samples = []
    for offset, (record, source_time) in enumerate(
        zip(selected_records, selected_times)
    ):
        channel_values = _channels(
            record,
            channel_field,
            "%s[%d]" % (stream_name, first + offset),
            indices,
            int(pwm_lower_bound),
            int(pwm_upper_bound),
        )
        replay_samples.append(
            RcReplaySample(
                elapsed=float(source_time - source_start_time),
                source_time=float(source_time),
                channels=channel_values,
            )
        )

    if stream_name == "rc_events":
        reference_candidates = list(selected_records)
        metadata_candidates = list(selected_records)
        event_has_position = any(
            _finite_vector(record, "position") is not None
            for record in selected_records
        )
        event_has_yaw = any(
            _finite_scalar(record, "yaw") is not None
            for record in selected_records
        )
        if samples and (not event_has_position or not event_has_yaw):
            sample_times = _ordered_times(samples, "samples")
            sample_window = _reference_records(
                samples, sample_times, source_start_time, source_end_time
            )
            reference_candidates.extend(
                _nearby_reference_records(
                    samples,
                    sample_times,
                    source_start_time,
                    source_end_time,
                    max_sample_gap,
                )
            )
            if not _unique_strings(metadata_candidates, "control_mode"):
                metadata_candidates = sample_window
    else:
        reference_candidates = list(selected_records)
        metadata_candidates = list(selected_records)

    reference_position = None
    reference_yaw = None
    for record in reference_candidates:
        if reference_position is None:
            reference_position = _finite_vector(record, "position")
        if reference_yaw is None:
            reference_yaw = _finite_scalar(record, "yaw")
        if reference_position is not None and reference_yaw is not None:
            break

    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    source_run_id = str(summary.get("run_id", "")).strip()
    return RecordedRcTimeline(
        samples=tuple(replay_samples),
        duration=float(duration),
        source_path=path,
        source_stream=stream_name,
        source_run_id=source_run_id,
        source_start_time=float(source_start_time),
        source_end_time=float(source_end_time),
        reference_position=reference_position,
        reference_yaw=reference_yaw,
        control_modes=_unique_strings(metadata_candidates, "control_mode"),
        effective_modes=_unique_strings(metadata_candidates, "effective_mode"),
    )


def wrapped_angle_error(angle: float, target: float) -> float:
    """Return the signed shortest ``angle - target`` error in radians."""

    difference = float(angle) - float(target)
    return math.atan2(math.sin(difference), math.cos(difference))
