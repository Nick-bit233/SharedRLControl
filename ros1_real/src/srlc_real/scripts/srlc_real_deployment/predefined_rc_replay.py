"""Pure helpers for deterministic, map-frame RC intent replay."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import math
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class IntentSample:
    """One analytical sample of the predefined map-frame intent."""

    progress: float
    position_xy: Tuple[float, float]
    velocity_xy: Tuple[float, float]
    complete: bool


@dataclass(frozen=True)
class SCurveIntent:
    """Smooth map-frame S curve with optional lateral control points.

    Without ``lateral_profile`` the geometric curve remains

        p(u) = start + u * (goal - start) + A * sin(2*pi*u)^3 * normal

    A profile replaces the sinusoid with shape-preserving cubic Hermite
    interpolation through ``(u, lateral_offset_m)`` knots.  Endpoint slopes
    are forced to zero, retaining the start-to-goal tangent.  Cubic smoothstep
    timing makes both endpoint velocities zero.  With ``arc_length_timing``,
    smoothstep advances normalized arc length instead of raw ``u`` so strong,
    asymmetric turns do not create speed spikes.
    """

    start_xy: Tuple[float, float]
    goal_xy: Tuple[float, float]
    lateral_amplitude: float
    duration: float
    lateral_profile: Tuple[Tuple[float, float], ...] = ()
    arc_length_timing: bool = False
    arc_length_samples: int = 2001
    _profile_progress: Tuple[float, ...] = field(init=False, repr=False)
    _profile_offset: Tuple[float, ...] = field(init=False, repr=False)
    _profile_slope: Tuple[float, ...] = field(init=False, repr=False)
    _arc_progress: Tuple[float, ...] = field(init=False, repr=False)
    _arc_distance: Tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dx = float(self.goal_xy[0]) - float(self.start_xy[0])
        dy = float(self.goal_xy[1]) - float(self.start_xy[1])
        if math.hypot(dx, dy) <= 1e-9:
            raise ValueError("S-curve start and goal must differ")
        if not math.isfinite(float(self.lateral_amplitude)):
            raise ValueError("lateral_amplitude must be finite")
        if not math.isfinite(float(self.duration)) or float(self.duration) <= 0.0:
            raise ValueError("duration must be finite and positive")
        if int(self.arc_length_samples) < 101:
            raise ValueError("arc_length_samples must be at least 101")

        profile = tuple(
            (float(point[0]), float(point[1])) for point in self.lateral_profile
        )
        object.__setattr__(self, "lateral_profile", profile)
        object.__setattr__(self, "arc_length_timing", bool(self.arc_length_timing))
        object.__setattr__(self, "arc_length_samples", int(self.arc_length_samples))
        if profile:
            if len(profile) < 2:
                raise ValueError("lateral_profile must contain at least two knots")
            if any(
                not (math.isfinite(u) and math.isfinite(offset))
                for u, offset in profile
            ):
                raise ValueError("lateral_profile values must be finite")
            progress = tuple(point[0] for point in profile)
            offset = tuple(point[1] for point in profile)
            if abs(progress[0]) > 1e-9 or abs(progress[-1] - 1.0) > 1e-9:
                raise ValueError("lateral_profile must start at u=0 and end at u=1")
            if abs(offset[0]) > 1e-9 or abs(offset[-1]) > 1e-9:
                raise ValueError("lateral_profile endpoint offsets must be zero")
            if any(right <= left for left, right in zip(progress, progress[1:])):
                raise ValueError("lateral_profile progress must be strictly increasing")
            slope = self._shape_preserving_slopes(progress, offset)
        else:
            progress = ()
            offset = ()
            slope = ()
        object.__setattr__(self, "_profile_progress", progress)
        object.__setattr__(self, "_profile_offset", offset)
        object.__setattr__(self, "_profile_slope", slope)

        count = int(self.arc_length_samples)
        arc_progress = tuple(index / float(count - 1) for index in range(count))
        arc_distance = [0.0]
        previous_speed = math.hypot(*self.tangent_at_progress(arc_progress[0]))
        step = 1.0 / float(count - 1)
        for u_value in arc_progress[1:]:
            current_speed = math.hypot(*self.tangent_at_progress(u_value))
            arc_distance.append(
                arc_distance[-1] + 0.5 * (previous_speed + current_speed) * step
            )
            previous_speed = current_speed
        object.__setattr__(self, "_arc_progress", arc_progress)
        object.__setattr__(self, "_arc_distance", tuple(arc_distance))

    @staticmethod
    def _shape_preserving_slopes(
        progress: Sequence[float], offset: Sequence[float]
    ) -> Tuple[float, ...]:
        """Return PCHIP-style slopes with diagonal endpoint tangents."""

        secants = [
            (offset[index + 1] - offset[index])
            / (progress[index + 1] - progress[index])
            for index in range(len(progress) - 1)
        ]
        slopes = [0.0] * len(progress)
        for index in range(1, len(progress) - 1):
            left = secants[index - 1]
            right = secants[index]
            if left * right <= 0.0:
                slopes[index] = 0.0
                continue
            left_width = progress[index] - progress[index - 1]
            right_width = progress[index + 1] - progress[index]
            weight_left = 2.0 * right_width + left_width
            weight_right = right_width + 2.0 * left_width
            slopes[index] = (weight_left + weight_right) / (
                weight_left / left + weight_right / right
            )
        return tuple(slopes)

    @property
    def displacement(self) -> Tuple[float, float]:
        return (
            float(self.goal_xy[0]) - float(self.start_xy[0]),
            float(self.goal_xy[1]) - float(self.start_xy[1]),
        )

    @property
    def normal(self) -> Tuple[float, float]:
        dx, dy = self.displacement
        length = math.hypot(dx, dy)
        return -dy / length, dx / length

    @property
    def arc_length(self) -> float:
        return self._arc_distance[-1]

    @property
    def lateral_bounds(self) -> Tuple[float, float]:
        if self._profile_offset:
            return min(self._profile_offset), max(self._profile_offset)
        amplitude = abs(float(self.lateral_amplitude))
        return -amplitude, amplitude

    def _profile_value_and_slope(self, progress: float) -> Tuple[float, float]:
        u = max(0.0, min(1.0, float(progress)))
        index = bisect.bisect_right(self._profile_progress, u) - 1
        index = max(0, min(index, len(self._profile_progress) - 2))
        left_u = self._profile_progress[index]
        right_u = self._profile_progress[index + 1]
        width = right_u - left_u
        phase = (u - left_u) / width
        phase_sq = phase * phase
        phase_cu = phase_sq * phase

        left_offset = self._profile_offset[index]
        right_offset = self._profile_offset[index + 1]
        left_slope = self._profile_slope[index]
        right_slope = self._profile_slope[index + 1]
        offset = (
            (2.0 * phase_cu - 3.0 * phase_sq + 1.0) * left_offset
            + (phase_cu - 2.0 * phase_sq + phase) * width * left_slope
            + (-2.0 * phase_cu + 3.0 * phase_sq) * right_offset
            + (phase_cu - phase_sq) * width * right_slope
        )
        slope = (
            (6.0 * phase_sq - 6.0 * phase) * left_offset
            + (3.0 * phase_sq - 4.0 * phase + 1.0) * width * left_slope
            + (-6.0 * phase_sq + 6.0 * phase) * right_offset
            + (3.0 * phase_sq - 2.0 * phase) * width * right_slope
        ) / width
        return offset, slope

    def lateral_offset_at_progress(self, progress: float) -> float:
        u = max(0.0, min(1.0, float(progress)))
        if self._profile_progress:
            return self._profile_value_and_slope(u)[0]
        sine = math.sin(2.0 * math.pi * u)
        return float(self.lateral_amplitude) * sine**3

    def lateral_slope_at_progress(self, progress: float) -> float:
        u = max(0.0, min(1.0, float(progress)))
        if self._profile_progress:
            return self._profile_value_and_slope(u)[1]
        phase = 2.0 * math.pi * u
        sine = math.sin(phase)
        return (
            float(self.lateral_amplitude)
            * 6.0
            * math.pi
            * sine**2
            * math.cos(phase)
        )

    def position_at_progress(self, progress: float) -> Tuple[float, float]:
        u = max(0.0, min(1.0, float(progress)))
        dx, dy = self.displacement
        nx, ny = self.normal
        lateral = self.lateral_offset_at_progress(u)
        return (
            float(self.start_xy[0]) + u * dx + lateral * nx,
            float(self.start_xy[1]) + u * dy + lateral * ny,
        )

    def tangent_at_progress(self, progress: float) -> Tuple[float, float]:
        """Return ``dp/du`` for the geometric curve.

        Both the sinusoidal fallback and the control-point profile have zero
        endpoint lateral slope, so the initial/final tangent follows the
        start-to-goal diagonal instead of introducing an immediate side-slip.
        """

        u = max(0.0, min(1.0, float(progress)))
        dx, dy = self.displacement
        nx, ny = self.normal
        lateral_tangent = self.lateral_slope_at_progress(u)
        return dx + lateral_tangent * nx, dy + lateral_tangent * ny

    def progress_at_arc_fraction(self, arc_fraction: float) -> float:
        fraction = max(0.0, min(1.0, float(arc_fraction)))
        target = fraction * self.arc_length
        index = bisect.bisect_left(self._arc_distance, target)
        if index <= 0:
            return 0.0
        if index >= len(self._arc_distance):
            return 1.0
        left_distance = self._arc_distance[index - 1]
        right_distance = self._arc_distance[index]
        if right_distance <= left_distance:
            return self._arc_progress[index]
        phase = (target - left_distance) / (right_distance - left_distance)
        return self._arc_progress[index - 1] + phase * (
            self._arc_progress[index] - self._arc_progress[index - 1]
        )

    def sample(self, elapsed: float) -> IntentSample:
        tau = max(0.0, min(1.0, float(elapsed) / float(self.duration)))
        timed_progress = tau * tau * (3.0 - 2.0 * tau)
        if tau <= 0.0 or tau >= 1.0:
            timed_progress_rate = 0.0
        else:
            timed_progress_rate = 6.0 * tau * (1.0 - tau) / float(self.duration)

        if self.arc_length_timing:
            progress = self.progress_at_arc_fraction(timed_progress)
            tangent_x, tangent_y = self.tangent_at_progress(progress)
            tangent_norm = math.hypot(tangent_x, tangent_y)
            progress_rate = (
                timed_progress_rate * self.arc_length / max(tangent_norm, 1e-12)
            )
        else:
            progress = timed_progress
            tangent_x, tangent_y = self.tangent_at_progress(progress)
            progress_rate = timed_progress_rate

        velocity_xy = (
            progress_rate * tangent_x,
            progress_rate * tangent_y,
        )
        return IntentSample(
            progress=progress,
            position_xy=self.position_at_progress(progress),
            velocity_xy=velocity_xy,
            complete=tau >= 1.0,
        )

    def sampled_max_speed(self, count: int = 10001) -> float:
        if count < 2:
            raise ValueError("count must be at least two")
        maximum = 0.0
        for index in range(count):
            elapsed = float(self.duration) * index / float(count - 1)
            vx, vy = self.sample(elapsed).velocity_xy
            maximum = max(maximum, math.hypot(vx, vy))
        return maximum


@dataclass(frozen=True)
class RcAxisEncoding:
    """RC channel mapping that is inverse-compatible with ``rc_input_node``."""

    channel_base: int = 1
    forward_channel: int = 2
    lateral_channel: int = 1
    vertical_channel: int = 3
    pwm_min: float = 1000.0
    pwm_mid: float = 1500.0
    pwm_max: float = 2000.0
    max_forward_speed: float = 1.0
    max_lateral_speed: float = 1.0
    max_vertical_speed: float = 0.4
    forward_reverse: bool = False
    lateral_reverse: bool = False
    vertical_reverse: bool = False

    def __post_init__(self) -> None:
        if not (self.pwm_min < self.pwm_mid < self.pwm_max):
            raise ValueError("expected pwm_min < pwm_mid < pwm_max")
        if min(
            self.max_forward_speed,
            self.max_lateral_speed,
            self.max_vertical_speed,
        ) <= 0.0:
            raise ValueError("RC axis speed scales must be positive")

    def _index(self, channel: int) -> int:
        return int(channel) - int(self.channel_base)

    def _speed_to_pwm(self, speed: float, max_speed: float, reverse: bool) -> int:
        value = max(-1.0, min(1.0, float(speed) / float(max_speed)))
        if reverse:
            value = -value
        if value >= 0.0:
            pwm = self.pwm_mid + value * (self.pwm_max - self.pwm_mid)
        else:
            pwm = self.pwm_mid + value * (self.pwm_mid - self.pwm_min)
        return int(round(pwm))

    def encode_motion(
        self,
        source_channels: Iterable[int],
        body_velocity: Sequence[float],
    ) -> List[int]:
        """Copy auxiliary channels and overwrite only the three motion axes."""

        channels = [int(value) for value in source_channels]
        mapped_channels = (
            self.forward_channel,
            self.lateral_channel,
            self.vertical_channel,
        )
        required_index = max(self._index(channel) for channel in mapped_channels)
        if required_index < 0:
            raise ValueError("motion channel index precedes channel_base")
        if len(channels) <= required_index:
            channels.extend(
                [int(round(self.pwm_mid))] * (required_index + 1 - len(channels))
            )

        values = (
            self._speed_to_pwm(
                body_velocity[0], self.max_forward_speed, self.forward_reverse
            ),
            self._speed_to_pwm(
                body_velocity[1], self.max_lateral_speed, self.lateral_reverse
            ),
            self._speed_to_pwm(
                body_velocity[2], self.max_vertical_speed, self.vertical_reverse
            ),
        )
        for channel, value in zip(mapped_channels, values):
            channels[self._index(channel)] = value
        return channels


def map_velocity_to_body(
    velocity_map: Sequence[float],
    yaw_local: float,
    map_yaw: float = 0.0,
) -> Tuple[float, float, float]:
    """Express a map-frame velocity as body axes used by the RC bridge."""

    yaw_map = float(yaw_local) + float(map_yaw)
    cy = math.cos(yaw_map)
    sy = math.sin(yaw_map)
    vx = float(velocity_map[0])
    vy = float(velocity_map[1])
    vz = float(velocity_map[2]) if len(velocity_map) >= 3 else 0.0
    return cy * vx + sy * vy, -sy * vx + cy * vy, vz


def wrapped_angle_error(angle: float, target: float) -> float:
    """Return the signed shortest ``angle - target`` error in radians."""

    difference = float(angle) - float(target)
    return math.atan2(math.sin(difference), math.cos(difference))


def local_position_to_map(
    position_local: Sequence[float],
    map_yaw: float = 0.0,
    map_origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """Apply the same local-to-PCD transform used by ``map_lidar_node``."""

    cy = math.cos(float(map_yaw))
    sy = math.sin(float(map_yaw))
    x_local = float(position_local[0])
    y_local = float(position_local[1])
    z_local = float(position_local[2])
    return (
        cy * x_local - sy * y_local + float(map_origin_xyz[0]),
        sy * x_local + cy * y_local + float(map_origin_xyz[1]),
        z_local + float(map_origin_xyz[2]),
    )
