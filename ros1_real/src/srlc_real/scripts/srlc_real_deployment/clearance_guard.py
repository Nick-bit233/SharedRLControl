"""ROS-free collision confirmation and optional proximity protection."""

from dataclasses import dataclass
import math
import struct
from typing import Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]


class ClearanceState:
    NORMAL = "NORMAL"
    PROXIMITY_HOLD = "PROXIMITY_HOLD"
    PROXIMITY_ESCAPE = "PROXIMITY_ESCAPE"
    COLLISION = "COLLISION"


@dataclass(frozen=True)
class ClearanceGuardConfig:
    proximity_enabled: bool = False
    proximity_enter_clearance: float = 0.10
    proximity_release_clearance: float = 0.15
    proximity_release_duration: float = 0.20
    escape_dot_threshold: float = 0.05
    collision_clearance: float = 0.02
    collision_confirm_samples: int = 2
    immediate_collision_clearance: float = -0.03
    sample_timeout: float = 0.30


@dataclass(frozen=True)
class ClearanceGuardResult:
    state: str
    hold_position: Optional[Vector3]
    escape_direction: Optional[Vector3]


def _finite_vector(value: Sequence[float], name: str) -> Vector3:
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain three finite values")
    return vector  # type: ignore[return-value]


def _normalized_vector(value: Sequence[float], name: str) -> Vector3:
    vector = _finite_vector(value, name)
    norm = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have non-zero length")
    if math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return vector
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def project_velocity_away(
    model_velocity: Sequence[float], escape_direction: Sequence[float]
) -> Vector3:
    """Remove only the model-velocity component opposing the escape direction."""
    velocity = _finite_vector(model_velocity, "model_velocity")
    direction = _normalized_vector(escape_direction, "escape_direction")
    dot = sum(component * normal for component, normal in zip(velocity, direction))
    toward_component = min(0.0, dot)
    return tuple(
        component - toward_component * normal
        for component, normal in zip(velocity, direction)
    )  # type: ignore[return-value]


def _positive_finite_limit(value: float, name: str) -> float:
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite value") from exc
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return limit


def constrain_escape_velocity(
    model_velocity: Sequence[float],
    escape_direction: Sequence[float],
    *,
    lock_z: bool,
    max_xy_speed: float,
    max_z_speed: float,
) -> Vector3:
    """Apply the escape half-space to the velocity that PX4 will receive."""
    velocity = _finite_vector(model_velocity, "model_velocity")
    direction = _normalized_vector(escape_direction, "escape_direction")
    xy_limit = _positive_finite_limit(max_xy_speed, "max_xy_speed")
    z_limit = _positive_finite_limit(max_z_speed, "max_z_speed")

    if lock_z:
        horizontal_norm = math.hypot(direction[0], direction[1])
        if horizontal_norm <= 1e-12:
            raise ValueError(
                "escape_direction must have a usable horizontal component "
                "when Z is locked"
            )
        horizontal_direction = (
            direction[0] / horizontal_norm,
            direction[1] / horizontal_norm,
        )
        horizontal_dot = (
            velocity[0] * horizontal_direction[0]
            + velocity[1] * horizontal_direction[1]
        )
        toward_component = min(0.0, horizontal_dot)
        constrained_x = velocity[0] - toward_component * horizontal_direction[0]
        constrained_y = velocity[1] - toward_component * horizontal_direction[1]
        horizontal_speed = math.hypot(constrained_x, constrained_y)
        scale = min(1.0, xy_limit / horizontal_speed) if horizontal_speed else 1.0
        return (constrained_x * scale, constrained_y * scale, 0.0)

    projected = project_velocity_away(velocity, direction)
    horizontal_speed = math.hypot(projected[0], projected[1])
    scale = 1.0
    if horizontal_speed > 0.0:
        scale = min(scale, xy_limit / horizontal_speed)
    if abs(projected[2]) > 0.0:
        scale = min(scale, z_limit / abs(projected[2]))
    return tuple(component * scale for component in projected)  # type: ignore[return-value]


def _float32(value: float, name: str) -> float:
    try:
        quantized = struct.unpack("!f", struct.pack("!f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{name} must remain finite in float32") from exc
    if not math.isfinite(quantized):
        raise ValueError(f"{name} must remain finite in float32")
    return quantized


def clamp_px4_velocity(
    velocity: Sequence[float],
    *,
    max_xy_speed: float,
    max_z_speed: float,
) -> Vector3:
    """Match the navigator's float32 XY-scale and independent Z clip."""
    command = _finite_vector(velocity, "velocity")
    xy_limit = _positive_finite_limit(max_xy_speed, "max_xy_speed")
    z_limit = _positive_finite_limit(max_z_speed, "max_z_speed")
    cmd = [_float32(component, "velocity") for component in command]

    horizontal_speed = math.hypot(cmd[0], cmd[1])
    if horizontal_speed > xy_limit:
        scale = xy_limit / horizontal_speed
        cmd[0] = _float32(cmd[0] * scale, "velocity")
        cmd[1] = _float32(cmd[1] * scale, "velocity")
    cmd[2] = _float32(max(-z_limit, min(cmd[2], z_limit)), "velocity")
    return (cmd[0], cmd[1], cmd[2])


def _next_float32_toward_zero(value: float) -> float:
    """Return the adjacent finite float32 with no greater magnitude."""
    quantized = _float32(value, "velocity")
    if quantized == 0.0:
        return 0.0
    bits = struct.unpack("!I", struct.pack("!f", quantized))[0]
    toward_zero = struct.unpack("!f", struct.pack("!I", bits - 1))[0]
    return 0.0 if toward_zero == 0.0 else toward_zero


def _bound_quantized_velocity(
    velocity: Vector3,
    *,
    max_xy_speed: float,
    max_z_speed: float,
) -> Vector3:
    """Move a rounded command inward until its exact float32 values are bounded."""
    command = list(velocity)
    if abs(command[2]) > max_z_speed:
        command[2] = _next_float32_toward_zero(command[2])

    for _ in range(8):
        if math.hypot(command[0], command[1]) <= max_xy_speed:
            return (command[0], command[1], command[2])
        index = 0 if abs(command[0]) >= abs(command[1]) else 1
        nudged = _next_float32_toward_zero(command[index])
        if nudged == command[index]:
            break
        command[index] = nudged
    return (0.0, 0.0, 0.0)


def _effective_escape_direction(
    escape_direction: Sequence[float], *, lock_z: bool
) -> Vector3:
    direction = _normalized_vector(escape_direction, "escape_direction")
    if not lock_z:
        return direction
    horizontal_norm = math.hypot(direction[0], direction[1])
    if horizontal_norm <= 1e-12:
        raise ValueError(
            "escape_direction must have a usable horizontal component when Z is locked"
        )
    return (
        direction[0] / horizontal_norm,
        direction[1] / horizontal_norm,
        0.0,
    )


def _repair_quantized_half_space(
    velocity: Vector3, escape_direction: Vector3
) -> Vector3:
    """Nudge only opposing float32 components toward zero until dot >= 0."""
    command = list(velocity)

    def escape_dot() -> float:
        return sum(
            component * normal
            for component, normal in zip(command, escape_direction)
        )

    dot = escape_dot()
    if dot >= 0.0:
        return velocity

    opposing = sorted(
        (
            index
            for index, (component, normal) in enumerate(
                zip(command, escape_direction)
            )
            if component * normal < 0.0
        ),
        key=lambda index: abs(escape_direction[index]),
        reverse=True,
    )
    for index in opposing:
        normal_magnitude = abs(escape_direction[index])
        required_change = -dot / normal_magnitude
        target_magnitude = max(0.0, abs(command[index]) - required_change)
        corrected = _float32(
            math.copysign(target_magnitude, command[index]),
            "velocity",
        )
        if abs(corrected) > abs(command[index]):
            corrected = command[index]
        command[index] = corrected
        dot = escape_dot()

        for _ in range(2):
            if dot >= 0.0 or command[index] == 0.0:
                break
            command[index] = _next_float32_toward_zero(command[index])
            dot = escape_dot()
        if dot >= 0.0:
            return (command[0], command[1], command[2])

        command[index] = 0.0
        dot = escape_dot()
        if dot >= 0.0:
            return (command[0], command[1], command[2])

    return (0.0, 0.0, 0.0)


def finalize_px4_escape_velocity(
    model_velocity: Sequence[float],
    escape_direction: Sequence[float],
    *,
    lock_z: bool,
    max_xy_speed: float,
    max_z_speed: float,
) -> Vector3:
    """Return the final escape command after PX4-bound quantization and limits."""
    xy_limit = _positive_finite_limit(max_xy_speed, "max_xy_speed")
    z_limit = _positive_finite_limit(max_z_speed, "max_z_speed")
    effective_direction = _effective_escape_direction(
        escape_direction,
        lock_z=lock_z,
    )
    constrained = constrain_escape_velocity(
        model_velocity,
        effective_direction,
        lock_z=lock_z,
        max_xy_speed=xy_limit,
        max_z_speed=z_limit,
    )
    clamped = clamp_px4_velocity(
        constrained,
        max_xy_speed=xy_limit,
        max_z_speed=z_limit,
    )
    rechecked = constrain_escape_velocity(
        clamped,
        effective_direction,
        lock_z=lock_z,
        max_xy_speed=xy_limit,
        max_z_speed=z_limit,
    )
    quantized = clamp_px4_velocity(
        rechecked,
        max_xy_speed=xy_limit,
        max_z_speed=z_limit,
    )
    bounded = _bound_quantized_velocity(
        quantized,
        max_xy_speed=xy_limit,
        max_z_speed=z_limit,
    )
    safe = _repair_quantized_half_space(bounded, effective_direction)

    if clamp_px4_velocity(
        safe,
        max_xy_speed=xy_limit,
        max_z_speed=z_limit,
    ) != safe:
        return (0.0, 0.0, 0.0)
    if sum(
        component * normal
        for component, normal in zip(safe, effective_direction)
    ) < 0.0:
        return (0.0, 0.0, 0.0)
    return safe


class ClearanceGuard:
    """Own hard collision confirmation and the optional soft guard state."""

    def __init__(self, config: Optional[ClearanceGuardConfig] = None):
        self.config = config if config is not None else ClearanceGuardConfig()
        self._validate_config(self.config)
        self._last_source_stamp: Optional[float] = None
        self._collision_pending = 0
        self._collision_confirmed = False
        self._soft_state = ClearanceState.NORMAL
        self._hold_position: Optional[Vector3] = None
        self._release_started_at: Optional[float] = None

    def update(
        self,
        now: float,
        source_stamp: float,
        valid: bool,
        surface_clearance: float,
        escape_direction: Sequence[float],
        human_velocity_world: Sequence[float],
        px4_local_position: Sequence[float],
    ) -> ClearanceGuardResult:
        """Evaluate one source-stamped clearance sample and current control input."""
        accepted = self._accept_sample(
            now,
            source_stamp,
            valid,
            surface_clearance,
            escape_direction,
        )
        if accepted is None:
            return self._unusable_result()

        now_value, clearance, direction, new_source_frame = accepted
        if new_source_frame:
            if clearance <= self.config.immediate_collision_clearance:
                self._collision_confirmed = True
            elif clearance <= self.config.collision_clearance:
                self._collision_pending += 1
                if self._collision_pending >= self.config.collision_confirm_samples:
                    self._collision_confirmed = True
            else:
                self._collision_pending = 0
                self._collision_confirmed = False

        if self._collision_confirmed:
            self._release_started_at = None
            return ClearanceGuardResult(
                ClearanceState.COLLISION,
                self._hold_position,
                direction,
            )

        if not self.config.proximity_enabled:
            self._clear_soft_state()
            return ClearanceGuardResult(ClearanceState.NORMAL, None, direction)

        try:
            position = _finite_vector(px4_local_position, "px4_local_position")
        except ValueError:
            return self._unusable_soft_result()
        try:
            human_velocity = _finite_vector(
                human_velocity_world, "human_velocity_world"
            )
        except ValueError:
            human_velocity = None

        if self._soft_state == ClearanceState.NORMAL:
            if clearance > self.config.proximity_enter_clearance:
                return ClearanceGuardResult(ClearanceState.NORMAL, None, direction)
            self._soft_state = ClearanceState.PROXIMITY_HOLD
            self._hold_position = position

        if clearance > self.config.proximity_release_clearance:
            if self._release_started_at is None:
                self._release_started_at = now_value
            elapsed = now_value - self._release_started_at
            if elapsed >= self.config.proximity_release_duration or math.isclose(
                elapsed,
                self.config.proximity_release_duration,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                self._clear_soft_state()
                return ClearanceGuardResult(ClearanceState.NORMAL, None, direction)
        else:
            self._release_started_at = None

        human_dot = (
            sum(
                component * normal
                for component, normal in zip(human_velocity, direction)
            )
            if human_velocity is not None
            else -math.inf
        )
        if human_dot >= self.config.escape_dot_threshold:
            self._soft_state = ClearanceState.PROXIMITY_ESCAPE
            self._hold_position = position
        else:
            self._soft_state = ClearanceState.PROXIMITY_HOLD

        return ClearanceGuardResult(
            self._soft_state,
            self._hold_position,
            direction,
        )

    def _accept_sample(
        self,
        now: float,
        source_stamp: float,
        valid: bool,
        surface_clearance: float,
        escape_direction: Sequence[float],
    ) -> Optional[Tuple[float, float, Vector3, bool]]:
        try:
            now_value = float(now)
            stamp = float(source_stamp)
            clearance = float(surface_clearance)
            direction = _normalized_vector(escape_direction, "escape_direction")
        except (TypeError, ValueError):
            return None
        if not bool(valid) or not all(
            math.isfinite(value) for value in (now_value, stamp, clearance)
        ):
            return None
        if stamp > now_value or now_value - stamp > self.config.sample_timeout:
            return None
        if self._last_source_stamp is not None and stamp < self._last_source_stamp:
            return None

        new_source_frame = (
            self._last_source_stamp is None or stamp > self._last_source_stamp
        )
        if new_source_frame:
            self._last_source_stamp = stamp
        return now_value, clearance, direction, new_source_frame

    def _unusable_result(self) -> ClearanceGuardResult:
        if self._collision_confirmed:
            return ClearanceGuardResult(
                ClearanceState.COLLISION,
                self._hold_position,
                None,
            )
        return self._unusable_soft_result()

    def _unusable_soft_result(self) -> ClearanceGuardResult:
        self._release_started_at = None
        if self.config.proximity_enabled and self._soft_state != ClearanceState.NORMAL:
            self._soft_state = ClearanceState.PROXIMITY_HOLD
            return ClearanceGuardResult(
                ClearanceState.PROXIMITY_HOLD,
                self._hold_position,
                None,
            )
        return ClearanceGuardResult(ClearanceState.NORMAL, None, None)

    def _clear_soft_state(self) -> None:
        self._soft_state = ClearanceState.NORMAL
        self._hold_position = None
        self._release_started_at = None

    @staticmethod
    def _validate_config(config: ClearanceGuardConfig) -> None:
        finite_values = (
            config.proximity_enter_clearance,
            config.proximity_release_clearance,
            config.proximity_release_duration,
            config.escape_dot_threshold,
            config.collision_clearance,
            config.immediate_collision_clearance,
            config.sample_timeout,
        )
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError("clearance guard thresholds must be finite")
        if config.proximity_release_clearance <= config.proximity_enter_clearance:
            raise ValueError("proximity release clearance must exceed enter clearance")
        if config.proximity_release_duration < 0.0:
            raise ValueError("proximity release duration must be non-negative")
        if config.escape_dot_threshold < 0.0:
            raise ValueError("escape dot threshold must be non-negative")
        if config.collision_confirm_samples < 1:
            raise ValueError("collision confirm samples must be at least one")
        if config.immediate_collision_clearance > config.collision_clearance:
            raise ValueError(
                "immediate collision clearance must not exceed collision clearance"
            )
        if config.sample_timeout < 0.0:
            raise ValueError("sample timeout must be non-negative")
