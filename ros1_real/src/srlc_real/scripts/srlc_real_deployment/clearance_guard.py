"""ROS-free collision confirmation and optional proximity protection."""

from dataclasses import dataclass
import math
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

        now_value, clearance, direction = accepted
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
            human_velocity = _finite_vector(
                human_velocity_world, "human_velocity_world"
            )
            position = _finite_vector(px4_local_position, "px4_local_position")
        except ValueError:
            return self._unusable_soft_result()

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

        human_dot = sum(
            component * normal
            for component, normal in zip(human_velocity, direction)
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
    ) -> Optional[Tuple[float, float, Vector3]]:
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
        if self._last_source_stamp is not None and stamp <= self._last_source_stamp:
            return None

        self._last_source_stamp = stamp
        return now_value, clearance, direction

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
