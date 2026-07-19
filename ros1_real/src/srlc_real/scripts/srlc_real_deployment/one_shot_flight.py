"""Deterministic one-shot takeoff lifecycle for the real PX4 runtime.

This module deliberately contains no ROS imports.  The ROS node supplies a
snapshot of external state and executes the returned action, while this class
owns every state transition and the single immutable takeoff target.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


Vector3 = Tuple[float, float, float]


class LifecycleState:
    DISABLED = "DISABLED"
    WAIT_READY = "WAIT_READY"
    WAIT_ARMED = "WAIT_ARMED"
    WAIT_OFFBOARD = "WAIT_OFFBOARD"
    TAKEOFF = "TAKEOFF"
    ACTIVE = "ACTIVE"
    FAULT_LAND = "FAULT_LAND"
    FAULT_HOLD = "FAULT_HOLD"
    TERMINATED = "TERMINATED"


class FlightAction:
    STOP_STREAM = "STOP_STREAM"
    PRESTREAM_HOLD = "PRESTREAM_HOLD"
    TAKEOFF_HOLD = "TAKEOFF_HOLD"
    ACTIVE_CONTROL = "ACTIVE_CONTROL"
    FAULT_HOLD = "FAULT_HOLD"
    REQUEST_MODE = "REQUEST_MODE"


@dataclass(frozen=True)
class OneShotFlightConfig:
    enabled: bool = True
    takeoff_height: float = 1.0
    takeoff_lower_margin: float = 0.2
    takeoff_upper_margin: float = 0.2
    takeoff_max_abs_vz: float = 0.25
    takeoff_confirm_duration: float = 0.5
    takeoff_timeout: float = 15.0
    takeoff_max_overshoot: float = 0.5
    takeoff_max_xy_drift: float = 0.5
    takeoff_max_climb_speed: float = 0.4
    takeoff_max_vertical_accel: float = 0.5
    takeoff_max_tracking_error: float = 0.25
    input_recovery_grace: float = 1.0
    fault_response: str = "hold"
    fault_land_mode: str = "AUTO.LAND"
    fault_land_confirm_timeout: float = 2.0
    fault_land_retry_interval: float = 0.5
    fault_land_max_attempts: int = 3


@dataclass(frozen=True)
class FlightSnapshot:
    now: float
    connected: bool
    armed: bool
    mode: str
    position: Optional[Vector3]
    velocity: Optional[Vector3]
    odom_fresh: bool
    rc_fresh: bool
    lidar_fresh: bool
    landed: bool
    external_fault: Optional[str] = None


@dataclass(frozen=True)
class FlightDecision:
    state: str
    action: str
    reason: str
    target: Optional[Vector3]
    target_velocity: Optional[Vector3]
    request_mode: Optional[str]
    session_consumed: bool
    state_changed: bool


class OneShotFlightLifecycle:
    """One process lifetime permits at most one OFFBOARD flight session."""

    def __init__(self, config: OneShotFlightConfig):
        if config.fault_response not in {"auto_land", "hold"}:
            raise ValueError("fault_response must be 'auto_land' or 'hold'")
        if config.takeoff_height <= 0.0:
            raise ValueError("takeoff_height must be positive")
        if config.takeoff_lower_margin < 0.0 or config.takeoff_upper_margin < 0.0:
            raise ValueError("takeoff margins must be non-negative")
        if config.takeoff_max_climb_speed <= 0.0:
            raise ValueError("takeoff_max_climb_speed must be positive")
        if config.takeoff_max_vertical_accel <= 0.0:
            raise ValueError("takeoff_max_vertical_accel must be positive")
        if config.takeoff_max_tracking_error <= 0.0:
            raise ValueError("takeoff_max_tracking_error must be positive")
        if config.fault_land_max_attempts < 1:
            raise ValueError("fault_land_max_attempts must be at least one")

        self.config = config
        self.state = LifecycleState.WAIT_READY if config.enabled else LifecycleState.DISABLED
        self.reason = "WAIT_READY" if config.enabled else "FLIGHT_DISABLED"
        self.session_consumed = False
        self.takeoff_origin: Optional[Vector3] = None
        self.takeoff_target: Optional[Vector3] = None
        self.takeoff_command: Optional[Vector3] = None
        self.takeoff_command_velocity: Optional[Vector3] = None

        self._takeoff_started_at: Optional[float] = None
        self._takeoff_profile_updated_at: Optional[float] = None
        self._takeoff_decelerating = False
        self._takeoff_confirm_started_at: Optional[float] = None
        self._input_fault_started_at: Optional[float] = None
        self._last_safe_position: Optional[Vector3] = None
        self._fault_hold_target: Optional[Vector3] = None
        self._fault_reason: Optional[str] = None
        self._fault_land_started_at: Optional[float] = None
        self._last_land_request_at: Optional[float] = None
        self._land_request_attempts = 0

    def update(self, snapshot: FlightSnapshot) -> FlightDecision:
        previous_state = self.state

        if self.state == LifecycleState.DISABLED:
            return self._decision(
                FlightAction.STOP_STREAM,
                "FLIGHT_DISABLED",
                previous_state=previous_state,
            )

        if self.state == LifecycleState.TERMINATED:
            return self._decision(
                FlightAction.STOP_STREAM,
                self.reason,
                previous_state=previous_state,
            )

        if self.state == LifecycleState.FAULT_LAND:
            return self._update_fault_land(snapshot, previous_state)

        if self.state == LifecycleState.FAULT_HOLD:
            return self._update_fault_hold(snapshot, previous_state)

        if not self.session_consumed:
            return self._update_before_session(snapshot, previous_state)

        return self._update_active_session(snapshot, previous_state)

    def _update_before_session(
        self, snapshot: FlightSnapshot, previous_state: str
    ) -> FlightDecision:
        if snapshot.position is not None and snapshot.odom_fresh:
            self._last_safe_position = self._vec(snapshot.position)

        if not snapshot.connected:
            self.state = LifecycleState.WAIT_READY
            self.reason = "MAVROS_NOT_CONNECTED"
            return self._decision(
                FlightAction.STOP_STREAM,
                self.reason,
                previous_state=previous_state,
            )

        prestream_target = (
            self._vec(snapshot.position) if snapshot.position is not None else None
        )

        if snapshot.armed and snapshot.mode == "OFFBOARD":
            self.session_consumed = True
            readiness_reason = self._start_readiness_failure(snapshot)
            if readiness_reason is not None:
                return self._enter_fault(readiness_reason, snapshot, previous_state)
            self._start_takeoff(snapshot)
            return self._decision(
                FlightAction.TAKEOFF_HOLD,
                "TAKEOFF",
                target=self.takeoff_command,
                target_velocity=self.takeoff_command_velocity,
                previous_state=previous_state,
            )

        if snapshot.position is None or snapshot.velocity is None or not snapshot.odom_fresh:
            self.state = LifecycleState.WAIT_READY
            self.reason = "ODOM_NOT_READY"
        elif not snapshot.rc_fresh:
            self.state = LifecycleState.WAIT_READY
            self.reason = "RC_NOT_READY"
        elif not snapshot.lidar_fresh:
            self.state = LifecycleState.WAIT_READY
            self.reason = "LIDAR_NOT_READY"
        elif not snapshot.armed:
            self.state = LifecycleState.WAIT_ARMED
            self.reason = "WAIT_ARMED"
        else:
            self.state = LifecycleState.WAIT_OFFBOARD
            self.reason = "WAIT_OFFBOARD"

        action = (
            FlightAction.PRESTREAM_HOLD
            if prestream_target is not None
            else FlightAction.STOP_STREAM
        )
        return self._decision(
            action,
            self.reason,
            target=prestream_target,
            previous_state=previous_state,
        )

    def _start_readiness_failure(self, snapshot: FlightSnapshot) -> Optional[str]:
        if snapshot.position is None or snapshot.velocity is None or not snapshot.odom_fresh:
            return "ODOM_NOT_READY"
        if not snapshot.rc_fresh:
            return "RC_NOT_READY"
        if not snapshot.lidar_fresh:
            return "LIDAR_NOT_READY"
        if snapshot.external_fault:
            return str(snapshot.external_fault)
        if not snapshot.landed:
            return "NOT_LANDED_AT_START"
        return None

    def _start_takeoff(self, snapshot: FlightSnapshot) -> None:
        origin = self._vec(snapshot.position)
        self.takeoff_origin = origin
        self.takeoff_target = (
            origin[0],
            origin[1],
            origin[2] + float(self.config.takeoff_height),
        )
        self.takeoff_command = origin
        self.takeoff_command_velocity = (0.0, 0.0, 0.0)
        self._takeoff_started_at = float(snapshot.now)
        self._takeoff_profile_updated_at = float(snapshot.now)
        self._takeoff_decelerating = False
        self._takeoff_confirm_started_at = None
        self._input_fault_started_at = None
        self._last_safe_position = origin
        self.state = LifecycleState.TAKEOFF
        self.reason = "TAKEOFF"

    def _update_active_session(
        self, snapshot: FlightSnapshot, previous_state: str
    ) -> FlightDecision:
        if not snapshot.connected:
            return self._terminate("MAVROS_DISCONNECTED", previous_state)
        if not snapshot.armed:
            return self._terminate("DISARMED", previous_state)
        if snapshot.mode != "OFFBOARD":
            return self._terminate("OFFBOARD_LOST", previous_state)

        if snapshot.position is None or snapshot.velocity is None or not snapshot.odom_fresh:
            return self._enter_fault("ODOM_TIMEOUT", snapshot, previous_state)

        position = self._vec(snapshot.position)
        velocity = self._vec(snapshot.velocity)
        self._last_safe_position = position

        if snapshot.external_fault:
            return self._enter_fault(
                str(snapshot.external_fault), snapshot, previous_state
            )

        stale_reason = None
        if not snapshot.rc_fresh:
            stale_reason = "RC_TIMEOUT"
        elif not snapshot.lidar_fresh:
            stale_reason = "LIDAR_TIMEOUT"
        if stale_reason is not None:
            if self._input_fault_started_at is None:
                self._input_fault_started_at = float(snapshot.now)
            if (
                float(snapshot.now) - self._input_fault_started_at
                >= self.config.input_recovery_grace
            ):
                return self._enter_fault(stale_reason, snapshot, previous_state)
            self._takeoff_confirm_started_at = None
            self._pause_takeoff_profile(snapshot.now, position)
            return self._decision(
                FlightAction.FAULT_HOLD,
                "INPUT_RECOVERY_HOLD",
                target=position,
                previous_state=previous_state,
            )
        self._input_fault_started_at = None

        if self.state == LifecycleState.TAKEOFF:
            return self._update_takeoff(snapshot, position, velocity, previous_state)

        self.state = LifecycleState.ACTIVE
        self.reason = "ACTIVE"
        return self._decision(
            FlightAction.ACTIVE_CONTROL,
            "ACTIVE",
            target=self.takeoff_target,
            previous_state=previous_state,
        )

    def _update_takeoff(
        self,
        snapshot: FlightSnapshot,
        position: Vector3,
        velocity: Vector3,
        previous_state: str,
    ) -> FlightDecision:
        target = self.takeoff_target
        origin = self.takeoff_origin
        if target is None or origin is None or self._takeoff_started_at is None:
            return self._enter_fault(
                "TAKEOFF_SESSION_INVALID", snapshot, previous_state
            )

        if float(snapshot.now) - self._takeoff_started_at >= self.config.takeoff_timeout:
            return self._enter_fault("TAKEOFF_TIMEOUT", snapshot, previous_state)

        xy_drift = math.hypot(position[0] - origin[0], position[1] - origin[1])
        if xy_drift > self.config.takeoff_max_xy_drift:
            return self._enter_fault("TAKEOFF_XY_DRIFT", snapshot, previous_state)

        if position[2] > target[2] + self.config.takeoff_max_overshoot:
            return self._enter_fault("TAKEOFF_OVERHEIGHT", snapshot, previous_state)

        self._advance_takeoff_profile(snapshot.now, position)
        command = self.takeoff_command
        command_velocity = self.takeoff_command_velocity
        if command is None or command_velocity is None:
            return self._enter_fault(
                "TAKEOFF_SESSION_INVALID", snapshot, previous_state
            )

        inside_height_band = (
            target[2] - self.config.takeoff_lower_margin
            <= position[2]
            <= target[2] + self.config.takeoff_upper_margin
        )
        vertical_speed_ok = abs(velocity[2]) <= self.config.takeoff_max_abs_vz
        profile_complete = command == target and command_velocity == (0.0, 0.0, 0.0)
        if profile_complete and inside_height_band and vertical_speed_ok:
            if self._takeoff_confirm_started_at is None:
                self._takeoff_confirm_started_at = float(snapshot.now)
            elif (
                float(snapshot.now) - self._takeoff_confirm_started_at
                >= self.config.takeoff_confirm_duration
            ):
                self.state = LifecycleState.ACTIVE
                self.reason = "ACTIVE"
                return self._decision(
                    FlightAction.ACTIVE_CONTROL,
                    "ACTIVE",
                    target=target,
                    previous_state=previous_state,
                )
        else:
            self._takeoff_confirm_started_at = None

        self.state = LifecycleState.TAKEOFF
        self.reason = "TAKEOFF"
        return self._decision(
            FlightAction.TAKEOFF_HOLD,
            "TAKEOFF",
            target=command,
            target_velocity=command_velocity,
            previous_state=previous_state,
        )

    def _advance_takeoff_profile(self, now: float, position: Vector3) -> None:
        command = self.takeoff_command
        command_velocity = self.takeoff_command_velocity
        target = self.takeoff_target
        if (
            command is None
            or command_velocity is None
            or target is None
            or self._takeoff_profile_updated_at is None
        ):
            return

        now = float(now)
        elapsed = max(0.0, now - self._takeoff_profile_updated_at)
        dt = min(elapsed, 0.1)
        self._takeoff_profile_updated_at = now

        remaining = max(0.0, target[2] - command[2])
        if remaining <= 1e-9:
            self.takeoff_command = target
            self.takeoff_command_velocity = (0.0, 0.0, 0.0)
            return
        if dt <= 0.0:
            return

        max_speed = float(self.config.takeoff_max_climb_speed)
        max_accel = float(self.config.takeoff_max_vertical_accel)
        tracking_limit = float(self.config.takeoff_max_tracking_error)
        current_vz = max(0.0, float(command_velocity[2]))
        tracking_error = command[2] - position[2]
        if tracking_error >= tracking_limit:
            desired_vz = 0.0
        else:
            stopping_distance = current_vz * current_vz / (2.0 * max_accel)
            if self._takeoff_decelerating or remaining <= stopping_distance + 1e-9:
                self._takeoff_decelerating = True
                desired_vz = 0.0
            else:
                desired_vz = max_speed

        max_dv = max_accel * dt
        if desired_vz >= current_vz:
            next_vz = min(desired_vz, current_vz + max_dv)
        else:
            next_vz = max(desired_vz, current_vz - max_dv)

        advance = max(0.0, 0.5 * (current_vz + next_vz) * dt)
        unconstrained_z = min(target[2], command[2] + advance)
        lead_limited_z = min(target[2], position[2] + tracking_limit)
        next_z = max(command[2], min(unconstrained_z, lead_limited_z))
        if next_z + 1e-9 < unconstrained_z:
            next_vz = 0.0
            self._takeoff_decelerating = False

        if target[2] - next_z <= 1e-9:
            next_z = target[2]
            next_vz = 0.0
        elif next_vz <= 1e-9 and self._takeoff_decelerating:
            # Timer jitter can finish a braking segment just short of the
            # target.  Replan the small remainder instead of waiting forever.
            self._takeoff_decelerating = False

        self.takeoff_command = (target[0], target[1], next_z)
        self.takeoff_command_velocity = (0.0, 0.0, next_vz)

    def _pause_takeoff_profile(self, now: float, position: Vector3) -> None:
        if (
            self.state != LifecycleState.TAKEOFF
            or self.takeoff_origin is None
            or self.takeoff_target is None
        ):
            return
        paused_z = min(
            self.takeoff_target[2],
            max(self.takeoff_origin[2], float(position[2])),
        )
        self.takeoff_command = (
            self.takeoff_origin[0],
            self.takeoff_origin[1],
            paused_z,
        )
        self.takeoff_command_velocity = (0.0, 0.0, 0.0)
        self._takeoff_profile_updated_at = float(now)
        self._takeoff_decelerating = False

    def _enter_fault(
        self, reason: str, snapshot: FlightSnapshot, previous_state: str
    ) -> FlightDecision:
        self._fault_reason = str(reason)
        if self._fault_hold_target is None:
            if snapshot.position is not None and snapshot.odom_fresh:
                self._fault_hold_target = self._vec(snapshot.position)
            else:
                self._fault_hold_target = self._last_safe_position

        can_request_land = (
            self.config.fault_response == "auto_land"
            and snapshot.connected
            and snapshot.armed
            and snapshot.mode == "OFFBOARD"
        )
        if not can_request_land:
            self.state = LifecycleState.FAULT_HOLD
            self.reason = self._fault_reason
            return self._decision(
                FlightAction.FAULT_HOLD,
                self.reason,
                target=self._fault_hold_target,
                previous_state=previous_state,
            )

        self.state = LifecycleState.FAULT_LAND
        self.reason = self._fault_reason
        self._fault_land_started_at = float(snapshot.now)
        self._last_land_request_at = float(snapshot.now)
        self._land_request_attempts = 1
        return self._decision(
            FlightAction.REQUEST_MODE,
            self.reason,
            target=self._fault_hold_target,
            request_mode=self.config.fault_land_mode,
            previous_state=previous_state,
        )

    def _update_fault_land(
        self, snapshot: FlightSnapshot, previous_state: str
    ) -> FlightDecision:
        if snapshot.mode == self.config.fault_land_mode:
            return self._terminate("AUTO_LAND_CONFIRMED", previous_state)
        if not snapshot.connected:
            return self._terminate("MAVROS_DISCONNECTED", previous_state)
        if not snapshot.armed:
            return self._terminate("DISARMED", previous_state)
        if snapshot.mode != "OFFBOARD":
            return self._terminate("OFFBOARD_LOST", previous_state)

        started_at = (
            self._fault_land_started_at
            if self._fault_land_started_at is not None
            else float(snapshot.now)
        )
        if (
            float(snapshot.now) - started_at
            >= self.config.fault_land_confirm_timeout
        ):
            self.state = LifecycleState.FAULT_HOLD
            self.reason = "AUTO_LAND_UNCONFIRMED"
            return self._decision(
                FlightAction.FAULT_HOLD,
                self.reason,
                target=self._fault_hold_target,
                previous_state=previous_state,
            )

        request_mode = None
        if (
            self._land_request_attempts < self.config.fault_land_max_attempts
            and self._last_land_request_at is not None
            and float(snapshot.now) - self._last_land_request_at
            >= self.config.fault_land_retry_interval
        ):
            self._land_request_attempts += 1
            self._last_land_request_at = float(snapshot.now)
            request_mode = self.config.fault_land_mode

        action = (
            FlightAction.REQUEST_MODE
            if request_mode is not None
            else FlightAction.FAULT_HOLD
        )
        return self._decision(
            action,
            self._fault_reason or self.reason,
            target=self._fault_hold_target,
            request_mode=request_mode,
            previous_state=previous_state,
        )

    def _update_fault_hold(
        self, snapshot: FlightSnapshot, previous_state: str
    ) -> FlightDecision:
        if not snapshot.connected:
            return self._terminate("MAVROS_DISCONNECTED", previous_state)
        if not snapshot.armed:
            return self._terminate("DISARMED", previous_state)
        if snapshot.mode != "OFFBOARD":
            return self._terminate("OFFBOARD_LOST", previous_state)
        return self._decision(
            FlightAction.FAULT_HOLD,
            self.reason,
            target=self._fault_hold_target,
            previous_state=previous_state,
        )

    def _terminate(self, reason: str, previous_state: str) -> FlightDecision:
        self.state = LifecycleState.TERMINATED
        self.reason = str(reason)
        return self._decision(
            FlightAction.STOP_STREAM,
            self.reason,
            previous_state=previous_state,
        )

    def _decision(
        self,
        action: str,
        reason: str,
        *,
        target: Optional[Vector3] = None,
        target_velocity: Optional[Vector3] = None,
        request_mode: Optional[str] = None,
        previous_state: str,
    ) -> FlightDecision:
        return FlightDecision(
            state=self.state,
            action=action,
            reason=str(reason),
            target=target,
            target_velocity=target_velocity,
            request_mode=request_mode,
            session_consumed=bool(self.session_consumed),
            state_changed=self.state != previous_state,
        )

    @staticmethod
    def _vec(value: Vector3) -> Vector3:
        return (float(value[0]), float(value[1]), float(value[2]))
