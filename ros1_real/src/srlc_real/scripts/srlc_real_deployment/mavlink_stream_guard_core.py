"""Pure state machine for restoring critical PX4 MAVLink streams."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Optional, Tuple


class GuardState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    WAITING_SERVICE = "WAITING_SERVICE"
    REQUESTING = "REQUESTING"
    VERIFYING = "VERIFYING"
    HEALTHY = "HEALTHY"
    FAILED = "FAILED"


class GuardAction(str, Enum):
    REQUEST_STREAMS = "REQUEST_STREAMS"


@dataclass(frozen=True)
class StreamRequest:
    message_id: int
    rate_hz: float


def build_stream_requests(
    local_position_rate_hz: float,
    attitude_rate_hz: float,
) -> Tuple[StreamRequest, StreamRequest]:
    """Return the two PX4 streams required by MAVROS local-position output."""

    return (
        StreamRequest(message_id=32, rate_hz=float(local_position_rate_hz)),
        StreamRequest(message_id=30, rate_hz=float(attitude_rate_hz)),
    )


def apply_stream_requests(
    requests: Iterable[StreamRequest],
    send: Callable[[StreamRequest], bool],
) -> bool:
    """Send a complete stream batch and report whether every request succeeded."""

    success = True
    for request in requests:
        if not bool(send(request)):
            success = False
    return success


class MavlinkStreamGuardCore:
    """Connection-aware, bounded-retry stream recovery state machine."""

    def __init__(
        self,
        verify_timeout_sec: float,
        stale_timeout_sec: float,
        retry_interval_sec: float,
        max_attempts: int,
    ):
        self.verify_timeout_sec = float(verify_timeout_sec)
        self.stale_timeout_sec = float(stale_timeout_sec)
        self.retry_interval_sec = float(retry_interval_sec)
        self.max_attempts = int(max_attempts)

        self.connected = False
        self.state = GuardState.DISCONNECTED
        self.attempts = 0
        self._next_attempt_at = 0.0
        self._request_started_at: Optional[float] = None
        self._request_in_flight = False
        self._last_local_pose_at: Optional[float] = None
        self._last_imu_at: Optional[float] = None

    def on_connection(self, connected: bool, now: float) -> None:
        connected = bool(connected)
        if connected == self.connected:
            return

        self.connected = connected
        self.attempts = 0
        self._request_started_at = None
        self._request_in_flight = False
        self._last_local_pose_at = None
        self._last_imu_at = None

        if connected:
            self.state = GuardState.WAITING_SERVICE
            self._next_attempt_at = float(now)
        else:
            self.state = GuardState.DISCONNECTED

    def on_local_pose(self, now: float) -> None:
        self._last_local_pose_at = float(now)

    def on_imu(self, now: float) -> None:
        self._last_imu_at = float(now)

    def on_request_result(self, now: float, success: bool) -> None:
        if not self.connected or not self._request_in_flight:
            return

        self._request_in_flight = False
        now = float(now)
        if success:
            self._request_started_at = now
            self.state = GuardState.VERIFYING
            return

        if self.attempts >= self.max_attempts:
            self.state = GuardState.FAILED
        else:
            self.state = GuardState.REQUESTING
            self._next_attempt_at = now + self.retry_interval_sec

    def tick(self, now: float, service_available: bool) -> Optional[GuardAction]:
        now = float(now)
        if not self.connected or self.state == GuardState.FAILED:
            return None

        if self.state == GuardState.VERIFYING:
            if self._received_fresh_samples_after_request(now):
                self.state = GuardState.HEALTHY
                return None

            if now - self._request_started_at >= self.verify_timeout_sec:
                if self.attempts >= self.max_attempts:
                    self.state = GuardState.FAILED
                else:
                    self.state = GuardState.REQUESTING
                    self._next_attempt_at = now + self.retry_interval_sec
            return None

        if self.state == GuardState.HEALTHY:
            if self._streams_are_fresh(now):
                return None
            self.attempts = 0
            self._request_started_at = None
            self.state = GuardState.REQUESTING
            self._next_attempt_at = now

        if self._request_in_flight or now < self._next_attempt_at:
            return None

        if not service_available:
            self.state = GuardState.WAITING_SERVICE
            return None

        if self.attempts >= self.max_attempts:
            self.state = GuardState.FAILED
            return None

        self.attempts += 1
        self._request_in_flight = True
        self.state = GuardState.REQUESTING
        return GuardAction.REQUEST_STREAMS

    def _streams_are_fresh(self, now: float) -> bool:
        return (
            self._last_local_pose_at is not None
            and self._last_imu_at is not None
            and now - self._last_local_pose_at <= self.stale_timeout_sec
            and now - self._last_imu_at <= self.stale_timeout_sec
        )

    def _received_fresh_samples_after_request(self, now: float) -> bool:
        return (
            self._request_started_at is not None
            and self._last_local_pose_at is not None
            and self._last_imu_at is not None
            and self._last_local_pose_at > self._request_started_at
            and self._last_imu_at > self._request_started_at
            and self._streams_are_fresh(now)
        )
