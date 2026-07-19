"""Atomic immutable observations shared by recorder ROS callbacks."""

import threading
from dataclasses import dataclass, field, replace


_NAN_VECTOR = (float("nan"), float("nan"), float("nan"))


@dataclass(frozen=True)
class ClearanceObservation:
    valid: bool = False
    source_stamp: float = float("nan")
    source_frame_id: str = ""
    surface_clearance: float = float("nan")
    center_distance: float = float("nan")
    nearest_obstacle_point: tuple = _NAN_VECTOR
    escape_direction: tuple = _NAN_VECTOR


@dataclass(frozen=True)
class GuardObservation:
    source_valid: bool = False
    source_stamp: float = float("nan")
    source_frame_id: str = ""
    raw_state: str = "UNKNOWN"
    effective_state: str = "UNKNOWN"


@dataclass(frozen=True)
class RecorderObservations:
    raw_center_distance: float = float("inf")
    policy_min_distance: float = float("inf")
    front_distance: float = float("inf")
    clearance: ClearanceObservation = field(
        default_factory=ClearanceObservation
    )
    guard: GuardObservation = field(default_factory=GuardObservation)


class RecorderObservationStore:
    """Replace whole observations and return one coherent timer snapshot."""

    def __init__(self):
        self._lock = threading.Lock()
        self._observations = RecorderObservations()

    def replace_clearance(self, clearance):
        with self._lock:
            self._observations = replace(
                self._observations,
                clearance=clearance,
            )

    def replace_guard(self, guard):
        with self._lock:
            self._observations = replace(
                self._observations,
                guard=guard,
            )

    def replace_raw_center_distance(self, raw_center_distance):
        with self._lock:
            self._observations = replace(
                self._observations,
                raw_center_distance=float(raw_center_distance),
            )

    def replace_policy_ranges(self, policy_min_distance, front_distance):
        with self._lock:
            self._observations = replace(
                self._observations,
                policy_min_distance=float(policy_min_distance),
                front_distance=float(front_distance),
            )

    def read(self):
        with self._lock:
            return self._observations
