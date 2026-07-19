"""ROS-free lifecycle adapter for clearance-guard inputs."""

from typing import Sequence, Tuple


Vector3 = Tuple[float, float, float]


def lifecycle_lidar_fresh(
    clearance_guard_mode: str,
    range_fresh: bool,
    range_ready: bool,
    clearance_fresh: bool,
) -> bool:
    """Return lifecycle readiness without leaking shadow clearance state."""
    range_channel_fresh = bool(range_fresh) and bool(range_ready)
    if str(clearance_guard_mode).strip().lower() == "shadow":
        return range_channel_fresh
    return range_channel_fresh and bool(clearance_fresh)


def soft_guard_position(
    clearance_guard_mode: str,
    lifecycle_active: bool,
    px4_local_position: Sequence[float],
) -> Vector3:
    """Hide pre-ACTIVE positions from an enforce-mode soft guard.

    The collision-confirmation path does not depend on the PX4 position and
    therefore remains active while the lifecycle is waiting or taking off.
    Shadow mode keeps evaluating the hypothetical soft response preflight.
    """
    if str(clearance_guard_mode).strip().lower() != "shadow" and not bool(
        lifecycle_active
    ):
        return (float("nan"), float("nan"), float("nan"))
    return tuple(float(component) for component in px4_local_position)  # type: ignore[return-value]
