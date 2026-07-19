"""Detection and error reporting for removed real-flight safety controls."""

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple


LEGACY_SAFETY_BASENAMES = frozenset(
    {
        "enable_safety_stop",
        "safety_min_dist",
        "collision_dist",
    }
)

LEGACY_SAFETY_ENV_NAMES = frozenset(
    LEGACY_SAFETY_BASENAMES
    | {f"srlc_{name}" for name in LEGACY_SAFETY_BASENAMES}
)


@dataclass(frozen=True)
class LegacySafetyUses:
    ros_params: Tuple[str, ...]
    environment: Tuple[str, ...]

    @property
    def found(self):
        return bool(self.ros_params or self.environment)


def find_legacy_safety_config(
    ros_param_names: Sequence[str], environment: Mapping[str, str]
) -> LegacySafetyUses:
    """Find removed controls without interpreting or converting their values."""
    ros_params = []
    for param_name in ros_param_names:
        full_name = str(param_name)
        basename = full_name.rstrip("/").rsplit("/", 1)[-1].lower()
        if basename in LEGACY_SAFETY_BASENAMES:
            ros_params.append(full_name)

    environment_names = []
    for name, value in environment.items():
        if str(name).lower() not in LEGACY_SAFETY_ENV_NAMES:
            continue
        if value is None or not str(value).strip():
            continue
        environment_names.append(str(name))

    return LegacySafetyUses(
        ros_params=tuple(sorted(set(ros_params))),
        environment=tuple(sorted(set(environment_names))),
    )


def legacy_safety_migration_error(uses: LegacySafetyUses) -> str:
    """Return an explicit, non-converting migration error."""
    detected = []
    if uses.ros_params:
        detected.append("ROS params: " + ", ".join(uses.ros_params))
    if uses.environment:
        detected.append("environment: " + ", ".join(uses.environment))
    location_text = "; ".join(detected) if detected else "no legacy controls"
    return (
        "Removed safety controls detected ("
        + location_text
        + "). Delete enable_safety_stop and use enable_proximity_guard; delete "
        "safety_min_dist and configure proximity_enter_clearance plus "
        "proximity_release_clearance; delete collision_dist and configure "
        "collision_confirm_clearance plus collision_immediate_clearance. Values "
        "are not converted automatically."
    )
