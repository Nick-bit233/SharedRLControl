#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from srlc_real_deployment.clearance_guard import (  # noqa: E402
    ClearanceGuard,
    ClearanceGuardConfig,
    ClearanceState,
    project_velocity_away,
)


def _update(guard, **overrides):
    values = {
        "now": 1.0,
        "source_stamp": 1.0,
        "valid": True,
        "surface_clearance": 1.0,
        "escape_direction": (1.0, 0.0, 0.0),
        "human_velocity_world": (0.0, 0.0, 0.0),
        "px4_local_position": (1.0, 2.0, 3.0),
    }
    values.update(overrides)
    return guard.update(**values)


def test_defaults_keep_soft_proximity_off_while_hard_collision_stays_on():
    config = ClearanceGuardConfig()
    guard = ClearanceGuard(config)

    assert config.proximity_enabled is False
    assert config.proximity_enter_clearance == 0.10
    assert config.proximity_release_clearance == 0.15
    assert config.proximity_release_duration == 0.20
    assert config.escape_dot_threshold == 0.05
    assert config.collision_clearance == 0.02
    assert config.collision_confirm_samples == 2
    assert config.immediate_collision_clearance == -0.03
    assert {
        ClearanceState.NORMAL,
        ClearanceState.PROXIMITY_HOLD,
        ClearanceState.PROXIMITY_ESCAPE,
        ClearanceState.COLLISION,
    } == {
        "NORMAL",
        "PROXIMITY_HOLD",
        "PROXIMITY_ESCAPE",
        "COLLISION",
    }
    first = _update(
        guard,
        now=1.0,
        source_stamp=1.0,
        surface_clearance=0.01,
    )
    second = _update(
        guard,
        now=1.1,
        source_stamp=1.1,
        surface_clearance=0.01,
    )

    assert first.state == ClearanceState.NORMAL
    assert first.hold_position is None
    assert second.state == ClearanceState.COLLISION


def test_collision_threshold_is_inclusive_and_needs_two_increasing_stamps():
    guard = ClearanceGuard()

    first = _update(
        guard,
        now=2.0,
        source_stamp=2.0,
        surface_clearance=0.02,
    )
    second = _update(
        guard,
        now=2.1,
        source_stamp=2.1,
        surface_clearance=0.02,
    )

    assert first.state == ClearanceState.NORMAL
    assert second.state == ClearanceState.COLLISION


def test_duplicate_stamp_neither_confirms_nor_resets_pending_collision():
    guard = ClearanceGuard()

    first = _update(
        guard,
        now=3.0,
        source_stamp=3.0,
        surface_clearance=0.02,
    )
    duplicate = _update(
        guard,
        now=3.05,
        source_stamp=3.0,
        surface_clearance=1.0,
    )
    confirmed = _update(
        guard,
        now=3.1,
        source_stamp=3.1,
        surface_clearance=0.02,
    )

    assert first.state == ClearanceState.NORMAL
    assert duplicate.state == ClearanceState.NORMAL
    assert confirmed.state == ClearanceState.COLLISION


def test_distinct_valid_recovery_resets_pending_collision_confirmation():
    guard = ClearanceGuard()

    _update(guard, now=4.0, source_stamp=4.0, surface_clearance=0.02)
    recovered = _update(
        guard,
        now=4.1,
        source_stamp=4.1,
        surface_clearance=0.0200001,
    )
    next_low = _update(
        guard,
        now=4.2,
        source_stamp=4.2,
        surface_clearance=0.02,
    )

    assert recovered.state == ClearanceState.NORMAL
    assert next_low.state == ClearanceState.NORMAL


def test_confirmed_collision_remains_visible_until_distinct_valid_recovery():
    guard = ClearanceGuard()
    _update(guard, now=5.0, source_stamp=5.0, surface_clearance=0.02)
    confirmed = _update(
        guard,
        now=5.1,
        source_stamp=5.1,
        surface_clearance=0.02,
    )
    duplicate_recovery = _update(
        guard,
        now=5.15,
        source_stamp=5.1,
        surface_clearance=1.0,
    )
    invalid_recovery = _update(
        guard,
        now=5.2,
        source_stamp=5.2,
        valid=False,
        surface_clearance=1.0,
    )
    recovered = _update(
        guard,
        now=5.3,
        source_stamp=5.3,
        surface_clearance=1.0,
    )

    assert confirmed.state == ClearanceState.COLLISION
    assert duplicate_recovery.state == ClearanceState.COLLISION
    assert invalid_recovery.state == ClearanceState.COLLISION
    assert recovered.state == ClearanceState.NORMAL


def test_immediate_penetration_threshold_is_inclusive_in_one_frame():
    guard = ClearanceGuard()

    result = _update(
        guard,
        now=6.0,
        source_stamp=6.0,
        surface_clearance=-0.03,
    )

    assert result.state == ClearanceState.COLLISION


@pytest.mark.parametrize(
    "bad_overrides",
    [
        {"valid": False},
        {"surface_clearance": math.nan},
        {"surface_clearance": math.inf},
        {"source_stamp": math.nan},
        {"escape_direction": (math.nan, 0.0, 0.0)},
        {"now": 7.1, "source_stamp": 7.2},
        {"now": 7.1, "source_stamp": 6.9},
        {"now": 8.0, "source_stamp": 7.1},
    ],
    ids=[
        "invalid",
        "nan-clearance",
        "infinite-clearance",
        "nan-stamp",
        "nan-direction",
        "future",
        "backwards",
        "stale",
    ],
)
def test_unusable_samples_neither_increment_nor_reset_pending_collision(
    bad_overrides,
):
    guard = ClearanceGuard(ClearanceGuardConfig(sample_timeout=0.3))
    _update(guard, now=7.0, source_stamp=7.0, surface_clearance=0.01)

    unusable_values = {
        "now": 7.1,
        "source_stamp": 7.1,
        "surface_clearance": 0.01,
    }
    unusable_values.update(bad_overrides)
    unusable = _update(guard, **unusable_values)
    confirmed = _update(
        guard,
        now=8.1,
        source_stamp=8.1,
        surface_clearance=0.01,
    )

    assert unusable.state == ClearanceState.NORMAL
    assert confirmed.state == ClearanceState.COLLISION


def test_proximity_enter_threshold_is_inclusive_and_captures_local_hold():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))

    result = _update(
        guard,
        now=9.0,
        source_stamp=9.0,
        surface_clearance=0.10,
        px4_local_position=(-1.0, 2.5, 0.8),
    )

    assert result.state == ClearanceState.PROXIMITY_HOLD
    assert result.hold_position == (-1.0, 2.5, 0.8)
    assert result.escape_direction == (1.0, 0.0, 0.0)


def test_escape_dot_threshold_is_inclusive_and_updates_backup_hold():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))
    _update(
        guard,
        now=10.0,
        source_stamp=10.0,
        surface_clearance=0.10,
        px4_local_position=(0.0, 0.0, 1.0),
    )

    escaping = _update(
        guard,
        now=10.1,
        source_stamp=10.1,
        surface_clearance=0.11,
        escape_direction=(2.0, 0.0, 0.0),
        human_velocity_world=(0.05, 3.0, 0.0),
        px4_local_position=(0.2, 0.0, 1.0),
    )
    holding = _update(
        guard,
        now=10.2,
        source_stamp=10.2,
        surface_clearance=0.11,
        human_velocity_world=(0.049999, 0.0, 0.0),
        px4_local_position=(0.3, 0.0, 1.0),
    )

    assert escaping.state == ClearanceState.PROXIMITY_ESCAPE
    assert escaping.hold_position == (0.2, 0.0, 1.0)
    assert escaping.escape_direction == (1.0, 0.0, 0.0)
    assert holding.state == ClearanceState.PROXIMITY_HOLD
    assert holding.hold_position == (0.2, 0.0, 1.0)


def test_release_requires_clearance_strictly_above_threshold_for_full_duration():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))
    _update(guard, now=11.0, source_stamp=11.0, surface_clearance=0.10)

    exact_release = _update(
        guard,
        now=11.1,
        source_stamp=11.1,
        surface_clearance=0.15,
    )
    release_started = _update(
        guard,
        now=11.2,
        source_stamp=11.2,
        surface_clearance=0.150001,
    )
    too_soon = _update(
        guard,
        now=11.399,
        source_stamp=11.399,
        surface_clearance=0.150001,
    )
    released = _update(
        guard,
        now=11.4,
        source_stamp=11.4,
        surface_clearance=0.150001,
    )

    assert exact_release.state == ClearanceState.PROXIMITY_HOLD
    assert release_started.state == ClearanceState.PROXIMITY_HOLD
    assert too_soon.state == ClearanceState.PROXIMITY_HOLD
    assert released.state == ClearanceState.NORMAL
    assert released.hold_position is None


def test_dip_below_release_threshold_restarts_continuous_release_window():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))
    _update(guard, now=12.0, source_stamp=12.0, surface_clearance=0.10)
    _update(guard, now=12.1, source_stamp=12.1, surface_clearance=0.16)
    interrupted = _update(
        guard,
        now=12.2,
        source_stamp=12.2,
        surface_clearance=0.15,
    )
    restarted = _update(
        guard,
        now=12.3,
        source_stamp=12.3,
        surface_clearance=0.16,
    )
    too_soon = _update(
        guard,
        now=12.49,
        source_stamp=12.49,
        surface_clearance=0.16,
    )
    released = _update(
        guard,
        now=12.5,
        source_stamp=12.5,
        surface_clearance=0.16,
    )

    assert interrupted.state == ClearanceState.PROXIMITY_HOLD
    assert restarted.state == ClearanceState.PROXIMITY_HOLD
    assert too_soon.state == ClearanceState.PROXIMITY_HOLD
    assert released.state == ClearanceState.NORMAL


def test_unusable_soft_sample_forces_hold_and_restarts_release_window():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))
    _update(guard, now=13.0, source_stamp=13.0, surface_clearance=0.10)
    escaping = _update(
        guard,
        now=13.1,
        source_stamp=13.1,
        surface_clearance=0.16,
        human_velocity_world=(0.1, 0.0, 0.0),
        px4_local_position=(1.1, 2.0, 3.0),
    )
    unusable = _update(
        guard,
        now=13.25,
        source_stamp=13.1,
        surface_clearance=1.0,
        human_velocity_world=(0.1, 0.0, 0.0),
        px4_local_position=(1.2, 2.0, 3.0),
    )
    restarted = _update(
        guard,
        now=13.3,
        source_stamp=13.3,
        surface_clearance=0.16,
    )
    too_soon = _update(
        guard,
        now=13.49,
        source_stamp=13.49,
        surface_clearance=0.16,
    )
    released = _update(
        guard,
        now=13.5,
        source_stamp=13.5,
        surface_clearance=0.16,
    )

    assert escaping.state == ClearanceState.PROXIMITY_ESCAPE
    assert unusable.state == ClearanceState.PROXIMITY_HOLD
    assert unusable.hold_position == (1.1, 2.0, 3.0)
    assert restarted.state == ClearanceState.PROXIMITY_HOLD
    assert too_soon.state == ClearanceState.PROXIMITY_HOLD
    assert released.state == ClearanceState.NORMAL


def test_collision_interrupts_soft_release_continuity():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))
    _update(guard, now=14.0, source_stamp=14.0, surface_clearance=0.10)
    _update(guard, now=14.1, source_stamp=14.1, surface_clearance=0.16)

    collision = _update(
        guard,
        now=14.15,
        source_stamp=14.15,
        surface_clearance=-0.03,
    )
    recovery = _update(
        guard,
        now=14.4,
        source_stamp=14.4,
        surface_clearance=0.16,
    )

    assert collision.state == ClearanceState.COLLISION
    assert recovery.state == ClearanceState.PROXIMITY_HOLD


def test_invalid_soft_only_vectors_fail_safe_without_masking_hard_collision():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))
    first = _update(
        guard,
        now=15.0,
        source_stamp=15.0,
        surface_clearance=0.01,
        human_velocity_world=(math.nan, 0.0, 0.0),
        px4_local_position=(math.nan, 0.0, 0.0),
    )
    second = _update(
        guard,
        now=15.1,
        source_stamp=15.1,
        surface_clearance=0.01,
        human_velocity_world=(math.nan, 0.0, 0.0),
        px4_local_position=(math.nan, 0.0, 0.0),
    )

    assert first.state == ClearanceState.NORMAL
    assert second.state == ClearanceState.COLLISION


def test_velocity_projection_removes_only_component_toward_obstacle():
    root_half = math.sqrt(0.5)
    direction = (root_half, root_half, 0.0)
    velocity = (-2.0, 1.0, 3.0)
    dot = sum(v * n for v, n in zip(velocity, direction))
    expected = tuple(v - min(0.0, dot) * n for v, n in zip(velocity, direction))

    projected = project_velocity_away(velocity, direction)
    already_away = project_velocity_away((2.0, 1.0, -3.0), direction)

    assert projected == expected
    assert sum(v * n for v, n in zip(projected, direction)) == pytest.approx(0.0)
    assert already_away == (2.0, 1.0, -3.0)
