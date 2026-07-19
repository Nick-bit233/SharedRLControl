#!/usr/bin/env python3

import math
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from srlc_real_deployment.clearance_guard import (  # noqa: E402
    ClearanceGuard,
    ClearanceGuardConfig,
    ClearanceState,
    clamp_px4_velocity,
    constrain_escape_velocity,
    finalize_px4_escape_velocity,
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


def test_unusable_samples_neither_increment_nor_reset_pending_collision():
    bad_override_cases = [
        {"valid": False},
        {"surface_clearance": math.nan},
        {"surface_clearance": math.inf},
        {"source_stamp": math.nan},
        {"escape_direction": (math.nan, 0.0, 0.0)},
        {"now": 7.1, "source_stamp": 7.2},
        {"now": 7.1, "source_stamp": 6.9},
        {"now": 8.0, "source_stamp": 7.1},
    ]
    for bad_overrides in bad_override_cases:
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

        assert unusable.state == ClearanceState.NORMAL, bad_overrides
        assert confirmed.state == ClearanceState.COLLISION, bad_overrides


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


def test_fresh_duplicate_remains_soft_usable_and_preserves_release_window():
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
    duplicate = _update(
        guard,
        now=13.25,
        source_stamp=13.1,
        surface_clearance=0.16,
        human_velocity_world=(0.1, 0.0, 0.0),
        px4_local_position=(1.2, 2.0, 3.0),
    )
    released = _update(
        guard,
        now=13.3,
        source_stamp=13.1,
        surface_clearance=0.16,
        human_velocity_world=(0.1, 0.0, 0.0),
        px4_local_position=(1.3, 2.0, 3.0),
    )

    assert escaping.state == ClearanceState.PROXIMITY_ESCAPE
    assert duplicate.state == ClearanceState.PROXIMITY_ESCAPE
    assert duplicate.hold_position == (1.2, 2.0, 3.0)
    assert released.state == ClearanceState.NORMAL


def test_backward_sample_forces_active_escape_to_captured_hold():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))
    _update(guard, now=13.5, source_stamp=13.5, surface_clearance=0.10)
    escaping = _update(
        guard,
        now=13.6,
        source_stamp=13.6,
        surface_clearance=0.11,
        human_velocity_world=(0.1, 0.0, 0.0),
        px4_local_position=(1.1, 2.0, 3.0),
    )
    backward = _update(
        guard,
        now=13.7,
        source_stamp=13.55,
        surface_clearance=0.11,
        human_velocity_world=(0.1, 0.0, 0.0),
        px4_local_position=(1.2, 2.0, 3.0),
    )

    assert escaping.state == ClearanceState.PROXIMITY_ESCAPE
    assert backward.state == ClearanceState.PROXIMITY_HOLD
    assert backward.hold_position == (1.1, 2.0, 3.0)
    assert backward.escape_direction is None


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


def test_nonfinite_human_velocity_enters_and_captures_proximity_hold():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))

    result = _update(
        guard,
        now=15.0,
        source_stamp=15.0,
        surface_clearance=0.10,
        human_velocity_world=(math.nan, 0.0, 0.0),
        px4_local_position=(-0.5, 0.25, 1.2),
    )

    assert result.state == ClearanceState.PROXIMITY_HOLD
    assert result.hold_position == (-0.5, 0.25, 1.2)
    assert result.escape_direction == (1.0, 0.0, 0.0)


def test_nonfinite_human_velocity_does_not_break_release_continuity():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))
    entered = _update(
        guard,
        now=15.5,
        source_stamp=15.5,
        surface_clearance=0.10,
        px4_local_position=(-0.5, 0.25, 1.2),
    )
    release_started = _update(
        guard,
        now=15.6,
        source_stamp=15.6,
        surface_clearance=0.16,
        human_velocity_world=(math.nan, 0.0, 0.0),
        px4_local_position=(-0.5, 0.25, 1.2),
    )
    released = _update(
        guard,
        now=15.8,
        source_stamp=15.8,
        surface_clearance=0.16,
        human_velocity_world=(math.nan, 0.0, 0.0),
        px4_local_position=(-0.5, 0.25, 1.2),
    )

    assert entered.state == ClearanceState.PROXIMITY_HOLD
    assert release_started.state == ClearanceState.PROXIMITY_HOLD
    assert released.state == ClearanceState.NORMAL


def test_nonfinite_px4_position_cannot_capture_hold_but_hard_collision_still_runs():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))
    first = _update(
        guard,
        now=16.0,
        source_stamp=16.0,
        surface_clearance=0.01,
        px4_local_position=(math.nan, 0.0, 0.0),
    )
    second = _update(
        guard,
        now=16.1,
        source_stamp=16.1,
        surface_clearance=0.01,
        px4_local_position=(math.nan, 0.0, 0.0),
    )

    assert first.state == ClearanceState.NORMAL
    assert second.state == ClearanceState.COLLISION


def test_nonfinite_px4_position_falls_back_to_existing_hold():
    guard = ClearanceGuard(ClearanceGuardConfig(proximity_enabled=True))
    entered = _update(
        guard,
        now=17.0,
        source_stamp=17.0,
        surface_clearance=0.10,
        px4_local_position=(0.1, 0.2, 0.3),
    )
    unusable_position = _update(
        guard,
        now=17.1,
        source_stamp=17.1,
        surface_clearance=0.10,
        human_velocity_world=(0.1, 0.0, 0.0),
        px4_local_position=(math.nan, 0.0, 0.0),
    )

    assert entered.state == ClearanceState.PROXIMITY_HOLD
    assert unusable_position.state == ClearanceState.PROXIMITY_HOLD
    assert unusable_position.hold_position == (0.1, 0.2, 0.3)


def test_velocity_projection_removes_only_component_toward_obstacle():
    root_half = math.sqrt(0.5)
    direction = (root_half, root_half, 0.0)
    velocity = (-2.0, 1.0, 3.0)
    dot = sum(v * n for v, n in zip(velocity, direction))
    expected = tuple(v - min(0.0, dot) * n for v, n in zip(velocity, direction))

    projected = project_velocity_away(velocity, direction)
    already_away = project_velocity_away((2.0, 1.0, -3.0), direction)

    assert projected == expected
    assert math.isclose(
        sum(v * n for v, n in zip(projected, direction)),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert already_away == (2.0, 1.0, -3.0)


def test_locked_z_escape_constraint_uses_actual_horizontal_half_space():
    root_half = math.sqrt(0.5)
    direction = (root_half, 0.0, root_half)

    constrained = finalize_px4_escape_velocity(
        (-1.0, 0.0, 0.0),
        direction,
        lock_z=True,
        max_xy_speed=0.5,
        max_z_speed=0.3,
    )

    assert constrained[2] == 0.0
    assert math.hypot(constrained[0], constrained[1]) <= 0.5
    assert sum(v * n for v, n in zip(constrained, direction)) >= -1e-12


def test_unlocked_escape_constraint_uniformly_scales_mixed_axis_projection():
    direction_raw = (0.06, 0.08, -1.0)
    direction_norm = math.sqrt(sum(value * value for value in direction_raw))
    direction = tuple(value / direction_norm for value in direction_raw)
    tangent_velocity = (6.0, 8.0, 1.0)
    model_velocity = tuple(
        value - 4.0 * normal
        for value, normal in zip(tangent_velocity, direction)
    )

    constrained = constrain_escape_velocity(
        model_velocity,
        direction,
        lock_z=False,
        max_xy_speed=0.5,
        max_z_speed=0.3,
    )
    independently_clamped = (0.3, 0.4, 0.3)

    assert sum(
        value * normal
        for value, normal in zip(independently_clamped, direction)
    ) < 0.0
    assert sum(v * n for v, n in zip(constrained, direction)) >= -1e-12
    assert math.hypot(constrained[0], constrained[1]) <= 0.5 + 1e-12
    assert abs(constrained[2]) <= 0.3 + 1e-12
    for actual, expected in zip(constrained, (0.3, 0.4, 0.05)):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def test_post_clamp_recheck_restores_actual_px4_escape_half_space():
    direction_raw = (0.12, 0.16, -1.0)
    direction_norm = math.sqrt(sum(value * value for value in direction_raw))
    direction = tuple(value / direction_norm for value in direction_raw)
    tangent_velocity = (3.0, 4.0, 1.0)
    model_velocity = tuple(
        value - 4.0 * normal
        for value, normal in zip(tangent_velocity, direction)
    )
    max_xy_speed = 10.0
    max_z_speed = 0.3

    constrained = constrain_escape_velocity(
        model_velocity,
        direction,
        lock_z=False,
        max_xy_speed=max_xy_speed,
        max_z_speed=max_z_speed,
    )
    legacy_clamped = clamp_px4_velocity(
        constrained,
        max_xy_speed=max_xy_speed,
        max_z_speed=max_z_speed,
    )
    final_velocity = finalize_px4_escape_velocity(
        model_velocity,
        direction,
        lock_z=False,
        max_xy_speed=max_xy_speed,
        max_z_speed=max_z_speed,
    )

    assert sum(v * n for v, n in zip(legacy_clamped, direction)) < -1e-9
    assert sum(v * n for v, n in zip(final_velocity, direction)) >= -1e-12
    assert math.hypot(final_velocity[0], final_velocity[1]) <= max_xy_speed
    assert abs(final_velocity[2]) <= max_z_speed


def test_final_escape_payload_is_float32_fixed_point_inside_half_space():
    cases = (
        (
            False,
            (-0.6370175412881358, 0.9989337332224479, -0.641218820288852),
            (3.846768111855119, -3.3752607770491423, 3.8992950511829445),
        ),
        (
            True,
            (0.16191838420990368, 0.1611201721950115, -0.1851560248140547),
            (-2.4231895445436242, 2.1561660425576914, -3.4810610469877314),
        ),
    )

    for lock_z, direction, model_velocity in cases:
        final_velocity = finalize_px4_escape_velocity(
            model_velocity,
            direction,
            lock_z=lock_z,
            max_xy_speed=0.5,
            max_z_speed=0.3,
        )
        actual_payload = clamp_px4_velocity(
            final_velocity,
            max_xy_speed=0.5,
            max_z_speed=0.3,
        )
        if lock_z:
            horizontal_norm = math.hypot(direction[0], direction[1])
            effective_direction = (
                direction[0] / horizontal_norm,
                direction[1] / horizontal_norm,
                0.0,
            )
        else:
            direction_norm = math.sqrt(sum(value * value for value in direction))
            effective_direction = tuple(
                value / direction_norm for value in direction
            )

        assert actual_payload == final_velocity
        assert sum(
            value * normal
            for value, normal in zip(actual_payload, effective_direction)
        ) >= 0.0
        assert math.hypot(actual_payload[0], actual_payload[1]) <= 0.5
        assert abs(actual_payload[2]) <= 0.3
        if lock_z:
            assert actual_payload[2] == 0.0


def test_locked_z_escape_constraint_rejects_vertical_escape_direction():
    try:
        finalize_px4_escape_velocity(
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 1.0),
            lock_z=True,
            max_xy_speed=0.5,
            max_z_speed=0.3,
        )
    except ValueError as exc:
        assert "horizontal" in str(exc)
    else:
        raise AssertionError("vertical locked-Z escape direction must be rejected")
