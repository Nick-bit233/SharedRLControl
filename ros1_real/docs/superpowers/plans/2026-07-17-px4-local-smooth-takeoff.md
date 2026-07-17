# PX4-local smooth one-shot takeoff implementation plan

**Goal:** Replace the one-step OFFBOARD takeoff target with a bounded PX4-local
trajectory while preserving raw Nokov use and the `-0.15 m` vision correction.

**Architecture:** Keep the pure lifecycle core responsible for the immutable
final target and generated takeoff command.  The ROS adapter supplies PX4-local
pose/local velocity to that core, retains raw Nokov odometry for the model and
physical safety, and publishes mixed position plus vertical-velocity
feed-forward `PositionTarget` messages.

**Stack:** ROS Noetic, rospy, MAVROS 1.20.1, PX4 v1.14.2, Python unittest/rostest.

## Task 1: Specify the new lifecycle behavior with failing tests

**Files:**

- Modify: `src/srlc_real/test/test_one_shot_flight.py`
- Modify: `src/srlc_real/test/test_one_shot_flight_ros.py`

Add assertions that the first OFFBOARD command equals the measured origin, the
generated height and velocity obey profile bounds, tracking lag pauses the
command, recovery resumes without a jump, and ACTIVE occurs only after the
profile reaches its final target.  Update the ROS test from a constant-target
contract to a monotonic-ramp contract.  Run the unit test and retain its
expected failures before production changes.

## Task 2: Implement the pure smooth-profile lifecycle

**Files:**

- Modify: `src/srlc_real/scripts/srlc_real_deployment/one_shot_flight.py`

Add profile configuration, commanded target/velocity state, bounded profile
integration, tracking-error limiting, pause/rebase behavior, and optional
vertical-velocity output in `FlightDecision`.  Keep the final target immutable
and all existing fault/session transitions intact.  Run lifecycle unit tests.

## Task 3: Separate raw Nokov data from PX4-local control feedback

**Files:**

- Modify: `src/srlc_real/scripts/real_navigation_node.py`

Subscribe independently to PX4 local odometry and local velocity.  Build the
lifecycle snapshot from PX4-local position/velocity while requiring the raw
Nokov stream to remain fresh.  Preserve raw Nokov use for observations,
map/geofence logic, and visualization.  Publish vertical feed-forward only for
takeoff decisions; use PX4-local fallbacks for control holds and final altitude.

## Task 4: Wire configuration and the hardware-free runtime

**Files:**

- Modify: `src/srlc_real/launch/real_px4.launch`
- Modify: `src/srlc_real/launch/dry_run_px4.launch`
- Modify: `src/srlc_real/cfg/tunnel/real_nav_px4.yaml`
- Modify: `docker-compose.real.yml`
- Modify: `src/srlc_real/scripts/srlc_real_fake_runtime_node.py`
- Modify: `src/srlc_real/test/test_launch_contracts.py`

Expose PX4-local topics and conservative profile defaults through every
deployment layer.  Publish fake `/mavros/local_position/velocity_local` and
support position plus velocity feed-forward.  Protect the existing Nokov
height correction with a contract test.  Run launch-contract tests.

## Task 5: Document and verify the complete change

**Files:**

- Modify: `README.md`

Document coordinate ownership, profile defaults, manual-arm/one-shot behavior,
and hardware verification commands.  Run Python compilation, all local unit
tests, launch XML parsing, Compose rendering when available, and the ROS smoke
test in the running test container when its environment permits.  Review the
final diff to confirm that `vision_z_offset = -0.15 m` is unchanged.
