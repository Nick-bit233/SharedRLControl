# PX4-local smooth one-shot takeoff design

## Goal

Make the existing one-shot OFFBOARD takeoff enter model control reliably and
smoothly, without changing the Nokov-to-PX4 visual height correction.  A node
restart remains mandatory before a second automatic takeoff attempt.

## Coordinate ownership

The two odometry streams have different jobs and must not be substituted for
each other globally:

- `/nokov/local_position/odom` remains the source for policy observations,
  map/LiDAR alignment, visualization, geofence, and physical-space altitude
  safety.  The existing `vision_z_offset = -0.15 m` in `nokov_node` is retained.
- `/mavros/local_position/odom` supplies the position used to pre-stream an
  OFFBOARD hold, capture the takeoff origin, construct the relative takeoff
  target, track the climb, hold the final altitude, and detect overshoot.
- `/mavros/local_position/velocity_local` supplies ENU/local-frame velocity for
  the vertical-speed settle gate.  MAVROS publishes body-frame linear velocity
  in `/mavros/local_position/odom.twist`, so that field is deliberately not used
  for the settle gate.

The takeoff target is `PX4 local position at the first valid OFFBOARD entry +
takeoff_height`.  Both feedback and target therefore share the same estimator
frame; the constant visual offset cannot create a 0.15 m target/feedback bias.

All three required streams (raw Nokov, PX4 local pose, and PX4 local velocity)
must be fresh before the one-shot session may begin.  Loss of any required
stream during the session follows the existing odometry-fault response.

## Smooth takeoff profile

The immutable final target remains the PX4-local origin plus the configured
height.  A separate commanded target starts at the measured PX4-local origin
on the OFFBOARD transition, preventing the current one-metre position step.

At each lifecycle update the commanded height advances along a monotonic
trapezoidal profile with:

- maximum climb speed: `0.4 m/s`;
- maximum vertical acceleration/deceleration: `0.5 m/s^2`;
- maximum commanded-height lead over measured PX4-local height: `0.25 m`.

The braking-limited desired speed is
`min(max_climb_speed, sqrt(2 * max_vertical_accel * remaining_height))`.
Acceleration is slew-limited per elapsed time.  The command never exceeds the
final target or measured height plus the tracking-error limit.  A local-frame
vertical velocity feed-forward accompanies the position target during TAKEOFF;
horizontal velocity fields remain ignored.

The final-height confirmation timer can start only after the generated command
has reached the immutable final target.  Existing measured-height band,
vertical-speed, timeout, drift, overshoot, collision, and fault-response gates
remain in force.

If a recoverable RC/LiDAR or proximity hold interrupts TAKEOFF, the generator
is rebased to the current PX4-local height with zero feed-forward velocity.  It
resumes from there after recovery instead of jumping to the final target or
integrating elapsed pause time.

## Lifecycle and safety invariants

- Arming remains manual.  The first armed OFFBOARD entry consumes the only
  session and starts the profile.
- Leaving OFFBOARD or disarming terminates the node lifecycle and stops the
  setpoint stream.  Re-entering OFFBOARD cannot start another takeoff until the
  navigator is restarted.
- The final target never changes after the session starts.
- `vision_z_offset = -0.15 m` is not removed or numerically changed.
- Physical geofence and altitude limits remain evaluated in raw Nokov space;
  flight-control tracking limits remain evaluated in PX4-local space.

## Configuration and observability

Add ROS/launch/Compose settings for the two PX4-local topics and the three
profile limits.  Startup logs identify both coordinate sources and profile
limits.  The fake runtime publishes local-frame velocity and models the mixed
position plus velocity-feed-forward setpoint so the ROS integration test can
exercise the new path without hardware.

## Verification

Unit tests cover zero-step start, acceleration/speed/deceleration bounds,
tracking-error pause, pause/resume, final-target immutability, final-profile
settling, and the one-shot terminal invariant.  Launch-contract tests cover the
new topic and profile wiring and protect the `-0.15 m` correction.  The ROS
smoke test verifies a monotonic rising target before ACTIVE and no second
OFFBOARD session.
