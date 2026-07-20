# SRLC ROS1 Noetic Real-PX4 Runtime

This workspace contains the Dockerized ROS1 Noetic runtime used for real PX4
and Nokov experiments. MAVROS, Nokov, the MAVLink stream guard, and the SRLC
navigation stack have separate launch entry points so they can be started and
stopped independently during bench tests.

## Default deployment

The Compose service uses host networking and these defaults:

```text
PX4_FCU_URL=udp://:14551@192.168.31.201:14550
PX4_GCS_URL=
NOKOV_SERVER=192.168.31.192
NOKOV_PORT=3883
NOKOV_TRACKER=uav_soccer
MAVLINK_LOCAL_POSITION_RATE_HZ=30.0
MAVLINK_ATTITUDE_RATE_HZ=20.0
SRLC_OUTPUT_DIR=/root/real_outputs
```

`PX4_GCS_URL` is empty by default so QGroundControl cannot overwrite stream
rates through the MAVROS GCS bridge. Enable it explicitly only when required.

Compose mounts:

- `${SRLC_CKPT_HOST_DIR:-../ros1/ckpts}` at `/root/real_assets/ckpts:ro`
- `${SRLC_MAP_HOST_DIR:-../ros1/real_maps}` at `/root/real_assets/maps:ro`
- `${SRLC_OUTPUT_HOST_DIR:-./results}` at `/root/real_outputs:rw`

## Build and enter the container

The source tree is copied into the image at build time; it is not bind-mounted
by Compose. Rebuild the release image after a change has passed real-flight
acceptance, or whenever the Dockerfile, dependencies, CMake, messages, or
compiled code change:

```bash
cd /home/nickbit/uav/SharedRLControl/ros1_real
docker compose -f docker-compose.real.yml down
docker compose -f docker-compose.real.yml build --no-cache real_runtime
docker compose -f docker-compose.real.yml up -d real_runtime
docker compose -f docker-compose.real.yml exec real_runtime bash
```

The default container command is `sleep infinity`; no flight process starts
until an operator runs a launch command.

The image intentionally includes the diagnostic packages installed in the
validated test container: ROS Noetic desktop-full, RQt/common plugins, MAVROS
debug symbols, and `net-tools`.

For a pre-release experiment, existing Python, launch, and YAML files can be
copied into a stopped navigation stack without rebuilding the image:

```bash
cd /home/nickbit/uav/SharedRLControl/ros1_real
SRLC_RUNTIME_CONTAINER=ros1_real_v2-real_runtime-1
docker exec "$SRLC_RUNTIME_CONTAINER" \
  pgrep -af '[r]eal_px4.launch|[/]real_navigation_node.py|[/]map_lidar_node.py'
docker cp src/srlc_real/. \
  "$SRLC_RUNTIME_CONTAINER":/root/catkin_ws/src/srlc_real/
```

The catkin Python relay and `rospack` both resolve files from
`/root/catkin_ws/src/srlc_real`, so these changes do not require `catkin_make`.
They are ephemeral and disappear when the container is recreated. Never copy
while the navigator or map LiDAR node is running. Rebuild/recreate instead if
the change affects anything outside those interpreted source/config files.

## Independent launch entry points

The project-owned launch files replace all hand edits under `/opt/ros`:

```bash
# MAVROS only
roslaunch srlc_real mavros_px4.launch

# VRPN client and Nokov bridge only
roslaunch srlc_real nokov.launch

# Critical MAVLink stream recovery only
roslaunch srlc_real mavlink_stream_guard.launch

# SRLC map/navigation/recorder stack; includes the guard, not MAVROS or Nokov
roslaunch srlc_real real_px4.launch
```

Do not copy `soccer.launch` into the MAVROS package and do not edit
`/opt/ros/noetic/share/mavros/launch/px4.launch`. Connection settings belong
to `srlc_real/mavros_px4.launch` and Compose environment variables.

### Link-only bench-test order

With propellers removed:

1. Start `roscore` in one shell.
2. Start `mavlink_stream_guard.launch`; it remains `DISCONNECTED` without an FCU.
3. Start `mavros_px4.launch` and power the FCU.
4. Start `nokov.launch`.
5. Inspect the guard and target topics.

```bash
rostopic echo -n 1 /srlc/mavlink_stream_guard/status
rostopic echo -n 1 /mavros/state
rostopic hz /mavros/local_position/pose
rostopic hz /mavros/imu/data
rostopic hz /vrpn_client_node/uav_soccer/pose
rostopic hz /mavros/vision_pose/pose
```

Expected rates after recovery:

- `/mavros/local_position/pose`: 25-35 Hz
- `/mavros/imu/data`: 15-25 Hz
- `/mavros/vision_pose/pose`: greater than 30 Hz while Nokov is healthy

The guard publishes one of:

```text
DISCONNECTED
WAITING_SERVICE
REQUESTING
VERIFYING
HEALTHY
FAILED
```

On each FCU connection it requests MAVLink `LOCAL_POSITION_NED` (ID 32) at
30 Hz and `ATTITUDE` (ID 30) at 20 Hz. It verifies that new local-pose and IMU
messages arrive, retries at most three times, and stops sending requests after
`FAILED` until the FCU reconnects or the guard restarts. It never changes PX4
mode, arm state, EKF parameters, or setpoints.

To attach QGroundControl through MAVROS for a deliberate test:

```bash
PX4_GCS_URL=udp://@localhost:14550 roslaunch srlc_real mavros_px4.launch
```

Recheck stream rates after attaching it.

## Starting the complete SRLC stack

Start MAVROS and Nokov independently first. Then launch the navigation stack:

```bash
roslaunch srlc_real real_px4.launch \
  post_takeoff_mode:=assist \
  takeoff_height:=1.0 \
  rviz:=true \
  record:=true
```

`real_px4.launch` starts its own guard by default. If a standalone guard is
already running, avoid duplicate node names with:

```bash
roslaunch srlc_real real_px4.launch start_stream_guard:=false
```

### Safety warning

SRLC no longer arms PX4.  The pilot must arm manually and then select OFFBOARD.
The first armed OFFBOARD transition starts exactly one automatic takeoff:

1. The node captures the current PX4-local XYZ and starts the command at that
   exact height; it does not send a one-metre position step.
2. It generates a monotonic climb limited by default to `0.4 m/s` and
   `0.5 m/s^2`, with no more than `0.25 m` command lead over PX4-local feedback.
3. It enters ASSIST only after the generated target reaches the final relative
   height and PX4-local height/vertical speed remain inside the configured band.
4. It then keeps that one PX4-local target z for DIRECT, ASSIST, input-recovery,
   proximity, and fault holds. Live Nokov/PX4 height samples cannot redefine it.
5. Any OFFBOARD loss or disarm after takeoff starts permanently terminates that
   navigator process.  Switching back to OFFBOARD cannot restart it; restart
   `srlc_real_navigator` (normally the full `real_px4.launch`) before another
   flight.

Bench tests must disable automatic takeoff:

```bash
SRLC_AUTO_TAKEOFF=false \
roslaunch srlc_real real_px4.launch record:=false rviz:=false
```

Before arming, verify `/tunnel_nav/lifecycle_state` progresses to `WAIT_ARMED`.
After arming it must report `WAIT_OFFBOARD`; do not select OFFBOARD while it
still reports `WAIT_READY`.

The smooth-profile defaults can be adjusted conservatively with
`SRLC_TAKEOFF_MAX_CLIMB_SPEED`, `SRLC_TAKEOFF_MAX_VERTICAL_ACCEL`, and
`SRLC_TAKEOFF_MAX_TRACKING_ERROR`.  Do not increase all three together on the
first hardware test; inspect the recorded PX4-local tracking error first.

Emergency faults default to requesting `AUTO.LAND`.  If PX4 does not confirm
the mode within two seconds, SRLC holds the last valid XYZ until the pilot
takes over.  Set `SRLC_FAULT_RESPONSE=hold` to skip the automatic landing
request and use the hold fallback directly.

### Proximity hold

The default map model is a `0.20 m` cube/sphere approximation: the PCD is
inflated by `0.10 m` on every axis on a `0.05 m` voxel grid. For the measured
minimum obstacle gap of `0.56 m`, this leaves `0.36 m` of centre travel width,
or `0.18 m` per side when centred.

After the aircraft rises `0.30 m` above its takeoff origin, protection follows
this lifecycle:

1. Collision detection has priority at `safety_distance <= collision_dist`.
2. `PROXIMITY_HOLD` enters at `safety_distance <= proximity_enter_dist` and
   captures one XY target. Its z target is the immutable takeoff altitude after
   TAKEOFF.
3. It remains held until distance is at least `proximity_release_dist`
   continuously for `proximity_release_duration`; falling below the release
   threshold resets that timer.
4. Disabling `enable_proximity_hold` bypasses only this protective hold;
   collision detection remains independent.

`safety_distance` is the minimum of the configured map-ray hits after vehicle
inflation. It is not policy output and is not the raw nearest PCD-point
distance. Recommended starting profiles are:

| Use | Proximity | Enter / release | Clear time | Collision | Notes |
|---|---:|---:|---:|---:|---|
| Dense-map default | on | `0.10 / 0.15 m` | `0.20 s` | on, `0.05 m` | Fits the `0.56 m` gap; theoretical release centring tolerance is about `±0.03 m`. |
| Open-space conservative | on | `0.15 / 0.20 m` | `0.30 s` | on, `0.05 m` | Do not use in the narrowest gap; it may be unable to release there. |
| Narrow-gap contingency | off | n/a | n/a | on, `0.05 m` | Removes `PROXIMITY_HOLD` but retains the hard collision fallback. |
| Bench-only, no map guard | off | n/a | n/a | off | No proximity or collision response; pilot/geofence/input faults still apply. |

Example explicit launch for the current dense map (also overrides stale
environment values in a container created from an older Compose file):

```bash
roslaunch srlc_real real_px4.launch \
  pcd_file:=/root/real_assets/maps/room601/0717_section_resampled_0p05_ascii_aligned_floor_level_z0.pcd \
  map_resolution:=0.05 \
  map_inflate_x:=0.10 map_inflate_y:=0.10 map_inflate_z:=0.10 \
  enable_proximity_hold:=true \
  proximity_enter_dist:=0.10 \
  proximity_release_dist:=0.15 \
  proximity_release_duration:=0.20 \
  enable_collision_detection:=true \
  collision_dist:=0.05
```

To disable only the protective hold in the final contingency profile, add
`enable_proximity_hold:=false`; do not also disable collision detection unless
the test is deliberately operating without any map-distance protection.

## Main data paths

External vision pose:

```text
/vrpn_client_node/uav_soccer/pose
  -> /mavros/vision_pose/pose
  -> PX4 estimator
```

Local estimate returned by PX4:

```text
PX4 LOCAL_POSITION_NED
  -> /mavros/local_position/pose
  -> /mavros/local_position/odom
  -> /mavros/local_position/velocity_local
```

Coordinate ownership is intentional:

- raw `/nokov/local_position/odom` remains the model, map/LiDAR, visualization,
  geofence, and physical-altitude source;
- `/mavros/local_position/odom` and
  `/mavros/local_position/velocity_local` are the feedback used only for
  OFFBOARD prestream, takeoff trajectory, settle detection, and altitude hold.

`nokov_node` still applies `vision_z_offset = -0.15 m` when sending visual pose
to PX4.  The correction represents the chosen physical reference and is not
removed.  Because takeoff origin, final target, and feedback are now all in the
same PX4-local frame, this constant correction no longer appears as a `0.15 m`
takeoff target/measurement mismatch.

Other important topics:

- `/nokov/local_position/odom`
- `/nokov/imu/data`
- `/mavros/state`
- `/mavros/rc/in`
- `/mavros/setpoint_raw/local`
- `/mavros/local_position/velocity_local`
- `/srlc/lidar/range_image`
- `/srlc/lidar/min_distance`
- `/srlc/lidar/min_safety_distance`
- `/tunnel_nav/status`
- `/tunnel_nav/lifecycle_state`
- `/tunnel_nav/effective_mode`
- `/tunnel_nav/session_consumed`
- `/tunnel_nav/fault_reason`
- `/tunnel_nav/policy_active`

`HEALTHY` confirms stream delivery, not estimator correctness. Before flight,
also check that position and quaternion values are finite and that PX4 estimator
status reports valid attitude, horizontal/vertical velocity, and position.

## Non-hardware smoke test

`dry_run_px4.launch` starts a fake MAVROS/PX4 runtime, including the message
interval service and the topics required by the stream guard:

```bash
roslaunch srlc_real dry_run_px4.launch \
  rviz:=false \
  record:=false \
  post_takeoff_mode:=direct \
  fake_forward_speed:=1.0 \
  motion_after:=4.0
```

Useful checks:

```bash
rostopic echo -n 1 /srlc/mavlink_stream_guard/status
rostopic echo /tunnel_nav/lifecycle_state
rostopic echo /tunnel_nav/status
rostopic echo /mavros/setpoint_raw/local
rostopic echo /nokov/local_position/odom
rostopic echo /mavros/local_position/odom
rostopic echo /mavros/local_position/velocity_local
```

## Verification after rebuilding

Inside the rebuilt container:

```bash
rospack find srlc_real
rospack find nokov_uav
rospack find vrpn_client_ros
rospack find mavros
dpkg-query -W ros-noetic-desktop-full ros-noetic-rqt \
  ros-noetic-rqt-common-plugins ros-noetic-mavros-dbgsym \
  ros-noetic-mavros-extras-dbgsym net-tools
dpkg -V ros-noetic-mavros
catkin_make -DCATKIN_ENABLE_TESTING=ON run_tests_srlc_real
catkin_test_results /root/catkin_ws/build/test_results
```

`dpkg -V ros-noetic-mavros` must print no modified package-owned files. The
project no longer requires `/opt/ros/noetic/share/mavros/soccer.launch`.

## Troubleshooting

- `DISCONNECTED`: check FCU power, `PX4_FCU_URL`, UDP ports, and LAN routing.
- `WAITING_SERVICE`: MAVROS is absent or `/mavros/set_message_interval` is unavailable.
- `VERIFYING`: requests succeeded but fresh pose/IMU data have not both arrived.
- `FAILED`: inspect MAVROS diagnostics and restart/reconnect only after fixing the link.
- No vision pose: check `nokov_node`, `NOKOV_SERVER`, `NOKOV_TRACKER`, and
  `/vrpn_client_node/uav_soccer/pose`.
- No local position with guard `HEALTHY`: inspect PX4 EKF status rather than
  repeatedly changing message intervals.
- Lifecycle remains `WAIT_READY` with valid Nokov data: verify both
  `/mavros/local_position/odom` and `/mavros/local_position/velocity_local` are
  fresh; the latter must be the local-frame topic, not `odom.twist`, which
  MAVROS publishes in the body frame.

For real-flight acceptance, test MAVROS restart, FCU power-cycle recovery, and
a ten-minute soak with zero MAVROS dropped packets, buffer overruns, and parse
errors before reinstalling propellers.
