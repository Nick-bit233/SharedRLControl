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

The source tree is copied into the image at build time. Rebuild after changing
anything under `src/`, the Dockerfile, or Compose configuration:

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
  post_takeoff_mode:=direct \
  takeoff_height:=1.2 \
  rviz:=true \
  record:=true
```

`real_px4.launch` starts its own guard by default. If a standalone guard is
already running, avoid duplicate node names with:

```bash
roslaunch srlc_real real_px4.launch start_stream_guard:=false
```

### Safety warning

The existing flight behavior is unchanged: `SRLC_AUTO_ARM` and
`SRLC_AUTO_TAKEOFF` still default to true once the required OFFBOARD conditions
are met. Bench tests must explicitly disable both:

```bash
SRLC_AUTO_ARM=false SRLC_AUTO_TAKEOFF=false \
roslaunch srlc_real real_px4.launch record:=false rviz:=false
```

The SRLC launch does not itself switch PX4 into OFFBOARD. Once the pilot or GCS
selects OFFBOARD, the configured navigation behavior applies.

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
```

Other important topics:

- `/nokov/local_position/odom`
- `/nokov/imu/data`
- `/mavros/state`
- `/mavros/rc/in`
- `/mavros/setpoint_raw/local`
- `/srlc/lidar/range_image`
- `/srlc/lidar/min_distance`
- `/tunnel_nav/status`
- `/tunnel_nav/lifecycle_state`

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

For real-flight acceptance, test MAVROS restart, FCU power-cycle recovery, and
a ten-minute soak with zero MAVROS dropped packets, buffer overruns, and parse
errors before reinstalling propellers.
