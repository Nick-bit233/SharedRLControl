# SRLC ROS1 Real PX4 Runtime

`ros1_real` is the ROS1 Noetic workspace for real PX4/Nokov SRLC experiments.
The existing `../ros1` workspace remains the Gazebo/simulation/batch-test entry.

## Layout

- `src/srlc_real`: real-flight ROS package.
- `src/nokov_uav`: Nokov to MAVROS vision/odom/IMU bridge.
- `src/vrpn_client_ros`: local VRPN client source used by Nokov.
- `Dockerfile` and `docker-compose.real.yml`: single-container real runtime.

## Default Assets

Compose mounts these host directories:

- `${SRLC_CKPT_HOST_DIR:-../ros1/ckpts}` to `/root/real_assets/ckpts:ro`
- `${SRLC_MAP_HOST_DIR:-../ros1/real_maps}` to `/root/real_assets/maps:ro`
- `${SRLC_OUTPUT_HOST_DIR:-./results}` to `/root/real_outputs:rw`

Default files:

- `/root/real_assets/ckpts/checkpoint_minrisk_0610.pt`
- `/root/real_assets/maps/room601/0624_section_resampled6w_ascii_aligned_yaw_m4p50.pcd`

## Build

```bash
cd SharedRLControl/ros1_real
docker compose -f docker-compose.real.yml build real_runtime
```

## Run

By default, `docker compose up` only starts an idle container. It does not run
`roslaunch` automatically.

```bash
cd SharedRLControl/ros1_real
docker compose -f docker-compose.real.yml up real_runtime
```

Enter the container and launch manually:

```bash
docker compose -f docker-compose.real.yml exec real_runtime bash
```

Inside that shell, ROS is already sourced by `/root/.bashrc`; run any launch
variant manually, for example:

```bash
roslaunch srlc_real real_px4.launch \
  start_mavros:=false start_nokov:=false record:=false rviz:=false \
  post_takeoff_mode:=assist takeoff_height:=1.0
```

To automatically run a specific launch command when the container starts, set
`SRLC_CONTAINER_COMMAND` explicitly:

```bash
SRLC_CONTAINER_COMMAND='roslaunch srlc_real real_px4.launch start_mavros:=false start_nokov:=false record:=false rviz:=false' \
docker compose -f docker-compose.real.yml up real_runtime
```

```bash
SRLC_CONTAINER_COMMAND='roslaunch srlc_real real_px4.launch' \
SRLC_POST_TAKEOFF_MODE=assist SRLC_TAKEOFF_HEIGHT=1.2 \
PX4_FCU_URL='udp://:14540@192.168.31.155:14557' \
NOKOV_SERVER=192.168.31.193 NOKOV_TRACKER=soccer \
docker compose -f docker-compose.real.yml up real_runtime
```

ROS XMLRPC defaults to `ROS_MASTER_URI=http://127.0.0.1:11311` and
`ROS_IP=127.0.0.1` so the container can contact its own `roslaunch` server on
any host. For remote ROS tools on another machine, override `ROS_IP` with the
host LAN address that those machines can reach.

## Main Topics

Inputs:

- `/nokov/local_position/odom`
- `/nokov/imu/data`
- `/mavros/state`
- `/mavros/rc/in`

Outputs:

- `/mavros/setpoint_raw/local`
- `/srlc/human_action`
- `/srlc/rc_status`
- `/srlc/lidar/range_image`
- `/srlc/lidar/min_distance`
- `/srlc/lidar/raycast_points`
- `/real_map/cloud`
- `/tunnel_nav/status`
- `/tunnel_nav/lifecycle_state`
- `/tunnel_nav/control_mode`
- `/tunnel_nav/policy_cmd`
- `/tunnel_nav/policy_active`

## Preflight Checks

Inside the container:

```bash
rospack find srlc_real nokov_uav vrpn_client_ros mavros
rostopic hz /nokov/local_position/odom
rostopic echo -n1 /mavros/state
rostopic info /mavros/setpoint_raw/local
rostopic hz /srlc/lidar/range_image
rostopic echo -n1 /srlc/lidar/min_distance
```
