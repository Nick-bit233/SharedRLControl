1. 容器状态与默认配置

宿主机进入容器：

docker exec -it ros1_real-real_runtime-1 bash

容器内工作区：

source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

rospack find srlc_real
rospack find mavros
rospack find nokov_uav
rospack find vrpn_client_ros

当前默认环境：

PX4_FCU_URL=udp://:14540@192.168.31.155:14557
NOKOV_SERVER=192.168.31.193
NOKOV_PORT=3883
NOKOV_TRACKER=soccer
SRLC_CHECKPOINT=/root/real_assets/ckpts/checkpoint_minrisk_0610.pt
SRLC_PCD_FILE=/root/real_assets/maps/room601/0624_section_resampled6w_ascii_aligned_yaw_m4p50_floor_level_z0.pcd
SRLC_POST_TAKEOFF_MODE=direct
SRLC_TAKEOFF_HEIGHT=1.2
SRLC_MAX_XY_SPEED=0.5
SRLC_MAX_Z_SPEED=0.3
SRLC_SAFETY_MIN_DIST=0.25
SRLC_COLLISION_DIST=0.15
SRLC_OUTPUT_DIR=/root/real_outputs

宿主机挂载关系：

ros1/ckpts      -> /root/real_assets/ckpts:ro
ros1/real_maps  -> /root/real_assets/maps:ro
ros1_real/results -> /root/real_outputs:rw

2. 启动真机链路

先确认 launch 会启动哪些节点：

roslaunch --nodes srlc_real real_px4.launch

默认会启动：

/mavros
/vrpn_client_node
/nokov_node
/map_lidar_node
/real_map_publisher
/srlc_alignment_viz_node
/rc_input_node
/srlc_real_navigator
/srlc_real_recorder

正式启动 direct 模式：

roslaunch srlc_real real_px4.launch \
post_takeoff_mode:=direct \
takeoff_height:=1.2 \
rviz:=false \
record:=true

启动 assist/RL 模式：

roslaunch srlc_real real_px4.launch \
post_takeoff_mode:=assist \
takeoff_height:=1.2 \
rviz:=false \
record:=true

如果现场已经单独启动了 MAVROS 或 NOKOV，避免重复启动：

roslaunch srlc_real real_px4.launch \
start_mavros:=false \
start_nokov:=false \
post_takeoff_mode:=direct

注意：该 launch 不会主动切 PX4 到 OFFBOARD，但一旦飞手切入 OFFBOARD，节点会自动请求 arm，并按起飞前 Nokov 位置锁定 x/y、按 当前 z +
takeoff_height 起飞。

3. 飞行前检查

MAVROS：

rostopic echo -n 1 /mavros/state
rostopic hz /mavros/rc/in
rostopic echo -n 1 /mavros/rc/in
rostopic info /mavros/setpoint_raw/local

期望：

/mavros/state.connected = True
/mavros/rc/in 有稳定输出
/mavros/setpoint_raw/local 有 MAVROS subscriber

NOKOV：

rostopic hz /vrpn_client_node/soccer/pose
rostopic hz /nokov/local_position/odom
rostopic echo -n 1 /nokov/local_position/odom
rostopic hz /nokov/imu/data

SRLC 输入输出：

rostopic hz /srlc/lidar/range_image
rostopic echo -n 1 /srlc/lidar/min_distance
rostopic echo /srlc/rc_status
rostopic echo /tunnel_nav/lifecycle_state
rostopic echo /tunnel_nav/status
rostopic echo /tunnel_nav/control_mode
rostopic echo /tunnel_nav/policy_active
rostopic echo /mavros/setpoint_raw/local

未切 OFFBOARD 前应看到：

/tunnel_nav/lifecycle_state: WAIT_OFFBOARD
/tunnel_nav/policy_active: False
/tunnel_nav/status: PX4_NOT_OFFBOARD 或 WAIT_OFFBOARD
setpoint 速度为 0 或锁高 hold

切入 OFFBOARD 后，正常状态流转：

WAIT_ARMED -> TAKEOFF_CLIMB -> TAKEOFF_SETTLE -> ACTIVE

direct 模式下，摇杆超过死区后：

/tunnel_nav/status 包含 DIRECT
/tunnel_nav/policy_active = False
/mavros/setpoint_raw/local 跟随 RC 速度输入

assist 模式下，摇杆超过死区且 LiDAR/odom/RC 都正常后：

/tunnel_nav/status 包含 ASSIST
/tunnel_nav/policy_active = True
/tunnel_nav/policy_cmd 有策略输出

4. 地图与坐标对齐

当前 PCD 地图由 SRLC_PCD_FILE 指定。启动时可覆盖：

roslaunch srlc_real real_px4.launch \
pcd_file:=/root/real_assets/maps/room601/0624_section_resampled6w_ascii_aligned_yaw_m4p50.pcd \
map_origin_x:=0.0 \
map_origin_y:=0.0 \
map_origin_z:=0.0 \
map_yaw_deg:=0.0

RViz 调试：

roslaunch srlc_real real_px4.launch rviz:=true

重点看这些 topic：

/real_map/cloud
/srlc/alignment/odom_map
/srlc/alignment/markers
/srlc/lidar/raycast_points
/srlc/lidar/min_distance

验收标准：无人机 marker 与 PCD 中真实起飞位置重合，前进方向与实验方向一致，/srlc/lidar/min_distance 随位置变化合理。

5. 常见故障定位

MAVROS_NOT_CONNECTED：

rostopic echo -n 1 /mavros/state
echo $PX4_FCU_URL

检查 PX4 地址、端口、同网段和 MAVLink 输出。

PX4_NOT_OFFBOARD：这是未切外部控制前的正常状态，由飞手/地面站切 OFFBOARD。

NO_ODOM 或 ODOM_TIMEOUT：

rostopic hz /vrpn_client_node/soccer/pose
rostopic hz /nokov/local_position/odom
rosparam get /nokov_node/tracker_name

检查 NOKOV_SERVER、NOKOV_TRACKER=soccer 和动捕刚体名是否一致。

NO_RC_ACTION 或 RC_ACTION_TIMEOUT：

rostopic hz /mavros/rc/in
rostopic echo /srlc/rc_status

默认 RC 映射：前后通道 2，左右通道 1，上下通道 3；当前 launch 默认 lateral_reverse=true。

NO_LIDAR：

rostopic hz /srlc/lidar/range_image
rostopic echo -n 1 /srlc/lidar/min_distance
rosparam get /map_lidar_node/pcd_file
rosparam get /map_lidar_node/odom_topic

通常是没有 Nokov odom、PCD 路径错误或地图坐标没对齐。

GEOFENCE_X/Y 或高度限制：默认边界是 x,y ∈ [-3,3]m，高度 0.5-3.0m。可启动时覆盖：

roslaunch srlc_real real_px4.launch \
geofence_x_min:=-2.0 geofence_x_max:=2.0 \
geofence_y_min:=-2.0 geofence_y_max:=2.0 \
min_altitude:=0.5 max_altitude:=3.0

6. 记录与结果

默认记录到容器：

ls -lh /root/real_outputs
python3 -m json.tool /root/real_outputs/<run_id>_*.json | sed -n '1,120p'

宿主机对应目录：

/home/nickbit/uav/SharedRLControl/ros1_real/results

如果 JSON 里 samples=0，说明 recorder 启动后没有收到 odom，或者 launch 很快退出。

7. 重新部署

ros1_real 源码是构建时复制进镜像的，修改 ros1_real/src/... 后需要重建镜像：

cd /home/nickbit/uav/SharedRLControl/ros1_real
docker compose -f docker-compose.real.yml down
docker compose -f docker-compose.real.yml build real_runtime
docker compose -f docker-compose.real.yml up -d real_runtime


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
  pcd_file:=/root/real_assets/maps/room601/0624_section_resampled6w_ascii_aligned_yaw_m4p50_floor_level_z0.pcd \
  start_mavros:=false start_nokov:=false record:=false rviz:=true \
  post_takeoff_mode:=direct takeoff_height:=1.0
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
