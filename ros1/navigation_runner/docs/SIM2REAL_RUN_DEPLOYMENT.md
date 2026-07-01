# SRLC 足球无人机 + 动捕真机部署手册

本文是当前真机实验的唯一部署说明。旧版“机载电脑运行 SRLC、使用 `/mavros/local_position/odom` 作为状态”的方案已经废弃。

当前真实实验设定：

- **推理电脑**：`192.168.31.xx`，运行 Docker，容器使用 `--net=host`。
- **足球无人机/机载芯片**：`192.168.31.155`，只负责 PX4/MAVROS 数传链路，接收速度 setpoint。
- **ROS 节点位置**：MAVROS、nokov bridge、SRLC 推理、PCD map LiDAR、RViz/recorder 均运行在推理电脑 Docker 容器内。
- **定位来源**：只使用 nokov 输出的 `/nokov/local_position/odom` 和 `/nokov/imu/data`；SRLC 不使用 `/mavros/local_position/odom`。
- **控制输出**：SRLC 只向 MAVROS 发布 `/mavros/setpoint_raw/local` 速度 setpoint。
- **assist 开关**：确认使用 `/mavros/rc/in` 的 **遥控器通道 9**。
- **急停/信号切断**：由无人机硬件、遥控器杆位和 PX4 failsafe 负责；SRLC 不再发布 `/experiment_control/stop`，也不主动请求 `AUTO.LOITER`/`POSCTL`/`AUTO.LAND`。

## 1. 当前 ROS 数据链路

```text
推理电脑 Docker(--net=host)

NOKOV / VRPN
  -> nokov_uav/sample.launch
  -> /nokov/local_position/odom   ─┬─ map_lidar_node.py
  -> /nokov/imu/data               │    -> /srlc/lidar/range_image
                                   │    -> /srlc/lidar/min_distance
                                   └─ tunnel_navigation.py state

MAVROS over LAN
  -> /mavros/state
  -> /mavros/rc/in
       └─ rc_input_node.py
            -> /srlc/human_action
            -> /srlc/rc_status

tunnel_navigation.py
  <- /srlc/lidar/range_image
  <- /srlc/human_action
  -> /mavros/setpoint_raw/local
  -> /tunnel_nav/status
  -> /tunnel_nav/lifecycle_state
  -> /tunnel_nav/control_mode
  -> /tunnel_nav/policy_active
```

关键约束：

1. `/nokov/local_position/odom` 是 SRLC、map LiDAR 和 RViz 对齐显示的唯一真实状态源。
2. `/mavros/local_position/odom` 即使存在也只作为 PX4/MAVROS 回报，不参与 SRLC 状态估计。
3. `tunnel_real_px4.launch` 不主动切 OFFBOARD；OFFBOARD 由遥控器/PX4 模式开关触发。
4. 进入 OFFBOARD 后，SRLC 会自动请求解锁，并以起飞前地面 Nokov 位置作为固定 `x/y` 目标、`当前 z + takeoff_height` 作为相对高度目标完成爬升与悬停，随后按 `post_takeoff_mode` 进入 DIRECT 或 ASSIST。
5. 输入死区、未进入 OFFBOARD、越界、定位/LiDAR/RC 超时都会使 SRLC 发布零速度/锁高 setpoint，并让 `/tunnel_nav/policy_active=False`。

## 2. 代码侧默认配置

真机入口：

```bash
roslaunch navigation_runner tunnel_real_px4.launch ...
```

当前默认已经对齐真机设定：

| 项目 | 当前默认 |
| --- | --- |
| odom | `/nokov/local_position/odom` |
| RC | `/mavros/rc/in` |
| 模式选择 | `post_takeoff_mode`，默认 `direct` |
| setpoint | `/mavros/setpoint_raw/local` |
| 自动 arm | OFFBOARD 后自动请求 |
| 自动 OFFBOARD | `false` |
| 要求 OFFBOARD 后才允许策略介入 | `true` |
| SRLC 外部 stop topic | 关闭 |
| SRLC 请求 PX4 hold/land | 关闭 |
| 起飞/平飞相对高度 | `1.2m` |
| z 轴共享控制 | `lock_z_control=true` |
| 默认水平边界 | `x,y ∈ [-3, 3]m` |
| 默认高度边界 | `z ∈ [0.5, 3.0]m` |

相关文件：

- `launch/tunnel_real_px4.launch`
- `cfg/tunnel/tunnel_nav_real_px4.yaml`
- `cfg/tunnel/rc_input_real_px4.yaml`
- `cfg/tunnel/map_lidar_real_px4.yaml`
- `scripts/rc_input_node.py`
- `scripts/tunnel_navigation.py`
- `scripts/srlc_dry_run_recorder.py`

## 3. 推理电脑 Docker 启动方式

推荐容器使用 host 网络，否则 MAVROS、nokov、RViz 和多机 ROS 通信容易被 Docker bridge 隔离。

```bash
docker run --rm -it \
  --net=host \
  --ipc=host \
  --shm-size=2g \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  -e ROS_IP=<推理电脑局域网IP> \
  -e ROS_HOSTNAME=<推理电脑局域网IP> \
  -v /path/to/SharedRLControl:/root/SharedRLControl \
  tunnel_comparison:20260415-ipcfix-cpu \
  bash
```
<!-- docker run --rm -it \
  --net=host \
  --ipc=host \
  --shm-size=2g \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  -e ROS_IP=192.168.31.144 \
  -e ROS_HOSTNAME=192.168.31.144 \
  -v /path/to/SharedRLControl:/root/SharedRLControl \
  tunnel_comparison:20260415-ipcfix-cpu \
  bash -->
  
如果在同一个容器内启动所有节点，也可以把 ROS master 固定为本机回环：

```bash
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=127.0.0.1
```

如果 RViz 或其他电脑需要跨主机访问 ROS topic，则使用推理电脑局域网 IP：

```bash
export ROS_MASTER_URI=http://<推理电脑IP>:11311
export ROS_IP=<推理电脑IP>
unset ROS_HOSTNAME
```

## 4. 容器内 workspace 准备

每个终端都先 source：

```bash
source /opt/ros/noetic/setup.bash
source ~/nokov_ws/devel/setup.bash
source ~/catkin_ws/devel/setup.bash
```

如果 `navigation_runner` 来自当前仓库：

```bash
cd /root/SharedRLControl/ros1
catkin_make -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash
```

确认包可见：

```bash
rospack find navigation_runner
rospack find nokov_uav
rospack find mavros
```

## 5. 启动顺序

建议使用 4 个终端，均在推理电脑 Docker 容器内执行。

### 5.1 终端 A：ROS master

```bash
source /opt/ros/noetic/setup.bash
roscore
```

如果后续第一个 `roslaunch` 自动启动了 roscore，也可以不单独开此终端；但真机实验建议显式启动，便于排查。

### 5.2 终端 B：MAVROS 连接足球无人机

默认按 LAN UDP 连接机载地址 `192.168.31.155`：

```bash
source /opt/ros/noetic/setup.bash
roslaunch mavros px4.launch \
  fcu_url:=udp://:14540@192.168.31.155:14557 \
  gcs_url:=
```

如果实验现场已有固定 MAVROS launch 文件，使用现场 launch，但必须保证以下 topic 可用：

```bash
rostopic echo -n 1 /mavros/state
rostopic echo -n 1 /mavros/rc/in
rostopic info /mavros/setpoint_raw/local
```

期望：

- `/mavros/state.connected=True`
- `/mavros/rc/in` 能看到遥控器通道 PWM
- `/mavros/setpoint_raw/local` 有 subscriber，即 MAVROS 正在接收 raw setpoint

### 5.3 终端 C：nokov 动捕 bridge

当前 nokov 源码发布的话题为：

```cpp
pub_imu  = nh.advertise<sensor_msgs::Imu>("nokov/imu/data", 1);
pub_odom = nh.advertise<nav_msgs::Odometry>("nokov/local_position/odom", 1);
```

启动：

```bash
source /opt/ros/noetic/setup.bash
source ~/nokov_ws/devel/setup.bash
roslaunch nokov_uav sample.launch
```

确认：

```bash
rostopic hz /nokov/local_position/odom
rostopic echo -n 1 /nokov/local_position/odom
rostopic hz /nokov/imu/data
```

要求：

- odom frame 与场地 local/map 坐标一致，位置连续、低延迟、无跳变。
- 起飞点附近的 z 与实际高度一致。
- yaw 朝向与实验前进方向可通过 `map_yaw_deg` 和 `map_origin_x/y/z` 对齐。

### 5.4 终端 D：SRLC 真机链路


```bash
source /opt/ros/noetic/setup.bash
source ~/nokov_ws/devel/setup.bash
source ~/catkin_ws/devel/setup.bash

# direct
roslaunch navigation_runner tunnel_real_px4.launch \
  start_mavros:=false \
  rviz:=true \
  record:=true \
  takeoff_height:=0.8 \
  lock_z_control:=true \
  odom_topic:=/nokov/local_position/odom \
  rc_topic:=/mavros/rc/in \
  setpoint_raw_topic:=/mavros/setpoint_raw/local \
  post_takeoff_mode:=direct \
  enable_safety_stop:=false \
  enable_collision_detection:=false
```

```bash
# assist RL
roslaunch navigation_runner tunnel_real_px4.launch \
  start_mavros:=false \
  rviz:=true \
  record:=true \
  takeoff_height:=0.8 \
  lock_z_control:=true \
  odom_topic:=/nokov/local_position/odom \
  rc_topic:=/mavros/rc/in \
  setpoint_raw_topic:=/mavros/setpoint_raw/local \
  post_takeoff_mode:=assist \
  enable_safety_stop:=false \
  enable_collision_detection:=false
```

如需让该 launch 同时启动 MAVROS，可显式打开：

```bash
roslaunch navigation_runner tunnel_real_px4.launch \
  start_mavros:=true \
  fcu_url:=udp://:14540@192.168.31.155:14557 \
  rviz:=false \
  record:=true
```

不要同时在其他终端重复启动 MAVROS。

## 6. 起飞、OFFBOARD 与自动模式流程

`tunnel_real_px4.launch` 启动后：

- **不会自动切 OFFBOARD**
- **会持续发送被动 hold setpoint，保证 PX4 可以接受 OFFBOARD**
- **检测到 OFFBOARD 后会自动请求解锁，并飞向“起飞前地面 `x/y` + `当前 z + takeoff_height`”的固定起飞点**
- **不需要命令行向 SRLC 发送起飞指令**

推荐流程：

1. 启动 MAVROS、nokov、SRLC。
2. 检查 `/tunnel_nav/lifecycle_state`，应处于 `WAIT_OFFBOARD`。
3. 检查 `/tunnel_nav/status`，确认不是 `NO_ODOM`、`NO_LIDAR`、`MAVROS_NOT_CONNECTED`、`NO_RC_ACTION`。
4. 飞手或地面站按现场流程切入 OFFBOARD/外部速度控制。
5. SRLC 自动请求 arm；PX4 报告 armed 后爬升到 `当前 z + takeoff_height` 并悬停。
6. 起飞后等待 `post_takeoff_mode_delay`，默认 `2s`。
7. SRLC 自动进入 `post_takeoff_mode` 指定模式：默认 `DIRECT`，也可启动时设为 `assist`。
8. 摇杆输入超过 `assist_input_deadzone_norm` 后才发布非零速度 setpoint；低于死区保持悬停。

## 7. 无输入/idle 状态确认

启动 SRLC 后、未进入 OFFBOARD 或摇杆处于死区时，无人机不应受到模型前进指令影响。OFFBOARD 起飞完成后，控制模式由 `post_takeoff_mode` 决定。

检查命令：

```bash
rostopic echo /srlc/rc_status
rostopic echo /tunnel_nav/lifecycle_state
rostopic echo /tunnel_nav/control_mode
rostopic echo /tunnel_nav/policy_active
rostopic echo /tunnel_nav/status
rostopic echo /mavros/setpoint_raw/local
```

期望：

- `/tunnel_nav/policy_active=False`
- `/tunnel_nav/status` 显示 `WAIT_OFFBOARD`、`TAKEOFF_CLIMB`、`TAKEOFF_SETTLE`、`INPUT_DEADZONE` 或 `PX4_NOT_OFFBOARD`
- DIRECT 且摇杆超过死区时，`/mavros/setpoint_raw/local` 的 `velocity.x/y` 跟随 RC；ASSIST 且摇杆超过死区时由策略输出；死区/未 OFFBOARD 时为 0
- 没有默认前进速度；真实部署不启动 fake RC，也不启用 user_model 输入

## 8. 自动模式是否生效的确认方法

### 8.1 RC bridge 层

```bash
rostopic echo /srlc/rc_status
```

RC 新鲜时：

```text
/srlc/rc_status: "RC vx=... vy=... vz=..."
```

RC 超时时：

```text
/srlc/rc_status: "RC_TIMEOUT"
```

### 8.2 策略层

```bash
rostopic echo /tunnel_nav/policy_active
rostopic echo /tunnel_nav/status
rostopic echo /tunnel_nav/policy_cmd
rostopic echo /tunnel_nav/control_mode
```

只有同时满足以下条件，`policy_active` 才应为 `True`：

1. MAVROS connected。
2. PX4 mode 为 `OFFBOARD`。
3. nokov odom 新鲜。
4. PCD LiDAR range image 新鲜。
5. `post_takeoff_mode=assist` 且 `/tunnel_nav/control_mode=ASSIST`。
6. 摇杆输入超过死区。
7. 没有越界、低/高高度、碰撞/近障碍停止。

`post_takeoff_mode=direct` 时，如果上述基础安全 gate 通过且摇杆超过死区，`/tunnel_nav/status` 应显示 `DIRECT`，`policy_active=False`，`/mavros/setpoint_raw/local` 直接跟随 RC 速度输入。

如果触及地图边界，`/tunnel_nav/status` 会显示 `GEOFENCE_X` 或 `GEOFENCE_Y`，`/tunnel_nav/policy_active=False`，最终 setpoint 为零速度/锁高；这就是 SRLC 的“自动退出速度输出”。飞手应切回手动/POSCTL 或重新进入安全区域后再继续。

## 9. 外部切断信号如何确认

硬件遥控器/PX4 failsafe 切断后，SRLC 不主动请求模式切换，但可以从 ROS 侧确认系统已经不再处于可介入状态。

检查：

```bash
rostopic echo /mavros/state
rostopic echo /mavros/rc/in
rostopic echo /srlc/rc_status
rostopic echo /tunnel_nav/status
rostopic echo /tunnel_nav/lifecycle_state
rostopic echo /tunnel_nav/policy_active
```

常见确认信号：

- RC 断开：`/srlc/rc_status=RC_TIMEOUT`，`/tunnel_nav/status` 通常回到 `RC_ACTION_TIMEOUT`。
- MAVROS 断开：`/tunnel_nav/status` 出现 `MAVROS_NOT_CONNECTED`。
- PX4 退出 OFFBOARD：`/mavros/state.mode` 不再是 `OFFBOARD`，`/tunnel_nav/status=PX4_NOT_OFFBOARD`。
- SRLC 策略退出：`/tunnel_nav/policy_active=False`，`/mavros/setpoint_raw/local` 速度为 0。

## 10. PCD 地图与坐标对齐

默认地图：

```text
$(rospack find navigation_runner)/cfg/real_maps/dry_run/real_map_dry_run_6x6x5_ascii.pcd
```

该地图来自真实 merged PCD 的中心 `6m x 6m x 5m` 裁剪。重新生成：

```bash
rosrun navigation_runner crop_real_pcd_map.py \
  --input "$(rospack find navigation_runner)/cfg/real_maps/merged/real_map_merged_ascii.pcd" \
  --output "$(rospack find navigation_runner)/cfg/real_maps/dry_run/real_map_dry_run_6x6x5_ascii.pcd" \
  --center 0 0 2 \
  --size 6 6 5
```

对齐参数：

```bash
map_origin_x:=0.0
map_origin_y:=0.0
map_origin_z:=0.0
map_yaw_deg:=0.0
```

含义：

- `map_origin_*`：把 nokov/local odom 原点平移到 PCD 地图中的起飞点。
- `map_yaw_deg`：把无人机起飞朝向/实验前进方向对齐到 PCD 地图障碍方向。

RViz 中必须同时检查：

```bash
/real_map/cloud
/srlc/alignment/odom_map
/srlc/alignment/markers
/srlc/lidar/raycast_points
```

验收标准：

1. 起飞点处无人机 marker 与 PCD 中实际起飞位置重合。
2. 无人机前进方向 marker 指向实验障碍方向。
3. `/srlc/lidar/raycast_points` 与 PCD 障碍表面相交合理。
4. 推动前进输入时，`/srlc/lidar/min_distance` 与前方障碍距离变化一致。

## 11. 飞行前检查清单

### 11.1 网络

```bash
ping 192.168.31.155
rostopic list | grep -E 'mavros|nokov|srlc|tunnel_nav'
```

### 11.2 MAVROS

```bash
rostopic echo -n 1 /mavros/state
rostopic hz /mavros/rc/in
rostopic info /mavros/setpoint_raw/local
```

### 11.3 nokov

```bash
rostopic hz /nokov/local_position/odom
rostopic echo -n 1 /nokov/local_position/odom
```

不要用以下 topic 判断 SRLC 状态：

```bash
/mavros/local_position/odom
```

### 11.4 RC 输入

```bash
rostopic echo /mavros/rc/in
rostopic echo /srlc/rc_status
```

拨动摇杆，确认 `/srlc/rc_status` 中的 `vx/vy/vz` 跟随变化。

### 11.5 SRLC gate

```bash
rostopic echo /tunnel_nav/status
rostopic echo /tunnel_nav/lifecycle_state
rostopic echo /tunnel_nav/control_mode
rostopic echo /tunnel_nav/policy_active
rostopic echo /mavros/setpoint_raw/local
```

未 OFFBOARD 时应看到 `PX4_NOT_OFFBOARD` 或 `WAIT_OFFBOARD`，且速度为 0。OFFBOARD 后应依次看到 `WAIT_ARMED`、`TAKEOFF_CLIMB`、`TAKEOFF_SETTLE`、`ACTIVE`；其中起飞阶段 setpoint 会锁定起飞前 Nokov 地面 `x/y`，避免目标点漂移。

### 11.6 速度方向

首次上桨前建议无桨或架高测试：

1. 切 OFFBOARD。
2. 等待 `/tunnel_nav/lifecycle_state=ACTIVE`。
3. 轻推前进摇杆。
4. 确认 `/mavros/setpoint_raw/local.velocity.x/y` 与 RViz 中期望前进方向一致。

## 12. 记录与数据检查

`tunnel_real_px4.launch record:=true` 会启动 `srlc_dry_run_recorder.py`，默认输出：

```text
/tmp/srlc_real
```

记录内容包括：

- nokov odom 位置/速度/yaw
- 原始 `/mavros/rc/in` PWM
- `/tunnel_nav/control_mode`
- `/srlc/human_action`
- `/tunnel_nav/policy_cmd`
- `/mavros/setpoint_raw/local`
- `/srlc/lidar/min_distance`
- 前向 LiDAR 距离估计
- `/tunnel_nav/status`

实验后检查：

```bash
ls -lh /tmp/srlc_real
python3 -m json.tool /tmp/srlc_real/<run_id>_*.json | sed -n '1,80p'
```

## 13. 软件 dry-run 入口

无真实无人机/MAVROS/nokov 时，可用 fake MAVROS + fake odom + 真实 PCD 进行端到端软件 dry-run：

```bash
roslaunch navigation_runner tunnel_dry_run_px4.launch \
  mode:=fake_mavros \
  rviz:=true \
  record:=true
```

dry-run 会显式覆盖真机默认定位源：

```text
odom_topic:=/mavros/local_position/odom
rc_topic:=/mavros/rc/in
setpoint_raw_topic:=/mavros/setpoint_raw/local
```

其中 `/mavros/local_position/odom` 由 `mavros_fake_node.py` 发布，并根据 `/mavros/setpoint_raw/local` 积分移动；`/mavros/rc/in` 由 `srlc_fake_rc_node.py` 发布模拟摇杆输入。不要在 dry-run 中使用 `/nokov/local_position/odom`，否则没有真实 nokov 节点时 `tunnel_navigation.py` 会一直等待 odom。

按当前真机地图/权重配置做软件验证的示例：

```bash
roslaunch navigation_runner tunnel_dry_run_px4.launch \
  rviz:=true \
  record:=true \
  fake_forward_stick:=0.8 \
  fake_lateral_stick:=0.1
```

该模式仅用于验证：

- 模型 checkpoint 可加载
- fake RC 摇杆输入可触发
- PCD raycast LiDAR 正常
- geofence/输入死区/锁高逻辑正常
- `/mavros/setpoint_raw/local` 输出符合预期

离线 smoke check：

```bash
rosrun navigation_runner srlc_dry_run_smoke_check.py \
  --pcd-file "$(rospack find navigation_runner)/cfg/real_maps/dry_run/real_map_dry_run_6x6x5_ascii.pcd" \
  --checkpoint "$(rospack find navigation_runner)/cfg/ckpts/checkpoint_tunnel_M3_21500.pt" \
  --launch-file "$(rospack find navigation_runner)/launch/tunnel_dry_run_px4.launch" \
  --device cpu
```

## 14. 常见故障

### `/tunnel_nav/status=NO_ODOM`

真机模式下，SRLC 没收到 `/nokov/local_position/odom`。检查：

```bash
rostopic hz /nokov/local_position/odom
rosparam get /tunnel_navigator/odom_topic
```

dry-run 模式下，SRLC 应使用 fake MAVROS odom。检查：

```bash
rostopic hz /mavros/local_position/odom
rosparam get /tunnel_navigator/odom_topic
rosparam get /map_lidar_node/odom_topic
```

期望两个参数都是 `/mavros/local_position/odom`。

### `/tunnel_nav/status=PX4_NOT_OFFBOARD`

MAVROS 已连接，但 PX4 当前不是 OFFBOARD。SRLC 会保持零速度/锁高 setpoint 流，不执行 DIRECT/RL 速度输出。由飞手/地面站按现场流程切 OFFBOARD。

### `/tunnel_nav/control_mode` 与期望不符

检查启动参数：

```bash
rosparam get /tunnel_navigator/post_takeoff_mode
rostopic echo /tunnel_nav/lifecycle_state
rostopic echo /tunnel_nav/control_mode
```

`post_takeoff_mode` 只在起飞完成并等待 `post_takeoff_mode_delay` 后生效；未到 `ACTIVE` 前不会输出 DIRECT/RL 速度。

### 边界内却显示 `GEOFENCE_X/Y`

检查 nokov 坐标原点是否与配置边界一致。默认 geofence 是 nokov/local 坐标：

```bash
rosparam get /tunnel_navigator/geofence_x
rosparam get /tunnel_navigator/geofence_y
rostopic echo -n 1 /nokov/local_position/odom
```

### RViz 地图和无人机不重合

调整：

```bash
map_origin_x:=...
map_origin_y:=...
map_origin_z:=...
map_yaw_deg:=...
```

并观察 `/srlc/alignment/markers`。

### MAVROS 连接不上 `192.168.31.155`

确认推理电脑和无人机在同一局域网，防火墙未拦截 UDP，现场 PX4/MAVLink 端口与 `fcu_url` 一致。必要时把 `fcu_url` 改为现场给定的 MAVROS launch 参数。

## 15. 最小验收标准

上机前至少满足：

1. `rostopic hz /nokov/local_position/odom` 稳定。
2. `rostopic echo /mavros/state` 显示 connected。
3. 未 OFFBOARD 时 `/tunnel_nav/lifecycle_state=WAIT_OFFBOARD`，setpoint 速度为 0。
4. 切入 OFFBOARD 后，SRLC 自动请求 arm，并进入 `TAKEOFF_CLIMB`/`TAKEOFF_SETTLE`。
5. 起飞延迟结束后，`/tunnel_nav/control_mode` 等于 `post_takeoff_mode`。
6. 未 OFFBOARD 或摇杆处于死区时，`/tunnel_nav/policy_active=False` 且 setpoint 速度为 0。
7. `post_takeoff_mode=direct` + 摇杆超过死区后，`/tunnel_nav/status=DIRECT`，`policy_active=False`，`/mavros/setpoint_raw/local` 跟随 RC 速度。
8. `post_takeoff_mode=assist` + 摇杆超过死区后，`/tunnel_nav/policy_active=True`，`/tunnel_nav/policy_cmd` 和 `/mavros/setpoint_raw/local` 更新。
9. 触发 geofence 后，`/tunnel_nav/status=GEOFENCE_X/Y`，`policy_active=False`，setpoint 速度为 0。
10. 关闭遥控器或退出 OFFBOARD 后，`/srlc/rc_status=RC_TIMEOUT` 或 `/tunnel_nav/status` 反映 `PX4_NOT_OFFBOARD`/`MAVROS_NOT_CONNECTED`/`RC_ACTION_TIMEOUT`，SRLC 不继续输出速度。


battery charge

11.3v - 12.3v 
<11.2 need charge
charge time: 10min
full charge can fly 10min (on air)
