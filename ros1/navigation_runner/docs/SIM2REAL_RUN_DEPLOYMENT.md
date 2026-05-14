## SRLC模型真实无人机部署实验 设计稿

目标：实现真实ROS环境下已经训练好的模型推理和用户输入结合的共享控制，并通过加载真实地图构建真机运行环境，
进行dry-run以确认此套部署方案可以无缝迁移到真实无人机实验

## 范围

- 模型：使用经过ConstrainedResidualPPO训练完成的权重（对应训练中的实验SharedRLControl/isaac-training/experiments/04_tunnel_task），动作网络使用Beta分布
    - 权重文件位置：/home/haoming/wht/IsaacLab_drones_5.1/SharedRLControl/ros1/navigation_runner/cfg/ckpts/checkpoint_tunnel_M3_21500.pt

- 地图：使用/home/haoming/wht/IsaacLab_drones_5.1/SharedRLControl/ros1/real_maps/merged中的真实地图pcd文件，pcd地图已经经过下采样为ascii编码，
但可能仍然需要进一步处理（如去除中心范围外的噪声等）

- 无人机控制器：使用marvos px4控制器（速度控制模式），在接入真实遥控器信号前，使用键盘输入来模拟所有遥控信号

- 激光雷达：使用模拟数据，即根据无人机定位信息和真实pcd地图，生成激光雷达扫描点云信号，部署到真机时，无人机实时定位数据将由一套动作捕捉系统提供。
    - dry-run时，无人机位置信息需要对齐，但我不清楚对齐方式，给我一套合理的方案

## 在当前仓库中有实现，但不要考虑的范围

- 其他类型的模型结构，以及使用Usermodel数据集代替人类输入的控制
- 安全围栏（safety_mode）机制的接入
- IPC控制算法的接入 

## 实验设计

- 地图大小：仅包含中央约6*6*5m的矩形区域，忽略外部的非动作捕捉区域，为无人机的速度控制添加限制，如果位置超出边界，立刻刹车悬停
    - 因为高度限制低于训练和仿真测试，无人机起飞高度修改为2m，考虑到此不一致性对模型的影响，应加入一个配置，开启后可禁止所有z轴控制（保持高度平飞）

- 用户输入：默认情况下，起飞后测试员输入向起飞朝向前进的指令（将此方向对齐为模型推理时接受输入坐标系的x方向），
可能伴随小幅度的y方向指令抖动，观察无人机能否绕过真实地图前方的障碍物（严格按照指令速度控制将会碰撞）。

- 共享控制切换需求：
    1）起飞后才能接通共享控制的介入
    2）飞行过程中，可以通过特定频道的遥控信号打开和关闭共享控制模型的介入
    3）输入死区：接受到遥控信号的量过小时，阻止共享控制模型的介入，保持无人机悬停指令
    4）紧急停止按钮：提供一行指令允许立刻终止所有控制并自动降落

- 实验过程设想：
    - 发送无人机起飞指令
    - 不开启共享控制，操作员控制无人机穿越障碍，如果发生碰撞（无人机使用球形桨叶保护，真实障碍物为软体），紧急停止并记录结果
    - 重新起飞（从同一起飞点）
    - 操作员确认起飞后，开启共享控制模式，然后进行相同的输入操作，记录结果

## 可视化

- 在dry-run和真实部署时，可以在rivz中看到可视化数据，包括：地图点云、无人机位置、激光雷达射线

## 第一阶段 dry-run 入口

当前第一阶段使用 fake MAVROS 和 fake odom，但接入真实 PCD 地图数据：

```bash
roslaunch navigation_runner tunnel_dry_run_px4.launch \
    mode:=fake_mavros \
    pcd_file:="$(rospack find navigation_runner)/cfg/real_maps/dry_run/real_map_dry_run_6x6x5_ascii.pcd" \
    rviz:=true
```

默认链路为：`mavros_fake_node.py` 发布 `/mavros/state`、MAVROS 服务和 `/mavros/local_position/odom`，`srlc_fake_rc_node.py` 发布 `/mavros/rc/in`，`rc_input_node.py` 转换为 `/srlc/human_action` 与 `/srlc/assist_enable`，`map_lidar_node.py` 用真实 PCD + fake odom 生成 `/srlc/lidar/range_image`，`tunnel_navigation.py` 加载策略并向 `/mavros/setpoint_raw/local` 输出速度/锁高 setpoint。

常用调参项：

- `map_origin_x/y/z`、`map_yaw_deg`：把 fake local ENU 起飞点对齐到 PCD 地图。
- `fake_forward_stick`、`fake_lateral_stick`：模拟测试员前进和小幅横向扰动。
- `assist_input_deadzone_norm`：输入过小时保持悬停，不执行共享控制输出。
- `lock_z_control:=true`、`takeoff_height:=2.0`：禁用 z 轴共享控制并保持 2m 平飞。
- `geofence_x_min/max`、`geofence_y_min/max`、`min_altitude`、`max_altitude`：dry-run 边界保护。

真实 merged map 的默认 dry-run 裁剪图由以下命令生成：

```bash
rosrun navigation_runner crop_real_pcd_map.py \
    --input "$(rospack find navigation_runner)/cfg/real_maps/merged/real_map_merged_ascii.pcd" \
    --output "$(rospack find navigation_runner)/cfg/real_maps/dry_run/real_map_dry_run_6x6x5_ascii.pcd" \
    --center 0 0 2 \
    --size 6 6 5
```

启动前可先离线检查 PCD、raycast shape、checkpoint 和 launch XML：

```bash
rosrun navigation_runner srlc_dry_run_smoke_check.py \
    --pcd-file "$(rospack find navigation_runner)/cfg/real_maps/dry_run/real_map_dry_run_6x6x5_ascii.pcd" \
    --checkpoint "$(rospack find navigation_runner)/cfg/ckpts/checkpoint_tunnel_M3_21500.pt" \
    --launch-file "$(rospack find navigation_runner)/launch/tunnel_dry_run_px4.launch" \
    --device cpu
```

## 真机部署入口：机载电脑 + nokov + MAVROS/PX4

当前真机目标链路如下：

```text
nokov/VRPN -> nokov_uav/nokov_node
  -> /mavros/vision_pose/pose
  -> /mavros/local_position/odom  ─┬─ map_lidar_node.py -> /srlc/lidar/range_image
                                  └─ tunnel_navigation.py state

PX4/MAVROS -> /mavros/rc/in -> rc_input_node.py
  -> /srlc/human_action
  -> /srlc/assist_enable
  -> /experiment_control/stop

tunnel_navigation.py
  -> /mavros/setpoint_raw/local  (PX4 OFFBOARD velocity setpoint)
```

`nokov_ws/src/nokov_uav/src/nokov.cpp` 的默认话题已经与 SRLC 对齐：

- 输入：`/vrpn_client_node/soccer/pose`、`/vrpn_client_node/soccer/twist`、`/vrpn_client_node/soccer/accel`
- 输出给 MAVROS/PX4 external vision：`/mavros/vision_pose/pose`
- 输出给 SRLC/map LiDAR：`/mavros/local_position/odom`
- 输出 IMU：`/mavros/imu/data`

### 真机理论正确启动步骤

以下步骤应在机载电脑上执行，开发机没有 MAVROS/PX4 环境时只做代码和参数准备。

1. 配置 ROS 网络，避免 hostname 自连失败：

   ```bash
   export ROS_MASTER_URI=http://127.0.0.1:11311
   export ROS_HOSTNAME=127.0.0.1
   export ROS_IP=127.0.0.1
   ```

   多机 ROS 时，把 `ROS_MASTER_URI`、`ROS_IP` 改为机载电脑实际 IP，并确认 `ping` 和 `rostopic list` 双向可用。

2. source 所有工作空间：

   ```bash
   source /opt/ros/noetic/setup.bash
   source ~/nokov_ws/devel/setup.bash
   source ~/catkin_ws/devel/setup.bash
   ```

   如果仓库路径保持为 `SharedRLControl/ros1`，则 `navigation_runner` 应位于机载 `catkin_ws/src`，`nokov_ws` 单独编译。

3. 启动 MAVROS 连接 PX4。若已经由系统服务启动，则跳过；否则先单独启动 MAVROS：

   ```bash
   roslaunch mavros px4.launch fcu_url:=/dev/ttyACM0:921600
   ```

   也可以在第 5 步用 `start_mavros:=true` 让 SRLC launch 代为 include MAVROS，但不要两边重复启动。确认：

   ```bash
   rostopic echo -n 1 /mavros/state
   rostopic echo -n 1 /mavros/rc/in
   ```

4. 启动 nokov/VRPN 定位桥：

   ```bash
   roslaunch vrpn_client_ros sample.launch server:=<NOKOV_SERVER_IP>
   ```

   该 launch 会启动 `vrpn_client_node` 和 `nokov_node`。确认：

   ```bash
   rostopic hz /vrpn_client_node/soccer/pose
   rostopic hz /mavros/local_position/odom
   rostopic echo -n 1 /mavros/local_position/odom
   ```

   理论要求：无人机位姿和速度在 `map` frame 下连续、低延迟、无跳变；起飞点附近 odom 可视为 local ENU 原点或通过 `map_origin_x/y/z` 对齐到 PCD 地图起飞点。

5. 启动 SRLC 真机部署链路：

   ```bash
   roslaunch navigation_runner tunnel_real_px4.launch \
       start_mavros:=false \
       pcd_file:="$(rospack find navigation_runner)/cfg/real_maps/dry_run/real_map_dry_run_6x6x5_ascii.pcd" \
       checkpoint:="$(rospack find navigation_runner)/cfg/ckpts/checkpoint_tunnel_M3_21500.pt" \
       takeoff_height:=2.0 \
       lock_z_control:=true \
       rviz:=true \
       record:=true
   ```

   默认真机话题：

   | 功能 | 默认话题 |
   | --- | --- |
   | nokov odom | `/mavros/local_position/odom` |
   | PX4 RC 输入 | `/mavros/rc/in` |
   | human action | `/srlc/human_action` |
   | assist 开关 | `/srlc/assist_enable` |
   | LiDAR range image | `/srlc/lidar/range_image` |
   | PX4 速度 setpoint | `/mavros/setpoint_raw/local` |
   | 记录输出 | `output_dir`，默认 `/tmp/srlc_real` |

### 起飞与 PX4 遥控器切换流程

SRLC 真机配置默认 **不会自动 arm，也不会自动切 OFFBOARD**：

- `auto_arm=false`
- `auto_offboard=false`
- `hold_on_stop=true`

推荐流程：

1. 起飞前保持 SRLC assist 开关关闭，estop/reset 通道按 `cfg/tunnel/rc_input_real_px4.yaml` 标定。
2. 用 PX4 遥控器或地面站按原流程解锁、起飞并稳定到约 `2m`。
3. 确认 `/mavros/local_position/odom`、`/srlc/lidar/range_image`、`/srlc/human_action` 都在更新。
4. 在 PX4 允许 OFFBOARD 的前提下切入 OFFBOARD/外部速度控制模式；SRLC 在 assist 关闭时只发布零 setpoint，不主动请求 LOITER，飞手仍可保持 PX4 遥控器控制。
5. 打开 assist 开关后，`rc_input_node.py` 发布 `/srlc/assist_enable=True`；当摇杆输入幅值超过 `assist_input_deadzone_norm` 后，模型输出才会写入 `/mavros/setpoint_raw/local`。
6. 关闭 assist 开关会回到飞手控制；estop 通道或 `/experiment_control/stop=True` 会触发 SRLC 请求 `AUTO.LOITER`，失败时尝试 `POSCTL` fallback，并持续输出零速度。

### 上机前检查清单

1. **RC 通道标定**：检查 `cfg/tunnel/rc_input_real_px4.yaml` 中 `forward_channel`、`lateral_channel`、`vertical_channel`、`assist_channel`、`estop_channel`、`reset_channel` 与真实遥控器一致。用：

   ```bash
   rostopic echo -n 1 /mavros/rc/in
   rostopic echo /srlc/rc_status
   ```

2. **坐标对齐**：RViz 中 `/real_map/cloud`、`/srlc/alignment/odom_map`、`/srlc/lidar/raycast_points` 必须重合。若起飞点不在 PCD 期望位置，调 `map_origin_x/y/z`；若前进方向不对，调 `map_yaw_deg`。
3. **高度策略**：默认 `takeoff_height=2.0`、`lock_z_control=true`、`height_control=false`，即共享控制只负责 x/y，PX4/定位系统保持高度。
4. **边界保护**：默认 geofence 是 local ENU `x,y ∈ [-3,3]`、`z ∈ [0.5,5.0]`。越界会停止共享控制输出并请求 hold。
5. **速度限制**：默认 `max_xy_speed_real=1.0`、`max_z_speed_real=0.3`。首次真机建议进一步降低 `max_xy_speed_real:=0.3~0.5`。
6. **OFFBOARD setpoint 方向**：无桨或架高测试时，轻推前进摇杆，确认 `/mavros/setpoint_raw/local/velocity.x` 与期望前进方向一致，再上桨。
7. **记录文件**：每次实验后检查 `/tmp/srlc_real/*.json`，确认 `assist_enabled`、`human_action`、`policy_cmd`、`setpoint_velocity`、`min_distance` 和 `front_distance` 被记录。

## 飞行过程记录

真机实验过程中，记录以下信息：
- 飞行位置和真实速度（相对起飞位置），原始遥控指令速度，模型输出指令速度
- 激光雷达数据中，距离所有障碍物的最小距离，以及与机头前进方向最靠近的扫描点获得的与障碍物的距离
