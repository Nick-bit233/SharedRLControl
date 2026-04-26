# Tunnel RL vs IPC — ROS1/Gazebo 部署与对比实验指南

> 将 Isaac Sim 训练的隧道避障 RL 策略部署到 ROS1 (Noetic) + Gazebo，
> 并与 slope_inspection 的 IPC 算法在同一环境中进行公平对比
>
> 当前分支的诊断结论与推荐基线见 `TUNNEL_DIAGNOSIS.md`。本文旧版本里关于
> Tanh residual、旧出生点和旧目标线的描述已经失效，请以当前 launch/config
> 与诊断文档为准。当前默认 Gazebo 起飞点是 `spawn_x=-8.5`，这是针对现有
> PCD/world 资产组合做过实测验证的安全微调，不代表 Isaac Sim 训练起点本身。

## 概览

本部署包提供三个功能：
1. **RL 部署**：将 `ConstrainedResidualPPO_Beta` 策略网络部署到 ROS1/Gazebo 仿真环境
2. **RL vs IPC 对比**：在相同 CERLAB Gazebo 仿真器和同一隧道地图中，对比 RL 策略和 slope_inspection IPC 算法
3. **键盘控制模式**：用键盘提供人类输入指令，可实时切换 RL 辅助，用于交互式调试和演示

**对比实验架构：**
```
               ┌── RL 模式 ─────────────────────┐
               │ lidar_sim_node (PCD→PointCloud2)│
               │ map_manager (占用地图+RayCast)   │
               │ tunnel_navigation.py             │
               │   UserModel → Policy → cmd_vel   │
               │   LiDAR via map_manager/raycast  │
               └───────────┬─────────────────────┘
                           │
  同一个 Gazebo 仿真器 ←───┤──── /CERLAB/quadcopter/cmd_vel
  同一张 PCD 隧道地图      │
                           │
               ┌── IPC 模式 ────────────────────┐
               │ ipc_node (C++ 原版)              │
               │   ROG-Map + A* + CIRI + MPC      │
               │   PositionCommand → cmd_bridge    │
               └───────────┬─────────────────────┘
                           │
            Bridge 适配层（5 个节点）:
            - rc_sim_node: 模拟遥控器模式切换
            - lidar_sim_node: PCD → PointCloud2
            - imu_bridge_node: odom → IMU
            - cmd_bridge_node: PositionCommand → cmd_vel
            - mavros_fake_node: 模拟 MAVROS 服务
```

## 1. 快速启动（Docker，推荐）

适用于开发机不是 Ubuntu 20.04 的情况。需要 Docker + nvidia-container-toolkit + `slope_inspection:test` 基础镜像。

### 1.1 构建与启动

```bash
cd SharedRLControl

# 构建 Docker 镜像
docker build -f Dockerfile.tunnel_comparison -t tunnel_comparison:latest .

# 使用 docker-compose 持久化启动（支持 GPU 渲染、X11 转发、代码热编辑）
xhost +local:docker
docker compose -f docker-compose.tunnel.yml up -d

# 进入容器
docker exec -it tunnel_debug bash
```

> 当前已验证并持久化了一个本地修复镜像：`tunnel_comparison:20260415-ipcfix`。
> 这份 image 包含容器内重新编译后的 `slope_ws` / `ipc_node` 修复，`docker-compose.tunnel.yml`
> 现在默认就固定到这个 tag。为了兼容旧命令，本机的 `tunnel_comparison:latest` 也已被重新标记到同一镜像。

### 1.2 运行 RL 模式

```bash
# 容器内（带 GUI + RViz）
roslaunch navigation_runner tunnel_comparison.launch method:=rl gui:=true

# 无 GUI（headless + Xvfb）
roslaunch navigation_runner tunnel_comparison.launch method:=rl gui:=false

# 显式指定 M3 最终模型（当前默认也是这份权重）
roslaunch navigation_runner tunnel_comparison.launch method:=rl gui:=false \
    checkpoint:="$(rospack find navigation_runner)/cfg/ckpts/checkpoint_tunnel_M3_21500.pt"
```

> **注意**: RL 模式现在同时启动 `lidar_sim_node`（PCD → PointCloud2）和
> `map_manager`（占用地图 + RayCast 服务）。`lidar_sim_node` 提供实时点云数据，
> `map_manager` 使用预构建 PCD 地图和实时点云来维护占用栅格。
> 当前 `tunnel_comparison:20260415-ipcfix` 镜像内的 PyTorch 是 CPU-only，因此推荐明确使用
> `device:=cpu`。如果误传 `device:=cuda:0`，当前的 batch 脚本和 `tunnel_navigation.py`
> 都会自动回退到 `cpu`，避免 RL 节点因 CUDA 不可用而退出。

### 1.3 运行 IPC 模式

```bash
export ROS_HOSTNAME=127.0.0.1 ROS_IP=127.0.0.1
source /opt/ros/noetic/setup.bash
source /root/slope_ws/devel/setup.bash
source /root/catkin_ws/devel/setup.bash

roslaunch navigation_runner tunnel_comparison.launch method:=ipc gui:=true
```

> 如果是在容器里手动分步调试，`slope_ws` 必须先于 `catkin_ws` source；反过来会让
> `navigation_runner` 从 `ROS_PACKAGE_PATH` 中消失。当前 IPC 启动链还会自动完成
> `takeoff -> Manual -> Hover -> Pilot -> AutoPilot`，并在 AutoPilot 后复用与 RL
> 模式相同的 `UserModelTunnel` 输入流，无需再手动给一次起飞命令或额外发送固定前进摇杆。
> 当前默认 startup timing 也已经压缩为：`takeoff_wait=0.6s`、`init_delay=0.1s`、
> `switch_delay=0.15s`，用于减少 AutoPilot 之前的意外位移。

### 1.4 关键 launch 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `method` | `rl` | `rl` 或 `ipc` |
| `gui` | `false` | Gazebo GUI（需 X11） |
| `checkpoint` | `cfg/ckpts/checkpoint_tunnel_M3_21500.pt` | RL 模型路径 |
| `tunnel_map` | `cfg/tunnel/tunnel_map_default.pcd` | 隧道 PCD 地图 |
| `device` | `cpu` | `cpu` 或 `cuda:0` |
| `keyboard` | `false` | 键盘控制模式（仅 RL） |
| `rviz` | `true` | 启动 RViz |
| `record` | `true` | 记录飞行数据 |
| `output_dir` | `/root/results` | 数据输出目录 |
| `gazebo_z_mode` | `alt_hold` | RL z 速度执行模式：`alt_hold`、`policy`、`policy_clamped` 或 `blend` |
| `gazebo_policy_z_takeoff_gate` | `true` | `policy`/`policy_clamped`/`blend` 下先用高度保持起飞，接近 `takeoff_height` 后再执行 policy z |

### 1.5 键盘控制模式

键盘模式用人类实时输入替代程序化的 UserModelTunnel，可随时开关 RL 辅助。

```bash
# 需要 xterm（容器内已安装）和 X11 显示
roslaunch navigation_runner tunnel_comparison.launch method:=rl keyboard:=true gui:=true
```

启动后 CERLAB 的 Qt 键盘控制窗口会自动弹出。操作流程：

1. **等待 Gazebo 和各节点就绪**（控制台出现 `[TunnelNav] KEYBOARD MODE (DIRECT)`）
2. **聚焦 CERLAB 键盘窗口**，按 `Z` 键触发起飞
3. 起飞完成后，使用 CERLAB 键盘控制飞行（见下方键位表）
4. 此时是 **DIRECT 模式**：键盘输入直接传递给无人机
5. 在另一终端发布 `rostopic pub /tunnel_nav/assist_toggle std_msgs/Empty` 开启 RL 辅助
6. RL 辅助开启后，键盘输入作为"人类意图"经 RL 策略修正后再发给无人机
7. 再次发布同一话题关闭 RL 辅助，回到 DIRECT 模式

**CERLAB 键盘键位（由 uav_simulator 的 keyboard_control 节点提供）：**

| 按键 | 功能 |
|------|------|
| Z | 起飞 |
| X | 降落 |
| H | 悬停（零速度） |
| I/K | 前进/后退（pitch → linear.x） |
| J/L | 左移/右移（roll → linear.y） |
| W/S | 上升/下降（linear.z） |
| A/D | 左偏航/右偏航（angular.z） |
| T | 切换位置控制模式 |
| 松开所有键 | 自动悬停 |

**RL 辅助切换：**

```bash
# 切换 DIRECT ↔ RL 辅助
rostopic pub /tunnel_nav/assist_toggle std_msgs/Empty

# 查看当前模式
rostopic echo /tunnel_nav/assist_active
```

**相关话题：**

| 话题 | 类型 | 说明 |
|------|------|------|
| `/keyboard/cmd_vel` | TwistStamped | CERLAB 键盘输出（重映射） |
| `/CERLAB/quadcopter/cmd_vel` | TwistStamped | 最终控制指令（Gazebo 插件接收） |
| `/tunnel_nav/assist_toggle` | Empty | RL 辅助开关（发布一次切换一次） |
| `/tunnel_nav/assist_active` | Bool | 当前 RL 辅助状态（latched） |

## 2. 分步调试

当需要逐步检查各个组件时，在 **4-5 个终端** 中分别启动：

```bash
# 终端 1: Gazebo 仿真器
roslaunch uav_simulator start_headless.launch gui:=true

# 终端 2: LiDAR 点云模拟（RL 和 IPC 都需要）
rosrun navigation_runner lidar_sim_node.py \
    _pcd_file:="$(rospack find navigation_runner)/cfg/tunnel/tunnel_map_default.pcd" \
    _rate:=10 _max_range:=50.0

# 终端 3: 占用地图服务（RL 需要，IPC 不需要）
rosparam load $(rospack find navigation_runner)/cfg/tunnel/occupancy_map_tunnel.yaml /occupancy_map
rosparam set /occupancy_map/prebuilt_map_directory \
    "$(rospack find navigation_runner)/cfg/tunnel/tunnel_map_default.pcd"
rosrun map_manager occupancy_map_node

# 终端 4: RL 导航节点
rosparam load $(rospack find navigation_runner)/cfg/tunnel/tunnel_nav_param.yaml /tunnel_navigator
rosparam set /tunnel_navigator/checkpoint_path \
    "$(rospack find navigation_runner)/cfg/ckpts/checkpoint_tunnel_M3_21500.pt"
rosrun navigation_runner tunnel_navigation.py __name:=tunnel_navigator

# 终端 5: RViz
# RL:
rviz -d $(rospack find navigation_runner)/cfg/tunnel/tunnel.rviz
# IPC:
rviz -d $(rospack find navigation_runner)/cfg/tunnel/ipc_tunnel.rviz
```

**注意**：分步启动时，参数必须手动加载到对应命名空间。使用 `roslaunch` 则自动完成。

## 3. Docker 开发环境

### 3.1 docker-compose.tunnel.yml

`docker-compose.tunnel.yml` 提供持久化开发容器：

| 特性 | 说明 |
|------|------|
| GPU 渲染 | `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=all` |
| X11 转发 | 挂载 `/tmp/.X11-unix`，Gazebo/RViz 显示在宿主机 |
| 代码热编辑 | Python 脚本、配置、launch 文件以 volume 挂载，宿主机编辑即时生效 |
| Checkpoint 挂载 | `isaac-training/outputs/` → `/root/catkin_ws/src/navigation_runner/checkpoints/`（只读） |
| 持久化结果 | 宿主机 `ros1/results/` bind mount → `/root/results` 与 `/root/catkin_ws/results` |
| 智能显示 | `entrypoint.sh` 自动检测：有 X11 → 用 GPU；无 X11 → Xvfb + 软件渲染 |

### 3.2 宿主机编辑 → 容器热更新

以下路径的修改无需重建镜像，实时生效：
```
宿主机                                              → 容器内
ros1/navigation_runner/scripts/                     → /root/catkin_ws/src/navigation_runner/scripts/
ros1/navigation_runner/cfg/                         → /root/catkin_ws/src/navigation_runner/cfg/
ros1/navigation_runner/launch/                      → /root/catkin_ws/src/navigation_runner/launch/
ros1/uav_simulator/worlds/                          → /root/catkin_ws/src/uav_simulator/worlds/
```

> 注意：当前 `docker-compose.tunnel.yml` **不会**把 `slope_inspection/IPC` 或
> `slope_inspection/rog_map` 挂载进 `/root/slope_ws/src/`。因此这两处 C++ 源码的修改
> 不会像 `navigation_runner` 脚本/配置那样自动热更新到容器；若要让源码修复生效，
> 需要在容器内重编 `ipc`（或重建镜像）。本次 IPC segfault 的“现成容器修复”是
> `cfg/tunnel/ipc_gazebo_param.yaml` 中启用 `rog_map.frontier_extraction_en`。
>
> 若要让 `slope_inspection/IPC/include/callback.cpp` 之类的源码修复真正进入当前容器里的
> `ipc_node`，需要先把完整 `slope_inspection` 源码同步到 `/root/slope_ws/src/slope_inspection`，
> 再在容器内执行：
> `catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCATKIN_WHITELIST_PACKAGES="cmake_utils;quadrotor_msgs;mars_planning_utils;mars_base;rog_map;ipc" -j2`
>。另外，`slope_inspection/mars_base/CMakeLists.txt` 现在补了
> `add_dependencies(mars_base ...)`，否则新容器里可能再次遇到 `BuckParam.h` 等消息头生成顺序问题。

## 4. 话题与服务参考

### RL 模式话题

**订阅：**

| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/CERLAB/quadcopter/odom` | `nav_msgs/Odometry` | 30 Hz | 四旋翼里程计（body-frame twist） |
| `/pcl_render_node/cloud` | `sensor_msgs/PointCloud2` | 10 Hz | lidar_sim_node 发布的环境点云 |

**发布：**

| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/CERLAB/quadcopter/cmd_vel` | `geometry_msgs/TwistStamped` | 20 Hz | 速度指令（body frame） |
| `/tunnel_nav/policy_cmd` | `geometry_msgs/TwistStamped` | 20 Hz | RL policy 原始世界系速度输出（用于和实际 cmd_vel 对比） |
| `/tunnel_nav/z_policy_active` | `std_msgs/Bool` | 20 Hz | policy z 是否已通过起飞门控并接管执行 |
| `/CERLAB/quadcopter/takeoff` | `std_msgs/Empty` | 一次 | 起飞指令（CERLAB 插件） |
| `/tunnel_nav/lidar_cloud` | `sensor_msgs/PointCloud2` | 30 Hz | LiDAR 射线命中点（RViz 红色点云） |
| `/tunnel_nav/cmd_vel_vis` | `visualization_msgs/MarkerArray` | 20 Hz | RL 速度指令箭头（绿色） |
| `/tunnel_nav/human_cmd_vis` | `visualization_msgs/MarkerArray` | 20 Hz | 用户模型指令箭头（蓝色） |
| `/tunnel_nav/status` | `std_msgs/String` | 20 Hz | 状态文本：位置、指令、最近障碍距离 |
| `/tunnel_nav/collision` | `std_msgs/Bool` | latched | 碰撞检测（min_dist < collision_dist 时发布 True） |

**服务调用：**

| 服务 | 类型 | 说明 |
|------|------|------|
| `/occupancy_map/raycast` | `map_manager/RayCast` | LiDAR 射线投射（map_manager 提供） |

### Gazebo 仿真器话题（CERLAB 插件）

| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/CERLAB/quadcopter/odom_raw` | `nav_msgs/Odometry` | 1000 Hz | 原始里程计 |
| `/CERLAB/quadcopter/odom` | `nav_msgs/Odometry` | 30 Hz | 降频里程计 |
| `/CERLAB/quadcopter/cmd_vel` | `geometry_msgs/TwistStamped` | — | 速度指令输入 |
| `/CERLAB/quadcopter/takeoff` | `std_msgs/Empty` | — | 起飞指令 |
| `/CERLAB/quadcopter/reset` | `std_msgs/Empty` | — | 重置无人机位置 |

### 调试命令

```bash
# 检查节点是否运行
rosnode list | grep -E "tunnel_navigator|occupancy_map|gazebo"

# 检查参数是否正确加载
rosparam get /tunnel_navigator/takeoff_height    # 应为 4.0
rosparam get /tunnel_navigator/checkpoint_path   # 应为有效路径
rosparam get /tunnel_navigator/control_freq      # 应为 20.0

# 检查话题频率
rostopic hz /CERLAB/quadcopter/odom              # ~30 Hz
rostopic hz /CERLAB/quadcopter/cmd_vel           # ~20 Hz
rostopic hz /tunnel_nav/lidar_cloud              # ~30 Hz

# 查看实时状态
rostopic echo /tunnel_nav/status

# 查看速度指令
rostopic echo /CERLAB/quadcopter/cmd_vel

# 检查 RayCast 服务
rosservice list | grep raycast
rosservice info /occupancy_map/raycast
```

## 5. 参数配置

参数文件：`navigation_runner/cfg/tunnel/tunnel_nav_param.yaml`

Launch 文件通过 `<rosparam>` 将 YAML 加载到 `/tunnel_navigator/` 命名空间，
然后用 `<param>` 覆盖 `checkpoint_path` 和 `device`（来自 launch 参数）。

### 关键参数

```yaml
# 策略参数 — 必须与训练一致，不可修改
action_limit: 2.0       # m/s
lidar_range: 4.0         # 米
lidar_vfov: [-10.0, 20.0]  # 度
lidar_vbeams: 4
lidar_hres: 10.0         # 度 → 36 水平波束

# 部署参数 — 可调
control_freq: 20.0       # Hz
takeoff_height: 5.0      # 米（与当前隧道训练初始化高度一致）
deterministic: true      # 确定性输出
user_model_simple: false
user_model_profile: m3_diverse # m3_diverse | legacy_perlin | simple
user_model_speed: 2.0    # m/s
user_model_freq_base: 0.1
user_model_freq_scale: 0.2
user_model_vx_bias: 1.5
user_model_vx_amp: 0.5
user_model_vy_amp: 2.0
user_model_vz_amp: 0.2  # 对齐 M3 tunnel offline dataset 的小幅 z 输入
user_model_smoothness_base: 0.4
user_model_smoothness_scale: 0.5
user_model_laziness: 0.3
user_model_seed: 42      # RL / IPC 共用，确保输入序列可复现
safety_min_dist: 0.3     # 米，安全停止距离（0 = 禁用）
collision_dist: 0.05     # 米，碰撞判定距离（低于此值 = 任务失败）
```

**安全机制说明：**
- `min_dist > safety_min_dist`：正常控制
- `collision_dist < min_dist < safety_min_dist`：安全停止（零速指令），距离恢复后自动继续
- `min_dist < collision_dist`：碰撞！发布 `/tunnel_nav/collision` (True)，永久停止。对比实验中视为任务失败

### UserModel 模式

- **m3_diverse 模式**（默认）：对齐 `trajectory_gen_tunnel.yaml` 的 feasible-diverse pilot，
  `vx≈1.5±0.5`、`vy` 为宽 Perlin 扰动、`vz≈±0.2` 小幅扰动。这是 M3 ROS1 批量主实验推荐设置；如需复现旧 ROS1 高度中性输入，可显式设 `user_model_vz_amp:=0.0` / `--user-model-vz-amp 0.0`。
- **legacy_perlin 模式**：旧 ROS1 online user model，`vx=user_model_speed`、
  `vy=user_model_speed*Perlin`、`vz=0`。仅建议用于 ablation / 历史结果复现。
- **simple 模式** (`user_model_simple: true` 或 `user_model_profile:=simple`)：
  `vx=user_model_speed`，`vy=vz=0`，用于 sanity check。
- `user_model_seed` 现在同时传给 RL 与 IPC；两种方法在相同 seed 下会复用同一条
  `UserModelTunnel` 指令序列

**M3 注意事项**：`checkpoint_tunnel_M3_21500.pt` 的训练主线继承了
`tunnel_m2_diverse_pilot`，训练时使用 offline feasible-diverse pilot dataset。
当前 ROS1 批量实验默认使用 `m3_diverse` online 近似，而不是旧 `legacy_perlin`。
它用于验证 M3 模型在 ROS1/Gazebo 中面对 M3-aligned pilot 分布的表现；若需要严格复现
M3 offline dataset 的逐样本输入，需要另行接入 offline pilot replay。

## 6. 架构说明

### 6.1 观测构建（与训练一致）

```
odom (ROS, body-frame twist)
  → rotation matrix → vel_world → quat_rotate_inverse → vel_body (3D)
                                                       → ang_vel_body (3D)
  → quaternion [w,x,y,z] (4D)
  ──────────────────────────────────── state (10D)

UserModelTunnel → human_action (3D, 体帧)

map_manager/raycast → hit points → distance to drone
  → (range - dist) / range → lidar (1, 36, 4)
```

**注意**：CERLAB Gazebo 插件发布 body-frame twist。导航节点先用旋转矩阵转到世界系，
再用 `quat_rotate_inverse` 转回体帧（与训练时 Isaac Sim 的处理保持一致）。

### 6.2 残差动作（Residual Action）

当前隧道策略使用的是 **`ConstrainedResidualPPO_Beta`**。  
`human_action` 不是旧版文档里那种显式 `atanh -> tanh` 残差公式，而是直接作为观测输入拼接进特征提取器：

```
feature = concat(lidar_cnn, human_action, state)
actor(feature) → Beta distribution parameters
deterministic deployment → distribution mean
mean in (0,1) → linear map to [-action_limit, action_limit]
```

它仍然是“以 human_action 为辅助输入的残差式策略”，但当前实现**不是**旧的 Tanh residual 公式。

### 6.3 坐标系

- **体帧 (body)**: 前=x, 左=y, 上=z — 观测空间
- **世界帧 (world)**: ENU — 动作输出空间
- **四元数**: 训练使用 [w,x,y,z]（scalar-first），ROS 使用 [x,y,z,w]，导航节点内部自动转换
- **cmd_vel**: 策略输出世界帧速度 → 用 yaw 旋转矩阵转为 body 帧 → 发布给 CERLAB 插件；默认 `gazebo_z_mode=alt_hold` 会用高度保持覆盖 z，`policy`/`policy_clamped`/`blend` 可用于 A/B 验证完整或部分 z 速度执行。为避免起飞低空阶段偏离训练初始高度，`gazebo_policy_z_takeoff_gate=true` 时会先用高度保持，达到 `takeoff_height - gazebo_policy_z_gate_tolerance` 后再让 policy z 接管。

### 6.3.1 隧道地图坐标系

PCD 地图和 Gazebo 世界**必须**使用同一坐标系（世界帧 ENU）。

IsaacSim 训练环境使用 `map_range = [6.0, 12.0, 5.0]`（config 坐标 `[x, y, z]`），
但在 IsaacSim 内部轴映射为 `[y, x, z]`，即：
- 前进方向（config y = 12.0 半轴）= 24m
- 侧向（config x = 6.0 半轴）= 12m
- 高度（config z = 5.0 半轴）= 10m

在 Gazebo 中，无人机朝 +X 方向飞行，因此：

**PCD 地图 (`tunnel_map_default.pcd`)**:
- 当前默认实验使用固定的预生成 PCD / Gazebo world 配对
- 当前 `ros1/README.md` 中记录的生成命令为：`generate_tunnel_map.py --seed 288 -n 15 --cuboid-ratio 0.5`
- X ∈ [-12, 12]（24m，前进方向），Y ∈ [-6, 6]（12m，侧向），Z ∈ [0, 10]（10m，高度）
- 出生区域：X ∈ [-12, -6]
- 结构：地面、天花板、Y=±6 侧壁、X=-10 后墙，以及与 `tunnel_pcd_match_static.world` 对齐的一组固定障碍物

**Gazebo 世界 (`tunnel_pcd_match_static.world`)**:
- 与默认 PCD 地图配对
- 包含一组与默认 PCD 对齐的固定静态障碍物、地面、侧壁和后墙
- 无天花板（Gazebo 中方便观察；lidar_sim 通过 PCD 处理天花板检测）

**无人机出生位置**: `(-8.5, 0.0, 0.1)`，yaw=0（朝 +X 方向），随后起飞到 `z=5.0`  
> 这里的 `-8.5` 是当前 Gazebo 侧安全起飞补偿值；如果做和 Isaac Sim 的严格 airborne 对齐，应从起飞稳定后的空中段开始比较。  
**目标**: `X ≥ 12.0`

**重新生成**:
```bash
cd ros1/navigation_runner/scripts/tunnel_deployment
python3 generate_tunnel_map.py \
  -o ../../cfg/tunnel/tunnel_map_default.pcd \
  -w ../../../uav_simulator/worlds/generated_env/tunnel_pcd_match_static.world \
  --seed 42 -n 170
# 调整障碍物数量: -n 80 (阶段3), -n 120 (阶段4), -n 170 (阶段5/默认)
```

### 6.4 控制流程（tunnel_navigation.py 主循环）

```
1. _takeoff()      — 等待 odom → 发送 Empty 到 /takeoff → 等 3s
2. _raycast_callback (30Hz) — 调用 RayCast 服务 → 更新 self.raypoints_np (numpy)
                              → 发布 PointCloud2 到 /tunnel_nav/lidar_cloud
3. _control_callback (20Hz) — if collision: permanent stop
                              if safety_stop: zero cmd (距离恢复后自动继续)
                              else: _build_obs → policy → _publish_cmd → _publish_vis
4. _safety_check (10Hz bg)  — 向量化 min_dist 计算：
                              min_dist < collision_dist → 碰撞 (任务失败)
                              min_dist < safety_min_dist → 安全停止
                              min_dist >= safety_min_dist → 正常
```

## 7. RViz 可视化

当前有两套 RViz 配置，不要混用：

### 7.1 RL 模式

- 配置文件：`cfg/tunnel/tunnel.rviz`
- Fixed Frame：`map`

| 显示项 | 话题 | 颜色/样式 | 说明 |
|--------|------|-----------|------|
| Grid | — | 灰色 | 参考网格 |
| TF | — | 轴 | 坐标系 |
| Occupancy Map | `/occupancy_map/map_vis` | 黑色方块 | 预构建障碍物地图 |
| LiDAR Points | `/tunnel_nav/lidar_cloud` | 红色点 | 实时 LiDAR 射线命中点 |
| RL Command | `/tunnel_nav/cmd_vel_vis` | 绿色箭头 | RL 策略输出方向 |
| Human Command | `/tunnel_nav/human_cmd_vis` | 蓝色箭头 | UserModel 前进方向 |
| Drone Odom | `/CERLAB/quadcopter/odom` | 箭头轨迹 | 无人机位姿历史 |

### 7.2 IPC 模式

- 配置文件：`cfg/tunnel/ipc_tunnel.rviz`
- Fixed Frame：`world`

| 显示项 | 话题 | 说明 |
|--------|------|------|
| Raw Cloud | `/pcl_render_node/cloud` | PCD 重建后的激光点云 |
| IPC Path | `/ipc/path` | IPC 当前规划路径 |
| A* Path | `/astar/path` | A* 初始路径 |
| SFC | `/ipc/sfc` | 灰色安全走廊，可视效果应接近 `slope_inspection/README.md` |
| ROG Maps | `/rog_map/occ` `/rog_map/unk` `/rog_map/inf_occ` `/rog_map/inf_unk` | 占据/未知/膨胀地图 |
| Goal / Goal Free | `/ipc/goal` `/ipc/goal_free` | 目标点可视化 |
| Drone Odom | `/CERLAB/quadcopter/odom_raw` | 无人机位姿 |

> IPC 原版可视化 publisher 把 marker 固定发在 `world`，但 Gazebo 里程计和点云链路是
> `map -> base_link`。`tunnel_ipc_sim.launch` 现在会额外发布一个恒等静态 TF
> `world -> map`，这样 `/pcl_render_node/cloud`、`/ipc/sfc`、`/astar/path` 和
> `/rog_map/*` 才能在同一个 RViz 视图里同时显示。

## 8. 自动化对比实验

### 8.1 自动对比脚本

```bash
# 先启动 Gazebo
roslaunch uav_simulator start_headless.launch gui:=true &

# 运行自动对比（各 5 轮，每轮 60s 超时）
python3 $(rospack find navigation_runner)/scripts/run_comparison.py \
    --methods rl,ipc \
    --n-trials 5 \
    --timeout 60 \
    --output-dir /root/results
```

### 8.2 分析结果

```bash
python3 $(rospack find navigation_runner)/scripts/analyze_results.py \
    --data-dir /root/results \
    --pcd-file $(rospack find navigation_runner)/cfg/tunnel/tunnel_map_default.pcd
```

**对比指标：**

| 指标 | 说明 |
|------|------|
| 成功率 | 到达目标 (x ≥ 15m) / 总试验次数 |
| 最大前进距离 | 沿隧道前进的最大 X 坐标 |
| 平均飞行速度 | 有效飞行段的平均速度 |
| 安全裕度 | 与最近障碍物的最小距离 |
| 指令平滑度 | 相邻帧指令变化量方差 |

## 9. IPC 桥接层

5 个 Python 节点将 CERLAB Gazebo 的接口适配为 IPC 期望的输入格式：

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `rc_sim_node.py` | 自动起飞 + 回放共享 usermodel RC 输入 | 定时器 + `UserModelTunnel` | `/CERLAB/quadcopter/takeoff`, `/mavros/rc/in` |
| `lidar_sim_node.py` | LiDAR 点云模拟 | PCD + odom | `/pcl_render_node/cloud` |
| `imu_bridge_node.py` | IMU 桥接 | odom + acc | `/mavros/imu/data` |
| `cmd_bridge_node.py` | 指令转换（yaw-only XY + 高度保持） | `PositionCommand` | `/CERLAB/.../cmd_vel` |
| `mavros_fake_node.py` | MAVROS 服务模拟 | 服务请求 | 返回 success=true |

**RC 模拟模式切换流程：**
```
启动 → 等待 Gazebo/takeoff subscriber → 发送一次 /CERLAB/quadcopter/takeoff →
等待 0.6s 起飞 → 0.1s 内进入 Manual → 每个模式保持 0.15s 完成边沿切换 →
AutoPilot(ch10高) → 共享 `UserModelTunnel` 摇杆输入
```

**当前已确认的 IPC 迁移关键修复：**
1. `rc_sim_node.py` 现在会先自动发布一次起飞命令，否则 CERLAB 插件停留在
   `LANDED_MODEL`，`cmd_vel` 再正确也不会起飞。
2. RC 摇杆 PWM 映射已按 `slope_inspection/IPC/include/callback.cpp` 的符号约定修正；
   若不取反，IPC 会把“前进”解释成负向摇杆。
3. `cmd_bridge_node.py` 不再把世界速度旋转到完整机体系，而是与 RL 路径一致，只按
   yaw 旋转 XY，并用 `PositionCommand.position.z + velocity.z` 做高度控制。
   这修复了“`/planning/pos_cmd.velocity.x` 为正，但 `/CERLAB/quadcopter/cmd_vel.linear.x`
   变成负值”的桥接错误。
4. IPC 模式默认使用 `ipc_tunnel.rviz`，并在 launch 里补上 `world -> map` 静态 TF；
   这修复了“话题其实在发，但 RViz 看不到点云/安全走廊”的可视化问题。
5. IPC 的 `rc_sim_node.py` 不再在 AutoPilot 后只发送固定 `forward_stick=0.8`。
   现在它会复用 RL 隧道部署的 `UserModelTunnel`，并把同一条体坐标输入序列映射到
   RC 通道；在默认 `user_model_simple=false` 下，运行时表现为：
   - `channel[1]` 固定为前进对应的 PWM（因为 `vx=user_model_speed`）
   - `channel[0]` 随 Perlin 横向扰动变化
   这才与 RL 测试时的 usermodel 分布一致。若只想手动回退到旧的恒定前进模式，
   才需要显式设置 `use_user_model:=false`。
6. `rc_sim_node.py` 的 startup waiting 已从“起飞后 4s + 多轮 3s 模式等待”压缩为
   “起飞后 0.6s + 0.1/0.15s 的最小边沿脉冲”，避免无人机在真正进入 AutoPilot 前
   就因为 Hover/Pilot 停留过久而跑偏。实际 probe 结果：
   - 旧时序：第一次 AutoPilot RC 前，`x` 约前移 `0.81m`
   - 新时序：第一次 AutoPilot RC 前，`x` 约前移 `0.21m`
   - AutoPilot 首次触发时间也从约 `5.15s` 提前到约 `3.73s`
7. 当前主机侧 `slope_inspection/IPC/include/callback.cpp` 已支持“在 `UAV_Pilot` 中，若
   Channel 10 持续保持高电平，也允许补锁存 AutoPilot”；但**当前 Docker 容器里的预编译
   `ipc_node` 不一定带着这版补丁**。因此 `rc_sim_node.py` 现在会发一组对旧/新二进制都兼容
   的握手：`AUTOPILOT EDGE1` → `AUTOPILOT REARM` → `AUTOPILOT EDGE2` → `AUTOPILOT ACTIVE`。
   这样即使容器里的 `ipc_node` 仍只认严格的 Channel 10 上升沿，也能稳定看到
   `[IPC FSM]: Pilot --> Auto_pilot.`，避免“RC 已打印 AutoPilot，但 IPC FSM 仍停在
   `Pilot`、`/ipc/sfc` 始终不发”的回归。
8. IPC launch 下的 `lidar_sim_node.py` 现在显式发布 `frame_id=world` 的世界坐标点云。
   之前它一直输出 `base_link` 机体系点云；RViz 因为有 TF，看起来仍然“显示正确”，但
   `ROGMap::updateMap()` 会把点坐标直接当成世界坐标用，结果 IPC 内部建图完全错位，
   继而引发 `No sfc!`、`Can not find a grid free near odom!`、CIRI `nan planes` 乃至
   后续 `rog error! new goal is free and unk in inf map!`。
9. `ipc_gazebo_param.yaml` 当前继续保持 `frontier_extraction_en: true`。虽然主机侧
   `ipc_fsm.h` 已有 UNKNOWN/FRONTIER 兼容分支，但 `docker-compose.tunnel.yml`
   **不会热挂载** `slope_inspection/IPC` 到容器里的 `/root/slope_ws/src/`，因此当前容器
   实际运行的预编译 `ipc_node` 仍需要这个运行时 workaround 才能避免首帧点云 segfault。
10. AutoPilot 模式下若当前没有有效 SFC，`TimerCallback()` 现在会让 MPC **保持当前位置**
    而不是继续沿 ref-path 无约束前冲；同时 `GenerateAPolytope()` 在 seed line 退化成单点时
    会退回局部 box corridor，避免 CIRI 在切入 AutoPilot 的前几帧直接生成 NaN 平面。

IPC 参数文件：`cfg/tunnel/ipc_gazebo_param.yaml`

## 10. 文件结构

```
ros1/navigation_runner/
├── scripts/
│   ├── tunnel_navigation.py          # RL 主控制节点（20Hz 循环）
│   ├── tunnel_deployment/            # RL 策略模块
│   │   ├── __init__.py
│   │   ├── policy_net.py             # 独立策略网络（ConstrainedResidualPPO）
│   │   ├── user_model.py             # UserModelTunnel（前进指令生成）
│   │   ├── quat_utils.py             # 四元数工具函数
│   │   └── generate_tunnel_map.py    # 隧道 PCD 地图生成器
│   ├── ipc_bridge/                   # IPC 桥接层
│   │   └── __init__.py
│   ├── rc_sim_node.py                # 遥控器模拟
│   ├── lidar_sim_node.py             # LiDAR 点云模拟
│   ├── imu_bridge_node.py            # IMU 数据桥接
│   ├── cmd_bridge_node.py            # 命令格式转换
│   ├── mavros_fake_node.py           # MAVROS 服务模拟
│   ├── flight_recorder.py            # 飞行数据记录（50Hz → .npz）
│   ├── run_comparison.py             # 自动化多轮对比
│   └── analyze_results.py            # 数据分析 + 绘图
├── launch/
│   ├── tunnel_comparison.launch      # 统一对比启动（RL 或 IPC）
│   └── tunnel_ipc_sim.launch         # IPC 仿真启动（含桥接层）
├── cfg/tunnel/
│   ├── tunnel_nav_param.yaml         # RL 导航参数
│   ├── occupancy_map_tunnel.yaml     # 占用地图参数
│   ├── ipc_gazebo_param.yaml         # IPC Gazebo 参数
│   ├── ipc_tunnel.rviz               # IPC 专用 RViz 配置（Fixed Frame=world）
│   ├── tunnel_map_default.pcd        # 默认隧道障碍物地图
│   └── tunnel.rviz                   # RViz 显示配置
├── cfg/ckpts/
│   └── checkpoint_tunnel_M3_21500.pt # M3 最终版隧道模型权重
├── docs/
│   └── TUNNEL_DEPLOYMENT.md          # 本文档
└── ...

ros1/uav_simulator/
├── launch/
│   ├── start.launch                  # Gazebo（带 GUI）
│   └── start_headless.launch         # Gazebo（支持 headless 模式）
├── worlds/
│   └── generated_env/
│       └── tunnel_pcd_match_static.world  # 由 generate_tunnel_map.py 生成，与 PCD 匹配
├── plugins/libquadcopterPlugin.so    # CERLAB 四旋翼 Gazebo 插件
└── scripts/entrypoint.sh             # Docker 智能入口（Xvfb/X11 检测）

Dockerfile.tunnel_comparison          # Docker 镜像构建
docker-compose.tunnel.yml             # 持久化开发容器
```

## 11. 训练→部署一致性清单

| 项目 | 训练值 | 部署值 | 匹配？ |
|------|--------|--------|--------|
| action_limit | 2.0 m/s | 2.0 m/s | ✅ |
| lidar_range | 4.0 m | 4.0 m | ✅ |
| lidar_vfov | [-10°, 20°] | [-10°, 20°] | ✅ |
| lidar_vbeams | 4 | 4 | ✅ |
| lidar_hres | 10° (36 beams) | 10° (36 beams) | ✅ |
| state 维度 | 10D (body) | 10D (body) | ✅ |
| human_action 维度 | 3D | 3D | ✅ |
| 四元数约定 | [w,x,y,z] | [w,x,y,z] | ✅ |
| 归一化方向 | (range-d)/range | (range-d)/range | ✅ |
| Beta min_concentration | 2.0 | 2.0 | ✅ |
| 确定性推理 | Beta mode | Beta mode | ✅ |
| residual_scale | 从 checkpoint | 从 checkpoint | ✅ |
| pilot 分布 | offline diverse dataset: `vx≈1.5±0.5`, broad `vy`, small `vz≈±0.2` | `m3_diverse`: `vx≈1.5±0.5`, broad `vy`, `vz_amp=0.2` | ✅ 主分布近似；非逐样本 replay |

## 12. 常见问题

### Q: Gazebo 打开后无人机不动

1. 检查 checkpoint 是否加载：
   ```bash
   rosparam get /tunnel_navigator/checkpoint_path
   # 应返回有效文件路径
   ```
2. 检查起飞是否完成：日志应显示 `[TunnelNav] Takeoff command sent, waiting 3s...`
3. 检查 RayCast 服务：`rosservice list | grep raycast`
4. 检查状态话题：`rostopic echo /tunnel_nav/status -n 3`

### Q: SAFETY_STOP 频繁触发

- 查看状态：`rostopic echo /tunnel_nav/status` 中 `min_d` 值
- 降低阈值：在 `tunnel_nav_param.yaml` 中调小 `safety_min_dist`（如 0.15）
- 或启动时覆盖：`rosparam set /tunnel_navigator/safety_min_dist 0.15`
- 设为 0 可完全禁用安全停止（仅保留碰撞检测）

### Q: 碰撞（COLLISION）后无人机不动了

这是预期行为。当 `min_dist < collision_dist`（默认 0.05m），系统判定为碰撞，永久停止。
- 查看碰撞话题：`rostopic echo /tunnel_nav/collision`
- 对比实验中，碰撞 = 试验失败，flight_recorder 会记录
- 如需调整碰撞判定距离：修改 `collision_dist` 参数

### Q: 策略行为异常（乱飞、不避障）

- 确认参数正确加载：`rosparam list /tunnel_navigator`
- 确认 `deterministic: true`
- 检查 LiDAR 数据有效：`rostopic echo /tunnel_nav/lidar_cloud -n 1 | head`
- 在 RViz 中观察 LiDAR 红点是否围绕无人机分布合理

### Q: RViz 看不到内容

- 先确认没有把 RL / IPC 的 RViz 配置混用：
  - RL：`cfg/tunnel/tunnel.rviz`，Fixed Frame=`map`
  - IPC：`cfg/tunnel/ipc_tunnel.rviz`，Fixed Frame=`world`
- IPC 模式下应同时能看到 `/pcl_render_node/cloud`、`/ipc/sfc`、`/astar/path`、
  `/rog_map/*`；若你是手动分步启动而不是用 launch，请补上：
  `rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 world map`
- 检查话题是否发布：
  - RL：`rostopic hz /tunnel_nav/lidar_cloud`
  - IPC：`rostopic hz /pcl_render_node/cloud && rostopic hz /ipc/sfc`

### Q: IPC 模式 ipc_node 启动后崩溃 (segfault)

**当前已确认根因（2026-04-14）**：`slope_inspection/IPC/src/ipc_fsm.h` 的
`LocalPcCallback()` 会无条件执行
`boxSearch(..., FRONTIER, pc_unk_)`。但 Gazebo 隧道参数里曾把
`rog_map/frontier_extraction_en` 设为 `false`。

`ProbMap` 只有在 `frontier_extraction_en=true` 时才会分配 `FreeCntMap`，因此首帧点云一到就会走到：

`ProbMap::boxSearch(FRONTIER)` → `ProbMap::isFrontier()` → `FreeCntMap::getFreeCnt(this=0x0)` → `SIGSEGV`

gdb 回溯关键词：
- `rog_map::ProbMap::boxSearch`
- `rog_map::FreeCntMap::getFreeCnt (this=0x0)`
- `ipc::IPCFSMClass::LocalPcCallback`

**修复**
1. 源码侧：`LocalPcCallback()` 在 frontier 提取关闭时回退到 `UNKNOWN` 搜索；
   修复位于 `slope_inspection/IPC/src/ipc_fsm.h`。
2. 运行侧：`cfg/tunnel/ipc_gazebo_param.yaml` 现在显式启用
   `rog_map.frontier_extraction_en: true`，与当前 Docker 里的预编译 `ipc_node`
   保持一致，因此不重编镜像也能直接跑通。
3. 仍保留 `map_size: [10,10,5] + map_sliding: true`；这解决的是另一个历史上会
   导致 IPC 启动即崩的 A* 内存问题，不能删。

**验证结果**
- `roslaunch navigation_runner tunnel_comparison.launch method:=ipc gui:=false rviz:=false`
  下，`ipc_node` 不再在首帧 `/pcl_render_node/cloud` 后退出
- `/planning/pos_cmd` 与 `/CERLAB/quadcopter/cmd_vel` 已恢复发布
- 历史验证文件：`/root/results/ipc_20260414_155721_trial002.npz`
  （证明 segfault 与落盘链路已修复，但当时仍未完成前进）
- 在后续修复自动起飞、RC 极性、`cmd_bridge_node.py` 速度坐标系和 IPC RViz/TF
  之后，已完成新的基线验证：
  - 录制文件：`/root/results/ipc_20260415_045621_trial003.npz`
  - `goal_reached=True`
  - `samples=3288`
  - `total_time=65.759s`
  - `max_x≈67.48`

**结论**
- “ipc-node 被杀死、无法拿到数据”以及“起飞后停在原地、RViz 没有点云/安全走廊”
  这两类问题都已确认属于**迁移实现 bug / 桥接错误**，而不是 IPC 基线本身无法在
  当前 Gazebo 隧道环境工作。
- 当前默认 tunnel map / spawn 组合下，IPC 基线已经可以自动起飞、显示原版风格
  可视化，并完整记录一轮成功穿隧数据。
- 在补上 AutoPilot 锁存修复并重编容器内 `ipc_node` 之后，`rosout` 已可稳定看到
  `[IPC FSM]: Pilot --> Auto_pilot (latched high Channel 10).`，说明“RC 已切到
  AUTOPILOT，但 IPC 仍卡在 Pilot”这一问题已经被单独修复。
- 之后新增的后继问题也已继续定位并修复：根因不是 RC mode 本身，而是 IPC 桥接层把
  `lidar_sim_node.py` 的**机体系点云**直接喂给了 `ROGMap::updateMap()`，而后者内部假定输入
  已经是**世界坐标**。这会让 RViz“看着对”，但 IPC 内部 occupancy / unknown / SFC 全部错位。
- 修复后，干净重跑中已经不再出现
  `Can not find a grid free near odom!`、CIRI `nan in generated planes`、
  `rog error! new goal is free and unk in inf map!` 或 MPC `Primal Infeasible` 这些故障签名；
  `/ipc/sfc` 重新稳定发布在约 `10Hz`，且 world-frame Marker 已可直接抓到。
- 当前切入 AutoPilot 时仍可能先出现**一次**
  `[fsm]: No sfc! Hold current position.`，这是正常的 warm-up 保护：表示第一帧有效 corridor
  还没准备好，FSM 会先悬停等待，而不是像之前那样继续无约束冲向障碍物。

**历史问题（仍值得保留）**
- RC 消息时序问题：`rc_sim_node.py` 现在会等待仿真时间 > 2s 再发送模式切换。
- `rog_map/map_size` 过大：A* 的 `GL_SIZE` 会爆炸式增长；日志中 `GL_SIZE_`
  应保持在约 `68 68 34`。

### Q: RL 模式 OccMap 报错 "Please check body to camera matrix!"

- 已修复：`occupancy_map_tunnel.yaml` 现在包含 `body_to_camera` 身份矩阵
- 如仍出现，检查 YAML 是否正确加载：`rosparam get /occupancy_map/body_to_camera`

### Q: RL 模式 RViz 看不到占用地图

- 确认 `lidar_sim_node` 在运行：`rosnode info /lidar_sim`
- 确认点云发布：`rostopic hz /pcl_render_node/cloud`（应 ~10Hz）
- 确认 OccMap 使用 pointcloud 模式：`rosparam get /occupancy_map/sensor_input_mode`（应为 1）

### Q: LiDAR 点云与 Gazebo 障碍物不一致

**原因**：PCD 文件和 Gazebo 世界使用不同的障碍物布局。`lidar_sim_node` 从 PCD 生成点云，
而 Gazebo 的深度相机 (`/camera/depth/points`) 渲染实际世界模型。

**解决方案**：使用 `generate_tunnel_map.py --world-output` 同时生成 PCD 和匹配的 Gazebo 世界：
```bash
python3 generate_tunnel_map.py -o tunnel_map.pcd -w tunnel.world --seed 42
```
确保 `tunnel_comparison.launch` 使用 `tunnel_pcd_match_static.world`（已默认配置）。

### Q: Docker 中 Gazebo 卡住

- 确保 `entrypoint.sh` 已设置虚拟显示：`echo $DISPLAY`
- 无 GPU 时应看到 `:99`（Xvfb）；有 X11 应看到宿主机的 `:1` 等
- 检查 Gazebo 是否能找到插件：`echo $GAZEBO_PLUGIN_PATH`

### Q: IPC 起飞后不动 / 看不到点云或灰色安全走廊

按下面顺序检查，基本对应这次迁移里已经踩到的 4 个坑：

1. **确认自动起飞是否发生**
   - Gazebo 日志应出现 `Quadrotor takes off!!` 和 `Entering flying model!`
   - 若没有，说明 `/CERLAB/quadcopter/takeoff` 没发出去；CERLAB 插件在 landed
     状态下不会响应 `cmd_vel`
2. **确认 RC 摇杆极性没有反**
   - `rostopic echo /mavros/rc/in -n 1`
   - 当前 `rc_sim_node.py` 已按原版 IPC 的 `RCCallback` 取反编码 PWM；如果你手动改过，
     正向前进必须仍然对应 IPC 里的正向 joystick x
3. **确认桥接后的命令方向正确**
   - `rostopic echo /planning/pos_cmd -n 3`
   - `rostopic echo /CERLAB/quadcopter/cmd_vel -n 3`
   - 正常情况下，`velocity.x > 0` 时桥接后的 `cmd_vel.linear.x` 也应为正，
     且 `linear.z` 不应无故持续大幅为负
4. **确认 IPC 可视化链路**
   - `rostopic hz /pcl_render_node/cloud`
   - `rostopic hz /ipc/sfc`
   - `rostopic hz /rog_map/occ`
   - RViz 必须使用 `ipc_tunnel.rviz`，Fixed Frame=`world`

### Q: IPC 在真正进入 AutoPilot 前就已经明显向前/向上跑偏

- 这通常不是 usermodel 本身的问题，而是 RC startup timing 太长，导致无人机在
  `Hover` / `Pilot` 状态停留过久
- 当前默认值已经压缩为：
  - `takeoff_wait=0.6`
  - `init_delay=0.1`
  - `switch_delay=0.15`
- 若你手动改大了这些值，第一次 AutoPilot RC 触发前就可能再次出现明显前移/上冲
- 可直接检查：
  `rostopic echo -n 1 /mavros/rc/in`
  中 `channels[10] > 1500` 是否在起飞后很快出现；若迟迟没有，说明 RC mode 切换又被拖慢了

### Q: `[RC Sim] -> AUTOPILOT` 已出现，但 IPC FSM 没有打印 `Pilot --> Auto_pilot`

- 这类现象在当前迁移链里已经复现并确认过：RC 侧 `Channel 10` 明明已经持续拉高，
  但原版 `slope_inspection/IPC/include/callback.cpp` 只在**单次上升沿**上切换
  `UAV_Pilot -> UAV_AutoPilot`。如果那个边沿在 Gazebo/桥接时序中被错过一次，
  IPC 就会永久卡在 `Pilot`，于是 `/ipc/sfc` 也不会再发布。
- 当前修复分成两层：
  1. `rc_sim_node.py` 现在会发一组兼容旧/新 `ipc_node` 的握手：
     `AUTOPILOT EDGE1` → `AUTOPILOT REARM` → `AUTOPILOT EDGE2` → `AUTOPILOT ACTIVE`
  2. `callback.cpp` 在 `UAV_Pilot` 中对“持续高电平的 Channel 10”补做锁存
- 修复后的关键验证信号：
  - 即使当前容器里的 `ipc_node` 仍是旧二进制，也应能在 `rosout` 看到
    `[IPC FSM]: Pilot --> Auto_pilot.`
  - 若容器里的 `ipc_node` 已重编进“持续高电平补锁存”补丁，则会看到
    `[IPC FSM]: Pilot --> Auto_pilot (latched high Channel 10).`
  - `rostopic info /ipc/sfc` 能看到 `/ipc_node` 作为 publisher
- **注意**：仅修改宿主机里的 `slope_inspection/IPC` 源码并不会自动影响当前 Docker 容器。
  若容器里还是旧的 `ipc_node` 二进制，这个问题会原样复现；请按上文“Docker 开发环境”
  中的 whitelist `catkin_make` 命令重编。

### Q: 进入 AutoPilot 后出现 `No sfc!` / CIRI `nan in generated planes` / `Can not find a grid free near odom!`

- 这一串报错现在已经确认根因并修复。真正的问题不是 RC 没切成功，而是 IPC 桥接层之前把
  `lidar_sim_node.py` 生成的 **body-frame 点云**直接发布给了 `ipc_node`；而
  `ROGMap::updateMap()` 会把点坐标当作 **world-frame** 使用。结果是：
  - RViz 因为 TF 存在，点云看起来仍然“贴着无人机”
  - 但 IPC 内部的 occupancy / unknown map 实际上已经错位
  - 切到 AutoPilot 后就会出现 `No sfc!`、CIRI NaN、`find free near odom` 失败甚至
    后面的 `rog error`
- 当前修复包含 4 层：
   1. IPC launch 下的 `lidar_sim_node.py` 改为发布 `frame_id=world` 的世界坐标点云
   2. `ipc_gazebo_param.yaml` 保持 `frontier_extraction_en: true`，与当前容器里的
      预编译 `ipc_node` 保持兼容
   3. `TimerCallback()` 在没有有效 SFC 时强制 hold current position
   4. `GenerateAPolytope()` 在 seed line 退化成单点时，退回局部 box corridor
- 修复后的验证信号：
  - `/ipc/sfc` 恢复稳定发布（约 `10Hz`）
  - 可直接 `rostopic echo -n 1 /ipc/sfc` 抓到 world-frame Marker
  - 干净重跑里不再出现 CIRI NaN、`Can not find a grid free near odom!`、
    `rog error! new goal is free and unk in inf map!` 或 MPC `Primal Infeasible`
- 仍可能看到一次 `[fsm]: No sfc! Hold current position.`；这是切入 AutoPilot 的
  第一拍保护，不是失败。只要随后 `/ipc/sfc` 开始发布，说明 corridor 已经恢复正常。

### Q: 手动跑完一次后没有看到 `.npz` 结果文件

- `tunnel_comparison.launch` 与 `tunnel_ipc_sim.launch` 现在默认把结果写到
  `/root/results`
- 在 Docker compose 环境下，宿主机 `SharedRLControl/ros1/results/` 会同时挂载到
  容器内的 `/root/results` 与 `/root/catkin_ws/results`，因此两条路径看到的是同一份结果
- 若是更早的旧运行，文件可能在 `/tmp/flight_data`
- `flight_recorder.py` 现在会在收到 `/flight_recorder/stop` **或节点 shutdown**
  时保存缓冲数据
- 如需在手动单轮测试中立即落盘，可执行：
  `rostopic pub -1 /flight_recorder/stop std_msgs/Bool "data: true"`

## 自动批量实验与分析（2026-04-15）

### 统一终止与单轮自动清场

- `flight_recorder.py` 现在不再只是被动录制器，而是单轮实验的**统一终止器**：
  - 到达 `goal_x` → 记为 `goal_reached`
  - 最近障碍距离 `< collision_dist` → 记为 `collision`
  - 可选 `timeout_sec > 0` → 记为 `timeout`
- 终止时会执行以下动作：
  1. 发布 latched `/experiment_control/stop`
  2. 保存 `.npz` 与 `run_summary.json`
  3. 若 launch 中 `shutdown_on_complete=true`，则以 `required` 节点身份退出，触发整个
     `roslaunch` 自动 shutdown，确保本轮 Gazebo / bridge / controller 全部清干净
- recorder 现在会在**收到 odom 后立即开始录制**，但碰撞判定仍要等无人机相对初始高度
  至少离地 `0.5m` 后才启用。这样既能避免把 PCD 里的地面点云误判成“起点即 collision”，
  也不会在 IPC AutoPilot 没切成功时整轮实验都没有 recorder、导致自动终止完全失效。
- RL / IPC 两条链都已接入同一个 stop hook：
  - `tunnel_navigation.py` 收到 stop 后改为 pose hold
  - `cmd_bridge_node.py` 收到 stop 后忽略新的 `PositionCommand` 并发布 hold pose
  - `rc_sim_node.py` 收到 stop 后把 RC motion sticks 拉回中性

### 新的批量实验脚本

- 新增脚本：`scripts/batch_tunnel_experiments.py`
- 每个 batch 的流程：
  1. 调用 `tunnel_deployment/generate_tunnel_map.py` 生成新的 `.pcd + .world + obstacles.json`
  2. 对该 batch 生成一组 `user_model_seed`
  3. 对每个 `run_idx`，用**同一个** `user_model_seed` 依次运行 RL 和 IPC，保证输入公平
  4. 每轮 run 使用全新的 `roslaunch navigation_runner tunnel_comparison.launch ...`，
     单轮结束后由 recorder 自动关掉整套 launch
- 推荐用法：
  ```bash
  python3 scripts/batch_tunnel_experiments.py \
      --num-batches 5 \
      --runs-per-batch 10 \
      --device cpu \
      --output-dir /root/results/tunnel_batch_001
  ```
- 常用参数：
  - `--device`：当前 `tunnel_comparison:20260415-ipcfix` 镜像内的 PyTorch 是 CPU-only；若未重建
    CUDA 版镜像，请保持 `--device cpu`。脚本现在会把无效的 `cuda:*` 请求自动回退到 `cpu`
  - `--launch-timeout`：批处理外层 watchdog
  - `--goal-x` / `--collision-dist`：统一终止阈值
  - `--gazebo-z-mode`：默认 `alt_hold`；可设为 `policy`/`policy_clamped`/`blend` 评估执行完整 3D policy velocity 的影响
  - `--disable-gazebo-policy-z-takeoff-gate`：关闭默认起飞门控，用于复现从低空立即执行 policy z 的纯 policy mode
  - `--user-model-speed` / `--user-model-freq-*`：共享 RL / IPC 输入配置
  - `--num-obstacles` / `--cuboid-ratio`：每批次地图生成参数

### 地图生成元数据

- `generate_tunnel_map.py` 现在新增 `--metadata-output`，可同时导出障碍物 JSON 元数据。
- batch runner 会把每个 batch 的地图元数据额外复制成根目录下的
  `b000_obstacles.json`、`b001_obstacles.json` ……
  这是为了兼容 `SharedRLControl/tunnel_test/render_viz.py` 的 batch obstacle 查找逻辑。

### 批量分析与可视化回放

- `scripts/analyze_results.py` 现在同时支持：
  - **旧模式**：直接分析一个平铺的 `.npz` 目录
  - **batch 模式**：递归读取 `batch_manifest.json` / `run_summary.json`
- 默认输出：
  - `analysis/metrics.csv`
  - `analysis/summary.json`
  - `analysis/comparison_plots.png`
  - `analysis/compare_results_ros1.json`
  - `analysis/render_data/`（给 `render_viz.py` 用的 portable bundle）
- `compare_results_ros1.json + analysis/render_data/` 已验证可直接被
  `SharedRLControl/tunnel_test/render_viz.py` 使用，例如：
  ```bash
  python3 SharedRLControl/tunnel_test/render_viz.py \
      /root/results/tunnel_batch_001/analysis/compare_results_ros1.json \
      --trial 0 --static
  ```
  或输出视频：
  ```bash
  python3 SharedRLControl/tunnel_test/render_viz.py \
      /root/results/tunnel_batch_001/analysis/compare_results_ros1.json \
      --trial 0 --fps 20
  ```

## 附录：Git 分支

部署代码位于 `deploy/ros1-tunnel` 分支：
```bash
cd SharedRLControl
git checkout deploy/ros1-tunnel
```
