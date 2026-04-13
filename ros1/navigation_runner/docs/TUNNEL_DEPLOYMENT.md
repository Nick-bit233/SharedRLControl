# Tunnel RL vs IPC — ROS1/Gazebo 部署与对比实验指南

> 将 Isaac Sim 训练的隧道避障 RL 策略部署到 ROS1 (Noetic) + Gazebo，
> 并与 slope_inspection 的 IPC 算法在同一环境中进行公平对比

## 概览

本部署包提供三个功能：
1. **RL 部署**：将 `ConstrainedResidualPPO` 策略网络部署到 ROS1/Gazebo 仿真环境
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

### 1.2 运行 RL 模式

```bash
# 容器内（带 GUI + RViz）
roslaunch navigation_runner tunnel_comparison.launch method:=rl gui:=true

# 无 GUI（headless + Xvfb）
roslaunch navigation_runner tunnel_comparison.launch method:=rl gui:=false
```

> **注意**: RL 模式现在同时启动 `lidar_sim_node`（PCD → PointCloud2）和
> `map_manager`（占用地图 + RayCast 服务）。`lidar_sim_node` 提供实时点云数据，
> `map_manager` 使用预构建 PCD 地图和实时点云来维护占用栅格。

### 1.3 运行 IPC 模式

```bash
roslaunch navigation_runner tunnel_comparison.launch method:=ipc gui:=true
```

### 1.4 关键 launch 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `method` | `rl` | `rl` 或 `ipc` |
| `gui` | `false` | Gazebo GUI（需 X11） |
| `checkpoint` | `cfg/tunnel/checkpoint_best.pt` | RL 模型路径 |
| `tunnel_map` | `cfg/tunnel/tunnel_map_default.pcd` | 隧道 PCD 地图 |
| `device` | `cpu` | `cpu` 或 `cuda:0` |
| `keyboard` | `false` | 键盘控制模式（仅 RL） |
| `rviz` | `true` | 启动 RViz |
| `record` | `true` | 记录飞行数据 |
| `output_dir` | `/tmp/flight_data` | 数据输出目录 |

### 1.5 键盘控制模式

键盘模式用人类实时输入替代程序化的 UserModelTunnel，可随时开关 RL 辅助。

```bash
# 需要 xterm（容器内已安装）和 X11 显示
roslaunch navigation_runner tunnel_comparison.launch method:=rl keyboard:=true gui:=true
```

启动后会自动打开一个 xterm 终端窗口用于键盘控制。操作流程：

1. **等待 Gazebo 和各节点就绪**（控制台出现 `[TunnelNav] KEYBOARD MODE`）
2. **聚焦 xterm 窗口**，按 `T` 键触发起飞
3. 起飞完成后，使用 WASD/QE 手动控制无人机
4. 按 `R` 键开启 RL 辅助 — 策略会修正你的键盘输入
5. 再按 `R` 关闭 RL 辅助，回到纯键盘控制
6. 按 `ESC` 退出

**键位表：**

| 按键 | 功能 |
|------|------|
| W/S | 前进/后退（body X） |
| A/D | 左移/右移（body Y） |
| Q/E | 上升/下降（body Z） |
| T | 起飞 |
| R | 切换 RL 辅助 |
| ESC | 退出 |

**不使用 launch（手动启动）：**

```bash
# 终端 A: 先启动 Gazebo + RL 节点
roslaunch navigation_runner tunnel_comparison.launch method:=rl keyboard:=true gui:=true

# 或者手动启动 teleop 节点（如果不使用 xterm launch-prefix）
rosrun navigation_runner tunnel_keyboard_teleop.py
```

**相关话题：**

| 话题 | 类型 | 说明 |
|------|------|------|
| `/tunnel_nav/user_cmd` | TwistStamped | 键盘→导航节点的速度指令 |
| `/tunnel_nav/takeoff_cmd` | Empty | 起飞触发 |
| `/tunnel_nav/assist_toggle` | Empty | RL 辅助开关 |
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
    "$(rospack find navigation_runner)/cfg/tunnel/checkpoint_best.pt"
rosrun navigation_runner tunnel_navigation.py __name:=tunnel_navigator

# 终端 5: RViz
rviz -d $(rospack find navigation_runner)/cfg/tunnel/tunnel.rviz
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
| 持久化结果 | Docker named volume `tunnel_results` → `/root/results` |
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
takeoff_height: 4.0      # 米（CERLAB 插件默认悬停高度）
deterministic: true      # 确定性输出
user_model_simple: true  # 恒速前进
user_model_speed: 2.0    # m/s
safety_min_dist: 0.3     # 米，安全停止距离（0 = 禁用）
collision_dist: 0.05     # 米，碰撞判定距离（低于此值 = 任务失败）
```

**安全机制说明：**
- `min_dist > safety_min_dist`：正常控制
- `collision_dist < min_dist < safety_min_dist`：安全停止（零速指令），距离恢复后自动继续
- `min_dist < collision_dist`：碰撞！发布 `/tunnel_nav/collision` (True)，永久停止。对比实验中视为任务失败

### UserModel 模式

- **simple 模式** (`user_model_simple: true`)：恒速前进 `user_model_speed` m/s，适合直隧道
- **online 模式** (`user_model_simple: false`)：Perlin 噪声生成多样化指令，更接近训练分布

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

```
human_action_norm = human_action / action_limit     ∈ (-1, 1)
ha_pre_tanh = atanh(ha_norm)

network(state, lidar) → loc_delta, scale
loc = loc_delta × residual_scale + ha_pre_tanh
action = tanh(loc) × action_limit                   → 世界帧 m/s
```

策略以 human_action 为基线，学习残差修正来避障。

### 6.3 坐标系

- **体帧 (body)**: 前=x, 左=y, 上=z — 观测空间
- **世界帧 (world)**: ENU — 动作输出空间
- **四元数**: 训练使用 [w,x,y,z]（scalar-first），ROS 使用 [x,y,z,w]，导航节点内部自动转换
- **cmd_vel**: 策略输出世界帧速度 → 用 yaw 旋转矩阵转为 body 帧 → 发布给 CERLAB 插件

### 6.3.1 隧道地图坐标系

PCD 地图和 Gazebo 世界**必须**使用同一坐标系（世界帧 ENU）。

IsaacSim 训练环境使用 `map_range = [6.0, 12.0, 5.0]`（config 坐标 `[x, y, z]`），
但在 IsaacSim 内部轴映射为 `[y, x, z]`，即：
- 前进方向（config y = 12.0 半轴）= 24m
- 侧向（config x = 6.0 半轴）= 12m
- 高度（config z = 5.0 半轴）= 10m

在 Gazebo 中，无人机朝 +X 方向飞行，因此：

**PCD 地图 (`tunnel_map_default.pcd`)**:
- 由 `scripts/tunnel_deployment/generate_tunnel_map.py --seed 42` 生成
- X ∈ [-12, 12]（24m，前进方向），Y ∈ [-6, 6]（12m，侧向），Z ∈ [0, 10]（10m，高度）
- 出生区域：X ∈ [-12, -6]，无障碍物
- 障碍区域：X ∈ [-6, +12]，170 个随机圆柱
- 结构：地面、天花板、Y=±6 侧壁、X=-10 后墙

**Gazebo 世界 (`tunnel_pcd_match_static.world`)**:
- 由同一脚本 `--world-output` 生成（同 seed=42，障碍物位置完全一致）
- 包含：地面、侧壁、后墙、170 个圆柱模型
- 无天花板（Gazebo 中方便观察；lidar_sim 通过 PCD 处理天花板检测）

**无人机出生位置**: (-8.0, 0.0, 0.1)，yaw=0（朝 +X 方向），位于出生区域内
**目标**: X ≥ 10.0（穿过 18m 障碍区域）

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

RViz 配置文件：`cfg/tunnel/tunnel.rviz`

| 显示项 | 话题 | 颜色/样式 | 说明 |
|--------|------|-----------|------|
| Grid | — | 灰色 | 参考网格 |
| TF | — | 轴 | 坐标系 |
| Occupancy Map | `/occupancy_map/map_vis` | 黑色方块 | 预构建障碍物地图 |
| LiDAR Points | `/tunnel_nav/lidar_cloud` | 红色点 | 实时 LiDAR 射线命中点 |
| RL Command | `/tunnel_nav/cmd_vel_vis` | 绿色箭头 | RL 策略输出方向 |
| Human Command | `/tunnel_nav/human_cmd_vis` | 蓝色箭头 | UserModel 前进方向 |
| Drone Odom | `/CERLAB/quadcopter/odom` | 箭头轨迹 | 无人机位姿历史 |

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
    --output-dir /tmp/flight_data
```

### 8.2 分析结果

```bash
python3 $(rospack find navigation_runner)/scripts/analyze_results.py \
    --data-dir /tmp/flight_data \
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
| `rc_sim_node.py` | 模拟遥控器模式切换 | 定时器 | `/mavros/rc/in` |
| `lidar_sim_node.py` | LiDAR 点云模拟 | PCD + odom | `/pcl_render_node/cloud` |
| `imu_bridge_node.py` | IMU 桥接 | odom + acc | `/mavros/imu/data` |
| `cmd_bridge_node.py` | 指令转换 | `PositionCommand` | `/CERLAB/.../cmd_vel` |
| `mavros_fake_node.py` | MAVROS 服务模拟 | 服务请求 | 返回 success=true |

**RC 模拟模式切换流程：**
```
启动 → 2s 静默 → Manual(ch4高) → 3s → Hover(ch5高) → 3s →
Pilot(ch5高) → 3s → AutoPilot(ch10高) → 持续前进摇杆
```

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
│   ├── checkpoint_best.pt            # 训练好的 RL 模型权重（1.6MB）
│   ├── tunnel_map_default.pcd        # 默认隧道障碍物地图
│   └── tunnel.rviz                   # RViz 显示配置
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
| residual_scale | 从 checkpoint | 从 checkpoint | ✅ |

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

- 确认 Fixed Frame 为 `map`
- 手动添加话题：Add → By Topic → 选择 `/tunnel_nav/lidar_cloud` 等
- 检查话题是否发布：`rostopic hz /tunnel_nav/lidar_cloud`

### Q: IPC 模式 ipc_node 启动后崩溃 (segfault)

**可能原因 1**: RC 消息时序问题 — `rc_sim_node` 在仿真时间 0 时发送消息导致 IPC 的 RCCallback 访问空向量。
- 已修复：`rc_sim_node.py` 等待仿真时间 > 2s 后再发布

**可能原因 2**: `rog_map/map_size` 过大导致 A* 内存爆炸 — IPC 的 A* 路径规划器用 `rog_map` 的 `map_size_d` 和 `inflation_resolution` 来分配网格：
  `GL_SIZE = (map_size / resolution + 1)^3`，每个 GridNode 约 64 字节 + malloc 开销。
  - `[50,30,10]` @ 0.15 → 4,499,538 节点 ≈ 360MB — **导致崩溃**
  - `[10,10,5]` @ 0.15 → 157,216 节点 ≈ 10MB — 安全
  - 原版 slope_inspection 使用 `[8,8,4]` + `map_sliding: true`，地图自动跟随无人机
- 已修复：`map_size` 从 [50,30,10] 回退到 [10,10,5]，配合 `map_sliding: true`
- 首次 `updateMap` 时会打印 `cur_pose out of map range, reset the map` — 这是正常行为（地图滑动到无人机位置）
- 验证：日志中 `GL_SIZE_` 应为约 `68 68 34`

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

### Q: IPC 启动后不动

1. IPC 可执行：`rosrun ipc ipc_node` 是否能运行
2. MAVROS 模拟：`rosservice list | grep mavros`
3. RC 模式切换：`rostopic echo /mavros/rc/in -n 1`（应看到 channel 数据）
4. LiDAR 数据：`rostopic hz /pcl_render_node/cloud`

## 附录：Git 分支

部署代码位于 `deploy/ros1-tunnel` 分支：
```bash
cd SharedRLControl
git checkout deploy/ros1-tunnel
```
