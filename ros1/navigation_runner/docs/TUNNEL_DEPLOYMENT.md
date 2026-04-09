# Tunnel RL vs IPC — ROS1/Gazebo 部署与对比实验指南

> 将 Isaac Sim 训练的隧道避障 RL 策略部署到 ROS1 (Noetic) + Gazebo，
> 并与 slope_inspection 的 IPC 算法在同一环境中进行公平对比

## 概览

本部署包提供两个功能：
1. **RL 部署**：将 `ConstrainedResidualPPO` 策略网络部署到 ROS1/Gazebo 仿真环境
2. **RL vs IPC 对比**：在相同 CERLAB Gazebo 仿真器和同一隧道地图中，对比 RL 策略和 slope_inspection IPC 算法

**对比实验架构：**
```
               ┌── RL 模式 ─────────────────────┐
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

## 1. 系统要求

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 20.04 LTS |
| ROS | Noetic (`ros-noetic-desktop-full`) |
| Python | 3.8+ (ROS Noetic 默认) |
| PyTorch | ≥ 1.13 (CPU 或 CUDA) |
| Gazebo | 11 (随 ROS Noetic 安装) |

## 2. 安装步骤

### 2.1 安装 ROS Noetic（若未安装）

```bash
# 添加 ROS 源
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-noetic-desktop-full

# 初始化 rosdep
sudo rosdep init
rosdep update

echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2.2 安装 PyTorch

```bash
# CPU 版
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 或 CUDA 版 (根据你的 CUDA 版本选择)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2.3 创建 catkin 工作空间

```bash
mkdir -p ~/tunnel_ws/src
cd ~/tunnel_ws/src

# 方法 A: 软链接（开发时推荐）
ln -s /path/to/SharedRLControl/ros1/navigation_runner .
ln -s /path/to/SharedRLControl/ros1/map_manager .
ln -s /path/to/SharedRLControl/ros1/uav_simulator .
ln -s /path/to/SharedRLControl/ros1/onboard_detector .

# 方法 B: 直接拷贝
# cp -r /path/to/SharedRLControl/ros1/{navigation_runner,map_manager,uav_simulator,onboard_detector} .
```

### 2.4 安装 ROS 依赖

```bash
cd ~/tunnel_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 2.5 编译工作空间

```bash
cd ~/tunnel_ws
catkin_make
source devel/setup.bash

# 建议添加到 .bashrc
echo "source ~/tunnel_ws/devel/setup.bash" >> ~/.bashrc
```

## 3. 准备文件

### 3.1 Checkpoint 文件

训练好的模型文件位于：
```
SharedRLControl/isaac-training/outputs/curriculum_stage5/
  2026-03-24_14-02-37/wandb/run-20260324_140244-w85uemxv/files/checkpoint_best.pt
```

建议拷贝到工作空间：
```bash
mkdir -p ~/tunnel_ws/checkpoints
cp /path/to/checkpoint_best.pt ~/tunnel_ws/checkpoints/
```

### 3.2 隧道地图

使用预生成的默认地图（随代码提供）：
```
navigation_runner/cfg/tunnel/tunnel_map_default.pcd
```

或生成新的随机隧道地图：
```bash
cd ~/tunnel_ws/src/navigation_runner/scripts
python3 tunnel_deployment/generate_tunnel_map.py \
    -o ~/tunnel_ws/maps/my_tunnel.pcd \
    -n 60 \
    --resolution 0.15 \
    --seed 42
```

参数说明：
- `-n`：障碍物数量（默认 50）
- `--resolution`：点云分辨率，米（默认 0.15）
- `--seed`：随机种子（可复现）
- `--range`：半范围 x y z（默认 `6.0 12.0 5.0`）

## 4. 运行仿真

### 4.1 一键启动

```bash
roslaunch navigation_runner tunnel_sim.launch \
    checkpoint:=$HOME/tunnel_ws/checkpoints/checkpoint_best.pt \
    tunnel_map:=$(rospack find navigation_runner)/cfg/tunnel/tunnel_map_default.pcd
```

### 4.2 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `checkpoint` | (必填) | `.pt` 文件路径 |
| `tunnel_map` | (必填) | `.pcd` 隧道地图路径 |
| `device` | `cpu` | `cpu` 或 `cuda:0` |
| `user_model_simple` | `true` | `true`=恒速前进, `false`=Perlin 噪声 |
| `use_safety_shield` | `false` | 启用 ORCA 安全护盾 |
| `rviz` | `true` | 启动 RViz 可视化 |
| `use_px4` | `false` | 使用 PX4/MAVROS（真机） |

### 4.3 分步启动（调试用）

```bash
# 终端 1: Gazebo 仿真器
roslaunch uav_simulator start.launch

# 终端 2: 占用地图服务
rosparam load $(rospack find navigation_runner)/cfg/tunnel/occupancy_map_tunnel.yaml /occupancy_map
rosparam set /occupancy_map/prebuilt_map_directory "/path/to/tunnel_map.pcd"
rosrun map_manager occupancy_map_node

# 终端 3: RL 导航节点
rosparam load $(rospack find navigation_runner)/cfg/tunnel/tunnel_nav_param.yaml /tunnel_navigator
rosparam set /tunnel_navigator/checkpoint_path "/path/to/checkpoint_best.pt"
rosrun navigation_runner tunnel_navigation.py __name:=tunnel_navigator

# 终端 4: RViz（可选）
rviz -d $(rospack find navigation_runner)/cfg/tunnel/tunnel.rviz
```

## 5. 参数调优

参数配置文件：`navigation_runner/cfg/tunnel/tunnel_nav_param.yaml`

### 5.1 关键参数

```yaml
# 策略参数 — 必须与训练一致
action_limit: 2.0       # m/s，动作上限
lidar_range: 4.0         # 米，LiDAR 感知范围
lidar_vfov: [-10.0, 20.0]  # 度，垂直视场角
lidar_vbeams: 4          # 垂直波束数
lidar_hres: 10.0         # 度，水平分辨率

# 部署参数 — 可调
control_freq: 20.0       # Hz，控制频率（训练时 ~60Hz）
deterministic: true      # 确定性输出（推荐）
takeoff_height: 1.0      # 起飞高度（米）
user_model_speed: 2.0    # 前进速度（m/s）
```

### 5.2 UserModel 模式

- **simple 模式** (`user_model_simple: true`)：恒速前进，适合直隧道
- **online 模式** (`user_model_simple: false`)：Perlin 噪声生成多样化指令，更接近训练时的输入分布

## 6. 话题与服务

### 订阅
| 话题 | 类型 | 说明 |
|------|------|------|
| `/CERLAB/quadcopter/odom` | `nav_msgs/Odometry` | 四旋翼里程计 |

### 发布
| 话题 | 类型 | 说明 |
|------|------|------|
| `/CERLAB/quadcopter/cmd_vel` | `geometry_msgs/TwistStamped` | 速度指令 (Gazebo) |
| `/CERLAB/quadcopter/takeoff` | `std_msgs/Float64` | 起飞指令 |
| `/tunnel_nav/cmd_vel_vis` | `visualization_msgs/MarkerArray` | 动作可视化 |
| `/tunnel_nav/human_cmd_vis` | `visualization_msgs/MarkerArray` | 用户指令可视化 |

### 服务调用
| 服务 | 类型 | 说明 |
|------|------|------|
| `/occupancy_map/raycast` | `map_manager/RayCast` | LiDAR 射线投射 |

## 7. 项目文件结构

```
ros1/navigation_runner/
├── scripts/
│   ├── tunnel_navigation.py          # 主控制节点（隧道 RL 部署）
│   ├── tunnel_deployment/
│   │   ├── __init__.py
│   │   ├── policy_net.py             # 独立策略网络（从训练代码移植）
│   │   ├── user_model.py             # UserModelTunnel（前进指令生成）
│   │   ├── quat_utils.py             # 四元数工具函数
│   │   └── generate_tunnel_map.py    # 隧道地图 PCD 生成器
│   └── navigation.py                 # 原版 NavRL 节点（参考，未修改）
├── launch/
│   └── tunnel_sim.launch             # 完整仿真启动文件
├── cfg/tunnel/
│   ├── tunnel_nav_param.yaml          # 导航节点参数
│   ├── occupancy_map_tunnel.yaml      # 占用地图参数
│   ├── tunnel_map_default.pcd         # 默认隧道地图
│   └── tunnel.rviz                    # RViz 配置
└── docs/
    └── TUNNEL_DEPLOYMENT.md           # 本文档
```

## 8. 架构说明

### 8.1 观测构建（与训练一致）

```
odom (ROS) → vel_world → quat_rotate_inverse → vel_body (3D)
                                               → ang_vel_body (3D)
           → quaternion [w,x,y,z] (4D)
           ────────────────────────────── state (10D)

UserModelTunnel → human_action (3D, 体帧)

map_manager/raycast → distances → normalize → lidar (1, 36, 4)
```

### 8.2 残差动作（Residual Action）

```
human_action_norm = human_action / action_limit  →  ha ∈ (-1, 1)
ha_pre_tanh = atanh(ha_norm)

network → loc_delta, scale
loc = loc_delta × residual_scale + ha_pre_tanh
action = tanh(loc) × action_limit  →  world-frame velocity (m/s)
```

策略以 human_action 为基线，学习残差修正来避障。

### 8.3 坐标系

- **体帧 (body)**: 前=x, 左=y, 上=z — 观测空间
- **世界帧 (world)**: ENU (东-北-天) — 动作空间
- **四元数**: 训练使用 [w,x,y,z]（scalar-first），ROS 使用 [x,y,z,w]，节点内部自动转换

## 9. 常见问题

### Q: Gazebo 打开后无人机不动

检查：
1. checkpoint 路径是否正确：`rosparam get /tunnel_navigator/checkpoint_path`
2. 地图是否加载：查看 `/occupancy_map/map_vis` 话题
3. 节点是否正常：`rosnode info /tunnel_navigator`
4. 起飞是否完成：节点会先发送起飞指令，等待 5 秒后开始 RL 控制

### Q: 策略行为异常（乱飞、不避障）

- 确认 `action_limit`、`lidar_range` 等参数与训练配置一致
- 检查 `deterministic: true`（部署时推荐）
- 确认 LiDAR 归一化方向正确（近距离=高值）

### Q: RayCast 服务不可用

```bash
# 检查服务是否存在
rosservice list | grep raycast

# 检查地图是否加载
rostopic echo /occupancy_map/map_vis -n 1
```

### Q: 如何在真机上使用？

1. 设置 `use_px4:=true`
2. 确保 MAVROS 已连接并发送心跳
3. 将 LiDAR 数据替换为真实传感器（修改 RayCast 调用为真实点云处理）
4. 调低 `user_model_speed`（建议 ≤1.0 m/s 起步）

### Q: 如何更换隧道场景？

```bash
# 生成不同难度的隧道
python3 generate_tunnel_map.py -o easy_tunnel.pcd -n 30 --seed 1
python3 generate_tunnel_map.py -o hard_tunnel.pcd -n 80 --seed 2

# 启动时指定
roslaunch navigation_runner tunnel_sim.launch tunnel_map:=/path/to/hard_tunnel.pcd ...
```

## 10. 训练→部署一致性清单

| 项目 | 训练值 | 部署值 | 匹配？ |
|------|--------|--------|--------|
| action_limit | 2.0 m/s | 2.0 m/s | ✅ |
| lidar_range | 4.0 m | 4.0 m | ✅ |
| lidar_vfov | [-10°, 20°] | [-10°, 20°] | ✅ |
| lidar_vbeams | 4 | 4 | ✅ |
| lidar_hres | 10° | 10° | ✅ |
| state 维度 | 10D (body) | 10D (body) | ✅ |
| human_action 维度 | 3D | 3D | ✅ |
| 四元数约定 | [w,x,y,z] | [w,x,y,z] | ✅ |
| 归一化方向 | (range-d)/range | (range-d)/range | ✅ |
| residual_scale | 从 checkpoint 加载 | 从 checkpoint 加载 | ✅ |

## 附录：Git 分支

部署代码位于 `deploy/ros1-tunnel` 分支：
```bash
cd SharedRLControl
git checkout deploy/ros1-tunnel
```

---

## 11. RL vs IPC 对比实验

### 11.1 方案 A：Docker 快速部署（推荐）

适用于开发机不是 Ubuntu 20.04 的情况。需要 Docker 和 `slope_inspection:test` 镜像。

```bash
cd SharedRLControl

# 构建 Docker 镜像（包含 Gazebo + IPC + PyTorch + 所有桥接节点）
docker build -f Dockerfile.tunnel_comparison -t tunnel_comparison:latest .

# 启动容器（需要 X11 转发或 VNC 以查看 RViz/Gazebo）
xhost +local:docker
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/checkpoints:/root/checkpoints \
    -v /tmp/flight_data:/tmp/flight_data \
    tunnel_comparison:latest \
    bash

# 容器内：运行 RL 模式
roslaunch navigation_runner tunnel_comparison.launch \
    method:=rl \
    checkpoint:=/root/checkpoints/checkpoint_best.pt

# 容器内：运行 IPC 模式
roslaunch navigation_runner tunnel_comparison.launch method:=ipc
```

### 11.2 方案 B：原生 Ubuntu 20.04 部署

#### 11.2.1 额外依赖（在第 2 节基础上）

```bash
# IPC 依赖
sudo apt install ros-noetic-mavros-msgs

# OsqpEigen (IPC 的 MPC 求解器)
sudo apt install libeigen3-dev cmake
cd /tmp
git clone --recursive https://github.com/osqp/osqp.git
cd osqp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && sudo make install

git clone https://github.com/robotology/osqp-eigen.git /tmp/osqp-eigen
cd /tmp/osqp-eigen && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && sudo make install

# Python 依赖
pip3 install scipy matplotlib
```

#### 11.2.2 编译 IPC 包

```bash
cd ~/tunnel_ws/src

# 软链接 slope_inspection 的 IPC 包
ln -s /path/to/slope_inspection/IPC .
ln -s /path/to/slope_inspection/mars_planning_utils/mars_quadrotor_msgs ./quadrotor_msgs
ln -s /path/to/slope_inspection/mars_base .
ln -s /path/to/slope_inspection/rog_map .

# 编译
cd ~/tunnel_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

### 11.3 运行对比实验

#### 单独运行 RL

```bash
roslaunch navigation_runner tunnel_comparison.launch \
    method:=rl \
    checkpoint:=/path/to/checkpoint_best.pt
```

#### 单独运行 IPC

```bash
roslaunch navigation_runner tunnel_comparison.launch method:=ipc
```

#### 自动化对比（多次试验）

```bash
# 先启动 Gazebo 仿真器
roslaunch uav_simulator start.launch gui:=false

# 然后运行自动对比脚本
python3 $(rospack find navigation_runner)/scripts/run_comparison.py \
    --methods rl,ipc \
    --n-trials 5 \
    --timeout 60 \
    --output-dir /tmp/flight_data
```

### 11.4 分析结果

```bash
python3 $(rospack find navigation_runner)/scripts/analyze_results.py \
    --data-dir /tmp/flight_data \
    --pcd-file $(rospack find navigation_runner)/cfg/tunnel/tunnel_map_default.pcd

# 输出文件：
#   /tmp/flight_data/analysis/comparison_plots.png  — 可视化对比图
#   /tmp/flight_data/analysis/metrics.csv           — 详细指标表
```

**对比指标：**

| 指标 | 说明 |
|------|------|
| 成功率 | 到达目标 / 总试验次数 |
| 最大前进距离 | 沿隧道前进的最大 X 坐标 |
| 平均飞行速度 | 平均/最大速度 |
| 路径平滑度 | 加速度方差 |
| 安全裕度 | 与最近障碍物的最小距离 |
| 指令平滑度 | 相邻帧指令变化量 |

### 11.5 IPC 桥接层说明

5 个 Python 桥接节点将 CERLAB Gazebo 的接口适配为 IPC 期望的输入格式：

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `rc_sim_node` | 模拟遥控器 | 定时器 | `/mavros/rc/in` |
| `lidar_sim_node` | LiDAR 模拟 | PCD + odom | `/pcl_render_node/cloud` |
| `imu_bridge_node` | IMU 桥接 | odom + acc | `/mavros/imu/data` |
| `cmd_bridge_node` | 指令转换 | `PositionCommand` | `/CERLAB/.../cmd_vel` |
| `mavros_fake_node` | MAVROS 模拟 | 服务请求 | 返回 success=true |

**RC 模拟模式切换流程：**
```
启动 → 2s 静默 → Manual(ch4高) → 3s → Hover(ch5高) → 3s →
Pilot(ch5高) → 3s → AutoPilot(ch10高) → 持续前进摇杆
```

### 11.6 IPC 参数调优

IPC 参数文件：`cfg/tunnel/ipc_gazebo_param.yaml`

关键调整（已匹配 RL 约束条件）：
- `fsm.vel_ref: 2.0` — 匹配 RL 的 `action_limit: 2.0`
- `mpc.v{x,y,z}_max: ±3.0` — 速度限制接近 RL
- `rog_map.map_size: [15,25,5]` — 覆盖整个隧道
- `astar.map_size_{x,y,z}: 30/15/8` — 扩大路径搜索范围

## 12. 文件结构总览

```
ros1/navigation_runner/
├── scripts/
│   ├── tunnel_navigation.py          # RL 控制节点
│   ├── tunnel_deployment/            # RL 策略包
│   │   ├── policy_net.py             # 独立策略网络
│   │   ├── user_model.py             # 前进指令生成
│   │   ├── quat_utils.py             # 四元数工具
│   │   └── generate_tunnel_map.py    # 隧道地图生成器
│   ├── ipc_bridge/                   # IPC 桥接层
│   │   └── __init__.py
│   ├── rc_sim_node.py                # 遥控器模拟
│   ├── lidar_sim_node.py             # LiDAR 点云模拟
│   ├── imu_bridge_node.py            # IMU 数据桥接
│   ├── cmd_bridge_node.py            # 命令格式转换
│   ├── mavros_fake_node.py           # MAVROS 服务模拟
│   ├── comparison/                   # 对比实验工具
│   │   └── __init__.py
│   ├── flight_recorder.py            # 飞行数据记录
│   ├── run_comparison.py             # 自动化实验
│   └── analyze_results.py            # 数据分析 + 绘图
├── launch/
│   ├── tunnel_sim.launch             # RL 仿真启动
│   ├── tunnel_ipc_sim.launch         # IPC 仿真启动
│   └── tunnel_comparison.launch      # 统一对比启动
├── cfg/tunnel/
│   ├── tunnel_nav_param.yaml         # RL 导航参数
│   ├── occupancy_map_tunnel.yaml     # 占用地图参数
│   ├── ipc_gazebo_param.yaml         # IPC Gazebo 参数
│   ├── tunnel_map_default.pcd        # 默认隧道地图
│   └── tunnel.rviz                   # RViz 配置
├── docs/
│   └── TUNNEL_DEPLOYMENT.md          # 本文档
└── Dockerfile.tunnel_comparison      # Docker 一键部署
```

## 13. 常见问题（对比实验）

### Q: IPC 启动后不动或报错

检查：
1. IPC 编译是否成功：`rosrun ipc ipc_node` 是否可执行
2. MAVROS 模拟是否运行：`rosservice list | grep mavros`
3. RC 模式切换是否完成：查看 `rc_sim_node` 日志，应该依次显示 Manual→Hover→Pilot→AutoPilot
4. LiDAR 数据是否发布：`rostopic hz /pcl_render_node/cloud`

### Q: IPC 飞行效果不佳

可能原因：
- CERLAB Gazebo 的 PID 速度跟踪与 IPC 假设的双积分器模型有差异
- 调整 `ipc_gazebo_param.yaml` 中的 MPC 参数
- 降低 `vel_ref` 和 MPC 速度限制

### Q: 对比不公平？

确保：
- 同一张 PCD 地图
- 同一个 Gazebo 仿真器（相同物理参数）
- 同一起点和终点
- RL 的 `action_limit` 与 IPC 的 `vel_ref` 匹配
- 多次试验取统计值（≥5 次）
