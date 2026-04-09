# Tunnel RL Policy — ROS1/Gazebo 部署指南

> 将 Isaac Sim 训练的隧道避障 RL 策略部署到 ROS1 (Noetic) + Gazebo 仿真环境

## 概览

本部署包将 `ConstrainedResidualPPO` 策略网络从 Isaac Sim 训练环境迁移到标准 ROS1/Gazebo 仿真环境，用于后续真机实验前的集成验证。

**部署架构：**
```
┌──────────────────────────────────────────────────────┐
│                 tunnel_navigator (ROS node)           │
│  ┌───────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │UserModel  │→ │TunnelPolicy  │→ │  cmd_vel pub  │  │
│  │(前进指令) │  │(残差避障RL)  │  │ (速度指令)    │  │
│  └───────────┘  └──────────────┘  └───────────────┘  │
│        ↑               ↑                              │
│   body-frame     state + lidar                        │
└──────────────────────────────────────────────────────┘
         │                  │
         │    ┌─────────────┘
         │    │
    ┌────▼────▼────┐      ┌─────────────────┐
    │ Gazebo Sim   │      │ occupancy_map   │
    │ (四旋翼)     │ ←——→ │ (RayCast LiDAR) │
    └──────────────┘      └─────────────────┘
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
