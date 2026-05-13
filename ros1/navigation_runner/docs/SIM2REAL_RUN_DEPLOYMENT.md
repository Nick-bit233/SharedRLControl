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

## 飞行过程记录

真机实验过程中，记录以下信息：
- 飞行位置和真实速度（相对起飞位置），原始遥控指令速度，模型输出指令速度
- 激光雷达数据中，距离所有障碍物的最小距离，以及与机头前进方向最靠近的扫描点获得的与障碍物的距离
