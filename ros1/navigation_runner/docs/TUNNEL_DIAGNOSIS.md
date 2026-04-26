# ROS1/Gazebo 隧道部署诊断结论

> **仓库状态更新说明**  
> 本文的根因归因仍然有效，但其中的定量轨迹与基线参数对应的是诊断时使用的那一版默认地图快照。  
> 之后仓库已进一步更新：`ros1/README.md` 中的默认地图生成命令变为 `--seed 288 -n 30 --cuboid-ratio 0.`，`tunnel_comparison.launch` 的默认 `spawn_x` 也进一步调整为 `-8.5`，作为当前 Gazebo 侧的**安全起飞微调**。  
> 因此，下面提到的旧 `spawn_x=-10` 仍然是已确认的历史错误基线，而中间诊断时采用的 `-7.0` 应理解为**历史审计参考值**，不是当前仓库的默认起点。

## 结论

当前这条 `ROS1/Gazebo -> tunnel_navigation.py -> TunnelPolicyNet` 隧道部署链路里，**之前看到的“大幅偏离 Isaac Sim”主要不是模型本身的 sim-transfer 失效，而是实现与基线存在多处关键错误**。  

已经确认并修复的主问题包括：

| 类别 | 问题 | 影响 |
| --- | --- | --- |
| 推理实现 | `TunnelPolicyNet` 把特征拼接成了 `[cnn, state, human_action]`，而训练实际使用 `[cnn, human_action, state]` | 同一输入下 ROS1 standalone 输出与训练策略严重不一致 |
| 参考实验 | `compare_ipc_rl.py` 给策略喂的是 `prev_human_action`，而训练环境使用当前步 `human_action` | Isaac Sim 参考链路本身就和训练观测不一致 |
| 运行基线 | `tunnel_comparison.launch` 旧默认 `spawn_x=-10.0`、`goal_x=10.0` | `spawn_x=-10` 会把飞机放到后墙边界，实验一开始就不公平 |
| 参考初始条件 | `tunnel_terrain.py` 旧 `INIT_POS=[-8,0,4]` | 与训练环境 `[-7,0,5]` 不一致 |

修复这些问题后，Gazebo 实际运行轨迹与“相同策略 + 相同地图 + 相同 user model 的理想速度跟踪离线滚动”已经比较接近，说明**剩余问题更像是策略在该固定地图和当前 Perlin user model 下的行为结果，而不是 ROS1/Gazebo 独有的实现 bug**。

## 复现实验基线

本次诊断统一采用以下基线：

- 入口：`roslaunch navigation_runner tunnel_comparison.launch method:=rl`
- 当前默认权重：`ros1/navigation_runner/cfg/ckpts/checkpoint_tunnel_M3_21500.pt`
- 地图：`ros1/navigation_runner/cfg/tunnel/tunnel_map_default.pcd`
- Gazebo world：`ros1/uav_simulator/worlds/generated_env/tunnel_pcd_match_static.world`
- 算法：`ConstrainedResidualPPO_Beta`
- 训练入口：`isaac-training/experiments/04_tunnel_task/train.py`

诊断时的 airborne 对照主要围绕“旧错误基线 `spawn_x=-10`”与“中间审计参考值 `spawn_x=-7`”展开；**当前仓库默认起飞点已进一步调整为 `spawn_x=-8.5`**，用于避免当前 Gazebo PCD/world 资产组合下的起飞碰撞。

历史诊断时已确认：

- `ros1/navigation_runner/cfg/tunnel/checkpoint_best.pt`
- `shared_demos/ckpts/260331/checkpoint_final.pt`

是同一份权重（SHA256 一致）：

`5dc96155ec22cd8eca80cc6414d2ccb0f150e6fa7e774f7f0ee6280813858055`

## 关键证据

### 1. ROS1 standalone 推理之前是错的

对同一 checkpoint、同一 observation 张量做离线对比：

| 对比项 | mean abs diff | max abs diff |
| --- | ---: | ---: |
| 修复前 | `1.5619` | `4.7396` |
| 修复后 | `0.000333` | `0.002712` |

这说明此前 ROS1 standalone 模型实现本身就不等价，Gazebo 结果不能直接拿来和 Isaac Sim 做结论。

### 2. `prev_human_action` 不是小误差，而是实质性观测错误

把训练侧策略输入中的当前 `human_action` 换成上一时刻/零输入，动作变化量为：

- `mean_abs_diff ≈ 0.7432`
- `max_abs_diff ≈ 3.9195`
- 典型样本前向速度差可达 `≈ 1.32 m/s`

因此 `compare_ipc_rl.py` 之前那条参考链路也会放大“ROS1 vs Isaac Sim”假差异。

### 3. `spawn_x=-10` 会直接污染结论

用同一份 checkpoint、同一份 PCD、同一套 `UserModelTunnel` 做**理想速度跟踪**离线滚动：

| 起点 | 结果 |
| --- | --- |
| `(-10, 0, 5)` | 第 1 步即 `safety_stop`，`min_dist=0.1` |
| `(-7, 0, 5)` | 第 169 步在 `x≈8.59, y≈-1.67, z≈4.99` 触发 `safety_stop` |

这说明旧的 `spawn_x=-10` 默认值本身就会把实验放在错误起点上；即使没有 Gazebo 动力学，离线滚动也会立刻触发安全停止。

### 4. 修正后的 Gazebo 运行与理想滚动相近

在 Docker 容器里使用修正后的基线拉起 headless Gazebo 后：

- 地图加载成功
- Gazebo 插件成功起飞
- `tunnel_navigation.py` 能正常进入“构造观测 -> 策略推理 -> 发布动作”闭环

实际 Gazebo 运行中，节点状态最终停在：

- `x≈8.7, y≈-1.4, z≈5.0`
- `min_d≈0.10`
- `SAFETY_STOP`

而理想速度跟踪离线滚动停在：

- `x≈8.59, y≈-1.67, z≈4.99`
- `min_dist≈0.20`
- `safety_stop`

两者位置已经比较接近，说明**修正后的 Gazebo 执行端没有表现出“完全不同于策略本身”的异常轨迹**。  
换句话说，当前剩余偏差不支持“ROS1/Gazebo 动力学把一个本来正确的策略彻底带坏了”这一说法。

## 已修复代码项

- `ros1/navigation_runner/scripts/tunnel_deployment/policy_net.py`
  - 修正特征拼接顺序为 `[cnn, human_action, state]`
  - 修正 batch>1 调试日志崩溃
  - 修正 lazy layer materialize/load 顺序
- `tunnel_test/compare_ipc_rl.py`
  - 改为使用当前步 `human_action`
- `tunnel_test/tunnel_terrain.py`
  - `INIT_POS` 改为 `[-7.0, 0.0, 5.0]`
- `ros1/navigation_runner/launch/tunnel_comparison.launch`
  - 历史错误基线 `spawn_x=-10.0` 已被移除；诊断过程中先回到 `-7.0` 做对齐审计
  - 当前仓库默认值已进一步调整为 `spawn_x=-8.5`，作为 Gazebo 侧安全起飞微调
  - `flight_recorder goal_x` 改为 `12.0`

## 最终归因

### 对“之前的大差距”的归因

**主因是实现/配置错误，不是模型本身迁移到 ROS1/Gazebo 后天然失效。**

最关键的三个污染源是：

1. ROS1 standalone 策略网络实现错误；
2. Isaac Sim 参考脚本 `compare_ipc_rl.py` 的 `human_action` 时序错误；
3. ROS1 运行基线默认起点错误（`spawn_x=-10`）。

只要这三项任意一项没排掉，之前的“ROS1 明显比 Isaac Sim 差很多”都不能作为模型迁移失败的证据。

### 对“修正后仍然没有通关”的归因

当前证据更支持：

- 在固定 `tunnel_map_default.pcd`
- 当前 `UserModelTunnel`（Perlin 模式）
- 当前 checkpoint

这一组合下，策略本身就会在后段靠近 `x≈8.6~8.7, y≈-1.5` 的障碍区域并触发安全停止。  

因此，**修正后的剩余问题更像是策略/人机输入分布/固定地图实例上的行为问题，而不是 ROS1/Gazebo 迁移实现错误**。

## 仍需注意的事项

1. 本次没有重新完整跑一遍修正后的 Isaac Sim `compare_ipc_rl.py` 与 Gazebo 做逐帧对齐；如果后续要做论文级别对比，建议补这一组。
2. 本次 Gazebo 通过 `docker exec` 拉起 `roslaunch` 时，需要额外设置 `ROS_HOSTNAME=127.0.0.1 ROS_IP=127.0.0.1`，否则容器里的 `HOSTNAME` 不能自回连。这是 ROS 网络配置问题，不是隧道策略逻辑问题。
3. 当前默认起飞点 `spawn_x=-8.5` 是针对现有 Gazebo 资产的安全设计；后续如果要做 Isaac Sim / Gazebo 严格对比，应优先区分“起飞安全补偿”与“airborne 轨迹差异”这两个问题。
4. 当前 ROS1 默认权重已切到 M3 `checkpoint_tunnel_M3_21500.pt`。这份权重可被 ROS1 standalone loader 直接加载，但 M3 训练使用 offline feasible-diverse pilot dataset，而 ROS1 默认批量实验仍使用 online Perlin `UserModelTunnel`；默认结果应解释为 cross-pilot generalization，而不是 M3 训练输入分布的逐样本复现。
5. 如果后续目标是提高这张固定地图上的通过率，优先应该检查：
   - 当前 checkpoint 是否就是期望的最佳隧道权重；
   - `UserModelTunnel` 的 Perlin 分布是否是本次要验证的目标输入分布；
   - 是否需要针对固定地图做 deterministic/simpler human input 的对照（例如 `user_model_simple=true`）。
