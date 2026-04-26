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

## 当前 RL 外部安全护栏与碰撞判定设计

这里的“外部”指不属于 M3/RL policy 本身的机制。RL 模型只输出世界系速度命令；下面这些逻辑会在模型输出前后拦截、覆盖、终止或记录实验结果。

### 1. `TunnelNav` 在线安全护栏

`tunnel_navigation.py` 在 RL 模式中运行一个 10Hz 的 `_safety_check()` 后台线程：

| 状态/阈值 | 默认值 | 行为 |
| --- | ---: | --- |
| `safety_start_takeoff_delta` | `0.5m` | 起飞前不启用安全/碰撞检测，避免 PCD 地面点被误判为碰撞；超过初始高度 `+0.5m` 后永久启用 |
| `safety_min_dist` | `0.2m` | 可恢复安全停车：设置 `safety_stop=True`，控制循环发布 `_publish_stop()`，即当前位置 XY + `takeoff_height` 姿态保持 |
| `collision_dist` | `0.05m` | 不可恢复碰撞：设置 `collision=True`，发布 `/tunnel_nav/collision=True`，控制循环永久停止本 run 的 RL 推理 |

距离来源优先使用 Python `PcdRaycaster.nearest_distance()` 对同一份 PCD 地图做全图最近点查询；只有未启用 Python PCD raycaster 时，才 fallback 到稀疏 raycast hit 点距离。这个设计是为了避免“上一帧 raycast 命中点 + 当前高速位姿”产生 stale-hit 假碰撞。

控制循环中的优先级是：

1. `external_stop`：由 recorder 或其他实验控制节点请求停止。
2. `goal_reached`：到达 `goal_x` 后发布 stop。
3. `collision`：永久停止，并保持悬停。
4. `safety_stop`：可恢复悬停；距离恢复到 `safety_min_dist` 以上后继续模型推理。
5. 正常路径：构造观测、发布 human command、M3 推理、发布速度命令。

Gazebo 非 PX4 路径还有两层执行侧保护：

| 机制 | 位置 | 行为 |
| --- | --- | --- |
| 横向速度限幅 | `_publish_cmd()` | 将 heading-local 水平速度范数限制到 `gazebo_max_hvel`，当前为 `2.0m/s` |
| z 速度执行模式 | `_publish_cmd()` | `gazebo_z_mode=alt_hold` 时用高度保持覆盖 policy z；`policy`/`policy_clamped`/`blend` 可用于验证完整 3D policy velocity 或折中模式 |
| policy z 起飞门控 | `_publish_cmd()` | 默认 `gazebo_policy_z_takeoff_gate=true`，在 `z >= takeoff_height - gazebo_policy_z_gate_tolerance` 前仍使用 altitude hold，避免低空起飞阶段进入训练外分布 |
| 完整 policy 接管门控 | `_control_callback()` | 默认 `policy_takeoff_gate=true`，在 `z >= takeoff_height - policy_takeoff_gate_tolerance` 前不运行 policy、不推进 online user model、不发布 policy x/y/z，只持续发布 takeoff pose |
| 姿态翻倒恢复 | `_publish_cmd()` | `roll/pitch > tumble_deg` 时切到 pose hold；超过 `tumble_recover_timeout` 仍未恢复则声明 collision 并发布 `/tunnel_nav/collision=True` |

### 2. `flight_recorder.py` 实验指标与终止

`flight_recorder.py` 是 batch 指标的最终来源。它订阅：

| 输入 | 用途 |
| --- | --- |
| `/CERLAB/quadcopter/odom_raw` | 记录位置、速度、姿态；计算 goal/timeout |
| `/CERLAB/quadcopter/cmd_vel` | 记录最终发给 Gazebo 插件的本体系速度命令 |
| `/experiment_control/human_cmd` | 记录 RL/IPC 使用的 user model 输入 |
| `/tunnel_nav/policy_cmd` | 记录 RL policy 原始世界系速度输出，用于和实际执行的 `cmd_vel` 对比 |
| `/tunnel_nav/policy_active` | 记录完整 RL policy 是否已经通过起飞门控并接管控制 |
| `/tunnel_nav/z_policy_active` | 记录 `policy`/`policy_clamped`/`blend` 的 z 轴是否已经通过起飞门控 |
| `/tunnel_nav/collision` | 仅 RL 模式启用，接收 TunnelNav 外部碰撞判定 |

recorder 也会加载 PCD 并建立 `cKDTree`，每帧计算真实最近障碍距离：

- 起飞前：`min_obstacle_dist_monitored = inf`，不用于碰撞终止。
- 起飞后：`min_obstacle_dist_monitored = KDTree nearest distance`。
- 如果 `monitored_min_dist < collision_dist`，则终止为 `collision`。
- 如果收到 `/tunnel_nav/collision=True`，也终止为 `collision`。
- 如果 `x >= goal_x`，终止为 `goal_reached`。
- 如果超过 `timeout_sec`，终止为 `timeout`。

因此，当前 batch 里的 “collision” 有两条来源：

1. recorder 自己的 PCD KDTree 指标碰撞；
2. RL 专属的 `/tunnel_nav/collision` 外部碰撞信号。

IPC 模式没有 `/tunnel_nav/collision`，主要依赖 recorder 的 PCD KDTree 和 timeout/goal。

### 3. Gazebo 仿真器自身碰撞

Gazebo world 里的障碍物有 SDF collision geometry，真实接触会影响动力学，例如被墙/障碍物挡住、翻倒或速度 PID 积分异常。但当前 batch 指标**不直接订阅 Gazebo contact sensor**。仿真器碰撞会通过以下间接路径进入指标：

1. 机体靠近 PCD 障碍物到 `collision_dist` 以下，recorder 判为 collision。
2. 机体姿态翻倒，TunnelNav 的 tumble recovery 超时后发布 `/tunnel_nav/collision=True`。
3. 机体被卡住但未进入上述阈值，最终通常表现为 `timeout`。

### 4. 当前设计反思

| 问题 | 影响 | 当前处理/建议 |
| --- | --- | --- |
| PCD 包含地面，起飞前最近距离很小 | 曾导致 RL 一启动就被判 collision，模型完全不推理 | 已用 `safety_start_takeoff_delta=0.5` 让 TunnelNav 和 recorder 都从 airborne 后开始监控 |
| `safety_stop` 是可恢复护栏，不是终止事件 | 可能把“非常危险但未碰撞”的 episode 变成 timeout，影响 success/collision 解读 | 分析结果时同时看 `pct_close_*`、`min_obstacle_dist` 和 `termination_reason`；如果论文指标需要“护栏介入率”，应单独记录 safety_stop 时间占比 |
| recorder 与 TunnelNav 都能判 collision | 双通道提高安全性，但也可能出现来源不一致 | 当前两者都使用 PCD 最近距离，并共享 `collision_dist=0.05`；建议后续在 `run_summary.json` 增加 `collision_source` |
| Gazebo contact 没有直接进入 recorder | 真实接触可能只表现为 tumble 或 timeout | 若要严格统计物理碰撞，应增加 Gazebo contact topic 或插件输出，并接入 recorder |
| RL 的 `cmd[2]` 默认不直接执行 | 默认 `gazebo_z_mode=alt_hold` 仍使用高度保持覆盖 z 速度，闭环状态分布会偏离完整 3D 速度执行 | 已加入 `gazebo_z_mode=policy|policy_clamped|blend` 做 A/B；recorder 同时保存 `policy_cmd_*`、最终 `cmd_vel*` 和 `z_policy_active` |
| 低空就执行完整 policy z | policy mode 若从 `z≈0.4m` 开始接管 z，M3 会大量输出 `±1.96m/s`，这不是 user model z 采样问题，而是起飞阶段 OOD 推理 | 已默认启用 policy z 起飞门控；如需复现纯 policy mode，可显式关闭 `gazebo_policy_z_takeoff_gate` |
| 低空就运行/执行 policy x-y | 只门控 z 轴时，日志仍会出现 policy `cmd=[...]`，且 x/y 已经在低空执行 | 已默认启用完整 `policy_takeoff_gate`；gate 前不会 forward policy，也不会推进 online user model |
| `safety_min_dist=0.2` 小于训练碰撞半径 `0.3` | ROS1 护栏比训练终止更宽松，可能允许进入训练中已接近终止的区域 | 如果目标是保守验证，可考虑把 `safety_min_dist` 提到 `0.3` 并把它作为 safety-intervention，而不是 collision 指标 |

当前推荐解读是：`goal_reached/collision/timeout` 是 batch 终止标签；`min_obstacle_dist` 和接近障碍比例描述风险暴露；`safety_stop` 是外部护栏介入，不应被混同为模型自身成功避障。


## 仍需注意的事项

1. 本次没有重新完整跑一遍修正后的 Isaac Sim `compare_ipc_rl.py` 与 Gazebo 做逐帧对齐；如果后续要做论文级别对比，建议补这一组。
2. 本次 Gazebo 通过 `docker exec` 拉起 `roslaunch` 时，需要额外设置 `ROS_HOSTNAME=127.0.0.1 ROS_IP=127.0.0.1`，否则容器里的 `HOSTNAME` 不能自回连。这是 ROS 网络配置问题，不是隧道策略逻辑问题。
3. 当前默认起飞点 `spawn_x=-8.5` 是针对现有 Gazebo 资产的安全设计；后续如果要做 Isaac Sim / Gazebo 严格对比，应优先区分“起飞安全补偿”与“airborne 轨迹差异”这两个问题。
4. 当前 ROS1 默认权重已切到 M3 `checkpoint_tunnel_M3_21500.pt`，默认 user model 也已切到 `m3_diverse`：`vx≈1.5±0.5`、宽 `vy` Perlin、`vz≈±0.2` 小幅扰动。这份权重可被 ROS1 standalone loader 直接加载；`m3_diverse` 是对 M3 offline feasible-diverse pilot dataset 的在线近似，而不是逐样本 replay。
5. 如果后续目标是提高这张固定地图上的通过率，优先应该检查：
   - 当前 checkpoint 是否就是期望的最佳隧道权重；
   - `UserModelTunnel` 的 profile 是否是本次要验证的目标输入分布；
   - 是否需要针对固定地图做 deterministic/simpler human input 的对照（例如 `user_model_simple=true`）。
