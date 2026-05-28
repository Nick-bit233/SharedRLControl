# Intent Pilot 离线数据生成方案实现分析

## 1. 背景与定位

当前 `isaac-training` 中离线模拟输入数据集已经被整理为三类 generator：

- `legacy_perlin`: 旧版通用 Perlin / APF / filter 输入。
- `tunnel_perlin`: 当前 tunnel 任务主线使用的 forward-biased Perlin 输入。
- `intent_pilot`: 基于感知、意图和反应层的模拟 pilot 输入。

本文重点分析 `intent_pilot` 的实现原理，并从宏观实验目标判断它和 `tunnel_perlin` 哪一个更适合作为主线模拟用户方案。

结论先行：

- 当前训练主线和大规模 ablation 更推荐继续使用 `tunnel_perlin`。
- `intent_pilot` 更适合作为下一阶段研究型数据源，用来验证“意图感知/人类反应建模”是否能提升 shared-control 策略的鲁棒性和解释性。
- 如果只能选择一个作为当前 base 分支的默认方案，应选择 `tunnel_perlin`；如果目标转向论文中更强的人类行为建模贡献，则应逐步升级到 `intent_pilot`，但不能直接替换，需要额外验证。

## 2. 当前 intent_pilot 离线生成链路

当前 `intent_pilot` 的离线生成入口位于：

```text
SharedRLControl/isaac-training/src/datasets/trajectory_generator.py
```

配置入口为：

```text
SharedRLControl/isaac-training/configs/user_input/intent_pilot.yaml
```

核心配置：

```yaml
generator:
  kind: intent_pilot
  experiment: tunnel_intent
  assistant_policy: zero
  num_obstacles: 24
  record_privileged: false
```

执行流程如下：

1. `trajectory_generator.py` 读取 Hydra 配置。
2. `_generator_kind(cfg)` 返回 `intent_pilot`。
3. `generate_intent_pilot_dataset(cfg, device)` 被调用。
4. 该函数复用 `visualize_pilot_distributions.py` 中的轻量 tunnel rollout 组件：
   - `load_cfg()`
   - `generate_obstacle_field()`
   - `rollout_model()`
5. `rollout_model(model_name="intent", ...)` 实例化 `UserModelIntent`。
6. 每一步根据当前位置和轻量障碍场计算最近障碍距离/法向。
7. `UserModelIntent.step()` 输出 body-frame pilot action。
8. generator 将 rollout 结果转成 HDF5 schema：

```text
/velocities                 (N, T, 3)
/positions                  (N, T, 3)
/bboxes                     (N, 6)
/styles/noise_freq          (N,)
/styles/smoothness          (N,)
/styles/laziness            (N,)
/metadata attrs
/intent/intent_mode         (N, T)
/intent/react_mode          (N, T)
/intent/threat              (N, T)
```

其中 `/velocities` 是训练可直接消费的 `human_action`，而 `/intent/*` 是分析和可视化诊断字段。

## 3. UserModelIntent 的内部结构

`UserModelIntent` 位于：

```text
SharedRLControl/isaac-training/src/simulated_users/user_model_intent.py
```

它不是简单的随机速度生成器，而是由四个逻辑层组成。

### 3.1 Profile 层

每个并行环境在 reset 时采样一组 pilot profile 参数：

```text
alpha: assistant trust / correction sensitivity
beta: dwell / behavior persistence factor
psi: joystick tracking responsiveness
phi: desired speed scale
eta: aggressive-conservative interpolation
tau_perc: perception delay
sigma_perc: perception noise
d_react: reaction distance threshold
wrong_direction_prob: wrong avoidance direction probability
```

这些参数来自 `cfg.user_model.intent.*_range`。它们决定同一个 intent model 在不同 episode 中表现为保守、激进、迟钝、噪声更大或更容易错误反应的 pilot。

### 3.2 Perception 层

感知层由 `PilotPerceptionModel` 实现：

```text
SharedRLControl/isaac-training/src/simulated_users/pilot_perception.py
```

它维护延迟 buffer，将真实几何信息转换为 pilot 感知到的状态：

```text
true nearest obstacle distance / normal
  -> delayed distance / normal
  -> additive distance noise
  -> threat score
```

输出包括：

```text
threat
perceived_dist
perceived_normal
```

这使得 intent pilot 的反应不是直接依赖真实世界状态，而是依赖延迟和噪声后的主观感知。

### 3.3 Intent 层

intent 层维护离散意图模式：

```text
CRUISE
MANEUVER
STATION_KEEP
IDLE
```

采样逻辑由 `IntentMode` 和 `_sample_intent_mode()` 控制。模式先验可以被 tunnel 任务覆盖，例如 `tunnel_intent.yaml` 中的：

```yaml
tunnel_mode_prior: [0.80, 0.14, 0.03, 0.03]
```

这意味着 tunnel 场景下大部分 pilot 会保持向前巡航，少量执行 maneuver、station keeping 或 idle。

每个 mode 会生成不同 waypoint：

- `CRUISE`: 沿当前 heading 生成前方目标点。
- `MANEUVER`: 加入 lateral / vertical maneuver。
- `STATION_KEEP`: 回到 anchor。
- `IDLE`: 目标保持在当前位置。

随后 `_intent_to_velocity()` 将 waypoint error 转为 body-frame desired velocity，即：

```text
intent_velocity_body
```

### 3.4 Reactive 层

reactive 层维护反应模式：

```text
NONE
NO_REACT
LATE_REACT
EMERGENCY_STOP
FREEZE
EVADE
OVERCORRECT
SURGE
```

`_update_reactive_layer()` 根据 `threat`、perception profile 和 spontaneous reaction rate 触发反应。触发后，`_apply_reactive_overlay()` 会修改 intent velocity：

- `EMERGENCY_STOP`: 速度衰减。
- `FREEZE`: 速度置零。
- `LATE_REACT`: threat 足够高时才沿 perceived normal 修正。
- `EVADE`: 沿 perceived normal 避障，可能因为 `wrong_direction_prob` 走反方向。
- `OVERCORRECT`: 反向或震荡式过度修正。
- `SURGE`: 放大当前速度。

最终输出：

```text
final_velocity_body
```

### 3.5 Joystick Dynamics 层

最终 pilot action 不是直接等于 `final_velocity_body`，而是经过 joystick dynamics：

```text
J = J + (final_velocity - J) * p_gain
I = i_decay * I + (last_pilot_action - assistant_action) * (1 - alpha)
pilot_action = J + I * i_gain
```

这一步的意义是：

- `J` 提供低通响应，避免 action 过于跳变。
- `I` 表达 pilot 对 assistant 介入的累计偏差反应。
- `assistant_action` 会影响 pilot 后续行为。

当前离线 generator 固定 `assistant_policy=zero`，因此这个 feedback 通道还没有真实体现“policy 与 human 的闭环交互”。

## 4. 当前离线化边界

当前 `intent_pilot` 离线化采用的是：

```text
assistant_policy = zero
```

也就是生成数据集时默认 assistant 不介入。这样做的好处是实现简单、稳定、可复现，也能观察 intent pilot 自身的行为分布。

但它带来一个重要边界：

```text
当前 intent_pilot dataset 不是 policy-coupled human response dataset。
```

换言之，它记录的是“无 assistant 或零 assistant 下的 pilot 输入分布”，而不是“当前训练策略介入后，pilot 如何反应”的交互分布。

这对训练意义有直接影响：

- 作为 `human_action` 输入分布，它是有效的。
- 作为真实 shared-control 闭环人类模型，它是不完整的。
- 如果要研究 human-assistant conflict、trust adaptation、intervention response，则必须引入 scripted assistant 或 checkpoint replay 生成数据。

## 5. 与 tunnel_perlin 的对比

### 5.1 tunnel_perlin 的机制

`tunnel_perlin` 使用当前 `trajectory_generator.py` 中的 batched Perlin pipeline：

```text
Perlin noise
  -> per-channel amp_scale + bias
  -> deadband
  -> low-pass filter
  -> APF / map-boundary correction
  -> position integration
```

核心是 `directional_bias`：

```yaml
directional_bias:
  amp_scale: [0.25, 1.0, 0.2]
  bias:      [1.5, 0.0, 0.0]
```

这意味着：

- `vx` 有稳定正向偏置，保证 tunnel 任务可达。
- `vy` 保持较大横向多样性。
- `vz` 变化较小，避免过多无效爬升/下降。

它的设计目标非常明确：为 tunnel shared-control 训练提供可达、平滑、多样的人类输入。

### 5.2 intent_pilot 的机制优势

`intent_pilot` 相比 `tunnel_perlin` 有三个明显优势：

1. 行为语义更强  
   它不只是速度噪声，而是包含 cruise、maneuver、station keep、idle 等意图模式。

2. 可解释性更强  
   生成数据时可以记录 `intent_mode`、`react_mode`、`threat`，后续可以分析策略在不同 pilot 状态下的行为。

3. 更接近 shared-control 问题本质  
   它显式包含 assistant feedback、pilot trust/sensitivity、perception delay/noise 和 reactive avoidance。

这些特性对论文叙事有价值，尤其适合支撑“人类意图与安全辅助之间的共享控制”这一宏观问题。

### 5.3 intent_pilot 的当前风险

当前版本不能直接认为优于 `tunnel_perlin`，主要有四个风险。

第一，任务可达性不如 tunnel_perlin 显式保证。

`tunnel_perlin` 通过 forward bias 明确保证 x 方向推进；`intent_pilot` 虽有 tunnel mode prior，但 mode 切换、reactive overlay、station/idle 行为都可能降低稳定推进能力。

第二，离线数据与真实训练闭环之间存在偏差。

当前 `assistant_policy=zero`，所以 pilot 没有看到真实训练 policy 的 correction。训练时如果 assistant 策略大幅介入，offline intent action 不能反映 pilot 后续反应。

第三，轻量 obstacle field 与 Isaac 环境几何并不完全等价。

`intent_pilot` 数据生成复用 `visualize_pilot_distributions.py` 的 lightweight geometry，而不是完整 Isaac Sim 环境。它适合快速生成分布，但不等价于真实 sim rollout。

第四，参数空间更大，调参成本更高。

`intent_pilot` 包含 profile ranges、mode priors、dwell distribution、perception delay/noise、react mode logits 等多层参数。它更有表达力，也更容易引入难以解释的分布偏差。

## 6. 宏观实验目标下的选择

当前项目的宏观目标可以拆成两层：

1. 训练稳定、可复现实验主线  
   需要高吞吐、低随机风险、明确可达性、便于 ablation。

2. 研究 shared-control 中更真实的人类输入  
   需要表达意图、感知误差、反应延迟、与 assistant 的交互。

对于第一层，`tunnel_perlin` 更优。

原因：

- 它已经被设计成 tunnel-feasible distribution。
- 生成速度更快，支持大规模 batched generation 和 distributed merge。
- 参数少，行为更容易解释。
- 对训练主线的噪声较小，适合作为 base branch 的默认方案。
- 当前 ablation、curriculum、checkpoint、campaign 清理工作都更容易围绕它稳定推进。

对于第二层，`intent_pilot` 更有潜力。

原因：

- 它可以提供更强的行为语义。
- 它天然支持记录 intent/react diagnostics。
- 它更适合分析策略是否真正理解 pilot 意图，而不是只拟合 Perlin velocity。
- 它能支撑更强的论文叙事，但前提是做充分验证。

因此推荐决策是：

```text
当前默认训练主线：tunnel_perlin
研究扩展与下一阶段验证：intent_pilot
```

如果必须在两者中选择一个作为“当前更优方案”，答案是：

```text
tunnel_perlin 更优。
```

如果问题是“未来更值得投入的模拟用户方向”，答案是：

```text
intent_pilot 更值得投入，但需要先完成验证闭环。
```

## 7. 推荐推进路线

### 7.1 保持 tunnel_perlin 作为默认主线

短期内不要用 `intent_pilot` 替换所有训练 campaign。建议继续使用：

```text
data/user_inputs/tunnel_perlin_v1.h5
```

作为 tunnel、lagrangian、ablation、curriculum 的默认 user input dataset。

### 7.2 将 intent_pilot 作为独立实验分支

新增独立数据集：

```text
data/user_inputs/intent_pilot_v1.h5
```

先只接入少量实验：

- held-out evaluation
- robustness evaluation
- intent-specific training branch
- dataset distribution comparison

不要在未验证前替换主线 ablation 数据源。

### 7.3 建立替换门槛

只有当 `intent_pilot` 满足以下条件后，才建议作为主线候选：

1. 1500-step reachability 不显著低于 `tunnel_perlin`。
2. collision/risk 分布不出现异常集中。
3. `vx/vy/vz`、speed、delta_v 分布合理。
4. intent/react mode 分布与实验假设一致。
5. 使用 `intent_pilot` 训练的策略在 tunnel_perlin held-out 和 intent held-out 上都不退化。
6. 如果论文主张 human-assistant interaction，需要增加非 zero assistant policy 的 dataset generation。

### 7.4 下一步改进 intent_pilot

当前 `intent_pilot` v1 建议继续补强：

- 记录 `/intent/intent_velocity` 和 `/intent/perceived_dist`。
- 支持 `record_privileged=true` 写出 `/intent/critic_privileged`。
- 增加 `assistant_policy=scripted_safety`，模拟简单安全辅助介入。
- 增加 `assistant_policy=checkpoint_replay`，用已训练策略生成 policy-coupled human response dataset。
- 将 lightweight obstacle field 参数与真实 tunnel env 的 obstacle sampling 对齐。

## 8. 最终判断

从工程主线看：

```text
tunnel_perlin 是当前更好的默认训练输入方案。
```

从研究潜力看：

```text
intent_pilot 是更好的下一阶段研究方向。
```

从论文实验组织看，推荐把两者定位为：

```text
tunnel_perlin: stable base distribution
intent_pilot: semantically richer evaluation/training distribution
```

这样可以避免在主线训练还需要稳定时引入过高建模复杂度，同时保留 intent pilot 为论文提供更强 human-modeling 贡献的空间。
