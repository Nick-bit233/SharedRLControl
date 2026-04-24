# UAV Shared Autonomy 中 Pilot/User Model 实现方案调研

> 目标：调研在 UAV/飞行控制场景下"模拟人类操控员（pilot）输入"的现有学术与开源实现，
> 与本项目此前提出的 4 个改进方案 A/B/C/D 进行对照，
> 给出适合落地到 `src/core/user_model_*.py` 的推荐方案与接口伪代码。
>
> 范围严格限定在 **UAV / multirotor / quadrotor / fixed-wing teleoperation** 场景，
> 不包括桌面机械臂、轮式车、腰带辅助等其它 shared autonomy 工作。

---

## 0. 本项目当前 pilot model 与候选方案回顾

| 实现 | 文件 | 核心做法 | 已知问题 |
|------|------|----------|----------|
| `UserModel` (基线) | `src/core/user_model.py` | 3 维独立 1D Perlin 噪声 → 速度指令；按 episode 随机 octave/persistence/scale | 无意图、无回中、无离散事件、轴间独立 |
| `UserModelTunnel` | `src/core/user_model_tunnely.py` | 1D Perlin（横向/纵向通道）+ 注入 `vx = 1·max_speed` 的方向偏置 | 偏置硬编码、轴间不对称未建模 |
| `UserModelDiverse` | `src/core/user_model_diverse.py` | 4 模式 (perlin / straight / arc / hover) per-env 随机 | 模式之间无转移，单 episode 内行为同质 |
| 真人键盘录制 | `flight_recorder` (commit f6abc87) | 真人键盘输入回放 | 数据稀缺，难以批量并行 |

之前讨论中提出的 4 个改进路线：

- **A. Semi-Markov 推杆-保持-松开状态机**：把摇杆建模为 `idle → push → hold → release` 的离散事件 + 连续幅度。
- **B. Intent-driven 噪声控制器**：内部维护"目标位置 / 目标速度"，用 PD + OU + 反应延迟产生指令，目标会被 RL 助手的反馈反向修正。
- **C. 从真人录制学 generative pilot**：用 `flight_recorder` 的轨迹训一个 VAE / diffusion 生成器。
- **D. A/B/C 混合**：状态机驱动意图，B 实现连续段，C 提供风格分布。

---

## 1. 关键工作清单（按"pilot 模型"复杂度排序）

### 1.1 Reddy, Dragan, Levine 2018 — *Shared Autonomy via Deep RL* (RSS'18)
- **链接**：<https://arxiv.org/abs/1802.01744>
- **任务**：Lunar Lander + 真实 quadrotor (n=4)
- **Pilot 模拟**：用 *synthetic pilots* 而非真人来跑 ablation。论文区分三种 surrogate：
  - **Laggy pilot**：最优策略 + 输入延迟（每 4 步重复一个动作）。
  - **Noisy pilot**：最优策略 + ε-greedy 抖动（高概率随机离散动作）。
  - **Sensor-less pilot**：最优策略 + 屏蔽部分观测（无法看到地形/目标位置）。
  - 共同结构：**先有一个能完成任务的 oracle 策略**，然后施加单一形式的退化（延迟/噪声/感知缺失）。
- **特点**：纯离散动作空间（DQN 输出），pilot 模型简单可控，但缺乏 hold/release 事件和摇杆物理特性。
- **对应方案**：≈ B 的"退化版"（B 不带 PD 反馈学习，没有用户对助手的反应）

### 1.2 Schaff & Walter 2020 — *Residual Policy Learning for Shared Autonomy* (RSS'20)
- **链接**：<https://arxiv.org/abs/2004.05097>
- **任务**：6-DOF quadrotor reaching + Lunar Lander，**连续动作空间**
- **Pilot 模拟**：surrogate pilot = "**a noisy controller that knows the goal**"，做法是
  - 一个最优 PD/policy 跟踪 goal；
  - 加 Gaussian 噪声 + 间歇性掉落（drop-out）模拟人类不专心；
  - **Goal 仅 surrogate 知道，assistant 不知道**（goal-agnostic constraints）。
- **特点**：第一次把 surrogate pilot 用在 *连续控制* + 6-DOF quadrotor 上，证明了在不知道用户目标时只用"约束满足"奖励就能训出助手。
- **对应方案**：≈ B 的标准实现（PD + 噪声 + 隐藏目标）

### 1.3 Backman, Kulić, Chung 2021 — *Learning to Assist Drone Landings* (RA-L)
- **链接**：<https://arxiv.org/abs/2011.13146>
- **任务**：模拟器 (AirSim) + 真人键盘控制 quadrotor 降落
- **Pilot 模拟**：2 参数模型 `(α, β)`：
  - `α`：用户对助手的 *conformance*（顺从度）；
  - `β`：用户的 *proficiency*（基于 UAV 与平台相对深度修正自己 goal 估计的能力）。
  - 用 keyboard-style 离散输入。
- **对应方案**：A 的雏形 + B 的简版

### 1.4 Backman 2022/2023 — *Shared Autonomy Drone Landings* (Auton. Robot. 2023) ⭐ 最相关
- **链接**：<https://arxiv.org/abs/2202.02927> · <https://link.springer.com/article/10.1007/s10514-023-10143-3>
- **任务**：AirSim + 物理 UAV，n=28 真人 user study；joystick 控制
- **Pilot 模拟**（重要细节，论文 §3.1.1）：
  - **4 参数** `(α, β, Ψ, Φ)`，每 episode 从均匀分布独立采样：
    - `α ∈ [0,1]` *conformance*：对助手动作的顺从度
    - `β ∈ [0,1]` *proficiency*：通过深度感知修正 goal 估计的能力
    - `Ψ ∈ [0,1]` *aggressiveness*：摇杆推杆速率，决定 P-gain
    - `Φ ∈ [0,1]` *daringness/speed*：最大期望飞行速度
  - **状态机**：`Approach`（飞到当前 goal 估计上空）→ `Descent`（降落）。
  - **Goal 估计动态更新**（POMDP → MDP 重表述，goal 作为 critic 的特权信息）：
    ```
    Ĝ_{i+1} = Ĝ_i + α · (a_a - a_u)/K_α  +  β · (G - Ĝ_i)/K_β
    ```
  - **轨迹规划模块** 设置航点，desired velocity `V_t ∝ Φ`。
  - **摇杆物理建模**（关键!）：
    - *Joystick control*（拇指动作）：`J_{t+1} = J_t + (V_t - J_t)·P_gain`，`P_gain ∝ Ψ`。
      → 这就是"摇杆不能瞬间到位、有惯性"的物理模型，**是 Perlin 完全没有的部分**
    - *Adaptability control*（人对 assistant 偏差的反应）：积分器 `I_{t+1} = I_t + (a_u - a_a)(1-α)`，再衰减
    - 输出 `a_u = J_{t+1} + I_{t+1}·I_gain`
  - **两路随机数生成器**：决定性决策 vs 非决定性决策分开 seed，便于在不同模型间公平复测。
  - 助手动作再过 OU 噪声做探索；最终速度 = avg(pilot, assistant) + 受地面效应影响的 Gaussian 噪声。
- **结果**：仅用模拟 pilot 训出来的助手，让真人成功率从 **51.4% → 98.2%**。**完全 sim-only 训练就能 transfer 到真人。**
- **对应方案**：**A + B + D 的工业化实现**，并已被真人验证

### 1.5 Wang et al. 2021 — *GPA-Teleoperation: Gaze Enhanced Aerial Teleoperation* (RA-L)
- **链接**：<https://arxiv.org/abs/2109.04907>
- **重点**：用 **眼动 (gaze)** 作为额外的 intent 通道 → 助手能更早预判用户想去的位置。
- **Pilot 模拟**：基本不模拟 pilot，是真人实验。
- **对应方案**：与 A/B/C/D **正交**，提示我们 intent 信号可以多源（不只是摇杆）

### 1.6 Zhang et al. 2024 — *Safe and Stable Teleoperation of Quadrotor under Haptic Shared Autonomy*
- **链接**：<https://arxiv.org/abs/2403.15335>
- **重点**：用 CBF (Control Barrier Function) 在线修正人输入；haptic feedback 闭环。
- **Pilot 模拟**：纯真人实验，无 surrogate。
- **对应方案**：与 RL 训练流程不直接相关；提示我们"安全过滤 + 可微修正"也是一条路

### 1.7 旁证：Patrikar 等 *Predicting like a Pilot* (ICRA'22) / Pfeiffer 等 *Visual Attention Prediction in Drone Racing* (PLOS'22)
- **链接**：<https://arxiv.org/abs/2202.05140> · <https://doi.org/10.1371/journal.pone.0264471>
- **重点**：从 **真人飞行数据集** 学一个生成式 pilot 模型（用 GMM/序列模型预测 trajectory 或 attention）。
- **对应方案**：**C 的模板**（学习式 pilot），但目前都没用在 shared autonomy 闭环训练里。

### 1.8 开源仓库情况
- **OmniDrones / Aerial-Gym / NavRL (本项目上游)**：均无明确的"human pilot model"模块。NavRL 的人类输入只在 evaluation 用键盘喂进去。
- **PX4 SITL / RotorS**：有 joystick driver，但仅做 IO，不模拟人。
- **shared-autonomy / Reddy 的 GitHub**：<https://github.com/rddy/deepassist>，有 quadrotor demo，pilot 就是上面 1.1 描述的 noisy/laggy oracle。
- **结论**：UAV 仿真社区目前**没有现成可直接 import 的"高保真 joystick pilot 模型"包**，最具体的实现都散在论文作者的私有代码里。Backman 2023 是最值得直接重写的参考。

---

## 2. 对照表：现有工作 × 我们的方案 A/B/C/D

| 工作 | 状态机 (A) | 意图驱动 (B) | 学习式 (C) | 摇杆物理 | 真人验证 | 备注 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| Reddy 2018 | ✗ | △ (oracle+noise) | ✗ | ✗ | n=4 真飞 | 离散动作 |
| Schaff 2020 | ✗ | ✓ | ✗ | ✗ | 有 | 6-DOF quadrotor 连续动作 |
| Backman 2021 | △ | ✓ | ✗ | ✗ | n=15 | 2 参数 keyboard 模型 |
| **Backman 2023** | **✓** | **✓** | ✗ | **✓** | **n=28 joystick** | **方案 A+B+D 工业实现** |
| GPA-Teleop 2021 | ✗ | ✗ | ✗ | ✗ | 真人 | 多源 intent |
| Zhang 2024 (CBF) | ✗ | ✗ | ✗ | ✗ | 真人 | 安全过滤 |
| Patrikar / Pfeiffer | ✗ | ✗ | ✓ | ✗ | 数据集 | 离线学习 |
| **本项目 `UserModelDiverse`** | △ (单 episode 不切) | ✗ | ✗ | ✗ | ✗ | Perlin + 模式离散 |

观察：
- **没有任何 UAV 工作走 C 的纯学习式 pilot 路线** —— 数据量是主要障碍。
- **Backman 2023 已经把 A+B 实现到了可以 sim-only train、真人 transfer 的程度**，且参数化 4 维直接对应"用户多样性"。
- 本项目里 `UserModelDiverse` 的"4 模式"是**最朴素的 A**（无状态转移），离 Backman 还有结构性差距。

---

## 3. 推荐落地方案

> **结论：以 Backman 2023 的 4 参数 (α, β, Ψ, Φ) + 摇杆 P-controller 为底座，
> 但把 Backman 的"task-phase FSM"替换为 **意图层 ⊕ 反应层 + 感知子模型** 的三段式结构，
> 实现一个 `UserModelIntent`，作为 `UserModelDiverse` 的下一代默认 pilot。
> Perlin/直线/弧线 等几何模式保留为 fallback，作为 D 路线的混合采样源之一。**

底座沿用 Backman 的理由：
1. **唯一被真人闭环验证过 sim→real transfer 的实现**，证明这种参数化粒度足以覆盖真人多样性。
2. **接口与本项目高度兼容**：Backman 的 pilot 输出的就是 XYZ 速度，与 `user_model.py` 的 `step(drone_state) -> (N,3) velocity` 完全一致。
3. **可解释 + 可批量并行**：4 个参数都在 `[0,1]` 上均匀采样，可直接放到 `(N,k)` 的 GPU tensor 里 vectorize。
4. **C 路线（学习式 pilot）的成本/收益不划算**：UAV 社区都没搞，且我们 `flight_recorder` 数据规模有限。可以先把 C 当作 **风格采样器**（episode 开始时从录制库选一段当 desired trajectory），让 B 去跟踪。

### 3.1 必须改写 Backman 的两点

| Backman 2023 | 本项目场景 | 改写动机 |
|--------------|-----------|---------|
| 任务阶段 FSM `Approach → Descent` | 通用飞行无任务阶段 | 必须切换到"行为模式"层 |
| 固定外部 `G` + `Ĝ_t → G` 的单调收敛（β 项） | 没有外部固定 G | 改成滚动短视界 intent `W_t`（生成式而非估计式） |

### 3.2 三段式结构总览

```
                ┌──────────────────────────────────────────┐
                │  PerceptionModel   (τ_perc, σ_perc, d_react)
                │     true_dist_to_obstacles  →  threat_t  │
                └──────────────────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     │
   ┌─────────────────┐   ┌─────────────────┐            │
   │  Intent Layer   │   │ Reactive Layer  │ ◄──────────┘
   │ (semi-Markov)   │──►│ (preempting     │
   │ Cruise/Maneuver │   │  overlay)       │
   │ StationKeep/Idle│   │ NoReact/Late    │
   │ → W_t (rolling) │   │ EmergencyStop   │
   └─────────────────┘   │ Freeze/Evade    │
            │            │ Overcorrect/Surge│
            │            └─────────────────┘
            │                     │
            └─────────►  V_t  ◄───┘
                         │
              ┌──────────▼──────────┐
              │ Joystick P-controller│ (Backman 摇杆物理)
              │  + Adaptability I    │
              └──────────┬──────────┘
                         ▼
                    a_u (N, 3)
```

### 3.3 意图层（Intent Layer，自发 semi-Markov）

| 模式 | 含义 | dwell 分布 | `W_t` 生成器 |
|------|------|-----------|------------|
| `Cruise` | 朝某方向稳定推杆 | lognormal(μ=ln 5s, σ=0.6) | `pos + Φ·v_max·τ_h·ĥ_t`，`ĥ_t` 为 OU 漂移航向 |
| `Maneuver` | 主动转向/爬升/横移 | lognormal(μ=ln 1.5s, σ=0.5) | 参数化弧/螺旋/阶跃高度 |
| `StationKeep` | 想悬停在锚点 | lognormal(μ=ln 3s, σ=0.7) | `W_t = anchor`（进入时锁定） |
| `Idle` | 摇杆回中、无主动指令 | lognormal(μ=ln 1s, σ=0.5) | `W_t = pos_t` |

**转移**：semi-Markov，下一模式由 profile-依赖的类别分布采样；`η ∈ [0,1]` 控制保守 vs 激进 pilot 的模式分布偏好。

### 3.4 反应层（Reactive Layer，由感知触发的 overlay，可抢占）

| 反应模式 | 行为 | 制造 / 测试什么 |
|---------|------|----------------|
| `NoReact` | 完全不反应，意图层照常推杆 | 入侵型风险 — 助手必须主动接管 |
| `LateReact` | 反应阈值比正常 pilot 晚 τ_late | 助手必须能"补救"已经临近的危险 |
| `EmergencyStop` | 速度指令在 < 0.3 s 内 ramp → 0 | 助手要平稳吸收急停，不让无人机失稳 |
| `Freeze` | 摇杆瞬间回中并保持 | 助手要在"无 intent 信号"下自主脱险 |
| `Evade` | 朝感知到的障碍法向施加大幅横向推杆，方向**可能错** | 助手要识别躲避意图但纠正错误方向 |
| `Overcorrect` (PIO) | 意图反向 + 周期性振荡 | 助手要抑制振荡，不能跟着共振 |
| `Surge` | "fight" 反应：朝原方向推得更猛 | 助手必须能抗 pilot 错指令 |

依据：`fight/flight/freeze` 应激反应文献；`PIO` 是航空学经典 sim-to-real 失败模式；`NoReact / LateReact` 是 ADAS/AEB 数据集里的标准标签。

### 3.5 感知子模型（PerceptionModel）

```
perceived_dist_t = true_dist_{t - τ_perc}  +  N(0, σ_perc²)
threat_t         = ramp(d_react − perceived_dist_t)        # ∈ [0, 1]
```

参数 `(τ_perc, σ_perc, d_react)` 决定 pilot 类型：
- 老练：`τ_perc` 小、`σ_perc` 小、`d_react` 大 → 早反应、反应正确
- 新手：`τ_perc` 大、`σ_perc` 大、`d_react` 小 → 晚反应，易触发 `LateReact / Evade(错向)`
- 分心：`τ_perc` 时间相关跳变（建模"低头看手机"间隔） → 易触发 `NoReact`

反应模式触发：`P(react_mode | threat_t, profile) = softmax(w_profile · feat_t)`，进入后有最小 dwell 0.5–1 s。

**还要保留 spontaneous reactive**：偶尔在没有真实威胁时也触发 `Freeze` 或 `EmergencyStop`（"误判威胁"），让助手学会"pilot 急停 ≠ 一定有危险"。

### 3.6 滚动 intent 更新（替换 Backman Eq.1）

```
W_{t+1} = (1 − γ_α)·W_t                                   # 自然衰减
        + γ_α · π_mode( pos_t, heading_t, ξ_t )           # 模式驱动重采样
        + α · K_react · (a_a − a_u_prev)                  # 助手反馈（保留 Backman α 项）
```

**β 的语义改造**：原 Backman 的 β 是"通过感知收敛 goal 估计的能力"，在没有外部 G 的场景里改为 **模式承诺度**（mode-switch resistance），作用于 dwell 分布的乘子 —— β 高的 pilot 一旦进入 Cruise 就不愿被打断。

### 3.7 4(+) 参数语义对照

| 参数 | 原 Backman 语义 | 通用飞行下新语义 |
|------|----------------|------------------|
| `α` | 对助手的顺从度 | 不变（控制反馈强度 + adaptability 衰减） |
| `β` | 通过感知收敛 goal 估计的能力 | **模式承诺度**（dwell 期望长度乘子） |
| `Ψ` | 摇杆推杆陡峭度（P_gain） | 不变 |
| `Φ` | 期望最大速度 | 不变 |
| `η` *(新增)* | — | 模式分布偏好（保守 vs 激进） |
| `(τ_perc, σ_perc, d_react)` *(新增)* | — | perception profile，决定反应模式触发时机/正确率 |
| `wrong_direction_prob` *(新增)* | — | `Evade` 朝错方向躲的概率 |

### 3.8 与现有 reward 的耦合

Backman 的 `R_ActionDiff = -‖a_u - a_a‖` **不能直接复用**：他场景里 pilot 的 `Ĝ` 最终收敛到真 `G`，所以"少干预"是合理的；我们这里 pilot 不知道障碍且 intent 是滚动生成的，助手必须更主动。建议改成：

- `R_intent_match = -‖proj_safe(a_a) - W_t‖`：助手在**安全方向上的投影**应贴合 pilot intent
- `R_safety = +1 if 不撞墙 else -K`
- `R_minimal_intervention = -‖a_a - a_u‖ · safety_margin(s)`：仅当安全裕度足够时才奖励"少干预"

### 3.9 与 tunnel / 其他任务的契合
- 当前 `UserModelTunnel` 的"硬编码 vx=1·max_speed bias" → 退化为"`Cruise` 模式 + 朝向先验"的特例，通过 `mode_prior_override` hook 注入。
- inspection / free-flight 任务通过同一 hook 注入不同模式偏置。
- M2 失败案例 (`ana_docs/experiments/m2_diverse_pilot_analysis.md`) 可在该模型 `α=0`（不顺从）+ `wrong_direction_prob=0.5` 下重跑，作为对抗 stress test。

### 3.10 不推荐的路径
- 直接把 C 上线作为 default pilot：数据量不够 + 难以 vectorize + 模式易坍缩。
- 引入 haptic / gaze 等额外通道：当前任务没这个传感器。
- 让 pilot 完全感知障碍：助手无事可做，学到退化策略。
- 让 pilot 完全不感知障碍：助手永远在救火，学到"接管 = 总是对"，无法学协作。**必须保留 perception 子模型作为中间地带。**

---

## 4. 推荐方案接口与伪代码 sketch

> 仅作为接口与算法骨架，不在本调研中实现。
> 目标文件：
> - `isaac-training/src/core/user_model_intent.py` (主模型)
> - `isaac-training/src/core/pilot_perception.py` (感知子模型)
> - `isaac-training/src/core/pilot_modes.py` (模式枚举 + 转移矩阵)

### 4.1 模式与配置

```python
# isaac-training/src/core/pilot_modes.py  (sketch)

from enum import IntEnum

class IntentMode(IntEnum):
    CRUISE       = 0
    MANEUVER     = 1
    STATION_KEEP = 2
    IDLE         = 3

class ReactMode(IntEnum):
    NONE            = 0   # 反应层未激活，意图层直通
    NO_REACT        = 1   # 故意不反应（"没看见"）
    LATE_REACT      = 2   # 触发阈值延后
    EMERGENCY_STOP  = 3
    FREEZE          = 4
    EVADE           = 5
    OVERCORRECT     = 6   # PIO 振荡
    SURGE           = 7   # fight 反向
```

```python
# isaac-training/src/core/user_model_intent.py  (sketch — 不要直接运行)

from dataclasses import dataclass, field
import torch

@dataclass
class IntentPilotConfig:
    # --- Backman 4 参数（含语义改造） ---
    alpha_range: tuple = (0.0, 1.0)   # conformance to assistant
    beta_range:  tuple = (0.2, 1.0)   # mode commitment (dwell multiplier)
    psi_range:   tuple = (0.2, 1.0)   # joystick aggressiveness (P-gain)
    phi_range:   tuple = (0.4, 1.0)   # daringness / desired-speed scale
    eta_range:   tuple = (0.0, 1.0)   # 模式分布偏好（保守↔激进）

    # --- 摇杆物理 / adaptability ---
    P_gain_scale: float = 0.6
    I_gain:  float = 0.05
    I_decay: float = 0.9
    K_react: float = 0.05             # 助手反馈对 W_t 的更新强度（替换 K_alpha）
    gamma_alpha: float = 0.05         # W_t 自然衰减系数

    # --- 滚动 intent ---
    horizon_sec: float = 2.0
    heading_ou_tau: float = 1.5       # OU 漂移时间常数
    heading_ou_sigma: float = 0.3
    max_speed: float = 1.5

    # --- 模式 dwell（lognormal: mu_ln_sec, sigma_ln） ---
    dwell_lognormal: dict = field(default_factory=lambda: {
        IntentMode.CRUISE:       (1.6, 0.6),   # ~ exp(1.6) ≈ 5s
        IntentMode.MANEUVER:     (0.4, 0.5),   # ~ 1.5s
        IntentMode.STATION_KEEP: (1.1, 0.7),   # ~ 3s
        IntentMode.IDLE:         (0.0, 0.5),   # ~ 1s
    })

    # --- 感知子模型 ---
    tau_perc_range:   tuple = (0.05, 0.4)     # s
    sigma_perc_range: tuple = (0.0, 0.5)      # m
    d_react_range:    tuple = (0.6, 2.5)      # m

    # --- 反应层 ---
    react_min_dwell_sec: float = 0.5
    spontaneous_react_rate_hz: float = 0.05   # 无威胁也偶发误判
    wrong_direction_prob: float = 0.2         # Evade 朝错方向的概率

    sim_dt: float = 0.02
```

### 4.2 主类骨架

```python
class UserModelIntent:
    """
    意图层 ⊕ 反应层 + Perception 三段式 pilot, vectorized over N envs.
    Drop-in 兼容 UserModel*.reset / .step（新增可选 assistant_action / env_geom kwargs）。
    """

    def __init__(self, num_envs, device, cfg: IntentPilotConfig):
        self.N, self.dev, self.cfg = num_envs, device, cfg

        # --- per-env profile (episode 重采样) ---
        self.alpha = torch.zeros(self.N, device=device)
        self.beta  = torch.zeros(self.N, device=device)
        self.psi   = torch.zeros(self.N, device=device)
        self.phi   = torch.zeros(self.N, device=device)
        self.eta   = torch.zeros(self.N, device=device)
        self.tau_perc   = torch.zeros(self.N, device=device)
        self.sigma_perc = torch.zeros(self.N, device=device)
        self.d_react    = torch.zeros(self.N, device=device)

        # --- intent layer state ---
        self.intent_mode    = torch.full((self.N,), int(IntentMode.CRUISE),
                                          dtype=torch.long, device=device)
        self.dwell_remain   = torch.zeros(self.N, device=device)
        self.W_t            = torch.zeros(self.N, 3, device=device)   # rolling waypoint
        self.heading_dir    = torch.zeros(self.N, 3, device=device)   # OU drift
        self.anchor         = torch.zeros(self.N, 3, device=device)   # for STATION_KEEP

        # --- reactive layer state ---
        self.react_mode     = torch.zeros(self.N, dtype=torch.long, device=device)
        self.react_remain   = torch.zeros(self.N, device=device)

        # --- perception buffer (ring buffer for τ_perc 延迟) ---
        self.perc_buffer    = None      # 由首次 step() 建立

        # --- joystick / adaptability ---
        self.J = torch.zeros(self.N, 3, device=device)
        self.I = torch.zeros(self.N, 3, device=device)
        self.last_pilot_action = torch.zeros(self.N, 3, device=device)

    # ------------------------------------------------------------------
    def reset(self, pos, quat, env_ids=None, seed=None,
              mode_prior_override=None, anchor=None):
        """
        mode_prior_override: 任务可注入模式先验（如 tunnel 场景 boost CRUISE）
        anchor:              STATION_KEEP 模式锁定点（默认 = pos）
        """
        ids = env_ids if env_ids is not None else slice(None)
        # 1) profile 采样
        self._sample_profile(ids)
        # 2) 初始化意图层（按 η + override 抽模式）
        self._sample_intent_mode(ids, mode_prior_override)
        self._refresh_W_t(ids, pos)
        self.anchor[ids] = anchor[ids] if anchor is not None else pos[ids]
        # 3) reactive 重置
        self.react_mode[ids] = int(ReactMode.NONE)
        self.react_remain[ids] = 0
        # 4) joystick 状态清零
        self.J[ids] = 0; self.I[ids] = 0
        self.last_pilot_action[ids] = 0

    # ------------------------------------------------------------------
    def step(self, drone_state, drone_pos_w,
             assistant_action=None,
             env_geom=None):
        """
        env_geom: dict with optional 'nearest_obstacle_dist' (N,) and 'nearest_obstacle_normal' (N,3)
                  无该信息时反应层只走 spontaneous 通道。
        """
        a_a = assistant_action if assistant_action is not None else torch.zeros_like(self.J)
        a_u_prev = self.last_pilot_action

        # === (A) Perception: τ_perc 延迟 + σ_perc 噪声 → threat_t ===
        threat_t, threat_normal = self._perceive(env_geom)

        # === (B) Reactive layer: 触发 / 维持 / 退出 ===
        self._update_reactive_layer(threat_t)

        # === (C) Intent layer: dwell 倒计时 + semi-Markov 转移 + W_t 滚动更新 ===
        self._tick_intent_layer(drone_pos_w, a_a, a_u_prev)

        # === (D) Trajectory planner → V_t ===
        V_t_intent = self._intent_to_velocity(drone_pos_w)
        V_t = self._apply_reactive_overlay(V_t_intent, threat_normal)

        # === (E) 摇杆 P-controller (Backman) ===
        P = self.psi[:, None] * self.cfg.P_gain_scale
        self.J = self.J + (V_t - self.J) * P

        # === (F) Adaptability integrator ===
        self.I = self.cfg.I_decay * self.I + (a_u_prev - a_a) * (1.0 - self.alpha[:, None])

        a_u = (self.J + self.I * self.cfg.I_gain).clamp(-self.cfg.max_speed, self.cfg.max_speed)
        self.last_pilot_action = a_u
        return a_u

    # ------------------------------------------------------------------
    def privileged_info(self):
        """提供给 critic 的特权信息（POMDP → MDP）。"""
        return {
            "alpha": self.alpha, "beta": self.beta,
            "psi":   self.psi,   "phi":  self.phi, "eta": self.eta,
            "intent_mode":  self.intent_mode,
            "react_mode":   self.react_mode,
            "W_t":          self.W_t,
            "dwell_remain": self.dwell_remain,
        }

    # ---- private helpers (signatures only) ----
    def _sample_profile(self, ids): ...
    def _sample_intent_mode(self, ids, prior_override=None): ...
    def _refresh_W_t(self, ids, pos): ...
    def _perceive(self, env_geom): ...                 # 返回 threat_t (N,), threat_normal (N,3)
    def _update_reactive_layer(self, threat_t): ...    # 处理 spontaneous + threat-driven 触发, dwell 倒计时
    def _tick_intent_layer(self, pos, a_a, a_u_prev):  # dwell--, semi-Markov 转移, OU 航向漂移, W_t 更新
        ...
    def _intent_to_velocity(self, pos):                # 按 IntentMode 分发 W_t → V_t
        ...
    def _apply_reactive_overlay(self, V_t, threat_normal):
        # 按 ReactMode 修改 V_t：
        # EMERGENCY_STOP → ramp to 0; FREEZE → 0 + J 强制衰减; EVADE → 横向 ±normal (含 wrong_direction_prob); ...
        ...
```

### 4.3 与现有训练流水线的集成点

1. **`PPO` rollout 层**：把上一步 `assistant_action` 反馈给 `UserModelIntent.step(..., assistant_action=...)`。当前 `UserModel.step` 不接收这个参数 → 需加可选 kwarg，向后兼容。
2. **环境几何 hook**：env 需要在 step 时提供 `nearest_obstacle_dist / normal`（已有 ray-cast 的任务可直接复用）。无该信息时反应层降级为 spontaneous-only。
3. **Critic 输入**：把 `privileged_info()` 中 `intent_mode / react_mode / W_t / α / β` 接到 critic 的额外特征上（Backman 证明这能显著加速收敛）。
4. **Reward 改写**：见 §3.8，必须把 `R_ActionDiff` 替换成 safety-margin 加权版本，否则 pilot 不感知障碍 + 助手少干预 reward 会打架。
5. **Curriculum**（建议三阶段）：
   - **Stage 1 — Tame pilot**：`alpha_range=(0.7,1.0)`、反应层只允许 `EMERGENCY_STOP/FREEZE`、`wrong_direction_prob=0`、感知 `(τ_perc, σ_perc)` 取小值。
   - **Stage 2 — Realistic pilot**：放开 4 参数全 range，反应层加入 `LATE_REACT/EVADE`。
   - **Stage 3 — Adversarial pilot**：加入 `OVERCORRECT/SURGE`、`wrong_direction_prob=0.5`，作为 robust RL stress test。
6. **方案 D 混合**：以一定比例（如 70%）启用 `UserModelIntent`，30% 仍用 `UserModelDiverse` 的几何模式（直线/弧线/hover）保持探索覆盖。
7. **任务 hook**：`mode_prior_override` 让 task 注入场景偏置（tunnel boost `CRUISE` + 朝向先验；inspection boost `STATION_KEEP` 等）。

---

## 5. 后续实现与验证路线（feature/pilot-intent-model 分支）

**实现顺序建议**：

1. **M1 — 框架骨架**：`pilot_modes.py` + `IntentPilotConfig` + `UserModelIntent` 空骨架（`step()` 直接退化为 Backman 4 参数 single-mode CRUISE，验证接口与 PPO rollout 互通）。
2. **M2 — 意图层**：实现 semi-Markov FSM + lognormal dwell + OU 航向漂移 + 4 个 IntentMode 的 `_intent_to_velocity`。在不接 reactive 的情况下能跑通训练。
3. **M3 — 感知子模型 + 反应层**：加 `pilot_perception.py`，先实现 `EMERGENCY_STOP / FREEZE / NO_REACT` 三个最简单的 reactive，验证 reward 改写后助手行为变化。
4. **M4 — 完整反应层**：补 `LATE_REACT / EVADE / OVERCORRECT / SURGE`，加入 spontaneous reactive。
5. **M5 — Critic 特权信息接入**：把 `privileged_info()` 接到 critic，对比是否加速收敛。
6. **M6 — Curriculum**：实现 Stage 1/2/3 的配置切换。

**验证基线**：

- 复用 `flight_recorder` 录 5–10 段真人飞行 → 用作 `UserModelIntent` 的"参数辨识"基准（fit α/β/Ψ/Φ + perception 参数）。
- 在 `ana_docs/research/` 增加一份 `pilot_realism_metrics.md`，定义"pilot 真实度"指标：自相关函数、回中频率、推杆速率分布、轴间相关性、模式 dwell 直方图，对比 Perlin / Diverse / Intent / 真人 4 种 pilot。
- M2 失败案例 (`ana_docs/experiments/m2_diverse_pilot_analysis.md`) 用 `UserModelIntent` Stage 3 配置重跑，观察 `α=0 + wrong_direction_prob=0.5` 极端 stress 下助手是否仍能工作。
- A/B 测试：相同 reward 与 PPO 配置下，原 `UserModelDiverse` vs `UserModelIntent`（Stage 2）训出的助手在固定真人录制脚本上的成功率与碰撞率。

---

## 6. 参考链接

- [Reddy et al. 2018, RSS](https://arxiv.org/abs/1802.01744) · code: <https://github.com/rddy/deepassist>
- [Schaff & Walter 2020, RSS](https://arxiv.org/abs/2004.05097)
- [Backman et al. 2021, RA-L](https://arxiv.org/abs/2011.13146)
- [Backman et al. 2023, Auton. Robot.](https://link.springer.com/article/10.1007/s10514-023-10143-3) ⭐ 主参考
- [Wang et al. 2021, GPA-Teleop](https://arxiv.org/abs/2109.04907)
- [Zhang et al. 2024, Haptic CBF](https://arxiv.org/abs/2403.15335)
- [Patrikar et al. 2022, *Predicting like a Pilot*](https://arxiv.org/abs/2202.05140)
- [Pfeiffer et al. 2022, Visual Attention in Drone Racing](https://doi.org/10.1371/journal.pone.0264471)
