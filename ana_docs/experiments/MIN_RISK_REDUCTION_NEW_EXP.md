# MIN_RISK_REDUCTION_NEW_EXP：最小干预的反事实风险降低实验方案

> 目标：在现有 PPO-Beta residual shared-control 框架上，设计一个新的experiment_specs与环境的代码实现。该方案包括：
>
> 1. 动态风险估计函数；
> 2. 基于反事实风险降低的 reward 替换；
> 3. 基于 risk reduction / pilot risk 的 residual regularization 改进；
> 4. 对 PPO 训练流程、日志、消融和验证指标的实现约定。
>
> 本文档不包含具体 Python 代码。它面向后续负责实现的 agent，用于说明算法原理、接口需求和实验落地路线。当前只考虑**仿真训练和验证**，暂不涉及真机部署和 hard safety shield。

---

## 1. 背景与设计动机

当前 shared-control 模型的spec（基于SharedRLControl/isaac-training/experiment_specs/tunnel.py）的基本形式是：

- 输入：无人机局部观测、状态、LiDAR 信息、人类速度指令 `u_t`；
- 输出：辅助后的速度命令 `a_t`；
- 策略结构：以 human command 为中心的 residual policy；
- 分布形式：Beta 分布，天然适配有界动作空间；
- 当前 PPO actor loss：PPO clipped objective + 固定 residual regularization；
- 当前 reward：主要依赖人工启发式的安全惩罚、指令跟随、任务项和平滑项。

这个结构已经具备共享控制的基本形式：copilot 不直接完全替代人类，而是在 human command 周围学习 residual correction。

但是旧方案存在两个关键问题。

### 1.1 固定 residual regularization 不区分必要干预和不必要干预

当前 residual regularization 主要惩罚策略偏离 human command 的程度。其隐含假设是：偏离越小越好。

但在共享控制中，这个假设并不总是成立：

- 当 human command 指向安全区域时，偏离确实应该很小；
- 当 human command 在当前动力学状态下会导致碰撞时，偏离不但合理，而且必要；
- 当 residual 很大但没有降低风险时，这不是有效干预；
- 当 residual 很小但 human command 已经危险时，这是危险的不干预。

因此，应该惩罚的不是所有 residual，而是**不必要的 residual**；应该鼓励的不是任意偏离，而是**能够降低风险的必要干预**。

### 1.2 只看速度指令方向的风险函数不够真实

无人机处于速度控制模式时，命令 `q` 并不会瞬间变成实际速度。真实系统存在：

- 当前实际速度和惯性；
- 控制链路延迟；
- 低层速度控制器响应时间；
- 最大加速度或姿态变化限制；
- LiDAR / depth 感知延迟；
- 仿真与真实动力学响应差异。

因此，判断一个 human command 是否危险，不能只看 `u_t` 的方向是否指向障碍物，而应该问：

> 在当前实际速度、延迟和速度响应限制下，如果执行这个命令，未来短时间内无人机的可达轨迹是否会扫到障碍物？

本方案将原先的几何方向风险升级为：

> **dynamics-aware command-conditioned risk**

也就是动态感知的、候选命令条件化的短时轨迹风险。

---

## 2. 新问题定义：最小干预的反事实风险降低

### 2.1 基本变量

每个时间步记为：

- `x_t`：当前物理状态，例如位置、速度、yaw、上一条实际命令等；
- `z_t`：当前局部感知，例如 LiDAR distances、ray directions 或局部点云；
- `u_t`：human / pilot velocity command；
- `a_t`：copilot 输出的 assisted velocity command；
- `v_t`：当前实际速度；
- `q`：任意候选速度命令，可以是 `u_t` 或 `a_t`。

定义动态风险函数：

\[
\rho_{\mathrm{dyn}}(x_t,z_t,q) \in [0,1]
\]

表示：在当前状态和局部感知下，如果发送候选速度命令 `q`，考虑延迟、惯性和低层速度响应后，未来短时可达轨迹的局部碰撞风险。

### 2.2 反事实 pilot risk 与 assisted risk

对同一个状态，同时评估两个反事实风险：

\[
\rho^u_t = \rho_{\mathrm{dyn}}(x_t,z_t,u_t)
\]

\[
\rho^a_t = \rho_{\mathrm{dyn}}(x_t,z_t,a_t)
\]

其中：

- `rho_u`：如果直接执行 human command，会有多危险；
- `rho_a`：如果执行 assisted command，会有多危险。

反事实风险降低定义为：

\[
\Delta \rho_t = \rho^u_t - \rho^a_t
\]

解释：

- `Delta rho > 0`：辅助命令比人类原始命令更安全；
- `Delta rho = 0`：辅助没有产生明显风险改善；
- `Delta rho < 0`：辅助命令反而更危险。

### 2.3 最小干预原则

copilot 的目标不是单纯“避障”，也不是单纯“跟随人类”，而是：

> 在 human command 动态风险低时，尽量不干预；
> 在 human command 动态风险高时，允许并鼓励产生能够降低风险的 residual；
> 干预之后的命令必须比原始命令更安全，至少不能更危险；
> 在风险降低足够相似时，优先选择更接近 human command 的动作。

可以写成如下抽象目标：

\[
\max_\theta\; \mathbb{E}\left[
    r_{\mathrm{follow}}(u_t,a_t,v_t) +
    w_{\Delta \rho}(\rho^u_t-\rho^a_t) -
    w_{\mathrm{worse}}[\rho^a_t-\rho^u_t]_+ -
    \lambda(\rho^u_t)\|\delta_t\|^2
\right]
\]

其中：

\[
\delta_t = a_t - u_t
\]

或者在 Beta policy 的 normalized / modal action space 中定义：

\[
\delta^{\mathrm{mode}}_t = m_\theta(o_t)-\bar{u}_t
\]

这里 `m_theta(o_t)` 是策略分布的 deterministic mode，`bar u_t` 是归一化后的 human command。

### 2.4 为什么这比 PPO-Lagrangian 更适合该问题

PPO-Lagrangian 通常使用全局 dual variable 来调节某个平均约束，例如平均安全代价或平均 residual budget。

但 shared control 中 residual budget 应该是状态相关的：

- safe human command：预算接近 0；
- dangerous human command：预算应被放开；
- high inertia / delayed state：可能需要更早、更强的纠偏。

因此，本方案不使用全局在线 Lagrangian，而采用：

> **local risk-conditioned intervention budget**

即由 `rho_u` 控制每个状态下 residual regularization 的强度。

---

## 3. 动态风险估计函数设计

动态风险估计函数是本实验的核心。它需要回答：

> 对候选命令 `q`，在当前实际速度和延迟条件下，未来短时轨迹是否会进入障碍物附近？

### 3.1 输入接口

环境需要为风险估计模块提供以下量。

#### 必需输入

1. 当前实际速度

\[
v_t
\]

建议使用 world frame 或与 LiDAR 点云一致的局部 frame。

2. 候选速度命令

\[
q
\]

该候选命令可以是：

- human command `u_t`；
- assisted command `a_t`；
- 未来用于消融或采样的其他候选 command。

3. 延迟期间保持的命令

\[
q_{\mathrm{hold}}
\]

建议使用上一条实际发送给低层控制器的 command。如果环境中没有该记录，可以暂时用当前实际速度 `v_t` 近似，但最终建议显式维护 last applied command。

4. LiDAR distances 和 ray directions

- distances：`d_i`；
- ray directions：`e_i`；
- 局部障碍点可近似为：

\[
P_i = d_i e_i
\]

5. 风险模型参数

- `dt_risk`：风险预测离散步长；
- `T_horizon`：预测时域；
- `tau_delay`：总延迟；
- `tau_v`：速度响应时间常数；
- `a_max`：最大加速度；
- `v_max`：速度上限；
- `r_uav`：无人机等效半径；
- `margin_static`：静态安全裕度；
- `margin_speed`：速度相关裕度；
- `margin_time`：时间相关裕度；
- `risk_temperature`：风险 sigmoid 温度。

### 3.2 Frame 对齐要求

实现 agent 必须先确认所有风险估计输入处于同一坐标系。

允许两种方案：

#### 方案 A：全部使用 world frame

- `v_t`：world frame；
- `u_t`：由 body / yaw-aligned command 转为 world frame；
- `a_t`：策略输出后已经转为 world frame；
- LiDAR rays：也转换到 world frame。

#### 方案 B：全部使用 yaw-aligned local frame

- `v_t`：转换到 yaw-aligned local frame；
- `u_t`：保持 actor 输入的 command frame；
- `a_t`：转换回相同 local frame；
- LiDAR rays：保持 local frame。

推荐在初期使用与现有 reward / LiDAR 处理最一致的 frame，减少改动。无论选哪一种，必须保证：

\[
q, v_t, P_i
\]

在同一个 frame 下计算。

### 3.3 延迟 + 一阶速度响应模型

本方案不引入完整四旋翼动力学，而使用速度控制任务中足够实用的近似模型。

延迟期间，新命令尚未生效：

\[
q_k = q_{\mathrm{hold}},\quad k\Delta t < \tau_{\mathrm{delay}}
\]

延迟之后，速度按一阶响应追踪候选命令：

\[
\dot v = \mathrm{clip}_{a_{\max}}\left(\frac{q-v}{\tau_v}\right)
\]

离散化为：

\[
v_{k+1} = v_k + \Delta t\cdot
\mathrm{clip}_{a_{\max}}\left(\frac{q_k-v_k}{\tau_v}\right)
\]

\[
p_{k+1} = p_k + \Delta t\cdot v_{k+1}
\]

其中 `p_k` 是相对当前时刻的未来位移。

这个模型捕捉三个关键信息：

1. 新命令不会立刻生效；
2. 实际速度不会瞬间等于 commanded velocity；
3. 当前速度越大，未来短时扫掠距离越大。

### 3.4 轨迹 tube clearance

对候选命令 `q`，风险模型会生成一段未来相对轨迹：

\[
\Gamma(q)=\{p_1(q),p_2(q),...,p_K(q)\}
\]

将 LiDAR 返回点近似为局部障碍点：

\[
P_i=d_i e_i
\]

对每个未来轨迹点计算到最近障碍点的距离：

\[
d_k(q)=\min_i \|P_i-p_k(q)\|
\]

考虑无人机半径和安全裕度，定义 tube 半径：

\[
r_{\mathrm{tube}}(k)=
    r_{\mathrm{uav}}+
    m_{\mathrm{static}}+
    m_{\mathrm{speed}}\|v_k\|+
    m_{\mathrm{time}}k\Delta t
\]

然后 clearance 为：

\[
c_k(q)=d_k(q)-r_{\mathrm{tube}}(k)
\]

若：

\[
c_k(q)<0
\]

说明预测轨迹 tube 已经与障碍物膨胀区域相交。

### 3.5 风险映射

将 clearance 映射为 `[0,1]` 风险：

\[
r_k(q)=\sigma\left(\frac{-c_k(q)}{\tau_c}\right)
\]

其中 `tau_c` 是风险温度。

风险聚合建议使用 `max`：

\[
\rho_{\mathrm{full}}(q)=\max_{k=1,...,K} r_k(q)
\]

这样可以更敏感地捕捉任意时间点的 near-collision。

### 3.6 delay risk 与 post-command risk

由于延迟阶段当前动作无法改变，因此建议把风险拆成：

\[
\rho_{\mathrm{delay}}(q)=\max_{k\Delta t<\tau_{\mathrm{delay}}} r_k(q)
\]

\[
\rho_{\mathrm{post}}(q)=\max_{k\Delta t\ge\tau_{\mathrm{delay}}} r_k(q)
\]

以及：

\[
\rho_{\mathrm{full}}(q)=\max(\rho_{\mathrm{delay}}(q),\rho_{\mathrm{post}}(q))
\]

在实际使用中：

- risk reduction 使用 `post` 风险；
- absolute safety penalty 使用 `full` 风险；
- delay risk 单独记录并惩罚，用来鼓励策略提前干预。

具体定义：

\[
\rho^u_{\mathrm{post}} = \rho_{\mathrm{post}}(u_t)
\]

\[
\rho^a_{\mathrm{post}} = \rho_{\mathrm{post}}(a_t)
\]

\[
\rho^a_{\mathrm{full}} = \rho_{\mathrm{full}}(a_t)
\]

\[
\rho_{\mathrm{delay}} = \rho_{\mathrm{delay}}(a_t)
\]

反事实风险降低为：

\[
\Delta \rho_t = \rho^u_{\mathrm{post}} - \rho^a_{\mathrm{post}}
\]

---

## 4. 奖励函数替换方案

本实验的 reward 改进目标是：

> 用符合反事实风险降低思想的动态风险 reward 替换当前启发式 safety penalty。

不是简单把旧安全项叠加更多惩罚，而是逐步替换当前安全惩罚，让 reward 的安全部分围绕以下问题构造：

1. human command 是否动态危险？
2. assisted command 是否比 human command 更安全？
3. assisted command 是否仍然有绝对动态风险？
4. 当前状态是否已经进入延迟阶段无法及时改变的风险？

### 4.1 新 reward 总体形式

建议目标 reward 写成：

\[
r_t =
    r_{\mathrm{follow\_gated}}
    + r_{\mathrm{risk\_reduce}}
    + r_{\mathrm{risk\_worse}}
    + r_{\mathrm{abs\_risk}}
    + r_{\mathrm{delay\_risk}}
    + r_{\mathrm{task}}
    + r_{\mathrm{smooth}}
    + r_{\mathrm{terminal}}
\]

其中安全相关的新项为：

\[
r_{\mathrm{risk\_reduce}}
= w_{\Delta\rho}(\rho^u_{\mathrm{post}}-\rho^a_{\mathrm{post}})
\]

\[
r_{\mathrm{risk\_worse}}
= -w_{\mathrm{worse}}[\rho^a_{\mathrm{post}}-\rho^u_{\mathrm{post}}]_+
\]

\[
r_{\mathrm{abs\_risk}}
= -w_{\mathrm{abs}}\rho^a_{\mathrm{full}}
\]

\[
r_{\mathrm{delay\_risk}}
= -w_{\mathrm{delay}}\rho_{\mathrm{delay}}
\]

### 4.2 风险门控的跟随奖励

旧 reward 中的 following term 不应直接删除，因为它是 shared control 中保留 human intent 的核心。

但是当 human command 动态危险时，策略不应该被强制盲目跟随。因此引入 risk-gated following reward：

\[
r_{\mathrm{follow\_gated}}
= g_f(\rho^u_{\mathrm{post}}) r_{\mathrm{follow}}
\]

其中：

\[
g_f(\rho)=\mathrm{clip}(1-\alpha_f\rho, g_{\min},1)
\]

解释：

- `rho_u_post` 低：human command 动态安全，强跟随；
- `rho_u_post` 高：human command 动态危险，降低跟随权重；
- `g_min` 大于 0：即使高风险，也保留一定 intent pressure，避免策略学成长期 hover 或完全无视人类。

### 4.3 每个 reward 项的作用

#### 4.3.1 `risk_reduce`：鼓励有效干预

\[
w_{\Delta\rho}(\rho^u_{\mathrm{post}}-\rho^a_{\mathrm{post}})
\]

该项奖励 assisted command 相对于 human command 的风险降低。

如果 human command 本来安全，则 `rho_u_post` 低，此项不会鼓励大幅偏离。

如果 human command 危险，且 assisted command 使动态风险下降，则此项给正奖励。

#### 4.3.2 `risk_worse`：禁止干预变坏

\[
-w_{\mathrm{worse}}[\rho^a_{\mathrm{post}}-\rho^u_{\mathrm{post}}]_+
\]

这个项必须保留。否则策略可能在某些状态下输出比 human command 更危险的动作，只要平均风险降低还不错。

#### 4.3.3 `abs_risk`：避免相对改善但仍然危险

\[
-w_{\mathrm{abs}}\rho^a_{\mathrm{full}}
\]

有些情况下，assisted command 相比 human command 稍微安全，但仍然很危险。例如：

\[
\rho^u=0.95,\quad \rho^a=0.80
\]

这时 risk reduction 是正的，但 `rho_a` 仍然过高。因此需要 absolute risk penalty。

#### 4.3.4 `delay_risk`：鼓励提前干预

\[
-w_{\mathrm{delay}}\rho_{\mathrm{delay}}
\]

如果延迟阶段风险已经高，说明当前时刻的动作已经有点来不及。这项短期内看似不能由当前 action 改变，但通过 RL return 可以让策略在更早状态学会提前干预。

### 4.4 与旧安全惩罚的替换关系

建议不要在最终版本中长期叠加所有旧安全惩罚，否则会导致 reward 目标混乱。推荐三阶段替换：

#### 阶段 1：logging only

只计算动态风险，不改变 reward。

目的：检查动态风险是否能解释旧 policy 的碰撞、near miss 和高风险行为。

#### 阶段 2：并行弱引入

保留旧 safety reward，但降低其权重，同时加入新 risk terms。

目的：避免训练突然崩溃，验证新项的梯度方向。

#### 阶段 3：正式替换

移除或显著降低旧 safety penalty，以新动态反事实风险项为主。

保留非安全类 reward：

- task / progress；
- alive；
- terminal success / collision；
- height / boundary；
- smoothness；
- command-following。

### 4.5 推荐初始权重范围

以下不是最终值，只作为仿真起始范围：

| 参数 | 建议范围 | 作用 |
|---|---:|---|
| `w_delta_rho` | 0.3–1.0 | 奖励风险降低 |
| `w_worse` | 1.0–3.0 | 强惩罚 assisted risk 高于 pilot risk |
| `w_abs` | 0.2–1.0 | 惩罚 assisted command 的绝对动态风险 |
| `w_delay` | 0.2–0.8 | 惩罚已经来不及改变的风险 |
| `alpha_f` | 0.3–0.7 | human command 风险越高，following 越弱 |
| `g_min` | 0.25–0.5 | 高风险状态下仍保留最低跟随压力 |
| `w_smooth` | 保持旧值或略增 | 抑制急剧 residual 和震荡 |

建议初始使用温和设置：

- `w_delta_rho = 0.5`；
- `w_worse = 1.5`；
- `w_abs = 0.4`；
- `w_delay = 0.3`；
- `alpha_f = 0.5`；
- `g_min = 0.35`。

如果策略过于保守：降低 `w_abs`、`w_delay` 或提高 `g_min`。

如果策略仍然撞障碍：提高 `w_worse`、`w_abs`、`w_delay`，或增大动态风险模型的安全裕度。

---

## 5. Risk reduction based residual regularization

### 5.1 当前 residual regularization 的问题

旧 PPO 训练中，actor loss 近似为：

\[
\mathcal{L}_{\pi}
= \mathcal{L}_{\mathrm{PPO}}
+ \lambda_{\Delta}\mathbb{E}\|\Delta_\theta\|^2
\]

其中 residual penalty 是固定的，不随风险变化。

这意味着：

- 安全状态下 residual 被惩罚；
- 危险状态下 residual 也被同样惩罚。

这与 shared control 的“必要干预”原则冲突。

### 5.2 新 regularization 形式

将 residual regularization 改为：

\[
\mathcal{L}_{\mathrm{reg}}
=
\mathbb{E}\left[
    g_{\Delta}(\rho^u_{\mathrm{post}})
    \|\delta^{\mathrm{mode}}_t\|^2
\right]
\]

其中：

\[
g_{\Delta}(\rho)=
    g_{\mathrm{danger}} +
    (g_{\mathrm{safe}}-g_{\mathrm{danger}})(1-\rho)^p
\]

解释：

- `rho_u_post` 低：human command 动态安全，`g_delta` 接近 `g_safe`，强惩罚 residual；
- `rho_u_post` 高：human command 动态危险，`g_delta` 接近 `g_danger`，放松 residual penalty；
- `g_danger` 不应为 0，因为即使危险，也不希望策略产生任意大、任意突兀的动作。

推荐初始参数：

| 参数 | 建议值 | 说明 |
|---|---:|---|
| `g_safe` | 1.0 | 安全状态下完整 residual penalty |
| `g_danger` | 0.03–0.10 | 危险状态下保留少量 residual penalty |
| `p` | 1.0–2.0 | 控制 gate 曲线陡峭程度 |
| `reg_coeff` | 沿用旧值或略降 | 外层 residual regularization 系数 |

### 5.3 residual 应该如何度量

建议不要继续直接正则 raw network output，例如 `_mean_delta`。

更合理的是使用 deterministic policy mode 与 human command 的实际偏离：

\[
\delta^{\mathrm{mode}}_t = m_\theta(o_t)-\bar u_t
\]

其中：

- `m_theta(o_t)`：Beta distribution 的 deterministic mode；
- `bar u_t`：与 Beta action domain 一致的 normalized human command；
- 二者都应在相同 action space 中比较，例如 `[0,1]` 或 physical velocity space。

推荐使用 normalized `[0,1]` space 作为最小实现，因为当前 residual Beta module 本身就在 Beta natural domain 中加 residual。

如果后续希望正则项更具物理意义，也可以使用 physical velocity space：

\[
\delta^{\mathrm{phys}}_t = a^{\mathrm{mode}}_t-u_t
\]

但无论使用哪种空间，都必须在文档和实现中保持一致。

### 5.4 为什么 gate 只能使用 pilot risk

residual regularization gate 应使用：

\[
\rho^u_{\mathrm{post}}
\]

而不是：

\[
\rho^a_{\mathrm{post}}
\]

原因：

- `rho_u_post` 只依赖状态和 human command，是当前情境下“是否允许干预”的条件；
- `rho_a_post` 依赖 actor 输出，如果用它作为 gate，策略可能通过输出某些动作改变自己的正则强度；
- 使用 `rho_u_post` 更符合共享控制语义：human command 危险时才放开干预预算。

### 5.5 新 PPO actor loss

最终 actor loss：

\[
\mathcal{L}_{\pi}^{\mathrm{new}}
=
\mathcal{L}_{\mathrm{PPO}}
+
\lambda_{\Delta}
\mathcal{L}_{\mathrm{reg}}
\]

其中：

\[
\mathcal{L}_{\mathrm{reg}}
=
\mathbb{E}\left[
    g_{\Delta}(\rho^u_{\mathrm{post}})
    \|m_\theta(o_t)-\bar u_t\|^2
\right]
\]

这个修改非常小，但语义变化很大：

> residual budget 从固定全局预算，变成了由 human command 动态风险决定的局部预算。

---

## 6. 环境与 PPO 的接口约定

因为实现 agent 会在训练环境中完成风险和 reward 计算，PPO 代码只需要读取必要信号。因此需要明确 TensorDict / rollout data 中需要新增的字段。

### 6.1 环境需要计算并记录的量

每个 transition 需要记录：

| 字段名建议 | 含义 | 用途 |
|---|---|---|
| `pilot_risk_dyn_post` | `rho_u_post` | PPO residual gate；logging |
| `assist_risk_dyn_post` | `rho_a_post` | reward；logging |
| `assist_risk_dyn_full` | `rho_a_full` | reward；logging |
| `delay_risk` | `rho_delay` | reward；logging |
| `risk_reduction_dyn` | `rho_u_post - rho_a_post` | reward；logging |
| `min_clearance_pilot` | pilot trajectory 最小 clearance | analysis |
| `min_clearance_assist` | assist trajectory 最小 clearance | analysis |
| `follow_gate` | risk-gated following 权重 | analysis |

其中 `pilot_risk_dyn_post` 是 PPO algorithm 必须读取的字段，其他字段主要用于 reward、logging 和 validation。

### 6.2 PPO 需要读取的字段

PPO 训练只需要从 transition 中读取：

\[
\rho^u_{\mathrm{post}}
\]

也就是 `pilot_risk_dyn_post`。

用于：

\[
g_{\Delta}(\rho^u_{\mathrm{post}})
\]

注意：该风险值不需要对 actor 反向传播。它来自环境计算，应视为 detached scalar signal。

### 6.3 Reward 计算位置

动态风险 reward 应在环境 step / reward function 中计算，而不是在 PPO `_update()` 中计算。

原因：

- reward 需要 `u_t`、`a_t`、`v_t`、LiDAR、last command、终止信息等环境变量；
- PPO 只消费 rollout 中的 reward，并基于 reward 计算 GAE；
- 将风险估计放在环境端更便于 logging、可视化和消融。

### 6.4 需要保持不变的部分

最小实现版本中，建议暂时保持以下模块不变：

- actor 网络结构；
- critic 网络结构；
- Beta policy 参数化；
- PPO clipped objective；
- GAE；
- entropy loss；
- curriculum 中的 residual scale / concentration 设置。

这样可以将实验变量集中在：

1. risk function；
2. reward；
3. residual regularization。

---

## 7. 实验落地路线

### Stage 1：动态风险 logging only

目标：验证动态风险估计对env_tunnel的baseline训练过程中的缺陷是否有解释力。

改动：

- 实现动态风险估计函数；
- 对 `u_t` 和 `a_t` 计算动态风险；
- 不改变 reward；
- 不改变 PPO actor loss。

记录：

- `pilot_risk_dyn_post`；
- `assist_risk_dyn_post`；
- `assist_risk_dyn_full`；
- `delay_risk`；
- `risk_reduction_dyn`；
- `min_clearance_pilot`；
- `min_clearance_assist`。

判断标准：

- 碰撞或 near miss 前，`assist_risk_dyn_full` 是否升高；
- 策略危险不干预时，`pilot_risk_dyn_post` 是否高但 residual 很小；
- 策略无效干预时，`risk_reduction_dyn` 是否接近 0 或为负；
- `delay_risk` 是否能识别“已经来不及”的状态。

如果这些现象不成立，优先调试 risk function，而不是直接进入新训练。

### Stage 2：只替换安全 reward

目标：验证 counterfactual dynamic risk reward 是否能改善安全表现。

改动：

- 保持旧 fixed residual regularization；
- 将旧 safety penalty 替换或弱化为新动态风险 reward；
- 保留 task、following、terminal、smoothness 等非安全项。

建议先使用弱替换：

- 旧 safety penalty 权重降低；
- 新 risk terms 以中等权重加入。

稳定后再进入正式替换：

- 旧 safety penalty 移除或大幅降低；
- 新 counterfactual risk terms 成为主要安全 reward。

### Stage 3：只改 residual regularization

目标：单独验证 risk-conditioned residual regularization。

两种可选实验：

1. 旧 reward + 新 residual gate；
2. 新 reward + 旧 fixed residual regularization。

这两个对照有助于回答：

- 安全提升主要来自 reward 还是 residual gate？
- residual gate 是否减少不必要干预？
- residual gate 是否增加危险状态下的必要干预？

### Stage 4：完整模型

最终训练配置：

- 动态风险 reward；
- risk-gated following reward；
- counterfactual risk reduction；
- risk-worsening penalty；
- assisted absolute risk penalty；
- delay risk penalty；
- risk-conditioned modal residual regularization。

该版本是本文档定义的主要新方法。

### Stage 5：仿真鲁棒性增强

在完整模型稳定后，引入 dynamics randomization：

- `tau_delay` 随机化；
- `tau_v` 随机化；
- `a_max` 随机化；
- LiDAR noise / missing rays；
- sensor latency 近似；
- safety margin 随机化；
- obstacle density / spacing curriculum。

本阶段仍只考虑仿真。目标是提高对非理想动力学的鲁棒性。

---

## 8. 验证指标

任务基础指标和baseline的验证一致，即成功率、碰撞率和跟手性能等，可引入如下新指标，主要用于评价新的训练过程中是否满足了动态风险评估和最小干预的思想。

### 动态风险指标

| 指标 | 含义 |
|---|---|
| mean `pilot_risk_dyn_post` | human command 平均动态风险 |
| mean `assist_risk_dyn_post` | assisted command 平均动态风险 |
| mean `assist_risk_dyn_full` | assisted command full horizon 风险 |
| mean `delay_risk` | 延迟阶段风险暴露 |
| mean `risk_reduction_dyn` | 平均反事实风险降低 |
| risk-worsening rate | `rho_a_post > rho_u_post` 的比例 |
| near-miss exposure | 最小 clearance 低于阈值的比例 |

### 最小干预指标

定义 residual norm：

\[
\|\delta_t\| = \|a_t-u_t\|
\]

或使用 normalized modal residual：

\[
\|\delta^{\mathrm{mode}}_t\|
\]

建议记录：

| 指标 | 定义 | 目标 |
|---|---|---|
| intervention rate | `Pr(||delta|| > epsilon)` | 不能过高 |
| unnecessary intervention rate | `Pr(||delta|| > epsilon | rho_u_post < rho_safe)` | 越低越好 |
| unsafe non-intervention rate | `Pr(||delta|| < epsilon | rho_u_post > rho_danger)` | 越低越好 |
| residual in safe states | `E[||delta|| | rho_u_post < rho_safe]` | 低 |
| residual in dangerous states | `E[||delta|| | rho_u_post > rho_danger]` | 适中或较高 |
| risk reduction per intervention | `E[Delta rho / (||delta||+eps)]` | 越高越好 |

这些指标直接验证：

> 该干预时干预，不该干预时不干预；干预必须带来风险降低。

### 推荐阈值

初始可使用：

\[
\rho_{\mathrm{safe}}=0.2
\]

\[
\rho_{\mathrm{danger}}=0.7
\]

\[
\epsilon_{\delta}=0.05
\]

其中 `epsilon_delta` 应根据 action normalization 或 physical action scale 调整。



## 可视化建议

建议实现以下图表。

### 单 episode 时间序列

横轴为时间，纵轴分别绘制：

- `pilot_risk_dyn_post`；
- `assist_risk_dyn_post`；
- `assist_risk_dyn_full`；
- `delay_risk`；
- residual norm；
- collision / near-miss marker；
- human command 和 assisted command。

目标：观察策略是否在 high pilot risk 前后产生合理 residual。

### residual vs pilot risk scatter

横轴：`pilot_risk_dyn_post`。

纵轴：residual norm。

期望：

- baseline 可能分布混乱；
- 新方法应表现出：risk 越高，允许 residual 越大；risk 低时 residual 接近 0。

### risk reduction vs residual scatter

横轴：residual norm。

纵轴：`risk_reduction_dyn`。

期望：

- 高 residual 应对应正 risk reduction；
- 不应大量出现 high residual + low/negative risk reduction。

### 预测 trajectory tube overlay

在局部 top-down 图中显示：

- LiDAR 点；
- human command 对应预测轨迹 tube；
- assisted command 对应预测轨迹 tube；
- tube clearance；
- 风险最高的时间步。

这个图非常适合放进论文方法或实验分析。

---

## 可能失败模式与调参方向

### 策略学会 hover / stop

原因：

- risk reduction reward 让 zero command 看起来安全；
- absolute risk penalty 太强；
- following reward 在 high risk 下被压得太低；
- task progress 不够强。

解决：

- 提高 `g_min`；
- 降低 `w_abs` 或 `w_delay`；
- 增加 progress / traversal reward；
- 对长期低速增加轻微惩罚；
- 确保 zero command 在有惯性时仍通过 dynamic risk 表现出真实风险，而不是被误判为完全安全。

### 策略过度干预

原因：

- `g_danger` 太低；
- `w_delta_rho` 太高；
- risk function 过于敏感；
- old safety penalty 与 new safety penalty 叠加过强。

解决：

- 提高 `g_danger`；
- 降低 `w_delta_rho`；
- 提高 `risk_temperature`；
- 降低 safety margin；
- 尽快移除重复的旧 safety penalty。

### 策略仍然碰撞

原因：

- horizon 太短；
- delay / tau_v / a_max 估计过于理想；
- safety margin 太小；
- `w_abs` 或 `w_worse` 太低；
- LiDAR points 与 command frame 不一致。

解决：

- 增大 `T_horizon`；
- 增大 `tau_delay`；
- 增大 `tau_v` 或降低 `a_max`；
- 增大 `margin_static` / `margin_speed`；
- 提高 `w_abs` 和 `w_worse`；
- 首先检查 frame 对齐。

### reward 噪声大，训练不稳定

原因：

- LiDAR 点云稀疏或 noisy；
- max risk 过于尖锐；
- risk sigmoid temperature 太小；
- clearance 在阈值附近频繁跳变。

解决：

- 使用 percentile / softmax 聚合替代 hard max；
- 提高 `risk_temperature`；
- 对 LiDAR distances 做轻度平滑；
- 使用连续的 tube clearance penalty；
- 对 risk values 做 clip 和 logging 检查。

### risk reduction 为正但动作仍不合理

原因：

- 只优化相对风险，忽略绝对风险和任务；
- risk function 没有考虑 vertical / yaw / body radius；
- assisted command 绕得太远或太急。

解决：

- 保留 `abs_risk`；
- 保留 smoothness；
- 加强 following / progress；
- 检查 tube radius 和 command dimensions。

---

## 推荐配置起点

### 动态风险参数

| 参数 | 初始值建议 | 说明 |
|---|---:|---|
| `dt_risk` | 0.05 s | 风险 rollout 步长 |
| `T_horizon` | 1.5 s | 短时预测时域 |
| `tau_delay` | 0.15–0.25 s | 命令生效延迟 |
| `tau_v` | 0.3–0.5 s | 速度响应时间常数 |
| `a_max` | 1.0–2.0 m/s² | 最大加速度近似 |
| `v_max` | 与 action limit 一致 | 速度上限 |
| `r_uav` | 按模型尺寸 | 无人机等效半径 |
| `margin_static` | 0.10–0.20 m | 静态安全裕度 |
| `margin_speed` | 0.05–0.12 s | 速度相关裕度系数 |
| `margin_time` | 0.02–0.05 m/s | 时间相关裕度系数 |
| `risk_temperature` | 0.08–0.20 m | clearance 到 risk 的平滑程度 |

### Reward 参数

| 参数 | 初始值建议 |
|---|---:|
| `w_delta_rho` | 0.5 |
| `w_worse` | 1.5 |
| `w_abs` | 0.4 |
| `w_delay` | 0.3 |
| `alpha_f` | 0.5 |
| `g_min` | 0.35 |

### 13.3 Residual gate 参数

| 参数 | 初始值建议 |
|---|---:|
| `g_safe` | 1.0 |
| `g_danger` | 0.05 |
| `p` | 1.0 |
| `reg_coeff` | 沿用 baseline 或略降 |

---

##实现 checklist

### 风险估计模块

- [ ] 明确风险计算 frame；
- [ ] 读取当前实际速度 `v_t`；
- [ ] 读取 human command `u_t`；
- [ ] 读取 assisted command `a_t`；
- [ ] 维护 last applied command `q_hold`；
- [ ] 将 LiDAR distances + ray directions 转为局部障碍点；
- [ ] 实现延迟 + 一阶速度响应 rollout；
- [ ] 实现 trajectory tube clearance；
- [ ] 输出 `rho_full`、`rho_post`、`rho_delay`、`min_clearance`；
- [ ] 对 `u_t` 和 `a_t` 都计算动态风险；
- [ ] logging only 阶段验证风险值合理。

### Reward 模块

- [ ] 实现 risk-gated following reward；
- [ ] 实现 counterfactual risk reduction；
- [ ] 实现 risk-worsening penalty；
- [ ] 实现 assisted absolute risk penalty；
- [ ] 实现 delay risk penalty；
- [ ] 保留 task / terminal / smoothness / progress 等非安全项；
- [ ] 逐步替换旧 safety penalty；
- [ ] 记录所有 reward component。

### PPO residual regularization

- [ ] 从 rollout 中读取 `pilot_risk_dyn_post`；
- [ ] 使用 deterministic Beta mode 计算 modal residual；
- [ ] 用 `rho_u_post` 计算 residual gate；
- [ ] 替换 fixed residual regularization；
- [ ] logging `reg_gate`、`modal_residual`、`reg_loss`；
- [ ] 保证风险值不参与 actor 反向传播。

### Validation

- [ ] 使用v2数据集(SharedRLControl/isaac-training/data/user_inputs/tunnel_perlin_bounded_v2.h5)和相同环境参数训练baseline和改进后的模型，对比指标。
- [ ] 完成可视化绘图并总结结论

---

## 最终预期结果

理想情况下，完整方法相对于 baseline 应表现为：

1. 碰撞率降低；
2. near-miss / dynamic risk exposure 降低；
3. high pilot risk 状态下 residual 增大；
4. low pilot risk 状态下 residual 减小；
5. risk-worsening rate 降低；
6. risk reduction per intervention 提高；
7. task success 和 command following 不出现显著崩溃。

若只能实现部分目标，优先级如下：

1. 先证明动态风险能解释旧模型失败案例；
2. 再证明 counterfactual risk reward 能减少动态风险暴露；
3. 最后证明 risk-gated residual regularization 能减少不必要干预。


## 实现备忘录：新风险降低方案中的人类意图对齐

在实现新的“最小干预风险降低方法”时，**不要**将旧变量 `prev_human_action` 理解为一个任意的一步奖励延迟量。在当前环境的 step 执行顺序中，`_compute_state_and_obs()` 生成的人类动作实际上已经代表了**下一次策略决策**对应的人类意图。因此，当前名为 `prev_human_action` 的变量，更准确地表示为：在当前已执行辅助控制动作被下发时，策略所能够观测到的人类控制指令。

在新的实现中，建议从概念上将该变量重命名为：

[
u^{\mathrm{issue}}_t
]

或者在代码层面使用如下命名：

```text
human_action_issue
human_action_for_executed_action
intent_at_action_issue
```

所有新的共享控制相关项都应基于同一个“动作下发时刻（issue-time）”的人类意图：

[
r_{\mathrm{follow}}(u^{\mathrm{issue}}_t, a_t)
]

[
\rho^u_t = \rho_{\mathrm{dyn}}(x_t, z_t, u^{\mathrm{issue}}_t)
]

[
\rho^a_t = \rho_{\mathrm{dyn}}(x_t, z_t, a_t)
]

[
\Delta\rho_t = \rho^u_t - \rho^a_t
]

同时，基于风险的残差正则化门控（risk-conditioned residual regularization gate）也应建立在以下量之上：

[
\rho^u_t = \rho_{\mathrm{dyn}}(x_t, z_t, u^{\mathrm{issue}}_t).
]

这样可以确保以下各项都对应于同一次状态转移（transition）：

* 跟随奖励（following reward）
* 反事实驾驶员风险（counterfactual pilot risk）
* 辅助控制动作风险（assisted-command risk）
* 风险降低量（risk reduction）
* 残差正则化（residual regularization）

实现过程中应避免将当前新生成的人类动作 (u_{t+1}) 与已经执行完成的辅助动作 (a_t) 混合使用。

在第一版实现中，建议保持当前基线方法的语义不变，但明确重命名并记录相关变量。至少应记录以下内容：

```text
human_action_issue
human_action_next
agent_action_executed
pilot_risk_dyn_issue
assist_risk_dyn_issue
risk_reduction_dyn
```

如果动力学风险函数使用了状态（state）、姿态（orientation）或 LiDAR 数据，最稳妥的实现方式是同时缓存动作下发时刻对应的数据：

```text
state_issue
orientation_issue
lidar_issue
```

并基于同一个 issue-time 状态分别计算驾驶员指令风险和辅助控制指令风险：

[
\rho_{\mathrm{dyn}}(x^{\mathrm{issue}}_t, z^{\mathrm{issue}}_t, u^{\mathrm{issue}}_t)
]

[
\rho_{\mathrm{dyn}}(x^{\mathrm{issue}}_t, z^{\mathrm{issue}}_t, a_t).
]

这样可以避免新的反事实风险降低奖励在计算时出现以下不一致情况：

* 使用旧的人类意图与新的观测进行比较；
* 使用新的人类意图与旧的已执行动作进行比较。

从而保证风险降低量的计算严格对应于同一个决策时刻和同一次状态转移。
