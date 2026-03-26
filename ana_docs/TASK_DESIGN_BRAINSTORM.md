# 任务设计反思与重构方案 — 头脑风暴文档
<!-- filepath: /home/haoming/wht/IsaacLab_drones_5.1/SharedRLControl/ana_docs/TASK_DESIGN_BRAINSTORM.md -->

**日期**: 2026-03-26  
**背景**: 三轮训练迭代后，发现当前任务设计存在根本性缺陷，需要重新思考实验框架

---

## 1. 当前设计的系统性问题诊断

### 1.1 核心矛盾

当前项目的目标是训练一个**通用安全盾牌（Safety Shield）**：
> 在安全时忠实跟随人类输入，在危险时输出辅助修正信号帮助无人机避障

但当前的两个实验都无法实现这个目标：

| 问题维度 | Exp03（开放空间） | Exp04（隧道） |
|----------|-------------------|---------------|
| 人类输入分布 | `[max, noise, 0]`（仅前向+y噪声） | 同左 |
| 方向覆盖 | ❌ 从不产生后退/纯侧向/对角/悬停 | ❌ 同左 |
| 速度覆盖 | ❌ x方向始终为max_speed | ❌ 同左 |
| Z轴覆盖 | ❌ vz恒为0 | ❌ 同左 |
| 任务目标 | 无明确目标（无task_reward） | 到达x>10（方向性目标） |
| 泛化性 | ❌ 无法处理非前向输入 | ❌ 同左+隧道结构特化 |

**根本原因**：训练数据的人类输入分布极度狭窄，模型只见过"快速前进+小幅横向偏移"这一种指令模式。

### 1.2 奖励函数的结构性缺陷

```
当前奖励 = 0 (task_reward禁用)
          + 0.5/step (存活奖励)
          + 5.0 × safety_penalty
          - 15.0 × height_penalty
          + terminal: ±10 (碰撞/成功)
```

**问题**：
1. **没有跟随激励**：`enable_task_reward=False`，策略没有任何动机去跟随人类输入
2. **存活奖励与成功奖励失衡**：慢飞1000步(≈500) >> 快速通关650步(≈335)
3. **"成功"定义是方向性的**：`pos[0] > 10.0`只对前向飞行有意义
4. **安全惩罚是全局的，但跟随奖励是缺失的**：策略只学到了"远离障碍物"，没有学到"在安全的前提下跟随人类"

### 1.3 residual正则化为何不够

残差正则化（`reg_loss = ||δ||²`）的确会迫使网络输出接近零（即接近人类输入），但这只是一个**软约束在loss中**，不是奖励信号。在RL优化中：
- PPO的Actor loss推动策略追求高回报
- reg_loss推动策略输出接近人类
- **当这两者冲突时**（人类命令方向有障碍物），策略必须在"跟随人类获得低reg_loss"和"偏离人类获得高safety reward"之间选择
- 没有task_reward时，偏离几乎没有代价（只有reg_loss），但安全惩罚可以通过"不移动"来规避

---

## 2. 重新定义问题

### 2.1 科学问题

**"如何训练一个基于传感器的反应式安全滤波器，使其：(a) 对任意人类速度指令保持高保真跟随；(b) 仅在碰撞风险时施加最小必要修正；(c) 泛化到训练时未见过的障碍物密度和指令模式？"**

这是一个**有约束的多目标优化问题**，可以形式化为：

```
maximize  J_track(π) = E[Σ r_tracking(a_out, a_human)]     // 最大化跟随保真度
subject to J_collision(π) = E[Σ c_collision] ≤ ε            // 碰撞约束
minimize  J_intervention(π) = E[Σ ||residual||²]             // 最小化干预幅度
```

### 2.2 关键范式转变

| 维度 | 当前思路 | 应有思路 |
|------|---------|---------|
| 任务类型 | 导航任务（到达终点） | **持续跟随任务（时刻跟随人类指令）** |
| 成功度量 | 是否到达终点 | **跟随误差 + 碰撞率 + 干预频率** |
| 人类输入 | 固定前向+噪声 | **全方向、全速度、多模态** |
| 环境 | 固定隧道 | **随机障碍物场（密度可变）** |
| 奖励设计 | 存活 + 安全 | **跟随保真度 + 安全约束** |
| 评估 | 单一成功率 | **Pareto前沿（跟随 vs 安全 vs 干预）** |

---

## 3. 三个候选方案

### 方案A：危险感知跟随奖励（Danger-Aware Tracking Reward）⭐ 推荐

**核心思想**：奖励函数根据当前危险程度动态调节跟随强度

```python
# 危险水平：从LiDAR最近障碍物距离计算
danger_level = exp(-min_lidar_dist / safe_dist)  # ∈ [0, 1]

# 跟随奖励：安全时必须跟随，危险时允许偏离
r_tracking = -||a_out - a_human||² * (1.0 - 0.8 * danger_level)
#             ↑ 总是存在                  ↑ 危险时系数降至0.2

# 安全奖励：保持现有barrier function
r_safety = -exp(-dist / scale) * [分方向加权]

# 干预稀疏性：鼓励最小干预
r_sparsity = -||residual||²

# 总奖励
reward = r_tracking + α * r_safety + β * r_sparsity
```

**为什么这个设计有效**：
- **安全时**（`danger_level ≈ 0`）：`r_tracking ≈ -||deviation||²`，策略必须精确跟随
- **危险时**（`danger_level ≈ 1`）：`r_tracking` 系数降至0.2，允许偏离以避障
- **始终有跟随压力**：即使危险时，仍有20%的跟随梯度（避免完全无视人类输入）
- **自然学会"何时干预"**：不需要显式门控，奖励结构自然教会策略在安全时跟随、危险时修正

**优点**：
- 实现简单，修改现有奖励函数即可
- 物理直觉清晰：安全时听人话，危险时自主修正
- 无需额外架构变化
- 可调超参少：α, β, safe_dist

**缺点**：
- 需要仔细调节α, β的平衡
- danger_level的定义可能不够精确（LiDAR最小距离 ≠ 真实碰撞风险）

---

### 方案B：约束MDP + 拉格朗日松弛（CMDP/Lagrangian）

**核心思想**：将安全作为硬约束而非奖励项

```python
# 主目标：最大化跟随保真度
J_track = E[Σ r_tracking]   # r_tracking = -||a_out - a_human||²

# 安全约束：碰撞率 < ε
J_collision = E[Σ 1_{collision}] ≤ ε

# 拉格朗日形式
L(π, λ) = J_track(π) - λ * (J_collision(π) - ε)

# 双层优化
π* = argmax_π L(π, λ)    # 策略优化
λ* = argmin_λ≥0 L(π, λ)   # 乘子更新
```

**实现**：
- PPO更新策略π：`loss = -advantage + reg_loss`，其中advantage来自`r_tracking - λ * r_collision`
- 每N步更新λ：`λ ← max(0, λ + α_λ * (collision_rate - ε))`
- λ自动调节安全与跟随的平衡：碰撞多时λ增大（更重视安全），碰撞少时λ减小（更重视跟随）

**优点**：
- 数学上最优美，有理论保证（强对偶性）
- λ**自动学习**安全与跟随的平衡——无需手动调α, β
- 可以精确控制碰撞率上界ε
- 与CPO (Constrained Policy Optimization) 文献对接

**缺点**：
- 实现稍复杂（需要额外的λ优化循环）
- λ的学习率敏感，可能振荡
- 初始阶段碰撞率高时λ可能爆炸（需要clip）
- 单步碰撞信号太稀疏（可能需要连续化的近似碰撞成本）

---

### 方案C：门控残差架构（Gated Residual）

**核心思想**：显式学习"何时干预"和"如何干预"两个解耦模块

```python
class GatedResidualModule:
    def forward(self, lidar_features, state, human_action):
        # 干预门控：学习何时需要干预 [0, 1]
        gate = sigmoid(self.gate_net(lidar_features))  # scalar per action dim
        
        # 修正向量：学习如何修正
        correction = self.correction_net(lidar_features, state, human_action)
        
        # 最终输出
        return human_action + gate * correction
```

**训练损失**：
```python
loss = actor_loss                    # PPO目标
     + λ_track * tracking_loss       # 跟随保真度
     + λ_gate * gate_sparsity_loss   # L1(gate) 鼓励门保持关闭
     + λ_smooth * smoothness_loss    # 门控变化平滑
```

**优点**：
- **可解释性极强**：可以直接可视化gate值→理解策略在何时何处进行干预
- **自然的稀疏干预**：gate的L1正则化鼓励大部分时间gate=0（纯跟随）
- **架构先验**：门控结构直接编码了"大部分时间不需要干预"的假设
- 非常适合做论文的可视化分析

**缺点**：
- 引入额外网络参数（gate_net）
- 门控梯度可能消失（sigmoid饱和区）
- 需要从头设计训练流程（不能直接复用现有PPO）
- gate_net和correction_net可能协调困难

---

## 4. 人类输入多样化设计

**无论选择哪个方案，人类输入多样化都是必须的**。

### 4.1 多模态人类输入生成器

```python
class DiverseHumanModel:
    """生成覆盖全动作空间的人类输入"""
    
    MODES = [
        "perlin_3d",       # 三轴Perlin噪声（当前仅用了1D）
        "straight_random",  # 随机方向直线飞行
        "arc",             # 圆弧轨迹
        "waypoint",        # 随机航点序列
        "hover",           # 悬停（v≈0，测试零输入）
        "mixed_speed",     # 交替快慢速
    ]
    
    def generate(self, env_ids):
        mode = random.choice(self.MODES)  # 每episode随机选模式
        if mode == "perlin_3d":
            # 三通道独立Perlin噪声 → 全3D速度覆盖
            vx = perlin(seed_x) * max_speed
            vy = perlin(seed_y) * max_speed  
            vz = perlin(seed_z) * max_speed_z
        elif mode == "straight_random":
            # 随机方向 + 随机速度
            direction = random_unit_vector_3d()
            speed = uniform(0.3, 1.0) * max_speed
            v = direction * speed
        elif mode == "waypoint":
            # 生成3-5个随机航点，产生分段平滑轨迹
            waypoints = random_positions_in_obstacle_field()
            v = smooth_interpolate(waypoints)
        # ...
```

### 4.2 每种模式的训练价值

| 模式 | 覆盖范围 | 训练价值 |
|------|---------|---------|
| perlin_3d | 全3D空间，平滑变化 | 基础覆盖，测试跟踪能力 |
| straight_random | 任意固定方向 | 测试方向不变性 |
| arc | 曲线轨迹 | 测试动态跟踪+预测能力 |
| waypoint | 含转弯的路径 | 测试急转弯时的安全干预 |
| hover | 零速输入 | 测试策略在v=0时不乱动 |
| mixed_speed | 速度变化 | 测试对不同速度的适应 |

### 4.3 渐进式复杂度

课程学习可以在输入复杂度维度展开：
1. **S1**: 仅straight_random（简单直线） + 稀疏障碍物
2. **S2**: straight + perlin_3d + 中等障碍物
3. **S3**: 全部模式 + 密集障碍物

---

## 5. 环境重构建议

### 5.1 从隧道到通用障碍物场

隧道是一个人为设计的、高度特化的场景。建议改用**通用随机障碍物场**：

```python
class RandomObstacleField:
    """每个episode随机生成障碍物配置"""
    
    def __init__(self, cfg):
        self.density_range = [0.01, 0.08]  # 障碍物/m²
        self.size_range = [0.3, 1.5]       # 障碍物尺寸范围
        self.height_range = [2.0, 8.0]     # 障碍物高度范围
        self.map_size = [30, 30, 10]       # 场景大小
    
    def reset(self, env_ids):
        density = uniform(*self.density_range)
        num_obs = int(density * map_area)
        positions = random_positions(num_obs, self.map_size)
        sizes = uniform(*self.size_range, shape=(num_obs,))
        # ... 放置障碍物
```

**注意**：Isaac Sim中地形生成是预编译的（height field），不能每episode动态更换。但可以：
- 预生成多种密度的地形（10-20种）
- 每episode随机spawn到不同地形
- 或使用RigidObject动态放置障碍物（如果支持）

### 5.2 去掉"成功"的概念

在通用安全盾牌框架中，没有"到达终点"的概念。每个episode就是：
- **生成随机障碍物场**
- **生成随机人类输入序列**
- **跑N步（固定episode长度）**
- **评估指标**：跟随误差、碰撞数、干预量

episode结束条件仅有：碰撞（重置）、超时（自然结束）。

### 5.3 评估体系

```
Primary Metrics:
  - tracking_rmse: E[||a_out - a_human||]          # 跟随保真度
  - collision_rate: E[碰撞episode占比]               # 安全性
  - intervention_rate: E[||residual|| > threshold]  # 干预频率
  
Derived Metrics:
  - safety_efficiency: tracking_rmse at fixed collision_rate  # Pareto效率
  - direction_invariance: std(tracking_rmse | direction)       # 方向不变性
  - density_generalization: perf(train_density) vs perf(test_density)
```

---

## 6. 推荐实施路径

### 第一阶段：修复当前框架（1-2天）

目标：验证"跟随 + 安全"奖励设计的可行性

1. **启用task_reward**：`enable_task_reward: True`
2. **实现方案A**（Danger-Aware Tracking）：在现有env_tunnel.py上改
3. **扩展人类输入**：将tunnel user model改为至少支持3D Perlin噪声
4. **修复已知bug**：reg_coeff用eval成功率、调整存活奖励
5. **快速验证**：在现有隧道环境中跑2-3阶段课程

**验证标准**：
- S1-S2中tracking_rmse < 0.3且碰撞率 < 5%
- 从多个方向输入时，输出偏差不应比前向方向大2倍以上

### 第二阶段：环境泛化（3-5天）

目标：摆脱隧道依赖

1. **新建通用环境**（`env_general_safety.py`）：
   - 可配置密度的随机障碍物场
   - 无方向性"成功"指标
   - 多模态人类输入生成器
2. **设计新的课程**：按障碍物密度递增
3. **新增评估协议**：Pareto前沿分析

### 第三阶段：方法论对比（5-7天）

目标：产出可发表的消融实验

1. **Baseline对比**：
   - Pure human（无安全盾牌）
   - Potential Field（经典方法）
   - 全自主RL（无人类输入）
2. **方法对比**：
   - 方案A vs B vs C
   - Beta vs TanhNormal分布
   - 不同reg_coeff策略
3. **泛化测试**：训练密度 vs 测试密度的cross-evaluation matrix

---

## 7. 科研价值分析

### 7.1 潜在贡献点

1. **Danger-Aware Tracking Reward**：一种新的奖励设计，自然平衡跟随与安全
2. **多模态人类输入下的安全滤波器**：现有工作多假设人类输入为单一模式
3. **残差架构 + Beta分布**：有界动作空间下的安全残差策略
4. **系统性消融**：分布类型 × 奖励结构 × 约束方法的完整对比

### 7.2 与现有文献的关系

| 方向 | 代表工作 | 我们的区别 |
|------|---------|-----------|
| Shared Autonomy | Javdani (2015), Dragan (2013) | 他们预测意图→我们修正命令 |
| Safety Filters | Wabersich & Zeilinger (2021) | 基于模型→我们数据驱动 |
| Residual RL | Silver (2018) | 他们补充固定策略→我们修正人类实时输入 |
| Safe RL / CPO | Achiam (2017) | 通用safe RL→我们专注于人机协作 |
| Control Barrier Functions | Ames (2019) | 需要系统模型→我们端到端学习 |

**独特定位**：**数据驱动的、基于残差架构的、面向人类实时操控的安全滤波器**——结合了Residual RL + Safe RL + Shared Autonomy。

### 7.3 实验设计建议（面向论文）

一篇完整的论文需要回答：
1. **有效性**：安全盾牌是否降低了碰撞率？（vs pure human）
2. **保真度**：安全盾牌是否保持了人类意图？（tracking error多大）
3. **最小干预**：干预是否是必要的？（intervention rate的Pareto分析）
4. **泛化性**：是否对未见过的障碍物密度/人类输入模式有效？
5. **消融**：各组件（Beta分布、danger-aware reward、门控残差）的贡献？

---

## 8. 决策总结

### 短期（下一次迭代）推荐

**采用方案A（Danger-Aware Tracking Reward）+ 3D人类输入**：
- 在现有环境和架构上改动最小
- 直接解决"无跟随激励"和"输入分布窄"两个核心问题
- 可以在1-2天内实现并验证

### 核心改动清单

1. `tunnel.yaml`: `enable_task_reward: True`
2. `env_tunnel.py`: 实现danger-aware权重调制跟随奖励
3. `user_model_tunnely.py`: 扩展为3D Perlin噪声（vx, vy, vz独立）
4. `train.py`: 修复reg_coeff用eval成功率
5. 调整奖励权重：降低存活奖励(0.5→0.1)，提高成功奖励(10→100)
6. 新增评估指标：tracking_rmse, intervention_rate

### 中期（建立实验框架）推荐

新建experiment 05，基于通用障碍物场 + 多模态人类输入 + Pareto评估，为论文积累数据。
