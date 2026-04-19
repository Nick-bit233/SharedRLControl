# TanhNormal vs Beta 分布：适用性分析报告

> 针对 Shared RL Control 隧道避障任务（Constrained Residual PPO）

---

## 1. 当前实现分析：TanhNormal

### 1.1 数学定义

当前 Actor 使用 **TanhNormal** 分布（又称 Squashed Gaussian）：

```
x ~ Normal(loc, scale)        # 在 pre-tanh 空间采样
a = tanh(x)                   # 映射到 (-1, 1)
action = a * action_limit     # 缩放到 [-2.0, +2.0] m/s
```

概率密度通过变换分布的 Jacobian 修正计算：

```
log π(a|s) = log N(x; loc, scale) - Σ log(1 - tanh²(x_i))
```

### 1.2 在本项目中的特殊性：残差架构

当前实现有一层特殊的 **ResidualActionModule**，流程为：

```
网络输出 → [_loc_delta, raw_scale]
                 ↓
scale = softplus(raw_scale) + 1e-4      (最小值 ≈ 0.694)
                 ↓
loc = _loc_delta * residual_scale + atanh(human_action / action_limit)
                 ↓
TanhNormal(loc, scale) → 采样/确定性输出
```

**关键设计**：`loc` 在 pre-tanh 空间中工作，通过 atanh 将 human_action 映射进来，使得当 `_loc_delta ≈ 0` 时，`tanh(loc) ≈ human_action`（恒等初始化）。

### 1.3 当前 TanhNormal 的问题

#### 问题 A：scale 下界过高

```python
scale = softplus(raw_scale) + 1e-4
```

`softplus(0) = ln(2) ≈ 0.693`。初始化时网络输出接近 0，因此 **scale 起步就是 0.693**。

在 pre-tanh 空间中，scale = 0.693 意味着：
- 95% 的采样点落在 `loc ± 1.386` 范围内
- 经过 tanh 变换后，如果 loc = 0 (对应 human_action = 0)：
  - tanh(0 ± 1.386) = ±0.884
  - **对应 ±1.77 m/s 的随机偏差！**

对于隧道中障碍物间距 < 1m 的场景，**一个随机偏差就足以导致碰撞**。

#### 问题 B：tanh 变换的非线性 Jacobian

TanhNormal 的 `log_prob` 包含 Jacobian 修正项 `-log(1 - tanh²(x))`：

| pre-tanh 值 x | tanh(x) | Jacobian 修正 | 备注 |
|---------------|---------|-------------|------|
| 0.0 | 0.0 | 0.0 | 分布中心，无扭曲 |
| 1.0 | 0.762 | -0.69 | 轻微扭曲 |
| 2.0 | 0.964 | -2.63 | 严重扭曲 |
| 3.0 | 0.995 | -4.60 | 极端扭曲 |

当 human_action 接近 action_limit 时：
```
human_action_norm = 0.9 → atanh(0.9) = 1.472 → loc 已经偏离中心
```

这意味着 **loc 的取值范围在 pre-tanh 空间中是不均匀的**，对边界附近的动作，梯度计算会出现数值问题。

#### 问题 C：探索-利用矛盾

如 V2 分析报告所述：
- **好模型** (eval > 80%): 均值 loc 已接近最优 → 采样噪声全部有害 → 99% 碰撞
- **差模型** (eval < 30%): 均值远离最优 → 噪声偶尔有帮助

TanhNormal 的 scale 是可学习参数，理论上策略可以学会减小 scale。但实际训练中，PPO 的熵正则化 (`entropy_coeff = 1e-3`) 对抗 scale 收缩，加上用 Monte Carlo 估计的熵（`H ≈ -E[log π]`）误差较大，导致 **scale 实际上难以在训练中显著减小**。

#### 问题 D：log_prob 数值不稳定

代码中有大量针对 TanhNormal 的数值保护：

```python
action_safe = action.clamp(-1.0 + 1e-5, 1.0 - 1e-5)     # 防止 atanh(±1) = ±inf
log_probs = log_probs.clamp(min=-20.0)                     # 防止 -inf
entropy_action = entropy_action.clamp(-1.0 + 1e-5, 1.0 - 1e-5)  # 同上
```

这些都是 TanhNormal 固有的边界问题。当动作接近 ±1 时，`1 - tanh²(x) → 0`，`log_prob → -inf`。

---

## 2. Beta 分布分析

### 2.1 数学定义

Beta 分布定义在 **(0, 1)** 上：

```
a_raw ~ Beta(α, β)                         # 采样值在 (0, 1)
a = a_raw * (max - min) + min              # 线性缩放到 [min, max]
action = a * action_limit                  # → [-2.0, +2.0] m/s
```

参数化方式（常用于 RL）：

```
α = softplus(net_output_α) + 1.0    # 保证 α > 1
β = softplus(net_output_β) + 1.0    # 保证 β > 1
```

当 α, β > 1 时，Beta 分布是**单峰的**（U 形或 J 形分布只在 α < 1 或 β < 1 时出现）。

### 2.2 Beta 分布的理论优势

#### 优势 1：天然有界，无需变换

Beta 的 support 是 (0, 1)，通过线性缩放到 [-1, 1]。
- **无 tanh 变换** → 无 Jacobian 修正 → log_prob 简单且数值稳定
- **无边界 clamp 问题** → 不需要 `action.clamp(-1+eps, 1-eps)` 这类 hack

```
log π(a|s) = log Beta(a_raw; α, β)    # 解析表达式，无额外修正项
```

#### 优势 2：可控的边界行为

| 参数 | 分布形状 | 边界概率密度 |
|------|---------|-------------|
| α=β=1 | 均匀分布 | 1.0 (边界) |
| α=β=2 | 对称钟形 | 0 (边界) |
| α=β=5 | 窄钟形 | ≈ 0 (边界) |
| α=β=10 | 很窄钟形 | ≈ 0 (边界) |

当 α, β > 1 时，**边界处的概率密度为 0**。这意味着策略不太可能输出极端动作，对避障任务是有利的。

#### 优势 3：精确可控的方差

```
mean = α / (α + β)
variance = αβ / ((α+β)²(α+β+1))
```

通过增大 α+β（concentration），可以**精确控制**方差的大小：
- α=β=2: variance = 0.05, std ≈ 0.224 → 宽探索
- α=β=5: variance = 0.023, std ≈ 0.150 → 适度探索
- α=β=20: variance = 0.006, std ≈ 0.077 → 窄探索

相比 TanhNormal 中 scale 的非线性效应（tanh 压缩），Beta 的方差控制更直觉、更稳定。

#### 优势 4：解析熵

```
H(Beta(α,β)) = ln B(α,β) - (α-1)ψ(α) - (β-1)ψ(β) + (α+β-2)ψ(α+β)
```

其中 B 是 Beta 函数，ψ 是 digamma 函数。PyTorch 的 `Beta.entropy()` 有解析实现，不需要当前代码中的 Monte Carlo 估计：

```python
# 当前 (TanhNormal): Monte Carlo 熵估计
entropy_action = action_dist.rsample()
entropy_log_prob = action_dist.log_prob(entropy_action)
entropy_est = -entropy_log_prob                # ← 高方差估计

# Beta: 解析熵
entropy = action_dist.entropy()                # ← 精确值，零方差
```

这会显著减少熵正则化的梯度噪声。

### 2.3 Beta 分布的挑战与局限

#### 挑战 1：与残差架构的兼容性

当前的 ResidualActionModule 在 **pre-tanh 空间**中做加法：

```python
new_loc = _loc_delta * residual_scale + atanh(human_action_norm)
```

这个设计**依赖 tanh/atanh 的数学性质**来实现残差连接。Beta 分布没有类似的"pre-transform 空间"。

**解决方案**: 改为在 **action 空间**（[-1, 1]）中做残差:

```python
# Beta 版本的残差连接
human_action_norm = human_action / action_limit            # [-1, 1]
human_action_01 = (human_action_norm + 1) / 2              # [0, 1] for Beta
mean_shift = _mean_delta * residual_scale + human_action_01  # 在 [0, 1] 空间加法
mean_shift = mean_shift.clamp(0.01, 0.99)                    # 安全范围

# 网络输出 concentration 参数
alpha = mean_shift * concentration
beta = (1 - mean_shift) * concentration
```

其中 `concentration = softplus(net_concentration) + 2.0` 控制分布宽度。

**注意**: 这种方案的残差连接是**线性的**（在动作空间中加法），而非当前的**非线性的**（在 atanh 空间中加法）。线性残差的优势是梯度更均匀，但劣势是需要额外确保 mean 不超出 (0, 1)。

#### 挑战 2：reg_loss 的定义变化

当前 reg_loss 基于 pre-tanh 空间的 delta：
```python
reg_loss = minibatch["_loc_delta"].pow(2).sum(dim=-1).mean()
```

对于 Beta 分布，需要改为在 action 空间衡量偏差：
```python
reg_loss = minibatch["_mean_delta"].pow(2).sum(dim=-1).mean()  # 在 [0,1] 空间
```

含义不变（惩罚偏离 human_action 的程度），但数值尺度不同。

#### 挑战 3：log_prob 的梯度特性

Beta 分布的 log_prob 关于参数 α, β 的梯度涉及 digamma 函数：

```
∂ log Beta(x; α, β) / ∂α = log(x) - ψ(α) + ψ(α+β)
```

当 x 接近 0 时，`log(x) → -inf`。这意味着如果策略输出的动作接近 action space 的最小值，梯度同样会有问题。不过实际中，α, β > 1 保证了采样不太可能到达边界，所以这在实践中不太严重。

#### 挑战 4：对称性与偏移

TanhNormal 是关于 loc 对称的。当 loc = atanh(human_action)，采样点在 human_action 两侧均匀分布。

Beta 分布在 α ≠ β 时是不对称的。这对残差策略实际上是**有利的**——策略可以学习"向一侧偏移更多"（如接近障碍物时只向安全方向偏移），这在 TanhNormal 中做不到（除非 loc 移动，但 scale 两侧对称不变）。

---

## 3. 对比分析表

| 特性 | TanhNormal (当前) | Beta 分布 |
|------|-------------------|-----------|
| **Action 范围** | (-1, 1) via tanh | (0, 1) via definition, 线性缩放到 (-1,1) |
| **参数** | loc (均值), scale (标准差) | α, β (shape 参数) |
| **采样机制** | Normal → tanh | 直接 Beta 采样 |
| **log_prob** | Normal log_prob - Jacobian 修正 | 解析 Beta log_prob |
| **数值稳定性** | ⚠️ 边界处需 clamp | ✅ 天然有界 |
| **熵计算** | Monte Carlo 估计 (高方差) | ✅ 解析公式 (零方差) |
| **初始探索量** | scale ≈ 0.69 → 很大 | 可通过 α+β 精确控制 |
| **残差兼容性** | ✅ atanh 空间天然匹配 | ⚠️ 需改为 action 空间残差 |
| **梯度质量** | ⚠️ tanh 饱和区梯度消失 | ⚠️ 边界处 log(x) → -inf |
| **方差可控性** | 间接 (通过 scale 在 pre-tanh 空间) | ✅ 直接 (通过 concentration) |
| **不对称能力** | ❌ 固有对称 | ✅ α ≠ β 时不对称 |
| **RL 文献支持** | 广泛使用 (SAC, TD3, PPO) | 有研究，但不够主流 |
| **实现复杂度** | 低 (TorchRL 内置) | 中等 (需自定义分布类) |

---

## 4. 针对当前任务的评估

### 4.1 当前任务的关键特征

1. **有界连续动作空间**: [-2, +2] m/s 速度指令，3 维
2. **安全关键**: 碰撞代价极高 → 需要低方差、精确的动作
3. **残差学习**: 策略纠正 human 动作 → 大部分时间 delta 应该很小
4. **课程学习**: 从简单到困难 → 后期需要更精确的控制
5. **密集障碍物**: 后期阶段 120-170 个障碍物 → 容错空间极小

### 4.2 TanhNormal 对当前任务的适合度: ⚠️ 中等偏低

**适合的方面**:
- 与 atanh 残差架构数学上一致
- PPO + TanhNormal 是经过验证的组合
- 确定性推理 (mode/mean) 效果好 (eval 75%+)

**不适合的方面**:
- 训练时探索噪声过大，后期阶段 99%+ 碰撞 → **学习效率极低**
- scale 难以收缩到足够小 (softplus 下界 + 熵正则化对抗)
- Monte Carlo 熵估计增加梯度噪声
- 边界处的数值不稳定需要多处 clamp hack
- 对称采样：在不对称避障场景中次优

### 4.3 Beta 分布的预期改善: ✅ 有望显著改善

**最大的预期改善**:

1. **可控探索量** (🔴 关键): 后期阶段可以设置 concentration = 20+，使方差降至 0.006，对应 action 空间 ±0.15 m/s 的探索范围 → 大幅减少无效碰撞

2. **数值稳定性** (🟡 重要): 消除 5+ 处 clamp hack，log_prob 计算更干净

3. **精确熵控制** (🟡 重要): 解析熵替代 Monte Carlo，减少梯度噪声

4. **不对称能力** (🟢 加分): 在靠近障碍物一侧可以有更大的"避让"倾向

**预期风险**:
1. 残差架构需要重新设计 (从 atanh 空间 → action 空间)
2. 需要验证 reg_loss 在新参数化下的有效性
3. 实现工作量中等 (约 100-150 行新代码)

---

## 5. 推荐方案

### 5.1 方案 A: 渐进式改进 (低风险，中等收益)

保持 TanhNormal，但解决核心问题：

1. **在 SplitLayer 中添加 max_scale 参数**:
   ```python
   scale = softplus(raw_scale) + 1e-4
   if self.max_scale is not None:
       scale = torch.clamp(scale, max=self.max_scale)
   ```
2. **后期阶段配置 max_scale = 0.3**
3. **降低 entropy_coeff 到 1e-4 或 0**

预期效果: 后期碰撞率从 99% 降至 ~80%，中等改善。

### 5.2 方案 B: 切换到 Beta 分布 (中等风险，高收益) ← 推荐

实现新的 `BetaResidualPPO` 或修改现有的 `ConstrainedResidualPPO`：

**核心改动**:
1. 新的 `BetaSplitLayer`: 输出 mean_delta 和 concentration
2. 新的 `BetaResidualActionModule`: 在 [0,1] 空间做残差
3. ProbabilisticActor 使用 `TransformedDistribution(Beta, AffineTransform)` 或自定义包装
4. 解析熵替代 MC 估计
5. reg_loss 改为 action 空间的 mean_delta²

**具体参数化建议**:
```python
# 网络输出: [mean_delta(3), raw_concentration(3)]
mean_delta = network_output[:, :3]                    # 残差偏移
concentration = softplus(network_output[:, 3:]) + 2.0  # 最小 concentration = 2.0

# 残差连接 (action 空间)
human_action_01 = (human_action / action_limit + 1) / 2   # 映射到 [0, 1]
mean = (mean_delta * residual_scale + human_action_01).clamp(0.01, 0.99)

# Beta 参数
alpha = mean * concentration
beta = (1 - mean) * concentration
```

**Concentration 课程**:
- Stage 1-2: concentration 最小值 = 2 (宽探索)
- Stage 3-4: concentration 最小值 = 5 (适度探索)  
- Stage 5: concentration 最小值 = 10 (窄探索)

### 5.3 方案 C: 双阶段混合 (最低风险，但复杂)

- Stage 1-2: 使用 TanhNormal (宽探索，帮助早期学习)
- Stage 3-5: 切换到 Beta (窄探索，保护已学到的策略)
- 切换时需要将 TanhNormal 的参数迁移到 Beta 参数化

**不推荐**: 复杂度过高，且 Beta 在早期也完全能工作。

---

## 6. 结论

| | TanhNormal (现状) | TanhNormal + max_scale (方案A) | Beta (方案B) |
|--|-------------------|-------------------------------|--------------|
| 实现工作量 | 0 | 小 (~20 行) | 中 (~150 行) |
| 后期阶段训练效率 | 极低 (99% 碰撞) | 中等 (预计 80% 碰撞) | 高 (预计 30-50% 碰撞) |
| 数值稳定性 | 差 (多处 clamp) | 差 (不变) | 好 |
| 残差兼容性 | 原生 | 原生 | 需重新设计 |
| 预期最终性能 | ~75% at 120 obs | ~80% at 120 obs | ~85%+ at 170 obs |
| 风险 | 已知瓶颈 | 低 | 中 (需验证) |

**最终推荐**: **方案 B (Beta 分布)**，原因是：

1. 它直接解决了训练分析中最严重的问题 — 后期探索噪声过大
2. concentration 参数提供了比 max_scale 更优雅的探索控制
3. 解析熵和更简洁的 log_prob 减少了梯度噪声
4. 不对称采样能力对避障任务有额外帮助
5. 虽然需要重写残差模块，但整体架构变化可控

如果时间紧迫，可以先实施方案 A 作为快速修复，同时并行开发方案 B。
