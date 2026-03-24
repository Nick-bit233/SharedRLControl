# 课程学习训练分析报告 V2 — 第二轮训练

> 训练时间: 2026-03-23 ~ 2026-03-24  
> 分析重点: 训练/评估指标差异的根因分析、阶段退化模式、改进建议

---

## 1. 整体结果概览

| 阶段 | 障碍物 | 最佳 Eval 成功率 | 最终 Eval 成功率 | 训练成功率(最终) | 早停? | 状态 |
|------|--------|-----------------|-----------------|-----------------|-------|------|
| S1 | 30  | **97.7%** | 90.2% | N/A | 否 | ✅ 学习成功 |
| S2 | 50  | **93.4%** | 82.4% | N/A | 否(2次降级警告) | ✅ 良好 |
| S3 | 80  | 93.4%(继承) | 53.1% | N/A | **是**(5/5) | ⚠️ 未学习,持续退化 |
| S4 | 120 | **76.6%** | 75.4% | **0.59%** | 否(2次警告后恢复) | ✅ 有效学习 |
| S5 | 170 | **79.7%** | 71.5% | N/A | 未完成(GPU超时) | ⚠️ 开始退化 |

### 与第一轮训练对比

| 指标 | 第一轮 | 第二轮 | 改善 |
|------|--------|--------|------|
| S1 最佳成功率 | 97.7% | 97.7% | 持平 |
| S4 最佳成功率 | 1.95% | **76.6%** | **🚀 显著提升** |
| S5 最佳成功率 | 75.8%(峰值) | 79.7% | 略有提升 |
| 最终可用模型质量 | ~32% at 200 obs | **~75% at 120 obs** | 模型更可靠 |

**结论**: 第二轮改进（best checkpoint 传递 + 早停 + 高度修复 + 难度平缓化）效果显著。S4 从灾难性崩溃(1.95%)恢复到了 76.6% 的有效避障率。

---

## 2. 训练/评估指标差异 — 核心发现

### 2.1 现象

在 Stage 4 中观察到的极端差异:

```
训练指标 (episode/stats_success): 0.59%  ← 看起来完全失败
评估指标 (eval/stats_success):   75.4%  ← 实际表现良好
```

这不是 bug，而是**架构设计的预期行为**。以下是三个叠加原因:

### 2.2 原因 #1: 随机探索 vs 确定性推理

| | 训练 (Rollout) | 评估 (Eval) |
|--|----------------|-------------|
| **动作采样** | `ExplorationType.RANDOM` — 从 TanhNormal(loc, scale) 中采样 | `ExplorationType.DETERMINISTIC` — 直接使用 loc (均值) |
| **噪声来源** | `scale = softplus(net_output) + 1e-4`，最小值 ≈ 0.693 | 无噪声 |
| **效果** | 动作 = 均值 + 大量随机噪声 → 频繁撞障碍物 | 动作 = 均值 → 精确避障 |

**关键代码路径**:
```python
# 训练: SyncDataCollector 使用默认 RANDOM 采样
# → TanhNormal.sample() → loc + scale * noise

# 评估: train.py evaluate() 函数
with set_exploration_type(ExplorationType.DETERMINISTIC):
    trajs = env.rollout(policy=policy, ...)
# → TanhNormal 直接输出 loc，无采样噪声
```

**影响**: 在密集障碍物环境中，任何微小的动作偏差都可能导致碰撞。scale ≈ 0.7 的噪声对于障碍物间距 <1m 的隧道来说是致命的。

### 2.3 原因 #2: 环境条件差异

| | 训练环境 | 评估环境 |
|--|---------|---------|
| **起始位置 Y 轴** | `random in [-sx, sx]` (随机) | `y = 0` (固定中心) |
| **用户模型** | 随机行为 (无 seed) | 固定 seed=42 (可复现) |
| **环境数量** | 256 个并行环境 | 256 个相同配置环境 |

评估从隧道中心开始、使用确定性用户模型，天然比训练条件更简单。

### 2.4 原因 #3: 统计捕获机制

`EpisodeStats` (omni_drones/utils/torchrl/env.py) 只在 episode **结束时**捕获统计:

```python
def add(self, tensordict):
    done = next_tensordict.get("done")
    if done.any():
        self._stats.extend(next_tensordict[done].cpu().unbind(0))
```

训练时 99.4% 的 episode 因碰撞结束 → `stats_success` 统计几乎全为 0。这个数字是**技术上正确但实际意义有限的**。

### 2.5 重要推论

> **`episode/stats_success` 不应作为训练进度指标。** 
> 
> 唯一可靠的进度指标是 `eval/stats_success`，因为它反映的是**学到的策略 (loc/mean)** 的质量，而非**随机探索**的结果。

---

## 3. 各阶段详细分析

### Stage 1 (30 障碍物) — ✅ 成功

```
Eval: 34.4% → 95.7% → 97.7%↑ → 96.1% → 90.2%
Best: 97.7% at step 3000
```

- 快速学会基础避障，best checkpoint 机制正确保存了最佳模型
- 后期轻微过拟合 (90.2% < 97.7%)
- 碰撞率 9.4%，高度越界 0%

### Stage 2 (50 障碍物) — ✅ 良好

```
Eval: 86.3% → 84.8% → 85.2% → 93.4%↑ → 91.0% → 89.8% → 91.0% → 78.5%↓ → 82.4%
Best: 93.4% at step 3000
Early stopping warnings: 2/5 (未触发)
```

- 继承 S1 best (97.7%)，在 50 障碍物环境下初始为 86.3%
- 训练后提升到 93.4%，证明课程学习有效
- 后期波动较大，出现 2 次降级警告

### Stage 3 (80 障碍物) — ⚠️ 训练有害

```
Eval: 93.4% → 87.9%↓ → 80.1%↓ → 82.8% → 65.2%↓ → 69.9% → 53.1%↓
Best: 93.4% at step 0 (继承的 S2 模型，从未被超越!)
Early stopping: 5/5 触发, 回退到 best (93.4% 的 S2 模型)
```

**这是最关键的发现**: Stage 3 的训练**从未改善**模型。从第一次评估开始就持续退化。这意味着:

1. 继承的 S2 模型在 80 障碍物场景下已有 93.4% 的确定性成功率
2. 在此基础上继续训练反而损害了模型质量
3. 早停机制正确触发，保护了 S2 的 best 模型

**根因**: 以随机采样收集的训练数据中，99%+ 是碰撞轨迹。当模型已经很好时，这些"如何碰撞"的数据提供了错误的梯度方向，将参数推离了好的配置。

### Stage 4 (120 障碍物) — ✅ 有效学习

```
Eval: 4.3% → 13.7% → 33.6% → 74.6%↑ → 67.2% → 59.8% → 63.7% → 66.0% → 68.4% → 76.6%↑ → 75.0% → 71.1% → 75.4%
Best: 76.6% at step 9000
Training success: 0.59% (但这不重要!)
```

**Stage 4 是本次训练最成功的阶段**:

- 继承的 S2/S3 模型在 120 障碍物下只有 4.3% 成功率
- 通过 12000 步训练，eval 成功率提升到 76.6%
- 这证明: **即使训练期间 99.4% 的 episode 都碰撞了，策略的均值 (loc) 仍在正确的方向上改善**
- 学习率 3e-5 (低于 S3 的 5e-5) 可能是关键: 更新更小，不容易破坏已学到的知识

### Stage 5 (170 障碍物) — ⚠️ 开始退化 (未完成)

```
Eval: 79.7% → 72.7%↓ → 71.5%↓ (GPU 超时，仅 3 次评估)
Best: 79.7% at step 0 (继承的 S4 模型)
```

- 趋势与 Stage 3 相同: 继承了好模型，训练使其退化
- 如果继续训练，预期会触发早停
- above_bound 率 9.0% 表明高度逃逸仍是一个问题

---

## 4. 发现的代码逻辑问题

### 4.1 🐛 reg_coeff 课程使用了错误的指标

**文件**: `train.py` 第 466 行

```python
success_rate = info.get("episode/stats_success", None)  # ← 使用训练指标!
```

在 Stage 5 中，`episode/stats_success` 始终接近 0%，因此:
- EMA 永远不会达到 `promotion_threshold: 0.95`
- `reg_coeff` 永远停留在 `initial_reg_coeff: 0.01`
- **整个 reg_coeff 课程在实际训练中是死代码**

**修复**: 应使用 `eval/stats_success` 来驱动课程调整。

### 4.2 ⚠️ 高度惩罚在后期阶段过强

Stage 3 eval 数据: `penalty_height` = 1.79, reward 公式中的贡献: `-15 * 1.79 = -26.85`（终止步的惩罚）

- 这导致 return 深度负值 (-2993)，即使成功率还有 53%
- 高度惩罚的梯度信号可能 **overwhelm** 避障奖励信号
- 策略可能学会"先降低高度"而不是"避开障碍物"

### 4.3 ⚠️ "好模型 + 继续训练 = 退化" 模式

这是一个系统性问题，在 S3 和 S5 中重复出现:

| 条件 | S3 | S4 | S5 |
|------|----|----|-----|
| 初始 eval 成功率 | 93.4% (高) | 4.3% (低) | 79.7% (高) |
| 训练后趋势 | 退化 | 改善 | 退化 |
| 结论 | 有害 | 有益 | 有害 |

**规律**: 当模型初始 eval 成功率 > ~50% 时，在更难环境中继续训练会退化。当初始很低时，训练才有效。

这本质上是 **exploration noise 与 policy quality 的矛盾**:
- 好的策略 → 均值已接近最优 → 噪声只会偏离最优 → 99% 碰撞 → 梯度推动离开最优
- 差的策略 → 均值远离最优 → 噪声反而能探索到好方向 → 梯度有用

---

## 5. 改进建议

### 5.1 短期修复 (下一次训练前)

#### A. 修复 reg_coeff 课程指标
将 `train.py` 中 reg_scheduler 的输入从 `episode/stats_success` 改为 `eval/stats_success`:

```python
# 在 eval 之后更新 reg_scheduler
if reg_scheduler is not None and eval_info is not None:
    eval_success = eval_info.get("eval/stats_success", None)
    if eval_success is not None:
        new_reg = reg_scheduler.update(eval_success)
        ...
```

#### B. 后期阶段降低探索噪声
在 `SplitLayer` 中添加 scale 上限控制:

```python
class SplitLayer(nn.Module):
    def __init__(self, action_dim, max_scale=None):
        self.max_scale = max_scale
    
    def forward(self, x):
        loc, scale = x.split(self.action_dim, dim=-1)
        scale = F.softplus(scale) + 1e-4
        if self.max_scale is not None:
            scale = torch.clamp(scale, max=self.max_scale)
        return loc, scale
```

建议后期阶段 `max_scale` = 0.3 (vs 默认 ~0.7+)，减少探索噪声对好模型的破坏。

#### C. 降低高度惩罚权重
将 `-15.0 * penalty_height` 降低到 `-5.0 * penalty_height`，避免高度信号淹没避障信号。

#### D. 跳过无效的训练阶段
考虑在 Stage 3 type 的情况（初始成功率已经很高）中自动跳过训练，直接传递模型:

```python
# 在 run_curriculum.py 中
# 如果首次 eval 成功率 > threshold，跳过训练
if initial_eval_success > 0.85:
    print(f"Model already strong ({initial_eval_success:.1%}), skipping stage")
```

### 5.2 中期改进

#### A. 动态探索噪声调度
根据 eval 成功率自动调整 `max_scale`:
- eval_success < 30%: max_scale = 0.7 (多探索)
- eval_success 30-70%: max_scale = 0.5 (平衡)
- eval_success > 70%: max_scale = 0.3 (少探索)

#### B. 训练期间添加确定性评估日志
在 EpisodeStats 之外，周期性地在训练 rollout 中添加确定性策略的成功率日志。这能让 wandb 图表中的训练曲线更有意义:

```python
# 每 N 个 batch 做一次快速确定性评估 (不用完整 eval)
if i % 100 == 0:
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        quick_trajs = env.rollout(max_steps=200, policy=policy, ...)
```

#### C. 熵系数衰减
在后期阶段降低 entropy bonus (`entropy_coeff: 1e-3 → 1e-4`)，鼓励策略收敛而非继续探索。

#### D. 重新思考课程结构
当前 5 阶段结构暴露的问题:
- S1 → S2: ✅ 有效
- S2 → S3: ❌ 无效 (模型太好了，训练有害)
- S3 → S4: ✅ 有效 (模型变差了，训练改善)
- S4 → S5: ❌ 开始退化

可能更好的策略:
1. 减少阶段数 (3 阶段: 30→80→170)，跳过中间的"退化-恢复"循环
2. 或者: 当检测到初始成功率 > 80% 时，切换到更低噪声的 fine-tuning 模式

---

## 6. 总结

### 积极方面
- Best checkpoint + 早停机制运行正常
- S4 证明策略确实能学会在密集障碍物中导航 (76.6%)
- 高度逃逸问题得到部分缓解 (above_bound 大幅降低)

### 需要解决的问题 (按优先级)
1. **🔴 关键**: 探索噪声在后期阶段过大，破坏已学好的模型
2. **🔴 关键**: reg_coeff 课程使用了无效的训练指标
3. **🟡 重要**: 高度惩罚权重过大，干扰避障学习
4. **🟡 重要**: 训练指标 (episode/stats_success) 不反映真实进度
5. **🟢 优化**: 课程结构可简化，避免无效训练阶段
