# 训练诊断报告：Constrained Residual PPO 隧道避障任务

## 1. 训练曲线核心观察

| 指标 | 值/趋势 | 解读 |
|------|---------|------|
| collision_rate | 100% → 80%, 之后平台期并反弹至85% | 策略学到基础避障后停滞 |
| success | 0% → 25% (peak ~step800), 回落至15% | 学到的有效行为在后期被遗忘 |
| lambda | 0 → 10.0 (max clamp), ~step300 后永远饱和 | 约束机制失效，lambda=const |
| reg_loss | 0.2 → 1.3, 持续上升 | 残差幅度不断增大但未改善碰撞 |
| lambda_loss | 持续为负 (~-7) | collision_rate 远大于 target, 但 lambda 已无法继续增大 |
| diag_reward | ~-35 到 -45 | 80%碰撞率 × (-50) 主导 |

## 2. 根因分析（按严重程度排序）

### 🔴 根因 #1: 凸组合损失函数中 reg_loss 与 actor_loss 存在根本冲突

**这是最关键的问题。**

当前损失函数：
```python
loss_pi = (reg_loss + λ * actor_loss) / (1 + λ)
```

当 λ=10（已饱和）时权重分配：
- actor_loss 权重 = 10/11 ≈ **90.9%**
- reg_loss 权重 = 1/11 ≈ **9.1%**

**问题**：策略要避障就 **必须** 产生大残差（override 用户指令），但 reg_loss = ‖δ‖² 直接惩罚所有残差的幅度。即使 reg_loss 只占 9%，在 reg_loss=1.3 时其梯度依然显著，形成了一个恒定的"拉回零点"的力。

**后果**：
- 策略找到一个局部最优：输出中等大小的残差（reg_loss≈1.3），但这不足以避开障碍物
- 进一步增大残差 → reg_loss 梯度增大 → 被拉回 → 形成均衡，碰撞率卡在 ~80%
- Step 800 后 success 下降就是这个均衡点微调导致的震荡

**类比**：这就像你命令一个人"快跑去灭火"（actor_loss），但同时用绳子拴着他说"不许离开原位"（reg_loss）。即使绳子较松（9%），跑得越远绳子拉力越大，最终平衡在某个距离。

### 🔴 根因 #2: 用户模型全速推向障碍物，而安全奖励信号太弱且缺少方向信息

**用户模型行为**：
```python
target_vels = [vx=1.0*max_speed, vy=noise, vz=0]  # 永远全速前进
```
- vx = 2.0 m/s（全速），与障碍物无关
- LiDAR 探测距离 = 4m → 反应窗口仅 2 秒

**安全奖励 r_safety 的问题**：
1. **太近才触发**：`safe_zone=1.5m`，只在离障碍 1.5m 内产生惩罚。2m/s → 从触发到碰撞（0.3m）仅 0.6 秒 ≈ 30 步
2. **幅度太小**：r_safety 最大 ≈ -1.0/step vs crash_penalty = -50.0 (一次性)。如果安全区内存活 30 步，累积安全惩罚 = -30，而直接撞墙 = -50 + 0.1*29(是survival) ≈ -47。两者差距不大，避障的边际收益很低
3. **没有方向信息**：r_safety 告诉策略"你靠近障碍了"，但没有告诉"往哪偏移才对"。策略必须通过 trial-and-error 在 30 步的小窗口内学会正确方向

**后果**：策略缺乏足够的学习信号来建立"看到前方障碍 → 向左/右偏移"的映射。

### 🟡 根因 #3: Lambda 约束机制已经完全失效

Lambda 在 ~step300 饱和到 10.0 后就变成常数了。此后：
- 无论碰撞率是 80% 还是 90%，loss_pi 中的权重分配完全不变
- 约束机制退化为"固定权重 PPO + 固定权重 reg_loss"
- `lambda_lr=3e-4` 太快导致快速饱和，但即使调慢也只是延迟问题
- max clamp=10.0 太低，导致 reg_loss 永远占据 9%（不可忽略）

**关键理解**：碰撞率从 100% 降到 5% 是一个 **巨大的** 策略变化。Lagrange 方法在约束间隙很大时（constraint violation = 0.80 - 0.05 = 0.75）无法有效工作，因为 lambda 会直接爆到上界。Lagrange 方法更适合约束接近满足时的微调。

### 🟡 根因 #4: LiDAR 分辨率和覆盖不足

| 参数 | 值 | 问题 |
|------|-----|------|
| 水平分辨率 | 36束 (每10°) | 4m处两束间隔≈0.7m, 可能漏过0.4m宽的障碍 |
| 垂直分辨率 | 4束 (-10°~+20°) | 几乎无法感知正下方的障碍 |
| 探测距离 | 4m | 2m/s下仅2s反应时间 |
| 碰撞判定 | ray distance < 0.3m | 基于LiDAR射线代理，不是物理碰撞 |

LiDAR射线是稀疏的。障碍物宽度范围 `(0.4, 1.1)m`，窄障碍在 4m 处可能完全落在两根射线之间。策略在训练中可能遇到"LiDAR看不到、但0.3m阈值内检测到碰撞"的情况，产生 **不可预测的终止信号**，损害 critic 的学习。

### 🟢 根因 #5: 学习动态问题

1. **短 episode 主导训练**：80% 碰撞意味着大部分 episode ≈30-50 步就结束。GAE 的 bootstrap 依赖 critic 准确预测未来收益，但 crash 的位置高度随机（取决于障碍物布局和用户模型噪声），使 critic 难以学到有用的 value function
2. **crash penalty (-50) 方差极大**：reward 标准差很高，回报分布是双峰的（正常步 ≈ 0.1 vs crash 步 ≈ -50），这对 value normalization 和 advantage estimation 都不利
3. **10 epoch × 16 minibatch = 160 次更新/batch**：策略变化快，但 on-policy 数据是按旧策略收集的。在训练初期策略剧烈变化时，后几个 epoch 的 ratio 可能严重偏离 1.0，clip 频繁触发导致学习效率低

## 3. 解决方案

### 方案 A：修正损失函数（推荐，最关键的改动）

**核心思想**：将 reg_loss 从凸组合中分离，使用固定小系数，让 lambda 只控制 actor_loss 的有效学习率。

```python
# 之前（有冲突）：
loss_pi = (reg_loss + λ * actor_loss) / (1 + λ)

# 之后（解耦）：
loss_pi = actor_loss + α_reg * reg_loss   # α_reg 是固定常数, e.g. 0.01
```

这样 reg_loss 始终是轻微的正则化（防止残差爆炸），不会阻碍策略学习避障。

**Lambda 的新角色**：用来调节学习率或 clip 范围，而不是混在损失函数里。或者，完全移除 lambda 机制（见方案 B）。

### 方案 B：移除 Lagrange 约束，改用纯奖励驱动

**理由**：当约束间隙很大时（collision_rate >> target），Lagrange 方法不如直接在 reward 中编码奖惩来得有效。

```python
# 固定权重损失函数
loss = actor_loss + 0.01 * reg_loss + critic_loss + entropy_loss
```

安全激励完全通过 reward 传导：
- 增大 r_safety 的幅度和范围
- 适当降低 crash_penalty（从 -50 到 -10~-20），减少方差
- 增大 survival reward（从 0.1 到 0.5-1.0），让"活着"的信号更强

### 方案 C：改进奖励函数

1. **扩大安全区并增大惩罚力度**：
   ```python
   safe_zone = 3.0  # 从 1.5 扩大到 3.0（覆盖 75% 的 LiDAR 范围）
   r_safety_dist_scale = 1.0  # 从 0.5 增大，让远处也有梯度
   r_safety_weight = 5.0  # 从 1.0 增大到 5.0
   ```

2. **降低 crash penalty, 增大 survival reward**：
   ```python
   crash_penalty = -10.0  # 从 -50 降低，减小回报方差
   reward_survival = 0.5  # 从 0.1 增大，让"活着"更有价值
   ```

3. **添加方向性奖励（关键）**：
   ```python
   # 奖励远离障碍物的运动（基于 potential 的 shaping）
   delta_min_dist = min_dist_to_obs - prev_min_dist_to_obs
   reward_retreat = 0.5 * delta_min_dist.clamp(min=0.0)  # 远离障碍 → 正奖励
   ```

### 方案 D：降低任务难度 / 课程学习

1. **降低用户模型速度**：
   ```python
   # user_model_tunnely.py 中:
   target_vels = [vx=0.3*max_speed, vy=noise*0.5*max_speed, vz=0]
   ```
   从 2 m/s 降到 0.6 m/s，反应时间从 2s 变为 6.7s

2. **从空隧道开始**：
   ```yaml
   num_obstacles: 0  # 先学习基础飞行
   ```
   然后逐步增加：0 → 10 → 30 → 50 → 100 → 150

3. **增大 LiDAR 探测距离**：从 4m 增到 8-10m

## 4. 推荐实施优先级

| 优先级 | 改动 | 预期效果 | 风险 |
|--------|------|---------|------|
| **P0** | 修正损失函数：reg_loss 用固定小系数 | 解除 actor 学习的阻碍 | 低。如果 α_reg 过小，残差可能过大，但可通过 grad clip 控制 |
| **P0** | 降低用户模型前进速度至 0.5-1.0 m/s | 给策略更多反应时间 | 低 |
| **P1** | 扩大安全区 safe_zone → 3.0m | 更早的安全梯度信号 | 低 |
| **P1** | 降低 crash_penalty → -10, 增大 survival → 0.5 | 减少回报方差，增大活着的激励 | 需要调整约束阈值 |
| **P2** | 添加方向性奖励 (potential-based) | 提供"往哪避"的信息 | 中。需要记录 prev_min_dist |
| **P2** | 增大 LiDAR range → 6-8m | 更长的 look-ahead | 可能影响仿真性能 |
| **P3** | 减少 training_epoch_num → 5 | 减少 off-policy 退化 | 低 |
| **P3** | 课程学习：从 0 障碍开始 | 先学飞行再学避障 | 低 |

## 5. 损失函数改动的具体实现说明

### 当前结构
```
                    ┌─────────────┐
                    │  reg_loss   │ ←── ‖δ‖²
                    └──────┬──────┘
                           │ weight = 1/(1+λ)
         loss_pi = ────────┼────────
                           │ weight = λ/(1+λ)
                    ┌──────┴──────┐
                    │ actor_loss  │ ←── PPO clipped surrogate
                    └─────────────┘
问题: 当 δ 增大用于避障时, reg_loss 梯度反向拉回
```

### 建议新结构
```
loss = actor_loss + α_reg * reg_loss + critic_loss + entropy_loss

其中 α_reg = 0.01 (固定超参数)
```

不再需要 lambda 参数和 lambda 优化器。安全约束完全通过 reward 传导。

如果要保留 lambda:
```
loss = λ * actor_loss + α_reg * reg_loss + critic_loss + entropy_loss
```
Lambda 只调节 actor_loss 的 effective scale（不影响 reg_loss），碰撞高 → λ 大 → actor 更用力学。

## 6. 对 `diag_reward` vs `batch_mean_reward` 差异的最终解释

| 指标 | 统计范围 | 计算方式 | 典型值 |
|------|---------|---------|--------|
| `diag_reward` (wandb) | EpisodeStats → episode 结束时 | 最后一步的 self.reward | ~-50 (碰撞) 或 ~0.1 (存活) |
| `batch_mean_reward` | 32768 个 transition | reward.mean() | ~0.1 (98% 正常步 dominate) |

这两个指标度量完全不同的东西。`diag_reward` 是 episode 最后一步的瞬时 reward（碰撞步=-50），而 `batch_mean_reward` 是所有步的平均（被正常步稀释）。建议使用 collision_rate（现在的方式）或 `stats["return"]`（episode 累积回报）来作为训练效果的指标。

---

*报告生成时间: 2026-03-18*
*代码版本: Constrained Residual PPO with collision-rate-based lambda constraint*
