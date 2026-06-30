# PPO-Lagrangian 修订方案

> 基于 AsyncShield 论文的设计哲学，对 SharedRLControl 项目的 reward/constraint 架构进行调整

---

## 一、当前架构诊断

### 现有 reward/loss 结构

| 组件 | 位置 | 作用 | 问题 |
|---|---|---|---|
| `r_safety` | 环境reward | 指数距离惩罚，越远越好 | **无上界**，永远有动力远离障碍物 |
| `reward_task` | 环境reward | 速度方向对齐+速度大小匹配 | 和safety在同一个reward里打架 |
| `reg_loss` | PPO算法loss | `||mean_delta||²` 正则化 | 固定权重，不是Lagrangian，无法自适应 |
| `penalty_height` | 环境reward | 高度越界惩罚 | OK，但权重很大(8.0)，可能压过其他信号 |

### 核心矛盾

safety是reward（追求无上限的远离），跟随是regularization（固定系数的软约束）。优化器发现"远离障碍物"的收益远大于"跟人走"的惩罚，所以lambda趋向于忽略跟随。

---

## 二、修订原则

**对调 reward 和 constraint 的角色：**

| | 改动前 | 改动后 |
|---|---|---|
| **Reward** | 跟随 + 安全距离 + 高度 | **跟随 + 高度**（安全移出） |
| **Safety** | reward里的指数惩罚，越远越好 | **独立cost通道**，阈值触发，超过R_safe就不管了 |
| **跟随机制** | 环境reward + 算法正则化 | **环境reward为主**，正则化为辅 |
| **平衡手段** | 固定 `reg_coeff` | **自适应 lambda**，cost超标自动加强安全 |

核心原则（来自 AsyncShield）：**安全不是目标，是底线。跟随才是目标。**

---

## 三、具体修改

### 修改 1：环境 reward — 只保留跟随意图

**文件：`env_residual.py`，`_compute_state_and_obs` 方法中的 reward 计算部分**

```python
# ========== 新的 reward 设计 ==========

# a. 跟随意图奖励（主要 reward，追求最大化）
# a1. 方向对齐（flow reward，类比 AsyncShield 的 w_flow）
cosine_sim = torch.cosine_similarity(target_vel_w, drone_vel_w, dim=-1).unsqueeze(-1)
reward_direction = (cosine_sim + 1.0) / 2.0  # [0, 1]

# a2. 速度大小匹配
vel_error = (target_vel_w - drone_vel_w).norm(dim=-1, keepdim=True)
reward_speed_match = torch.exp(-2.0 * vel_error)

# a3. 动作平滑性（类比 AsyncShield 的 w_smooth）
action_diff = (self.agent_action - self.prev_action_command).norm(dim=-1, keepdim=True)
penalty_smoothness = (action_diff / self.max_action_vel) ** 2

# 综合跟随 reward
reward_following = 1.0 * reward_speed_match + 0.5 * reward_direction - 0.1 * penalty_smoothness

# b. 高度惩罚（保留，这个OK）
h_min, h_max = self.height_range[..., 0], self.height_range[..., 1]
z = self.drone.pos[..., 2:3].reshape(self.num_envs, 1)
penalty_height = (z - (h_max + 0.2)).clamp(min=0.0) + ((h_min - 0.2) - z).clamp(min=0.0)

# ========== 总 reward = 只有跟随 + 高度，不含安全 ==========
self.reward = reward_following - 8.0 * penalty_height
```

**关键改动：把 `r_safety` 完全从 reward 中移除。**

---

### 修改 2：安全成本 — 阈值触发式，作为独立 cost 输出

**文件：`env_residual.py`，在同一方法中新增 cost 计算**

```python
# ========== 安全成本（cost，不走 reward，走独立的 cost 通道）==========
# 类比 AsyncShield: c_t = I(d_min < R_safe) + alpha * max(0, R_safe - d_min)
R_safe = 1.5  # 安全半径阈值，超过此距离 cost = 0
alpha_cost = 2.0  # 连续惩罚缩放因子

if self.enable_lidar:
    ray_vecs_w = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)
    ray_dists = ray_vecs_w.norm(dim=-1).clamp_max(self.lidar_range)
    min_dist_to_obs, _ = ray_dists.min(dim=-1, keepdim=True)

    # 阈值触发：只在 d < R_safe 时产生 cost
    indicator = (min_dist_to_obs < R_safe).float()
    continuous_penalty = alpha_cost * (R_safe - min_dist_to_obs).clamp(min=0.0)
    self.safety_cost = indicator + continuous_penalty
else:
    self.safety_cost = torch.zeros(self.num_envs, 1, device=self.device)
```

---

### 修改 3：在环境中暴露 cost 给算法

**文件：`env_residual.py`，`_set_specs` 方法**

```python
# 在 reward_spec 中新增 cost 字段
self.reward_spec = Composite({
    "agents": Composite({
        "reward": Unbounded((1,)),
        "cost": Unbounded((1,)),  # 新增
    })
}).expand(self.num_envs).to(self.device)
```

**文件：`env_residual.py`，`_compute_reward_and_done` 方法**

```python
def _compute_reward_and_done(self):
    reward = self.reward
    terminated = self.terminated
    truncated = self.truncated
    return TensorDict(
        {
            "agents": {
                "reward": reward,
                "cost": self.safety_cost,  # 新增
            },
            "done": terminated | truncated,
            "terminated": terminated,
            "truncated": truncated,
        },
        self.batch_size,
    )
```

---

### 修改 4：PPO-Lagrangian — 自适应 lambda

**文件：`ppo_constrained_beta.py`，`__init__` 中新增 Lagrangian 变量**

```python
# 在 __init__ 中添加：
self.lambda_lag = torch.nn.Parameter(torch.tensor(1.0, device=device))
self.lambda_optimizer = torch.optim.Adam([self.lambda_lag], lr=1e-2)
self.cost_limit = cfg.get("cost_limit", 0.05)  # 期望的平均 cost 上限
```

**文件：`ppo_constrained_beta.py`，`_update` 方法中修改 loss 计算**

```python
def _update(self, minibatch):
    # ... 前面的 forward 部分不变 ...

    # 1. Entropy Loss
    entropy = action_dist.entropy()
    entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(entropy)

    # 2. Actor Loss (PPO clipped)
    # ... (保持不变) ...
    actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim

    # 3. 跟随意图的 reward（从环境来，已经在 GAE/advantage 中体现）
    #    advantage 本身就编码了跟随 reward，所以 actor_loss 已经在优化跟随

    # 4. 安全成本约束（Lagrangian）
    cost = minibatch[("agents", "cost")]  # 从环境拿到的 cost
    mean_cost = cost.mean()

    # lambda * mean_cost：lambda 自适应调节安全压力
    safety_loss = self.lambda_lag * mean_cost

    # 5. 正则化 loss 保留但降低权重（作为辅助，不是主要跟随信号）
    reg_loss = minibatch["_mean_delta"].pow(2).sum(dim=-1).mean()

    # 6. 总 policy loss
    loss_pi = actor_loss + safety_loss + 0.01 * reg_loss

    # 7. Critic Loss (不变)
    # ...

    # 8. Lambda 更新（梯度上升：使 lambda 在 cost > limit 时增大）
    #    目标：min -lambda * (mean_cost - cost_limit)
    #    即：cost > limit → lambda 增大 → 更重视安全
    lambda_loss = -self.lambda_lag * (mean_cost.detach() - self.cost_limit)

    # Total Loss
    loss = entropy_loss + loss_pi + critic_loss

    # Backward (policy + critic)
    # ... (不变) ...
    loss.backward()

    # 单独更新 lambda
    self.lambda_optimizer.zero_grad()
    lambda_loss.backward()
    self.lambda_optimizer.step()

    # Lambda 非负约束
    self.lambda_lag.data.clamp_(min=0.0)

    # ... 返回 infos，加上 mean_cost 和 lambda 的记录 ...
```

---

## 四、cost_limit 参数说明

`cost_limit` 是给安全成本设的"预算上限"。

| `cost_limit` 值 | 效果 |
|---|---|
| 很低（如 0.01） | 策略非常保守，几乎不允许触发安全约束 |
| 中等（如 0.05） | 允许偶尔触发，大部分时间跟随人类 |
| 很高（如 0.5） | 策略激进跟随，安全约束几乎不生效 |

**工作原理：**
- 实际 cost > cost_limit → lambda 增大 → 安全压力大 → 策略被迫更重视避障
- 实际 cost < cost_limit → lambda 减小 → 安全压力松 → 策略有更多空间去跟随

---

## 五、训练策略建议

1. **先不加 Lagrangian 跑一版**：只改 reward/cost 的角色对调，用固定权重近似看看效果是否改善
2. **确认 reward 信号有效后再加 Lagrangian**：避免同时改太多变量导致调不出来
3. **R_safe 的选择**：和无人机尺寸+安全裕度匹配，太大则频繁触发约束，太小则来不及反应
4. **cost_limit 的选择**：先用 0.05（即平均每步只有5%的时间触发cost），观察后调整

---

## 六、关于模型架构的建议

- **残差网络保留**：修正后reward驱动跟随，残差退化为"微调"角色，大部分时候输出接近0，只在需要避障时产生偏移
- **RNN建议暂时去掉**：保持MLP，先验证reward设计的正确性
- **先对齐目标，再对齐模型**：用最简单的结构验证问题定义是否正确
