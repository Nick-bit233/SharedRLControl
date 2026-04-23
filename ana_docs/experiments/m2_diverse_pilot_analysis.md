# M2 Diverse-Pilot 训练分析（second run, 20260422_233240）

## 1. 结论先行

- **轴序 / vx bias 两个 bug 已修复生效**：success 不再恒为 0（峰值 0.133，最终 ~0.05），collision 也能压到 1–3% 区间，说明轨迹缩放和 episode 长度都恢复正常。
- **但本次 from-scratch 训练事实上失败**：模型陷入“低碰撞 + 高超时 + 低成功”的安全悬停吸引子，关键证据是后期 `truncated ≈ 0.97–0.99`、`episode_len ≈ 1452–1489`、`success ≈ 3–13%`。
- **根因不是 pilot 速度，也不是 episode 长度，而是 reward 设计 + from-scratch 不稳定**：
  - `env_tunnel.py` 没有 success bonus，只有 `survival=0.2/step` + `task_reward≈0.4/step`，安全悬停 1500 步即可拿到 ~900 return，几乎与成功完成隧道持平。
  - eval 5 一次大碰撞惩罚把 return 砸到 –3778，引发 eval 6–10 的 value-function 崩塌（`terminated≈1.0, ep_len≈480`），策略被迫从废墟中重建，恢复后只能爬到“安全悬停”这个局部最优。
- **建议**：放弃 from-scratch 路线，改为 **从 M1 reg=0.005 best checkpoint warm-start**，叙事调整为“在更多样的 pilot 分布上做适配”，同时考虑给 env 加一个小的 success bonus 以摆脱悬停吸引子。

## 2. 训练曲线分级

| 阶段 | eval | iter (step×1e6) | succ | coll | term | trunc | ep_len | return | 解读 |
|------|------|----------------|------|------|------|-------|--------|--------|------|
| A 初始 | 1–4 | 0–49 | 0.000–0.004 | 0.84→0.26 | ↓ | ↑ | 710→1357 | 554→815 | 正常起步，碰撞快速下降 |
| B 崩塌 | 5–10 | 65–148 | 0 | 0.5→0.39 | **0.99–1.0** | ~0 | **480–561** | **–3778→–292** | 一次大惩罚后 value 崩，策略全部撞机 |
| C 重建 | 11–22 | 164–344 | 0.000–0.016 | 0.45–0.70 | 0.43–0.89 | 0.11–0.57 | 626–1158 | 5→449 | 缓慢恢复，开始出现首批成功 |
| D 提升 | 23–32 | 360–524 | **0.016–0.109** | 0.25–0.58 | 0.17–0.66 | 0.34–0.82 | 974–1369 | –377→667 | success 单调向上，碰撞继续下降 |
| E 收敛到悬停 | 33–41 | 540–655 | 0.027–0.133 | **0.008–0.05** | **0.01–0.20** | **0.96–0.99** | **1327–1489** | 416–957 | 碰撞极低、超时极高、成功停滞 |

最佳点是 eval 38：`success=0.133, collision=0.027, ep_len=1452, return=951`。但下一帧立即回落到 0.10/0.09/0.03，说明这只是 RNG 抖动，并非真正的能力达到 13%。

## 3. 为什么模型卡在“安全悬停”

reward 拆解（`src/envs/env_tunnel.py`，relevant lines 660–712）：

```
reward = enable_task_reward ? (1.0 * speed_match + 0.5 * direction) : 0
       + 0.2  (survival)
       + 3.0 * r_safety           (≤0)
       - 10  * penalty_height     (≥0)
crash bonus: -10  on terminated-by-crash
success bonus:  0  ← 这里没有！
```

观察后期评估：

- `diag_reward_task ≈ 0.40–0.65`（接近上限 1.5 的 1/3，因为速度跟得很差）
- `survival × 1500 = 300`
- 一个 1500 步全程不撞机的“安全悬停”可拿到 ≈ `300 + 1500*0.5 ≈ 1050`
- 一次成功（假设 800 步到达，速度跟得稍好 task=0.7）：`160 + 800*0.9 ≈ 880`，**还不如悬停**！
- 由于 `truncated = timeout | success`，到达终点不会带来任何 terminal bonus。

→ 在当前 reward 下，**安全悬停在严格意义上比成功更优**。M1 baseline 之所以能跑 95%+ 是因为它从已经爬出这个吸引子的 `paper-best` checkpoint 暖启动；本次 from-scratch + 多样 pilot 的组合让策略掉回了这个 reward landscape 的另一个局部极值。

## 4. 为什么 episode_len 永远顶到 1500

把 `debug_pos_world` 沿训练展开（取该指标的 scalar 值，眼下只用于趋势）：

| eval | 1 | 10 | 20 | 30 | 36 | 41 |
|------|---|----|----|----|----|----|
| pos  | 0.40 | –1.11 | 0.99 | 3.03 | 2.51 | 2.09 |

后期稳定在 2–3，远未达到“前进 17 m”所需位移。结合 `debug_vec_world` 后期只有 0.10–0.32（无人机实际世界速度模），**策略主动把 forward 速度压到 0.1–0.3 m/s**，即使 pilot 输入 vx∈[1.0, 2.0]。这是策略学到的“安全策略”，而不是 pilot 的问题，也不是 episode 长度的问题——把 max_episode_length 提到 3000 也不会改善，只会让悬停 return 更高。

## 5. 与 M1 直接对比

| 指标 | M1 reg=0.005 best | M2 second-run last | 趋势 |
|------|-------------------|--------------------|------|
| success | 0.95+ (peak 0.99) | 0.027 | ✗✗✗ |
| collision | 0.008–0.043 | 0.016 | ✓ |
| truncated | ~0.05 | 0.984 | ✗✗ |
| ep_len | ~700–800 | 1485 | ✗✗ |

M2 没有改善任何 paper-relevant 指标，而且和 M1 baseline 不同源——它是一次几乎完全失败的 from-scratch 训练，**不能直接拿来比较 pilot distribution 的影响**。

## 6. 推荐下一步（优先级排序）

### A. 立即跑：M2 = warm-start from M1-best on diverse pilot（首选）

理由：
- 论文叙事可以诚实地写成 “**adapt the M1 mainline to a more diverse pilot distribution**”，比 from-scratch 更符合 reviewer 期待（reviewer 关心的是“你的方法在更宽 pilot 分布下是否还行”，不是“你能不能 from-scratch 学会”）。
- 直接绕开 reward landscape 的悬停吸引子。
- 训练时间预计减半（10k iter 即可收敛）。

实操：
```bash
python experiments/04_tunnel_task/run_m2_diverse_pilot.py \
  --reg-coeff 0.005 \
  --resume-checkpoint outputs/tunnel_m1_noreg/reg_0.005/<best_run>/checkpoint_best.pt \
  --max-iterations 12010
```

需要把 runner 里的 `--resume-checkpoint` 接通（如尚未支持）。

### B. 同步小改：给 env 加一个 success bonus

仅对 from-scratch 路线必要；warm-start 路线可以不改。改动极小：

```python
# env_tunnel.py 第 712 行附近
success_bonus = 20.0
just_succeeded_mask = (success.unsqueeze(-1) & ~self.was_done_buf)
self.reward[just_succeeded_mask] += success_bonus
```

这会把“成功 800 步”的 return 从 ~880 提到 ~900，**真正大于** 悬停 1050 还差一点；可以更激进地把 bonus 提到 50 来彻底打破吸引子。如果担心改 reward 影响 M1 对比，把 `enable_success_bonus` 加成可配置 flag 即可。

### C. 不建议的方向（已排除）

- **再延长 episode 长度**：会让悬停 return 更高，恶化吸引子。
- **再调 pilot vx**：本次 mean=1.5 已经足够，限制是策略主动减速，不是 pilot 慢。
- **把 reg_coeff 拉大到 0.05+**：会强行把 policy 拽回 pilot 速度，但同时削弱避障能力，回到 paper-best 之前的状态。

### D. 关于本次 run 的处置

- **不进入对比表**：标记为 “from-scratch baseline failed due to safe-hover attractor，excluded from M2 main result”。
- 保留 train.log 用于 ablation 中 “reward landscape 论证” 的辅助证据，可以放进附录或 response letter。

## 7. SQL & plan 状态更新

- `tunnel-m2-diverse-pilot`: 仍标 `done`（实现已完成，bug 已修），但本 run 不算 final M2。
- `tunnel-m2-analysis`: 本文档完成，标 `done`。
- 新增 todo `tunnel-m2-warmstart`：以 M1 best 为起点重跑 M2，并视情况加 success bonus。
- M3 暂不启动，等 M2-warmstart 出结果。
