# Tunnel PPO-Lagrangian Experiment Guide

This document summarizes the design history, experimental evidence, and next-step training plan for the `04s_tunnel_lagrangian` experiment line. It is intended to be the working guide for continuing training without losing the lessons from the reward/cost redesign and the lambda stabilization debugging.

## 1. Experiment objective

The goal is to test whether PPO-Lagrangian can make safety behave like a constraint while preserving the main task objective: following the human command through the tunnel.

The target behavior is:

- `reward` optimizes following, progress, height stability, and smoothness.
- `cost` represents safety violation risk and is not mixed into reward.
- `lambda_lag` dynamically increases when rollout cost exceeds the budget and relaxes when cost is below budget.
- During training, both success/return and collision rate should improve, rather than safety dominating the policy or being ignored.

The current experiment line is intentionally additive:

- Baseline code remains in `experiments/04_tunnel_task/`, `src/envs/env_tunnel.py`, and `src/algos/ppo_constrained_beta.py`.
- The Lagrangian line uses `experiments/04s_tunnel_lagrangian/`, `src/envs/env_tunnel_lagrangian.py`, and `src/algos/ppo_constrained_beta_lagrangian.py`.

## 2. Design history and lessons learned

### 2.1 Initial proposal

The starting recommendation was to swap the roles of reward and safety:

| Component | Old role | New role |
|---|---|---|
| Following | Reward plus residual regularization | Main reward objective |
| Safety | Exponential reward penalty, always active over a wide distance | Independent cost channel |
| Trade-off | Fixed `reg_coeff` and reward weights | Adaptive Lagrangian multiplier |

The intended principle is: safety is a constraint, not the goal; following remains the goal.

### 2.2 Additive implementation

The implementation created a separate experiment path instead of modifying the original tunnel task:

| Area | File |
|---|---|
| New environment | `src/envs/env_tunnel_lagrangian.py` |
| New algorithm | `src/algos/ppo_constrained_beta_lagrangian.py` |
| New experiment entrypoints | `experiments/04s_tunnel_lagrangian/train.py`, `eval_video.py`, `run_curriculum.py` |
| New configs | `configs/experiment/tunnel_lagrangian*.yaml` |

The directory name uses `04s_tunnel_lagrangian` rather than a new numeric prefix such as `06_*`, to avoid future branch conflicts.

### 2.3 First failure mode: lambda explosion

The first run was intended to evaluate role-swap behavior, but its actual config still had `algo.use_lagrangian: true`.

Failure symptoms:

| Metric | Observation |
|---|---:|
| `lambda_lag` | Grew from about `2.6` to `1.6e4` |
| `safety_loss` | Grew to about `2e4` |
| `eval/success` | Fell from `14.1%` at iter 0 to near `0%` |
| `eval/collision` | Stayed high, roughly `55%` to `96%` |

Root causes:

1. Lambda was updated inside every PPO minibatch and epoch. With `training_epoch_num=10` and `num_minibatches=16`, this meant about 160 lambda updates per rollout batch.
2. Cost scale was not aligned with `cost_limit=0.05`. The old cost was `I(d < R_safe) + alpha * max(0, R_safe - d)`, so any violation produced cost at least `1`.
3. `safety_cost_radius=1.5m` was too large relative to the simulator collision distance of about `0.3m`, making cost active across much of the usable tunnel corridor.

### 2.4 Stabilization fixes

The current stable version applies three key fixes:

1. Normalize cost to `[0, 1]`:

   ```python
   soft_margin = safety_cost_radius - safety_collision_radius
   safety_cost = clamp(safety_cost_radius - min_dist, 0, soft_margin) / soft_margin
   ```

2. Update lambda once per rollout batch, not per PPO minibatch.
3. Bound lambda with a max clamp.

Current default config:

| Parameter | Current value | Rationale |
|---|---:|---|
| `env.safety_collision_radius` | `0.3` | Matches collision trigger scale |
| `env.safety_cost_radius` | `0.8` | 0.5m soft buffer before collision |
| `algo.use_lagrangian` | `true` | Main experiment target |
| `algo.cost_limit` | `0.05` | Allows limited normalized cost budget |
| `algo.lambda_init` | `0.5` | Nonzero initial safety pressure |
| `algo.lambda_lr` | `1e-3` | Slow enough for rollout-level updates |
| `algo.lambda_max` | `10.0` | Prevents another lambda explosion |
| `algo.reg_coeff` | `0.0` in current run | Tests whether reward plus Lagrangian is sufficient |

### 2.5 M2 diverse-pilot input alignment

The earlier 04s runs used the default online `UserModelTunnel` path. The current
04s config now aligns with experiment 04 M2 by default:

```yaml
user_model:
  offline_mode: true
  dataset_path: ${hydra:runtime.cwd}/data/trajectories_tunnel.h5
  sampling_mode: scaled
```

This means "M2-aligned input" refers to the existing M2 implementation:
`trajectory_gen_tunnel.yaml` generates `data/trajectories_tunnel.h5`, and
`UserModelTunnel(offline_mode=True)` samples from that dataset. It does **not**
mean directly replacing the environment's input model with
`src/core/user_model_diverse.py`, which is a separate online multi-modal input
model and should be treated as a distinct experiment if used later.

Use the 04s curriculum runner to create/reuse the dataset automatically:

```bash
cd isaac-training
python experiments/04s_tunnel_lagrangian/run_curriculum.py --end-stage 1
```

## 3. Current run analysis

Run path:

```text
outputs/lagrangian_curriculum_stage1/2026-04-30_03-49-10
```

The run had not fully finished when analyzed, but it had reached `checkpoint_10000.pt` and already showed clear trends.

Actual run configuration highlights:

| Setting | Value |
|---|---:|
| Experiment | `tunnel_lagrangian_stage1` |
| `env.num_obstacles` | `40` |
| `eval_interval` | `2000` |
| `save_interval` | `1000` |
| `reg_coeff` | `0.0` |
| `use_lagrangian` | `true` |
| `safety_collision_radius` | `0.3` |
| `safety_cost_radius` | `0.8` |

### 3.1 Eval curve

Metric key convention used in this guide:

| Guide shorthand | Training/eval log key |
|---|---|
| `eval/success` | `eval/success` |
| `eval/collision` | `eval/collision` |
| `eval/diag_safety_cost` | `eval/diag_safety_cost` |
| `eval/above_bound`, `eval/below_bound` | `eval/above_bound`, `eval/below_bound` |
| `episode_success`, `episode_collision` in training windows | `episode/stats_success`, `episode/stats_collision` |
| `episode_safety_cost`, `episode_min_dist` in training windows | `episode/stats_diag_safety_cost`, `episode/stats_diag_min_dist_to_obs` |

`eval/*` values are deterministic evaluation-rollout summaries produced by the training script. `episode/stats_*` values are on-policy training episode statistics and should be used for trend monitoring, not final checkpoint claims.

Checkpoint selection now reads `eval/success` while remaining compatible with older `eval/stats_success` logs, so new `checkpoint_best.pt` files are selected from deterministic eval metrics rather than on-policy training-window stats.

| iter | success | collision | terminated | episode_len | return | task_reward | safety_cost | min_dist | height_penalty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1719 | 0.8281 | 0.8281 | 433.2 | 622.0 | 0.9859 | 0.8638 | 0.408 | 0.000 |
| 2000 | 0.6836 | 0.3008 | 0.3164 | 553.4 | 582.8 | 0.6754 | 0.3416 | 0.986 | 0.127 |
| 4000 | 0.7617 | 0.2227 | 0.2383 | 577.8 | 514.4 | 0.6058 | 0.2770 | 1.043 | 0.160 |
| 6000 | **0.7930** | **0.1914** | 0.2070 | 591.9 | 531.3 | 0.5702 | 0.2517 | 0.989 | 0.181 |
| 8000 | 0.7305 | 0.2578 | 0.2695 | 555.6 | 570.8 | 0.5907 | 0.3539 | 0.868 | 0.055 |
| 10000 | 0.7813 | 0.2070 | 0.2188 | 571.6 | 571.1 | 0.6540 | 0.2751 | 0.962 | 0.098 |

Key deltas from iter 0 to iter 10000:

| Metric | Delta |
|---|---:|
| `eval/success` | `+60.9` percentage points |
| `eval/collision` | `-62.1` percentage points |
| `eval/diag_safety_cost` | `-0.589` |
| `eval/return` | `-51.0` |

The return decrease is not a failure by itself. Iter 0 has high return because identity following gives high task reward, but collision is very high. For this constrained task, success and collision should dominate checkpoint selection; return should be a tie-breaker.

### 3.2 Training windows

| window | rollout_cost mean | lambda mean / last | safety_loss mean | episode_success mean | episode_collision mean | episode_safety_cost mean | episode_min_dist mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-99 | 0.070 | 0.527 / 0.578 | 0.037 | 0.278 | 0.718 | 0.763 | 0.509 |
| 900-999 | 0.069 | 1.396 / 1.424 | 0.097 | 0.558 | 0.430 | 0.507 | 0.794 |
| 1900-1999 | 0.061 | 1.827 / 1.850 | 0.111 | 0.688 | 0.310 | 0.363 | 0.946 |
| 3900-3999 | 0.033 | 0.671 / 0.608 | 0.022 | 0.808 | 0.182 | 0.232 | 1.058 |
| 5900-5999 | 0.040 | 0.000 / 0.000 | 0.000 | 0.773 | 0.222 | 0.280 | 0.930 |
| 7900-7999 | 0.074 | 0.712 / 0.788 | 0.053 | 0.758 | 0.239 | 0.323 | 0.913 |
| 9900-9999 | 0.038 | 0.585 / 0.532 | 0.022 | **0.831** | **0.165** | **0.198** | 1.060 |

Interpretation:

- Lambda now behaves dynamically rather than exploding.
- When rollout cost is above `cost_limit=0.05`, lambda increases.
- When rollout cost stays below the budget, lambda decays and may hit zero.
- The zero floor can cause safety pressure to disappear temporarily, which likely explains the collision rebound around iter 8000.

## 4. What the current results indicate

### Finding 1: The Lagrangian mechanism is now functional

Observation: `lambda_lag` stays in a small dynamic range, roughly `0` to `1.85`, instead of exploding to `1e4+`.

Interpretation: Rollout-level lambda updates plus normalized cost fixed the optimizer-scale failure.

Implication: This experiment line is no longer blocked by the previous numerical/optimization pathology.

Next check: Continue tracking `rollout_mean_cost`, `lambda_lag_after_update`, and `eval/collision` through the final stage1 checkpoint.

### Finding 2: Safety improves without destroying task performance

Observation: Success rises from `17.2%` to `78.1%`, while collision falls from `82.8%` to `20.7%`.

Interpretation: The policy learns meaningful avoidance while still completing the tunnel task.

Implication: PPO-Lagrangian is a viable direction for this task, but this claim is still based on one main seed/run and needs held-out validation.

Next check: Verify the same checkpoint on held-out obstacle counts and seeds.

### Finding 3: Return is a misleading primary metric

Observation: Iter 0 has the highest raw eval return in the table, but also the worst collision.

Interpretation: The reward still contains strong task-following terms and does not encode the full safety objective.

Implication: Checkpoint selection must use a constrained score, not raw return alone.

Next check: Use the checkpoint score defined below.

### Finding 4: `reg_coeff=0.0` works in stage1 but may be risky

Observation: Current run uses `reg_coeff=0.0`; training succeeds, but `reg_loss` grows to large values in later windows.

Interpretation: The learned policy is willing to deviate far from human action to satisfy task and safety. This is not necessarily bad in stage1, but it may reduce robustness in stage2/stage3.

Implication: `reg_coeff` should remain an ablation variable, not a deleted concept.

Next check: Compare `reg_coeff=0.0`, `0.003`, and `0.01` with the same Lagrangian setup.

## 5. Recommended checkpoint score

Do not pick checkpoints by return alone. Use a constrained score:

```text
score = success
        - 0.5 * collision
        - 0.2 * above_bound
        - 0.2 * below_bound
```

Tie-breakers:

1. Higher `eval/success`.
2. Lower `eval/collision`.
3. Lower `eval/diag_safety_cost`.
4. Higher `eval/diag_reward_task`.
5. Higher return.

Current candidates:

| Checkpoint | Why use it | Risk |
|---|---|---|
| `checkpoint_6000.pt` | Best eval success and collision in available eval table | Later training may improve further |
| `checkpoint_10000.pt` | Better return/task reward and latest training window looks strong | Eval collision slightly worse than 6000 |
| Final checkpoint | Preferred if final eval stays good | Must verify final eval and held-out metrics |

Implementation note: checkpoint selection has been updated to consume the actual
`eval/success` key (while remaining compatible with historical
`eval/stats_success`) and to select by the constrained score above. New rich
checkpoints include `best_eval_score`, `best_eval_success`,
`best_eval_collision`, and `best_eval_info` metadata for curriculum handoff.

## 6. Next training plan

### 6.1 Finish the active run

Do not interrupt the current run unless it clearly collapses.

Completion criteria for stage1:

| Metric | Proceed threshold | Strong threshold |
|---|---:|---:|
| `eval/success` | `>= 0.75` | `>= 0.85` |
| `eval/collision` | `<= 0.25` | `<= 0.15` |
| `eval/diag_safety_cost` | `<= 0.35` | `<= 0.20` |
| `eval/above_bound + eval/below_bound` | `<= 0.05` | `<= 0.02` |

If the final checkpoint meets the proceed threshold, run held-out evals before entering stage2.

### 6.2 Held-out validation before stage2

Run the best checkpoint on at least:

| Validation | Purpose |
|---|---|
| `env.num_obstacles=40`, `eval_seed=42` | Reproduce training eval/user-command stochasticity |
| `env.num_obstacles=50`, `eval_seed=43` | Intermediate stress between stage1 and stage2 |
| `env.num_obstacles=60`, `eval_seed=44` | Exact stage2 obstacle count |
| `env.num_obstacles=65`, `eval_seed=45` | Optional beyond-stage2 stress |

Suggested command template:

```bash
cd isaac-training
python experiments/04s_tunnel_lagrangian/eval_video.py \
    experiment=tunnel_lagrangian_stage1 \
    +resume_checkpoint=/path/to/checkpoint.pt \
    env.num_envs=256 \
    +keep_num_envs=true \
    env.num_obstacles=50 \
    +eval_seed=43 \
    +video_dir=./eval_videos/lagrangian_stage1_obst50_seed43
```

`eval_video.py` defaults to reducing `env.num_envs` to 4 when rendering if `keep_num_envs` is not set. For metric validation, set `+keep_num_envs=true`; otherwise the result is useful for qualitative video inspection but too small for stable rate estimates. Use `+eval_seed=...` for rollout/user-command seeding; plain `seed=...` is not the parameter consumed by `eval_video.py`.

Or run the default validation grid:

```bash
cd isaac-training
python experiments/04s_tunnel_lagrangian/run_heldout_eval.py \
    --checkpoint /path/to/checkpoint_best.pt
```

Terrain/obstacle-layout caveat: `EnvTunnelLagrangian` currently constructs the terrain generator with a hardcoded `seed=0`. Therefore the validation grid above varies obstacle count and rollout/user-command seed, but not true terrain-layout seed. To claim layout generalization, first expose the terrain generator seed as a config parameter and then run a real terrain-seed sweep.

Proceed to stage2 only if held-out collision remains below `0.30` and success remains above `0.65`. For each held-out setting, use at least 256 parallel environments or multiple seeds; with 256 environments, a rate around 0.20 has a rough binomial standard error of about 2.5 percentage points, so changes smaller than 3 percentage points should not be over-interpreted from a single eval.

### 6.3 First improvement: prevent lambda from fully turning off

The main remaining instability is lambda touching zero. Add a small `lambda_min` if collision oscillation persists. This changes pure Lagrangian relaxation into a hybrid design with a fixed safety floor, so treat it as an ablation rather than the default conclusion.

Proposed change:

```yaml
algo:
  lambda_min: 0.2
```

Implementation:

```python
self.lambda_min = cfg.get("lambda_min", 0.0)
...
self.lambda_lag.clamp_(min=self.lambda_min, max=self.lambda_max)
```

Expected effect:

- Reduces collision rebound when rollout cost temporarily drops below the budget.
- Keeps Lagrangian adaptive, but preserves a safety floor.

Validation:

| Expected metric change | Accept if |
|---|---|
| Lower collision | `eval/collision` decreases by at least 3 percentage points |
| No major success loss | `eval/success` decreases by less than 5 percentage points |
| Lambda remains dynamic | `lambda_lag_after_update` does not stay pinned at `lambda_min` for the whole run |

### 6.4 Second improvement: light residual regularization ablation

The current successful run uses `reg_coeff=0.0`. Keep it as a valid baseline, but test whether a tiny regularizer improves robustness.

Sweep:

| Run | `reg_coeff` | Hypothesis |
|---|---:|---|
| R0 | `0.0` | Strong current baseline |
| R1 | `0.003` | Reduces extreme residuals with minimal safety cost |
| R2 | `0.01` | More identity bias, may improve generalization but hurt avoidance |

Decision criteria:

| Metric | Preferred direction |
|---|---|
| `eval/success` | Maximize |
| `eval/collision` | Minimize |
| `ppo_train/reg_loss` | Lower is better only if success/collision do not degrade |
| `debug_vec_policy` vs `debug_vec_target` | Avoid extreme divergence |

Do not restore large `reg_coeff` values such as `0.05` unless there is clear evidence that residual drift is causing failures.

### 6.5 Third improvement: stage2 curriculum

Only after stage1 passes held-out checks:

1. Start `tunnel_lagrangian_stage2` from the selected stage1 checkpoint.
2. Keep `lambda_lr=1e-3`, `lambda_max=10`, and `safety_cost_radius=0.8`.
3. If collision jumps above `0.40`, do not immediately increase radius. First try `lambda_min=0.2` or `cost_limit=0.04`.

Stage2 proceed thresholds:

| Metric | Minimum |
|---|---:|
| `eval/success` | `>= 0.60` |
| `eval/collision` | `<= 0.35` |
| `eval/diag_safety_cost` | `<= 0.40` |

## 7. Ablation matrix

Run the smallest matrix that can answer the next question.

### 7.1 Stability ablation

| ID | `lambda_min` | `cost_limit` | `reg_coeff` | Goal |
|---|---:|---:|---:|---|
| A0 | `0.0` | `0.05` | `0.0` | Current baseline |
| A1 | `0.2` | `0.05` | `0.0` | Test safety floor |
| A2 | `0.0` | `0.04` | `0.0` | Test stricter budget |
| A3 | `0.2` | `0.05` | `0.003` | Safety floor plus mild residual control |

Recommended priority: A1 first, then A3.

### 7.2 Cost-radius ablation

Only run this if the lambda floor does not reduce collision enough.

| ID | collision radius | cost radius | Hypothesis |
|---|---:|---:|---|
| C0 | `0.3` | `0.8` | Current baseline |
| C1 | `0.3` | `0.9` | Earlier safety pressure |
| C2 | `0.3` | `0.7` | Less conservative, more task-following |

Do not return to `1.5m` unless there is a new environment reason. It made cost active too often.

## 8. Monitoring checklist

Watch these metrics together:

| Metric | Healthy sign | Bad sign |
|---|---|---|
| `ppo_train/rollout_mean_cost` | Oscillates around `cost_limit` | Always far above or far below |
| `ppo_train/lambda_lag_after_update` | Moves up/down smoothly | Explodes, pins at max, or stays zero |
| `ppo_train/safety_loss` | Same order as actor/critic losses | Dominates by orders of magnitude |
| `episode/stats_success` | Upward trend | Improves then collapses |
| `episode/stats_collision` | Downward trend | Rebounds when lambda hits zero |
| `eval/diag_safety_cost` | Decreases over time | No improvement despite lambda growth |
| `eval/diag_reward_task` | Does not collapse | Safety avoids by refusing task |
| `eval/above_bound`, `eval/below_bound` | Near zero | Height instability |
| `ppo_train/reg_loss` | Stable or slowly increasing | Rapid growth with poor held-out generalization |

## 9. Recommended immediate actions

1. Let the active run finish.
2. Select the best checkpoint by constrained score, not return.
3. Run held-out evals at obstacle counts 40, 50, 60, and optionally 65.
4. If collision remains above `0.15`, implement and test `lambda_min=0.2`.
5. Run `reg_coeff` ablation only after deciding whether `lambda_min` is needed.
6. Enter stage2 only after held-out success/collision thresholds pass.

## 10. Concise summary

The new Lagrangian tunnel experiment is now viable. The first attempt failed because lambda was updated too often and the cost scale/radius were misaligned. After normalizing cost to the `0.3m` collision scale, using `0.8m` as a soft safety radius, and updating lambda once per rollout, training improved from `17.2%` to `78.1%` eval success and reduced collision from `82.8%` to `20.7%` by iter 10000. The next priority is not another major redesign, but stabilizing the safety floor and validating generalization before stage2.
