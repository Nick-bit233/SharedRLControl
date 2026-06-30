# 实验 04s Lagrangian RL 迭代优化计划

## 问题与目标

目标是在 `isaac-training/experiments/04s_tunnel_lagrangian` 这条 PPO-Lagrangian 实验线上，基于已有训练分析继续提升安全性与跟随性能：降低碰撞率，同时提高成功率，并监控速度跟随、轨迹稳定性、干预幅度等指标。

当前代码状态显示：

- 04s 使用独立实现：`experiments/04s_tunnel_lagrangian/train.py`、`eval_video.py`、`run_curriculum.py`，环境为 `src/envs/env_tunnel_lagrangian.py`，算法为 `src/algos/ppo_constrained_beta_lagrangian.py`。
- 04s 环境当前实例化的是 `UserModelTunnel`，并支持 `user_model.offline_mode` + `TrajectoryDataset`；但 `configs/experiment/tunnel_lagrangian*.yaml` 没有开启 offline dataset，因此默认仍使用旧的在线 tunnely user model。
- 实验 04 的 M2 阶段不是直接使用 `UserModelDiverse` 类，而是使用 `trajectory_gen_tunnel.yaml` 生成的 `data/trajectories_tunnel.h5`，再由 `UserModelTunnel(offline_mode=True, sampling_mode=scaled)` 采样；这是当前代码中“对齐 M2 diverse pilot 输入”的实际路径。
- 04s 指南已记录：Lagrangian 机制稳定后，stage1 成功率从约 17.2% 提升到 78.1%，碰撞率从 82.8% 降到 20.7%；下一步主要问题是输入分布未对齐 M2、best checkpoint 选择 key 错误、lambda 可能触零导致安全压力短暂消失，以及 held-out eval 覆盖不足。
- 已确认的实现口径：04s 对齐实验 04 的 M2，指采用 M2 当前实际路径，即 `trajectory_gen_tunnel.yaml` 生成的离线 `trajectories_tunnel.h5` + `UserModelTunnel(offline_mode=True)`，不是直接切换到 `src/core/user_model_diverse.py` 在线类。

拟采用的总体策略：

1. 先做正确性修复和输入对齐，让 04s 的训练/验证真实使用 M2 的 diverse pilot 分布。
2. 再修复 checkpoint 选择和评估工具，避免继续用错误指标驱动课程、early stopping 或阶段衔接。
3. 在稳定 baseline 上做小步 ablation：`lambda_min` 安全下限、轻量 `reg_coeff`、必要时 cost budget/radius，而不是重新大改 reward/cost。
4. 用 held-out obstacle count 和 seed 网格做进入 stage2/stage3 前的门槛验证。

## 计划待办

### 1. 对齐 04s 的 user model 输入到实验 04 M2

- 修改 `configs/experiment/tunnel_lagrangian.yaml` 或新增一个明确的 04s-M2 配置，使 04s 默认或可选启用：
  - `user_model.offline_mode: true`
  - `user_model.dataset_path: ${hydra:runtime.cwd}/data/trajectories_tunnel.h5`
  - `user_model.sampling_mode: scaled`
  - `gpu_cache_reserve_gb/min_scale_factor/preload_data` 与 `tunnel_m2_diverse_pilot.yaml` 保持一致。
- 为 04s 增加一个 runner（或扩展 `run_curriculum.py` 参数），复用 04 的 M2 数据集生成逻辑：
  - 如果 `data/trajectories_tunnel.h5` 不存在，则运行 `src/datasets/trajectory_generator.py --config-name=trajectory_gen_tunnel`。
  - 支持 `--regenerate-dataset`、`--skip-dataset`、`--dataset-path` 等安全选项。
- 在训练启动日志里显式打印 user model 来源，区分：
  - legacy online `UserModelTunnel`
  - M2 offline `trajectories_tunnel.h5`
  - 未来若直接接入 `UserModelDiverse` 在线类，需作为独立新实验命名，避免和 M2 offline dataset 混淆。
- 在 `eval_video.py` 中保持与训练一致的 offline dataset 加载路径，并在 metric validation 时要求 `+keep_num_envs=true`。

### 2. 修复 04s checkpoint 选择与 early-stopping 指标

- 修复 `experiments/04s_tunnel_lagrangian/train.py` 中的 key mismatch：
  - 当前代码读取 `eval/stats_success`。
  - 当前 eval 实际产出是 `eval/success`、`eval/collision`、`eval/above_bound`、`eval/below_bound` 等。
- 引入一个小的 helper，例如 `get_eval_metric(eval_info, "success")`，统一处理 `eval/success` 与历史 `eval/stats_success` 兼容，避免训练脚本、curriculum scheduler、early stopping 分散写死 key。
- 将 best checkpoint 选择从“仅 success”升级为指南中的 constrained score：
  ```text
  score = success - 0.5 * collision - 0.2 * above_bound - 0.2 * below_bound
  ```
  并用 tie-breaker 排序：更高 success、更低 collision、更低 `diag_safety_cost`、更高 `diag_reward_task`、更高 return。
- checkpoint metadata 中同时保存：
  - `best_eval_score`
  - `best_eval_success`
  - `best_eval_collision`
  - `best_eval_info`
  这样 `run_curriculum.py` 进入下一 stage 时可追溯为什么选中该 checkpoint。

### 3. 实现 Lagrangian 安全下限 ablation

- 在 `src/algos/ppo_constrained_beta_lagrangian.py` 中增加配置项：
  - `algo.lambda_min`，默认 `0.0`，保持当前行为不变。
- 将 lambda clamp 从 `min=0.0` 改为 `min=self.lambda_min`，并在 train info 中记录：
  - `lambda_min`
  - `lambda_lag_after_update`
  - `rollout_mean_cost`
- 在 04s configs 中不直接替换 baseline，而是新增或通过 runner 生成 ablation：
  - A0: `lambda_min=0.0, cost_limit=0.05, reg_coeff=0.0`
  - A1: `lambda_min=0.2, cost_limit=0.05, reg_coeff=0.0`
  - A3: `lambda_min=0.2, cost_limit=0.05, reg_coeff=0.003`
- 验收标准：
  - collision 至少下降约 3 个百分点；
  - success 下降小于 5 个百分点；
  - `lambda_lag_after_update` 不长期钉死在 `lambda_min`。

### 4. 跟随性能与 reward/diagnostic 指标补强

- 继续保留 safety cost 只进入 cost channel，不混入 reward。
- 不优先修改 reward 结构；先在 M2 输入分布 + checkpoint score + lambda floor 后观察是否仍出现安全悬停或过度干预。
- 增加或整理已有诊断输出，用于同时判断安全与跟随：
  - `eval/success`
  - `eval/collision`
  - `eval/diag_safety_cost`
  - `eval/diag_reward_task`
  - `eval/diag_penalty_smooth`
  - `eval_debug/vec_target/*`
  - `eval_debug/vec_policy/*`
  - `eval_debug/vec_world/*`
  - `eval/episode_len`
- 如果 M2 输入对齐后出现 M2 早期报告里的“低碰撞、高 timeout、低成功”安全悬停吸引子，再考虑添加可配置 success/progress shaping；该项应作为第二轮计划，不和第一轮 Lagrangian 修复混在一起。

### 5. Held-out validation 与阶段推进规则

- 为 04s 增加或整理一个评估脚本/命令模板，批量运行：
  - `env.num_obstacles=40, eval_seed=42`
  - `env.num_obstacles=50, eval_seed=43`
  - `env.num_obstacles=60, eval_seed=44`
  - 可选 `env.num_obstacles=65, eval_seed=45`
- metric eval 使用 `env.num_envs>=256` 和 `+keep_num_envs=true`；video eval 可以用小 env 数，只用于定性检查。
- 进入 stage2 的门槛：
  - held-out collision `< 0.30`
  - held-out success `> 0.65`
  - `above_bound + below_bound` 不显著增加。
- stage2 若 collision 跳到 `>0.40`，优先试 `lambda_min=0.2` 或 `cost_limit=0.04`，不立即把 `safety_cost_radius` 拉大。

### 6. 文档与结果记录

- 更新 `LAGRANGIAN_EXPERIMENT_GUIDE.md`：
  - 记录 04s 当前确实默认仍用 legacy online UserModelTunnel；
  - 明确“M2 对齐”在当前代码中表示使用 `trajectory_gen_tunnel.yaml` + `trajectories_tunnel.h5` + `offline_mode=True`；
  - 记录新 best-checkpoint score 与 ablation matrix；
  - 记录 held-out validation 命令模板和推进门槛。
- 如新增 runner/config，补充最短可运行命令，避免未来训练误用旧输入分布。

## 注意事项

- 不要直接复用旧 stage1 结果作为新结论，因为其 user input 分布与 M2 不一致。
- 不要把 `src/core/user_model_diverse.py` 直接等同于 M2；它是多模态在线输入模型，而本轮已确认的 M2 对齐路径是离线 tunnel trajectory dataset。
- 04s 与 04 的环境/算法存在 fork，修复应优先落在 04s 文件中，避免破坏已有 04 baseline，除非发现共享 bug（例如 metric helper）值得抽到共用模块。
- 所有训练配置修改应保持 Hydra override 友好，便于 ablation 脚本生成矩阵，而不是为每个超参组合复制大量配置文件。
