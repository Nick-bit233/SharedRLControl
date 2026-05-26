**这份计划的目标是整理 `isaac-training/src` 的职责边界，并收敛 4 个主线实验入口的重复训练框架。**

核心约束：
- 共享 runtime 放在 `src/core/`，不新建 `src/experiment_runtime/`。
- 当前混在 `src/core/` 的模拟用户 / pilot 组件迁移到 `src/simulated_users/`。
- 训练入口仍保留各自的 `@hydra.main` 和 CLI 用法。
- 本轮只做结构整理、导入迁移、训练框架抽取，不改 env / algo / reward / config 的行为语义。

---

## 0. 先锁定范围

**本轮处理两类目标。**

### A. 目录职责整理

从 `src/core/` 迁移到 `src/simulated_users/`：
```text
src/core/user_model.py              -> src/simulated_users/user_model.py
src/core/user_model_tunnely.py      -> src/simulated_users/user_model_tunnely.py
src/core/user_model_diverse.py      -> src/simulated_users/user_model_diverse.py
src/core/user_model_intent.py       -> src/simulated_users/user_model_intent.py
src/core/user_model_diagnostics.py  -> src/simulated_users/user_model_diagnostics.py
src/core/pilot_modes.py             -> src/simulated_users/pilot_modes.py
src/core/pilot_perception.py        -> src/simulated_users/pilot_perception.py
```

迁移后 `src/core/` 只保留训练 runtime、训练工具、profiler、curriculum 等训练框架相关代码。

### B. 训练入口收敛

本轮重点迁移这 4 个入口：
1. `experiments/04_tunnel_task/train.py`
2. `experiments/04s_tunnel_lagrangian/train.py`
3. `experiments/05_safety_shield/train.py`
4. `experiments/06_tunnel_intent_task/train.py`

**本轮不处理：**
- `01_simple_baseline`
- `02_residual_policy`
- `03_constrained_residual_policy`
- `configs/experiment/*` 的语义整理
- `src/envs/*` 的环境行为逻辑
- `src/algos/*` 的算法行为逻辑

**允许的 env / algo 改动：**
- 仅允许为了模拟用户迁移而修改 import 路径。
- 不改 reward、observation、action、reset、step、PPO 更新等行为逻辑。

**保留现状：**
- `04_real_room_task/train.py` 继续做薄包装，不单独扩展成新 runtime。

---

## 1. 做一次基线盘点

**操作**
```bash
cd SharedRLControl/isaac-training

rg "@hydra.main" experiments/*/train.py
rg "wandb.init\(|def evaluate\(|def save_env_image\(|SyncDataCollector\(|EpisodeStats\(" experiments/*/train.py
wc -l experiments/04_tunnel_task/train.py \
      experiments/04s_tunnel_lagrangian/train.py \
      experiments/05_safety_shield/train.py \
      experiments/06_tunnel_intent_task/train.py

rg "src.core.user_model|src.core.pilot_" src experiments
rg "from src.core.trainning_utils|import src.core.trainning_utils" src experiments
```

**目的**
- 记录当前 4 个目标 `train.py` 的重复点和体量。
- 记录模拟用户组件当前从 `src.core` 被引用的位置。
- 记录 `trainning_utils.py` 的历史拼写依赖，避免误删或一次性改坏算法/env 导入。

**验收标准**
- 能清楚看到 4 个目标 `train.py` 都内嵌了相似的 wandb / collector / evaluate / checkpoint 逻辑。
- 能列出所有需要改成 `src.simulated_users.*` 的 import。

---

## 2. 先搬模拟用户，清出 `src/core/`

**目标结构**
```text
isaac-training/src/
  core/
    __init__.py
    runner.py
    spec.py
    wandb_utils.py
    evaluation.py
    checkpointing.py
    collector.py
    training_utils.py
    trainning_utils.py
    curriculum.py
    profiler.py

  simulated_users/
    __init__.py
    user_model.py
    user_model_tunnely.py
    user_model_diverse.py
    user_model_intent.py
    user_model_diagnostics.py
    pilot_modes.py
    pilot_perception.py
```

**操作**
1. 新建 `src/simulated_users/__init__.py`。
2. 将 `user_model_*.py`、`user_model.py`、`pilot_*.py` 从 `src/core/` 移动到 `src/simulated_users/`。
3. 更新所有引用：
   - `src.core.user_model` -> `src.simulated_users.user_model`
   - `src.core.user_model_tunnely` -> `src.simulated_users.user_model_tunnely`
   - `src.core.user_model_diverse` -> `src.simulated_users.user_model_diverse`
   - `src.core.user_model_intent` -> `src.simulated_users.user_model_intent`
   - `src.core.pilot_modes` -> `src.simulated_users.pilot_modes`
   - `src.core.pilot_perception` -> `src.simulated_users.pilot_perception`
4. 修正迁移文件内部的相互导入，例如 `user_model_intent.py` 内部也应从 `src.simulated_users.pilot_modes` 导入。
5. 暂时保留 `src/core/trainning_utils.py`，因为 env / algo 仍依赖其中的 `ValueNorm`、`GAE`、`vec_to_world`、`vec_to_body` 等工具。
6. 可新增 `src/core/training_utils.py`，并让它从 `trainning_utils.py` re-export；后续再逐步改正拼写依赖。

**重要原则**
- 这一阶段只改 import 和文件位置，不改模拟用户行为。
- 不要同时重构 user model 内部逻辑。
- 不要删除 `trainning_utils.py`，除非所有旧导入都有兼容层。

**建议验证**
```bash
python3 -m py_compile \
  src/simulated_users/*.py \
  src/envs/*.py \
  src/algos/*.py

rg "src.core.user_model|src.core.pilot_" src experiments
```

**验收标准**
- `src/simulated_users/` 存在，并包含全部模拟用户 / pilot 组件。
- `src/core/` 中不再包含 `user_model*.py` 和 `pilot_*.py`。
- `rg "src.core.user_model|src.core.pilot_" src experiments` 不再命中。
- env / algo 中除 import 路径外没有行为级 diff。

---

## 3. 明确 `src/core/` 的职责

**`src/core/` 是训练 runtime 与训练工具目录。**

建议职责：
- `spec.py`：定义 `ExperimentSpec`、runtime hook / adapter 协议。
- `runner.py`：统一训练主循环。
- `wandb_utils.py`：统一 `wandb.init`、run name、resume id 处理。
- `evaluation.py`：统一 `evaluate()`、视频录制、eval stats 展平。
- `checkpointing.py`：统一 init/resume/final/best/latest checkpoint 与 marker 文件。
- `collector.py`：统一 `SyncDataCollector` / `EpisodeStats` 初始化和 episode stats 展平。
- `training_utils.py`：推荐的新工具入口。
- `trainning_utils.py`：历史拼写兼容入口，短期保留。
- `curriculum.py`：课程学习相关逻辑。
- `profiler.py`：性能分析相关逻辑。

**不放入 `src/core/` 的内容**
- 模拟用户模型。
- pilot perception / pilot modes。
- 具体实验脚本。
- 具体环境实现。
- 具体 PPO 算法实现。

**验收标准**
- `src/core/` 的文件名能直接表达训练框架职责。
- `src/core/` 中没有模拟用户组件。
- 不存在 `src/experiment_runtime/`。

---

## 4. 选一个 runtime 行为母版

**以 `04_tunnel_task/train.py` 为母版。**

**原因**
- 它是当前主线。
- checkpoint 语义相对完整。
- 已支持 `init_checkpoint` / `resume_checkpoint`。
- 已覆盖 tunnel / real_room 两种主场景逻辑分发。

**需要先统一的语义**
- `resume_checkpoint` 表示继续同一个中断训练，应恢复 policy、optimizer、iter、env_frames、best tracking、curriculum state。
- `init_checkpoint` 表示开启新 stage / 新 run，只加载 policy weights。
- checkpoint 中建议明确保存 `last_completed_iter`，runner 从 `last_completed_iter + 1` 开始，避免不同入口出现 off-by-one 语义差异。

**checkpoint 格式边界**
- 长期维护格式是 rich checkpoint：
  ```python
  {
      "policy": policy.state_dict(),
      "iter": global_iter,
      "last_completed_iter": global_iter,
      "env_frames": env_frames,
      "actor_optim": ...,
      "critic_optim": ...,
      "feature_extractor_optim": ...,
      "best_eval_score": ...,
      "best_eval_success": ...,
      "best_eval_collision": ...,
      "best_eval_info": ...,
      # optional: reg_scheduler / lambda_optimizer / experiment extras
  }
  ```
- legacy weights-only checkpoint 是裸 `policy.state_dict()`。
- legacy policy-wrapped checkpoint 是 `{"policy": policy.state_dict()}`，但没有 optimizer / iter / env_frames。
- `init_checkpoint` 可以接受 rich 和 legacy 格式，但只加载 policy weights，忽略 optimizer 和训练进度。
- `resume_checkpoint` 只接受 rich checkpoint。legacy 格式不具备断点续训所需的 optimizer、iter、env_frames、curriculum/best tracking 信息，必须改用 `init_checkpoint`。
- 新 runtime 保存的 periodic/latest/final checkpoint 必须全部使用 rich 格式。

**验收标准**
- 明确写下：`04_tunnel_task/train.py` 是 runtime 行为基准。
- Lagrangian 的特殊 checkpoint state 通过 adapter/hook 扩展，不反过来污染主路径。

---

## 5. 先抽 `ExperimentSpec`

**建议字段**
```python
ExperimentSpec(
    name: str,
    env_factory,
    policy_factory,
    dataset_loader=None,
    eval_summary_fn=None,
    checkpoint_adapter=None,
    hooks=(),
)
```

**建议协议**
- `env_factory(cfg, resources) -> env`
- `policy_factory(cfg, env) -> policy`
- `dataset_loader(cfg, hydra_cfg) -> object | None`
- `eval_summary_fn(eval_info) -> dict`，至少返回 `score`、`rank`、`success`、`collision` 等 best checkpoint 所需字段。
- `checkpoint_adapter` 负责：
  - load policy state
  - restore optimizer state
  - snapshot optimizer state
  - snapshot / restore experiment-specific state
- `hooks` 可提供：
  - `on_after_setup`
  - `on_before_training`
  - `on_before_train_step`
  - `on_after_train_step`
  - `on_before_eval`
  - `on_after_eval`
  - `on_before_checkpoint`
  - `on_after_checkpoint`
  - `on_after_training`

**当前 hook 使用建议**
- `sanity_check_fn` 只作为旧 wrapper 兼容字段保留，新入口不要继续使用。
- 残差模式检查、dataset/policy 一致性检查等实验特定 sanity check 放入 `on_after_setup` 或 `on_before_training` hook。
- `RegCoeffScheduler` 这类只对特定 policy/config 生效的调度器放入实验 hook，例如 `RegCoeffSchedulerHook`，不要在通用 runner 中硬编码。
- hook 需要随 checkpoint 保存的状态写入 `context["checkpoint_extra_state"]`。

**目的**
- 避免 `runner.py` 写死 tunnel / lagrangian / intent / shield 的分支。
- 让实验差异通过 spec / adapter / hook 注入，而不是继续复制 train.py。

**验收标准**
- `src/core/spec.py` 不为空，且定义清楚 runtime 扩展点。
- `src/core/runner.py` 不包含大量 `if experiment_name == ...` 分支。

---

## 6. 第一个落地：只迁移 `04_tunnel_task/train.py`

**操作**
1. 把以下通用逻辑移到 `src/core/` runtime：
   - wandb 初始化
   - profiling / env_test_mode 基础分支
   - Hydra output dir / `cfg.log_output_dir`
   - profiler 初始化与收尾
   - collector 初始化
   - `EpisodeStats`
   - 通用 `evaluate()`
   - episode stats 展平
   - best/latest/final checkpoint 写入
   - `best_checkpoint_path.txt` / `latest_checkpoint_path.txt` / `final_checkpoint_path.txt`
2. `04_tunnel_task/train.py` 保留：
   - `@hydra.main`
   - `init_simulation_app(cfg)`
   - env 选择：`tunnel` / `real_room`
   - algo 选择：beta / tanh normal
   - dataset loader 接线
   - `ExperimentSpec` 组装
   - `run_training(cfg, spec)` 调用
3. 先只让 tunnel 主线跑通，不马上迁移其他 train.py。

**特别注意**
- 当前 `src/core/trainning_utils.py` 中已有旧版 `evaluate()`，但它和目标 train.py 里的新版 evaluate 不完全一致。runtime 应以 `04_tunnel_task/train.py` 当前 evaluate 行为为准。
- `record_video`、`global_view`、`eval_visualization`、visibility restore 等细节要保留。

**验收标准**
- `04_tunnel_task/train.py` 明显缩短。
- 它不再包含完整训练循环主体。
- checkpoint marker 和 resume/init 语义与原来一致。

**建议验证**
```bash
python3 -m py_compile \
  src/core/*.py \
  experiments/04_tunnel_task/train.py

rg "wandb.init\(|def evaluate\(|SyncDataCollector\(|EpisodeStats\(" experiments/04_tunnel_task/train.py
```

预期：第二条不再命中，或只剩极少数必要引用。

---

## 7. 第二个落地：迁移 `05_safety_shield/train.py`

**操作**
- 只保留：
  - `EnvSafetyShield`
  - policy 选择
  - 特定打印 / 命名
  - `ExperimentSpec` 组装
- 删除重复的：
  - wandb
  - collector
  - `evaluate()`
  - checkpoint 保存逻辑
  - episode stats 展平逻辑

**验收标准**
- `05_safety_shield/train.py` 变成薄入口。
- 与 `04_tunnel_task/train.py` 结构一致，只是 env / policy / spec 不同。

**建议验证**
```bash
python3 -m py_compile experiments/05_safety_shield/train.py
rg "wandb.init\(|def evaluate\(|SyncDataCollector\(" experiments/05_safety_shield/train.py
```

预期：后者不再命中。

---

## 8. 第三个落地：迁移 `06_tunnel_intent_task/train.py`

**操作**
- 保留：
  - `EnvTunnelIntent`
  - `resolve_constrained_policy`
  - `load_trajectory_dataset`
  - intent 相关 observation / dataset 接线
  - `init_checkpoint` / `resume_checkpoint` 兼容语义
- 训练主循环交给 runtime。

**特别注意**
- intent 有 dataset loader 和额外 observation，不要把这些塞回 runtime 里写死。
- 通过 `ExperimentSpec`、`dataset_loader` 或 hook 注入。

**验收标准**
- `06_tunnel_intent_task/train.py` 和 `04_tunnel_task/train.py` 在结构上统一。
- intent 特性仍然独立存在。

**建议验证**
```bash
python3 -m py_compile experiments/06_tunnel_intent_task/train.py
```

---

## 9. 最后迁移 `04s_tunnel_lagrangian/train.py`

**这是最难的一步，放最后。**

**操作原则**
- 不强行把 Lagrangian 特殊逻辑塞进通用路径。
- 只把真正共用的部分迁走。
- Lagrangian 特有部分通过 adapter / hook 注入。

**建议抽成 adapter / hook 的内容**
- `lambda_lag` 兼容加载。
- `lambda_optimizer` 恢复与保存。
- Lagrangian 特定 rich checkpoint state。
- Lagrangian 特定 eval score / rank。
- 训练阶段中的自定义更新逻辑。

**验收标准**
- `04s_tunnel_lagrangian/train.py` 仍可能比其他 wrapper 稍长。
- 但不再重复 wandb / collector / 通用 evaluate / marker 写入这些公共部分。
- `runner.py` 没有出现 Lagrangian 专属 if 分支。

---

## 10. 做一次反向清理

迁移完 4 个 train.py 后，执行统一清理：

**操作**
```bash
rg "wandb.init\(" experiments/*/train.py
rg "def evaluate\(" experiments/*/train.py
rg "def save_env_image\(" experiments/*/train.py
rg "SyncDataCollector\(" experiments/*/train.py
rg "EpisodeStats\(" experiments/*/train.py
rg "src.core.user_model|src.core.pilot_" src experiments
find src/core -maxdepth 1 -type f | sort
find src/simulated_users -maxdepth 1 -type f | sort
```

**预期**
- 重复训练实现主要只存在于 `src/core/`。
- `experiments/*/train.py` 中不再保留多套重复实现。
- `src.core.user_model*` / `src.core.pilot_*` 旧路径不再被引用。

**验收标准**
- `train.py` 从“实现层”降为“配置/接线层”。
- `src/core/` 从“混合杂物目录”变成训练 runtime 目录。
- `src/simulated_users/` 独立承载模拟用户组件。

---

## 11. 控制 wrapper 体量

迁移结束后，建议给目标脚本设一个硬门槛：

| 文件 | 建议目标 |
| --- | --- |
| `04_tunnel_task/train.py` | `< 180` 行 |
| `05_safety_shield/train.py` | `< 120` 行 |
| `06_tunnel_intent_task/train.py` | `< 140` 行 |
| `04s_tunnel_lagrangian/train.py` | `< 220` 行 |

不是绝对值，但如果还远超这个范围，说明共享 runtime 抽取不充分，或者 spec / hook 边界还不够清楚。

---

## 12. 最终可验证的成果标准

最终整理完成后，至少满足下面这组可检查条件。

### A. 结构验证
```bash
find src/core -maxdepth 1 -type f | sort
find src/simulated_users -maxdepth 1 -type f | sort
test ! -d src/experiment_runtime
```

**预期**
- 共享 runtime 文件位于 `src/core/`。
- 模拟用户文件位于 `src/simulated_users/`。
- 不存在 `src/experiment_runtime/`。

### B. 重复消除验证
```bash
rg "wandb.init\(" experiments/*/train.py
rg "def evaluate\(" experiments/*/train.py
rg "def save_env_image\(" experiments/*/train.py
rg "SyncDataCollector\(" experiments/*/train.py
rg "EpisodeStats\(" experiments/*/train.py
```

**预期**
- 这些命中数显著下降，最好迁到 `src/core/` 中统一实现。

### C. 模拟用户导入验证
```bash
rg "src.core.user_model|src.core.pilot_" src experiments
rg "src.simulated_users" src experiments
```

**预期**
- 第一条不命中。
- 第二条能看到 env / diagnostics / visualization 等必要引用。

### D. 语法验证
```bash
python3 -m py_compile \
  src/core/*.py \
  src/simulated_users/*.py \
  src/envs/*.py \
  src/algos/*.py \
  experiments/04_tunnel_task/train.py \
  experiments/04s_tunnel_lagrangian/train.py \
  experiments/05_safety_shield/train.py \
  experiments/06_tunnel_intent_task/train.py
```

**预期**
- 全部通过。

### E. 入口保持验证
```bash
rg "@hydra.main" experiments/*/train.py
```

**预期**
- 每个实验仍是独立入口。
- CLI 用法不被破坏。

### F. 行为边界验证
```bash
git diff --stat -- src/envs src/algos configs/experiment
git diff -- src/envs src/algos configs/experiment
```

**预期**
- `src/envs/*`、`src/algos/*` 只有 import 路径变化。
- `configs/experiment/*` 不发生语义级大改动。

---

## 13. 推荐提交节奏

不要一次性做完再看。

**建议分 6 个 commit：**
1. `refactor: move simulated users out of core`
2. `refactor: add shared core runtime skeleton`
3. `refactor: migrate tunnel train entrypoint to core runtime`
4. `refactor: migrate safety shield and intent train entrypoints`
5. `refactor: migrate lagrangian train entrypoint with hooks`
6. `chore: remove duplicated train-layer helpers from experiment entrypoints`

这样出问题时更容易定位：第一步只负责文件归属和 import，后续步骤才处理训练 runtime 行为。

---

## 最后一句话

**这轮整理的成功标准，不是“train.py 变短了”，而是：`src/core/` 成为清晰的训练 runtime，`src/simulated_users/` 独立承载模拟用户组件，训练循环实现只保留一份，实验脚本只负责选环境 / 算法 / 配置，并且这些结果能被 grep、py_compile 和 git diff 直接验证。**
