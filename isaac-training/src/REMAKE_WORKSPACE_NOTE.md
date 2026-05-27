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

## 7. 下一阶段目标：统一 experiments 入口，而不是继续复制 train.py

前 0-6 步已经完成了 runtime skeleton 和 `04_tunnel_task/train.py` 的薄入口迁移。下一阶段不要继续按
“每个实验各迁移一个 `train.py`”推进，否则仍然会留下大量入口脚本、课程学习脚本、评估脚本和 tmux
包装脚本。

新的目标是：
- 训练只保留一个统一入口：`experiments/train.py`。
- 评估只保留一个统一入口：`experiments/eval.py`。
- 后台启动只保留一个统一入口：`experiments/launch.py`。
- 多阶段课程学习、sweep、held-out grid 只保留统一 orchestration：`experiments/campaign.py`。
- 实验差异只通过 `ExperimentSpec` / `CheckpointAdapter` / `RuntimeHook` / 配置文件注入。
- `src/core/runner.py` 只负责“一次已经启动的训练循环”，不负责 tmux、不负责多阶段、不负责命令行编排。

**重要边界**
- 不要把 tmux 集成进 `runner.py`。tmux 是进程启动方式，runner 是训练循环。
- 不要把 curriculum stage pipeline 塞进 hook。hook 适合单次训练进程内的调度，stage pipeline 是多进程/多配置编排。
- 不要让统一 `train.py` 写大量 `if experiment_name == ...` 分支。统一入口只负责根据 config 加载 spec。
- 旧实验目录里的 `train.py`、`eval_video.py`、`run_curriculum.py`、`run_tmux.py` 可以先保留为兼容 shim，最后再删除或标记废弃。

---

## 8. 目标结构

建议新增结构：

```text
isaac-training/
  experiments/
    train.py          # 统一训练入口
    eval.py           # 统一评估入口
    launch.py         # 统一前台/tmux/dry-run 启动入口
    campaign.py       # 统一多阶段 curriculum / sweep / eval-grid 编排入口

  experiment_specs/
    __init__.py
    registry.py
    tunnel.py
    tunnel_intent.py
    safety_shield.py
    tunnel_lagrangian.py

  src/
    core/
      runner.py
      evaluation.py
      checkpointing.py
      collector.py
      wandb_utils.py
      curriculum.py
      launching.py    # 可选：tmux/session/log helper，不能反向依赖 runner
      campaign.py     # 可选：stage graph/checkpoint marker helper
```

职责边界：
- `experiments/train.py`：Hydra 入口、启动 SimulationApp、加载 spec、调用 `run_training(cfg, spec)`。
- `experiments/eval.py`：Hydra 入口、启动 SimulationApp、加载 spec、加载 checkpoint、调用通用 evaluation helper。
- `experiments/launch.py`：把任意 train/eval/campaign 命令放进 tmux 或前台运行，写 log 和 wrapper script。
- `experiments/campaign.py`：按 stage/case/grid 逐个启动 train/eval，读取 checkpoint marker，把上一阶段 checkpoint 传给下一阶段。
- `experiment_specs/*.py`：只描述实验差异，包括 env factory、policy factory、dataset loader、adapter、hook。
- `src/core/runner.py`：不感知具体实验，不感知 tmux，不感知 campaign。

**为什么 `experiment_specs/` 与 `src/` 平行**
- `src/` 保持相对稳定的库代码：runtime、env、algo、simulated users、checkpoint/eval utilities。
- `experiment_specs/` 是实验装配层，会随着论文实验、ablation、算法变体、训练流程不断迭代。
- 把 spec 放在顶层可以降低对 `src` 稳定接口的扰动，也让“实验组合配置”和“可复用实现库”边界更清楚。

---

## 9. Spec registry 设计

统一入口不能直接 import 所有实验并写死分支。建议使用 registry：

```python
# experiment_specs/registry.py
SPEC_BUILDERS = {
    "tunnel": "experiment_specs.tunnel:build_spec",
    "tunnel_intent": "experiment_specs.tunnel_intent:build_spec",
    "safety_shield": "experiment_specs.safety_shield:build_spec",
    "tunnel_lagrangian": "experiment_specs.tunnel_lagrangian:build_spec",
}
```

配置文件指定：

```yaml
runtime:
  spec: tunnel
```

统一入口只做：

```python
from experiment_specs.registry import build_spec_from_cfg
from src.core.runner import run_training

spec = build_spec_from_cfg(cfg)
run_training(cfg, spec)
```

**验收标准**
- `experiments/train.py` 不包含 tunnel / intent / shield / lagrangian 专属分支。
- `src/core/runner.py` 不包含 tunnel / intent / shield / lagrangian 专属分支。
- 新增实验时只需要新增 `experiment_specs/<name>.py` 和配置里的 `runtime.spec`。

---

## 10. 统一 train.py 设计

统一 `experiments/train.py` 应该小于 250 行，理想情况下小于 120 行。

建议入口逻辑：
1. 设置 CUDA / PyTorch allocator 环境变量。
2. Hydra 加载 `configs/train.yaml`。
3. 校验 `runtime.spec`、`env`、`algo` 等必要配置。
4. 启动 `init_simulation_app(cfg)`。
5. `build_spec_from_cfg(cfg)`。
6. 调用 `run_training(cfg, spec)`。
7. finally 中关闭 SimulationApp。

建议命令：

```bash
python experiments/train.py experiment=tunnel
python experiments/train.py experiment=tunnel_lagrangian
python experiments/train.py experiment=safety_shield
python experiments/train.py experiment=tunnel_intent
python experiments/campaign.py campaign=tunnel_m2_diverse_pilot
```

旧入口策略：
- 历史 `experiments/*/train.py`、`eval_video.py`、`run_curriculum.py`、`run_matrix.py` 已移除。
- 项目脚本统一切到 `experiments/train.py` / `experiments/eval.py` / `experiments/campaign.py`。

**验收标准**
```bash
python3 -m py_compile experiments/train.py experiment_specs/*.py src/core/*.py

rg "if .*lagrangian|if .*tunnel|if .*safety|if .*intent" experiments/train.py src/core/runner.py
rg "wandb.init\(|def evaluate\(|SyncDataCollector\(|EpisodeStats\(" experiments/train.py
```

预期：
- 编译通过。
- `experiments/train.py` 和 `runner.py` 不出现实验专属分支。
- 训练循环相关重复实现不出现在统一入口里。

---

## 11. Curriculum / campaign 设计

课程学习要分成两类处理。

### A. 单次训练进程内的连续调度

适合 hook，例如：
- `RegCoeffSchedulerHook`
- residual regularization ramp
- 单个 policy 内部的温度、lambda、系数调度

这些逻辑运行在同一个训练进程中，保存在 `context["checkpoint_extra_state"]`。

### B. 多阶段训练 pipeline

不适合 hook，应由统一 `campaign.py` 处理，例如：
- stage1/2/3 使用不同障碍数量。
- stage1 训练完，把 best/final checkpoint 传给 stage2。
- 每个 stage 有不同 Hydra output dir / wandb group / seed。
- held-out eval grid 对同一 checkpoint 跑多组 obstacles/seed。

建议新增 campaign 配置，例如：

```yaml
campaign:
  name: tunnel_curriculum
  mode: train_stages
  checkpoint_policy: best   # best | final | latest
  stages:
    - name: stage1
      overrides:
        - experiment=tunnel
        - env.num_obstacles=30
        - curriculum.enable=false
    - name: stage2
      init_from_previous: true
      overrides:
        - env.num_obstacles=50
    - name: stage3
      init_from_previous: true
      overrides:
        - env.num_obstacles=80
        - curriculum.enable=true
```

建议命令：

```bash
python experiments/campaign.py campaign=tunnel_curriculum
python experiments/campaign.py campaign=tunnel_ablation_ours seed=42
python experiments/campaign.py campaign=tunnel_lagrangian_curriculum
```

**验收标准**
- 不再为每个实验维护单独 `run_curriculum.py`。
- 新增 stage 不需要新增一个 `configs/experiment/*_stageN.yaml`，除非确实有大量不可读 override。
- stage checkpoint 串接统一通过 `best_checkpoint_path.txt` / `final_checkpoint_path.txt` / `latest_checkpoint_path.txt`。
- campaign 只启动 `experiments/train.py` / `experiments/eval.py`，不直接 import env / policy / algo。

---

## 12. tmux / launch 设计

不要把 tmux 写进 `runner.py`。建议新增：

```text
experiments/launch.py
src/core/launching.py
```

`launch.py` 提供统一命令包装：

```bash
python experiments/launch.py campaign --tmux --session tunnel-m2 -- campaign=tunnel_m2_diverse_pilot
python experiments/launch.py eval --tmux --session eval-m2 -- experiment=tunnel eval.checkpoint=/path/to/ckpt
python experiments/launch.py campaign --tmux --session ablation -- campaign=tunnel_ablation_ours
python experiments/launch.py train --dry-run -- experiment=tunnel env_test_mode=true
```

建议功能：
- `--tmux`：后台启动。
- `--foreground`：默认前台运行。
- `--dry-run`：只打印命令和写 wrapper，不启动。
- `--attach`：启动后 attach。
- `--replace`：同名 session 存在时替换。
- `--log-dir`：统一保存 stdout/stderr。
- 自动生成 session name：`<mode>-<experiment>-<timestamp>`。

**为什么不默认所有非测试训练都 tmux 后台启动**
- debug 和 Hydra 错误会更难定位。
- Slurm/容器/IDE 里可能没有 tmux。
- 用户显式选择 `--tmux` 更清晰，也更符合进程管理边界。

**验收标准**
```bash
python experiments/launch.py train --dry-run -- experiment=tunnel env_test_mode=true
python experiments/launch.py eval --dry-run -- experiment=tunnel eval.checkpoint=/tmp/foo.pt
python experiments/launch.py campaign --dry-run -- campaign=tunnel_curriculum
```

预期：
- dry-run 能打印完整命令。
- 不启动 Isaac Sim。
- 不 import env / algo。
- tmux helper 代码不出现在 `runner.py`。

---

## 13. 统一 eval.py 设计

新增 `experiments/eval.py`，替代各实验重复的 `eval_video.py`、`run_heldout_eval.py`。

入口职责：
1. Hydra 加载 config。
2. 强制评估友好设置：`wandb.mode=disabled`、`record_video=true/false`、较小 `env.num_envs`。
3. 通过 `runtime.spec` 加载同一个 `ExperimentSpec`。
4. 构建 env / policy / dataset。
5. 通过 `checkpointing.py` 加载 checkpoint policy weights。
6. 调用 `src/core/evaluation.py` 运行 deterministic rollout。
7. 写出 `eval_info.json`、视频文件、可选 wandb artifact。

建议配置：

```yaml
eval:
  checkpoint: null
  output_dir: ./eval_outputs
  seed: 42
  num_envs: 4
  record_video: true
  global_view: false
  keep_num_envs: false
  grid: null
```

建议命令：

```bash
python experiments/eval.py experiment=tunnel eval.checkpoint=/path/to/checkpoint_best.pt
python experiments/eval.py experiment=tunnel_lagrangian eval.checkpoint=/path/to/checkpoint_best.pt eval.grid=heldout_tunnel
python experiments/eval.py experiment=tunnel eval.checkpoint=/path/to/checkpoint_best.pt eval.global_view=true
```

**需要补的 core 能力**
- `evaluation.py` 当前偏向 wandb video logging；需要增加“保存 mp4 到磁盘”的 helper。
- `checkpointing.py` 需要提供 `load_policy_for_eval(path, policy, cfg, adapter)`，只加载 policy，不恢复 optimizer。
- `ExperimentSpec` 可增加可选 `eval_hooks` 或继续复用 `hooks` 的 `on_before_eval/on_after_eval`。

**验收标准**
```bash
python3 -m py_compile experiments/eval.py src/core/evaluation.py src/core/checkpointing.py experiment_specs/*.py

rg "eval_video.py|run_heldout_eval.py" experiments -g "*.py"
python experiments/eval.py --cfg job experiment=tunnel eval.checkpoint=/tmp/foo.pt
```

预期：
- 统一 eval 入口可以 compose config。
- 旧 eval 脚本逐步降级为 shim 或删除。
- tunnel 和 lagrangian 都能通过同一入口构建 policy 并加载 checkpoint。

---

## 14. 用 Lagrangian 验证统一入口

`04s_tunnel_lagrangian` 是统一入口设计的关键验证，因为它算法和 checkpoint state 都和 tunnel 主线不同。

迁移目标：
- 不再从 `experiments/04s_tunnel_lagrangian/train.py` 作为专有入口启动训练。
- 从 `experiments/train.py experiment=tunnel_lagrangian` 启动。
- `experiments/train.py` 和 `src/core/runner.py` 都没有 Lagrangian 专属 if 分支。
- Lagrangian 差异全部在 `experiment_specs/tunnel_lagrangian.py`、adapter、hook 中。

建议拆分：
- `LagrangianCheckpointAdapter`
  - 兼容加载 `lambda_lag`。
  - 恢复 / 保存 `lambda_optimizer`。
  - 保存 Lagrangian 特定 rich checkpoint state。
- `lagrangian_eval_summary_fn`
  - 定义 Lagrangian best checkpoint 的 score / rank。
- `LagrangianRuntimeHook`
  - 如果训练 step 前后需要额外日志、约束统计或 lambda 调度，放在 hook。
- `policy_factory`
  - 直接加载 `src/algos/ppo_constrained_beta_lagrangian.py`。

验证命令：

```bash
python experiments/train.py experiment=tunnel_lagrangian env_test_mode=true record_video=false env.num_envs=1
python experiments/train.py experiment=tunnel_lagrangian max_iterations=2 eval_interval=1 save_interval=1 record_video=false env.num_envs=4
python experiments/eval.py experiment=tunnel_lagrangian eval.checkpoint=/path/to/checkpoint_best.pt eval.record_video=false
```

验收标准：
- env test 可以启动并退出。
- 2 iteration smoke test 能写出 latest/final/best marker。
- checkpoint 中包含 Lagrangian 需要的 optimizer / lambda state。
- resume checkpoint 可以从 latest rich checkpoint 继续。

---

## 15. 配置收敛策略

短期保留 `configs/experiment/*.yaml`，但减少 stage 配置数量。

推荐规则：
- “实验身份”放在 `configs/experiment/<experiment>.yaml`。
- “一次训练运行的微调”用 CLI override 或 campaign stage override。
- “多阶段顺序”放在 `configs/campaign/*.yaml`。
- `runtime.spec` 是每个实验配置必须显式声明的字段。
- `env.name` 只描述环境类型，不承担 spec 选择职责。

示例：

```yaml
# configs/experiment/tunnel.yaml
runtime:
  spec: tunnel

env:
  name: tunnel

algo:
  distribution: beta
  policy_mode: residual
```

```yaml
# configs/experiment/tunnel_lagrangian.yaml
runtime:
  spec: tunnel_lagrangian

env:
  name: tunnel_lagrangian

algo:
  distribution: beta
  policy_mode: residual
```

验收标准：
```bash
rg "runtime:" configs/experiment/tunnel.yaml configs/experiment/tunnel_lagrangian.yaml
rg "runtime.spec" configs/experiment
```

预期：
- 所有迁移到统一入口的实验配置都有 `runtime.spec`。
- stage yaml 数量开始下降，新增 stage 优先进入 `configs/campaign/*.yaml`。

---

## 16. 迁移顺序

建议按下面 commit 节奏推进。

### Commit A: 抽 tunnel spec 到 registry

操作：
- 新建 `experiment_specs/`。
- 把当前 `experiments/04_tunnel_task/train.py` 中的 `ResidualPolicySanityCheckHook`、dataset loader、env factory、policy factory、`build_spec()` 移到 `experiment_specs/tunnel.py`。
- 新建 `experiment_specs/registry.py`。
- `experiments/04_tunnel_task/train.py` 改为从 registry 加载 tunnel spec。

验证：
```bash
python3 -m py_compile experiments/04_tunnel_task/train.py experiment_specs/*.py src/core/*.py
python experiments/04_tunnel_task/train.py --cfg job experiment=tunnel | rg "runtime:|env:|algo:"
```

### Commit B: 新增统一 `experiments/train.py`

操作：
- 新建统一入口。
- `configs/experiment/tunnel.yaml` 增加 `runtime.spec: tunnel`。
- 旧 `04_tunnel_task/train.py` 可以保留为 shim。

验证：
```bash
python3 -m py_compile experiments/train.py
python experiments/train.py --cfg job experiment=tunnel | rg "runtime:|env:|algo:"
python experiments/train.py experiment=tunnel env_test_mode=true record_video=false env.num_envs=1
```

### Commit C: 用 Lagrangian 验证 spec 扩展

操作：
- 新建 `experiment_specs/tunnel_lagrangian.py`。
- 实现 `LagrangianCheckpointAdapter` / eval summary / hook。
- `configs/experiment/tunnel_lagrangian.yaml` 增加 `runtime.spec: tunnel_lagrangian`。

验证：
```bash
python3 -m py_compile experiment_specs/tunnel_lagrangian.py experiments/train.py
python experiments/train.py experiment=tunnel_lagrangian env_test_mode=true record_video=false env.num_envs=1
```

### Commit D: 新增统一 eval 入口

操作：
- 新建 `experiments/eval.py`。
- 补 `evaluation.py` 的磁盘视频保存 helper。
- 补 checkpoint eval-only loader。

验证：
```bash
python3 -m py_compile experiments/eval.py src/core/evaluation.py src/core/checkpointing.py
python experiments/eval.py --cfg job experiment=tunnel eval.checkpoint=/tmp/checkpoint.pt
```

### Commit E: 新增统一 launch 入口

操作：
- 新建 `src/core/launching.py`。
- 新建 `experiments/launch.py`。
- 从 `experiments/04a_tunnel_ablation/run_tmux.py` 迁移通用 tmux helper。

验证：
```bash
python experiments/launch.py train --dry-run -- experiment=tunnel env_test_mode=true
python experiments/launch.py eval --dry-run -- experiment=tunnel eval.checkpoint=/tmp/checkpoint.pt
```

### Commit F: 新增 campaign 入口并迁移 curriculum

操作：
- 新建 `configs/campaign/`。
- 新建 `experiments/campaign.py`。
- 把 `run_curriculum.py`、`run_matrix.py`、`run_heldout_eval.py` 中通用的 stage/checkpoint/grid 逻辑收敛。

验证：
```bash
python experiments/campaign.py --cfg job campaign=tunnel_curriculum
python experiments/campaign.py campaign=tunnel_curriculum --dry-run
```

### Commit G: 清理旧 wrapper

操作：
- 旧脚本转成 deprecated shim 或删除。
- 更新 README / run scripts。
- 删除明显重复的 eval/curriculum/tmux 包装。

验证：
```bash
rg "wandb.init\(|def evaluate\(|SyncDataCollector\(|EpisodeStats\(" experiments -g "train.py"
rg "eval_video.py|run_heldout_eval.py|run_curriculum.py|run_tmux.py" experiments -g "*.py"
```

---

## 17. 最终验收标准

### A. 统一入口

```bash
test -f experiments/train.py
test -f experiments/eval.py
test -f experiments/launch.py
test -f experiments/campaign.py
test -d experiment_specs
```

### B. runtime 边界

```bash
rg "tmux|subprocess|new-session" src/core/runner.py
rg "tunnel_lagrangian|safety_shield|tunnel_intent|04_tunnel" src/core/runner.py experiments/train.py
```

预期：
- 第一条不命中。
- 第二条不命中，或只出现在文档/注释之外的 registry 配置中。

### C. 训练重复消除

```bash
rg "wandb.init\(" experiments -g "train.py"
rg "def evaluate\(" experiments -g "train.py"
rg "SyncDataCollector\(" experiments -g "train.py"
rg "EpisodeStats\(" experiments -g "train.py"
```

预期：
- 统一入口没有这些重复实现。
- 旧入口如果暂时保留，也应是 shim，不再包含完整训练循环。

### D. eval 重复消除

```bash
rg "RenderCallback|imageio|eval_info.json|eval_video" experiments -g "*.py"
```

预期：
- 视频评估逻辑主要集中在 `experiments/eval.py` 和 `src/core/evaluation.py`。

### E. config 收敛

```bash
rg "runtime.spec" configs/experiment
find configs/campaign -maxdepth 1 -type f | sort
```

预期：
- 已迁移实验都显式声明 `runtime.spec`。
- 多阶段逻辑开始进入 `configs/campaign/`。

### F. smoke tests

```bash
python3 -m py_compile \
  experiments/train.py \
  experiments/eval.py \
  experiments/launch.py \
  experiments/campaign.py \
  src/core/*.py \
  experiment_specs/*.py

python experiments/train.py --cfg job experiment=tunnel
python experiments/train.py --cfg job experiment=tunnel_lagrangian
python experiments/launch.py train --dry-run -- experiment=tunnel env_test_mode=true
```

需要真实 Isaac 环境验证：
```bash
python experiments/train.py experiment=tunnel env_test_mode=true record_video=false env.num_envs=1
python experiments/train.py experiment=tunnel_lagrangian env_test_mode=true record_video=false env.num_envs=1
```

---

## 18. 控制文件体量

建议目标：

| 文件 | 目标 |
| --- | --- |
| `experiments/train.py` | `< 120` 行 |
| `experiments/eval.py` | `< 220` 行 |
| `experiments/launch.py` | `< 250` 行 |
| `experiments/campaign.py` | `< 300` 行 |
| `experiment_specs/tunnel.py` | `< 220` 行 |
| `experiment_specs/tunnel_lagrangian.py` | `< 300` 行 |

如果超过目标，优先把通用逻辑下沉到 `src/core/`，把实验差异留在 `experiment_specs/`。

---
