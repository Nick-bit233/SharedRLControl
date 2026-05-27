# Experiments

Unified experiment entrypoints live at the top of this directory. Historical
per-experiment scripts are compatibility shims only.

## 入口职责

- `experiments/train.py`：启动一次训练进程。
- `experiments/eval.py`：加载 checkpoint，运行评估并写出结果。
- `experiments/campaign.py`：按 campaign 配置串联多阶段课程训练。
- `experiments/launch.py`：统一封装 train/eval/campaign 的前台、tmux、dry-run 启动方式。

建议所有命令都从仓库的训练目录运行：

```bash
cd SharedRLControl/isaac-training
```

建议使用uv管理python环境：

```bash
source /path/to/env_isaaclab/bin/activate
```

下面示例用 `python` 表示当前已激活的项目 Python。

## Hydra 参数覆写

`train.py`、`eval.py` 和 `campaign.py` 都使用 Hydra 配置。`launch.py` 会把 `--` 后面的内容原样传给对应入口，因此配置和参数覆写都写在 `--` 后面。

常用覆写格式：

```bash
# 选择单次训练配置
experiment=tunnel
experiment=tunnel_lagrangian
# 选择课程配置
campaign=tunnel_curriculum
# 选择wandb模式
wandb.mode=online
wandb.mode=disabled
# 是否开启视频录制
record_video=false
```

注意：

- `experiment=<name>` 选择 `configs/experiment/<name>.yaml`。
- `campaign=<name>` 选择 `configs/campaign/<name>.yaml`。
- 嵌套字段用点号覆写，例如 `env.num_envs=4`、`algo.reg_coeff=0.0`。
- 需要新增不存在的 Hydra 字段时使用 `+key=value`，普通已有字段不要加 `+`。
- `launch.py` 自身参数写在 `--` 前，Hydra 覆写写在 `--` 后。

## 直接运行

调试时可以直接运行`train.py`，不经过 `launch.py`。

```bash
python experiments/train.py experiment=tunnel wandb.mode=disabled
```

快速环境检查：

```bash
python experiments/train.py \
  experiment=tunnel \
  env_test_mode=true \
  record_video=false \
  env.num_envs=1 \
  wandb.mode=disabled
```

Lagrangian 训练入口：

```bash
python experiments/train.py \
  experiment=tunnel_lagrangian \
  wandb.mode=disabled
```

## 使用 launch.py 启动训练

`launch.py train` 会构造 `experiments/train.py ...` 命令，并写出 wrapper 和 log 文件到 `outputs/launch_logs/`。

前台 dry-run，只打印命令、不启动 Isaac Sim：

```bash
python experiments/launch.py train --dry-run -- \
  experiment=tunnel \
  env_test_mode=true
```

前台真实启动：

```bash
python experiments/launch.py train --foreground -- \
  experiment=tunnel \
  wandb.mode=disabled
```

后台 tmux 启动并自动 attach：

```bash
python experiments/launch.py train --tmux --attach --session tunnel-m2 -- \
  experiment=tunnel_m2_diverse_pilot \
  wandb.mode=offline \
  record_video=false
```

替换已有同名 tmux session：

```bash
python experiments/launch.py train --tmux --replace --session tunnel-debug -- \
  experiment=tunnel \
  max_iterations=2 \
  eval_interval=1 \
  save_interval=1 \
  env.num_envs=4 \
  record_video=false
```

查看后台任务：

```bash
tmux ls
tmux attach -t tunnel-m2
```

## 使用 launch.py 启动评估

评估入口要求提供 `eval.checkpoint`。默认结果写到 `eval.output_dir`，其中包括 `eval_info.json`；开启视频时也会写出 mp4。

评估 dry-run：

```bash
python experiments/launch.py eval --dry-run -- \
  experiment=tunnel \
  eval.checkpoint=/path/to/checkpoint_best.pt
```

前台评估，不录视频：

```bash
python experiments/launch.py eval --foreground -- \
  experiment=tunnel \
  eval.checkpoint=/path/to/checkpoint_best.pt \
  eval.output_dir=./eval_outputs/tunnel_best_seed42 \
  eval.record_video=false \
  eval.seed=42 \
  eval.num_envs=4
```

Lagrangian checkpoint 评估：

```bash
python experiments/launch.py eval --foreground -- \
  experiment=tunnel_lagrangian \
  eval.checkpoint=/path/to/checkpoint_best.pt \
  eval.output_dir=./eval_outputs/lagrangian_best \
  eval.record_video=false
```

保持配置中的 `env.num_envs`，用于更稳定的批量指标：

```bash
python experiments/launch.py eval --foreground -- \
  experiment=tunnel \
  eval.checkpoint=/path/to/checkpoint_best.pt \
  eval.keep_num_envs=true \
  env.num_envs=256 \
  eval.record_video=false
```

## 使用 launch.py 启动 campaign 课程训练

`campaign.py` 负责多阶段训练编排。当前示例配置在 `configs/campaign/tunnel_curriculum.yaml`，会依次运行 stage1、stage2、stage3，并用上一阶段的 checkpoint 初始化下一阶段。

campaign dry-run：

```bash
python experiments/launch.py campaign --dry-run -- \
  campaign=tunnel_curriculum
```

前台运行 campaign：

```bash
python experiments/launch.py campaign --foreground -- \
  campaign=tunnel_curriculum
```

后台 tmux 运行 campaign：

```bash
python experiments/launch.py campaign --tmux --session tunnel-curriculum -- \
  campaign=tunnel_curriculum
```

直接调用 campaign 并 dry-run 查看每个 stage 的训练命令：

```bash
python experiments/campaign.py campaign=tunnel_curriculum --dry-run
```

campaign 配置中常见字段：

- `campaign.name`：campaign 名称。
- `campaign.mode`：当前支持 `train_stages`。
- `campaign.checkpoint_policy`：阶段串接使用的 checkpoint，可选 `best`、`final`、`latest`。
- `campaign.checkpoint_override`：传给下一阶段的参数名，默认 `init_checkpoint`。
- `campaign.output_root`：campaign 输出根目录。
- `campaign.stages[].overrides`：每个 stage 传给 `experiments/train.py` 的 Hydra 覆写。
- `campaign.stages[].init_from_previous`：是否使用上一阶段 checkpoint 初始化。

## launch.py 参数速查

`launch.py` 支持三个模式：

```bash
python experiments/launch.py train ...
python experiments/launch.py eval ...
python experiments/launch.py campaign ...
```

通用参数：

- `--foreground`：前台运行。
- `--tmux`：启动 tmux 后台 session。
- `--dry-run`：只打印命令并写 wrapper，不真正启动。
- `--attach`：tmux 启动后自动 attach。
- `--replace`：同名 tmux session 已存在时先替换。
- `--session <name>`：手动指定 tmux session 名称。
- `--log-dir <dir>`：指定 wrapper 和日志目录，默认 `outputs/launch_logs`。

示例 wrapper 和日志路径：

```text
outputs/launch_logs/train-tunnel-YYYYMMDD_HHMMSS.sh
outputs/launch_logs/train-tunnel-YYYYMMDD_HHMMSS.log
```
