# 宿主机批量 Docker 仿真环境配置指南

本文说明如何在一台只安装了 Docker 的新主机上，配置到可以像本机一样从宿主机运行：

```bash
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py ...
```

该脚本是 **宿主机 orchestrator**：宿主机只负责调用 Docker Compose、分批启动一次性容器、收集日志和结果；ROS1、Gazebo、PyTorch、`navigation_runner`、`uav_simulator`、`map_manager`、`onboard_detector`、IPC/slope 依赖都在容器内运行。

## 1. 总体结构

```text
宿主机
  ├─ Docker Engine / docker compose
  ├─ Python3 标准库运行 run_tunnel_batch_containers.py
  ├─ SharedRLControl 仓库
  ├─ slope_inspection 仓库或源码目录
  └─ ros1/results/             # bind mount，保存所有实验结果

容器 tunnel_batch
  ├─ /root/catkin_ws/src/navigation_runner
  ├─ /root/catkin_ws/src/uav_simulator
  ├─ /root/catkin_ws/src/map_manager
  ├─ /root/catkin_ws/src/onboard_detector
  ├─ /root/slope_ws/src         # 从宿主机挂载 slope_inspection
  ├─ /root/results              # 映射到宿主 ros1/results
  └─ Xvfb + Gazebo headless + ROS Noetic
```

`run_tunnel_batch_containers.py` 本身不依赖 ROS 和 PyTorch；它只用 Python 标准库和 Docker CLI。真正的 `navigation_runner` 编译环境在 CPU 镜像 `tunnel_comparison:20260415-ipcfix-cpu` 中。

## 2. 宿主机基础依赖

建议宿主机使用 Ubuntu 20.04/22.04 x86_64。必须安装：

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip ca-certificates curl
```

安装 Docker Engine 和 Compose plugin 后确认：

```bash
docker --version
docker compose version
docker run --rm hello-world
```

当前 `docker-compose.tunnel.yml` 默认使用 CPU 镜像，不再声明 `runtime: nvidia`，因此没有 NVIDIA GPU 的主机也可以直接运行。宿主机无需安装 NVIDIA driver 或 NVIDIA Container Toolkit。

## 3. 宿主机 Python 环境

宿主机只需要 Python3 标准库。推荐使用 venv 只是为了固定命令入口，不需要安装 ROS/PyTorch：

```bash
cd /path/to/SharedRLControl
python3 -m venv .venv-host
source .venv-host/bin/activate
python3 -V
```

无需 `pip install -r ...`。脚本使用的模块是 `argparse/json/os/shlex/subprocess/sys/time/datetime`，均为标准库。

可先检查脚本参数：

```bash
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py --help
```

## 4. 准备源码目录

推荐目录布局：

```text
/data/SharedRLControl
/data/slope_inspection
```

获取仓库：

```bash
cd /data
git clone <SharedRLControl repo url> SharedRLControl
git clone <slope_inspection repo url> slope_inspection
```

如果 `slope_inspection` 在其他路径，运行前设置：

```bash
export SLOPE_INSPECTION_HOST_PATH=/absolute/path/to/slope_inspection
```

`docker-compose.tunnel.yml` 默认挂载：

```yaml
${SLOPE_INSPECTION_HOST_PATH:-../slope_inspection}:/root/slope_ws/src
```

因此如果 `SharedRLControl` 和 `slope_inspection` 是兄弟目录，通常不用额外设置。

## 5. 准备 Docker 镜像

### 5.1 推荐：从已有机器导出/导入镜像

本项目当前推荐使用 CPU 镜像：

```text
tunnel_comparison:20260415-ipcfix-cpu
```

在已可运行的机器上导出：

```bash
docker save tunnel_comparison:20260415-ipcfix-cpu | gzip > tunnel_comparison_20260415-ipcfix_cpu.tar.gz
```

拷贝到新主机后导入：

```bash
gunzip -c tunnel_comparison_20260415-ipcfix_cpu.tar.gz | docker load
docker images | grep tunnel_comparison
```

如果导入后需要校验压缩包完整性：

```bash
sha256sum -c tunnel_comparison_20260415-ipcfix_cpu.tar.gz.sha256
```

这是最稳妥的方式，因为该镜像已经包含 ROS Noetic、Gazebo、PyTorch、CERLAB 仿真和 IPC 修复。

### 5.2 从 Dockerfile 构建

如果必须在新主机构建，注意 `Dockerfile.tunnel_comparison` 的基础镜像是：

```dockerfile
FROM slope_inspection:test
```

因此必须先准备 `slope_inspection:test`。完成后在仓库根目录执行：

```bash
cd /data/SharedRLControl
docker build -f Dockerfile.tunnel_comparison -t tunnel_comparison:20260415-ipcfix-cpu .
```

构建会把以下 ROS1 包复制进 `/root/catkin_ws/src` 并编译：

- `uav_simulator`
- `map_manager`
- `onboard_detector`
- `navigation_runner`

如果只是运行现有镜像，宿主机不需要 catkin/ROS 编译环境。

## 6. navigation_runner 编译与挂载逻辑

镜像构建时已经执行：

```bash
source /opt/ros/noetic/setup.bash
source /root/slope_ws/devel/setup.bash
cd /root/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

运行容器时，compose 会把宿主机的这些目录覆盖到容器内：

```text
./ros1/navigation_runner/scripts -> /root/catkin_ws/src/navigation_runner/scripts
./ros1/navigation_runner/cfg     -> /root/catkin_ws/src/navigation_runner/cfg
./ros1/navigation_runner/launch  -> /root/catkin_ws/src/navigation_runner/launch
./ros1/navigation_runner/docs    -> /root/catkin_ws/src/navigation_runner/docs
./ros1/uav_simulator/worlds      -> /root/catkin_ws/src/uav_simulator/worlds
```

因此：

- 修改 Python 脚本、YAML、launch、world/PCD 通常不需要重建镜像。
- 修改 C++、`.srv`、`package.xml`、`CMakeLists.txt` 后，需要进入容器重新 `catkin_make` 或重建镜像。
- `run_tunnel_batch_containers.py` 默认会在第一次 batch 前重编译挂载的 `/root/slope_ws` IPC 相关包，但不会重编译 `/root/catkin_ws`。

如需手动验证容器内 `navigation_runner`：

```bash
docker compose -f docker-compose.tunnel.yml run --rm tunnel_batch bash

source /opt/ros/noetic/setup.bash
source /root/slope_ws/devel/setup.bash
source /root/catkin_ws/devel/setup.bash
rospack find navigation_runner
roslaunch --nodes navigation_runner tunnel_comparison.launch
```

如需重编译 catkin workspace：

```bash
docker compose -f docker-compose.tunnel.yml run --rm tunnel_batch bash

source /opt/ros/noetic/setup.bash
source /root/slope_ws/devel/setup.bash
cd /root/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash
```

若希望重编译结果长期保留，建议重建镜像；默认 compose 没有把 `/root/catkin_ws/build` 和 `/root/catkin_ws/devel` 做 named volume。

## 7. 首次 smoke test

在仓库根目录：

```bash
cd /data/SharedRLControl
mkdir -p ros1/results
```

先只打印 Docker 命令，不运行：

```bash
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
    --num-batches 1 \
    --output-dir /root/results/smoke_dryrun \
    --methods rl \
    --master-seed 5716 \
    --runs-per-batch 1 \
    --no-rebuild-slope
```

真正运行一个最小 RL batch：

```bash
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
    --run \
    --num-batches 1 \
    --output-dir /root/results/smoke_rl_seed5716 \
    --methods rl \
    --master-seed 5716 \
    --runs-per-batch 1 \
    --launch-timeout 100 \
    --run-retries 1 \
    --no-rebuild-slope
```

说明：

- `--no-rebuild-slope` 适合只跑 `--methods rl` 的 smoke test。
- 如果要跑 `ipc` 或 `rl,ipc`，不要加 `--no-rebuild-slope`，让脚本先构建 `/root/slope_ws`。
- 输出会写入宿主机 `ros1/results/smoke_rl_seed5716`。

检查结果：

```bash
find ros1/results/smoke_rl_seed5716 -maxdepth 3 -type f | sort | sed -n '1,80p'
cat ros1/results/smoke_rl_seed5716/host_logs/*.status.json
```

## 8. RL + IPC 完整 batch 示例

确认 `slope_inspection` 挂载正确后运行：

```bash
export SLOPE_INSPECTION_HOST_PATH=/data/slope_inspection

python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
    --run \
    --num-batches 10 \
    --output-dir /root/results/replay_h5_mapconstrained_seed5716 \
    --methods rl,ipc \
    --master-seed 5716 \
    --runs-per-batch 10 \
    --input-source offline \
    --replay-dataset-path /root/catkin_ws/src/navigation_runner/cfg/ckpts/trajectories_tunnel.h5 \
    --replay-start-offset 0 \
    --map-sampling-mode constrained \
    --min-obstacle-spacing 0.6 \
    --local-density-window 3.0 \
    --max-obstacles-per-window 3 \
    --max-local-area-fraction 0.35 \
    --require-connectivity \
    --gazebo-z-mode policy_clamped \
    --gazebo-policy-z-max 0.50 \
    --safety-min-dist 0.35 \
    --launch-timeout 100 \
    --run-retries 1
```

默认行为：

- 先启动一个 `tunnel_batch_slope_build` 容器，白名单编译 IPC 相关包。
- 然后每个 batch 启动一个一次性容器。
- 全部 batch 成功后再启动 `tunnel_batch_analysis` 容器。
- 日志写入 `<output>/host_logs/`。

## 9. 续跑与单 batch 重跑

续跑已有输出：

```bash
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
    --run \
    --resume-from /root/results/replay_h5_mapconstrained_seed5716 \
    --num-batches 10 \
    --start-batch 7 \
    --end-batch 9 \
    --methods rl,ipc \
    --master-seed 5716 \
    --runs-per-batch 10 \
    --run-retries 1
```

只重跑某个 batch：

```bash
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
    --run \
    --resume-from /root/results/replay_h5_mapconstrained_seed5716 \
    --num-batches 10 \
    --batch-index 3 \
    --methods rl,ipc \
    --master-seed 5716 \
    --runs-per-batch 10 \
    --run-retries 1
```

## 10. 手动 debug 容器

只在需要 Gazebo GUI/RViz 时使用：

```bash
cd /data/SharedRLControl
xhost +local:docker
docker compose -f docker-compose.tunnel.yml --profile debug up -d tunnel_debug
docker exec -it tunnel_debug bash
```

容器内：

```bash
source /opt/ros/noetic/setup.bash
source /root/slope_ws/devel/setup.bash
source /root/catkin_ws/devel/setup.bash

roslaunch navigation_runner tunnel_comparison.launch method:=rl gui:=true rviz:=true
```

headless batch 不需要 X11，也不需要 `xhost`。

## 11. 常见问题

### `docker compose` 找不到

安装 Compose plugin，不要依赖旧版 `docker-compose`：

```bash
sudo apt-get install docker-compose-plugin
docker compose version
```

### 仍然出现 `Unknown runtime specified nvidia`

说明目标主机使用的不是当前 CPU 版 `docker-compose.tunnel.yml`。更新仓库后确认：

```bash
docker compose -f docker-compose.tunnel.yml config | grep -E 'image:|runtime:|NVIDIA'
```

正常情况下只应看到 `tunnel_comparison:20260415-ipcfix-cpu`，不应再出现 `runtime: nvidia` 或 `NVIDIA_*` 环境变量。

### `/root/slope_ws/devel/setup.bash` 不存在

说明 IPC/slope workspace 没构建，或 named volume 是空的。运行 `rl,ipc` 时不要加 `--no-rebuild-slope`，让脚本执行默认的 slope rebuild；只跑 RL smoke test 时可以用 `--no-rebuild-slope`，但镜像本身仍应包含基础 `/root/slope_ws/devel`。

### `slope_inspection` 路径不对

设置绝对路径：

```bash
export SLOPE_INSPECTION_HOST_PATH=/absolute/path/to/slope_inspection
docker compose -f docker-compose.tunnel.yml config | grep slope_ws -n
```

### Python 脚本提示输出目录非法

`--output-dir` 必须是容器 `/root/results/...` 或宿主仓库 `ros1/results/...` 下的路径。推荐始终使用：

```bash
--output-dir /root/results/<experiment_name>
```

### Gazebo headless 启动失败

确认容器 entrypoint 启动了 Xvfb；检查 batch 日志：

```bash
tail -n 200 ros1/results/<experiment>/host_logs/batch_000.docker.log
```

必要时进入容器手动检查：

```bash
docker compose -f docker-compose.tunnel.yml run --rm tunnel_batch bash
echo $DISPLAY
glxinfo -B || true
```

### 修改了 C++ 或 srv 后不生效

重建镜像或进入容器重编译 `/root/catkin_ws`。Python/launch/yaml 修改会通过 bind mount 立即生效。

## 12. 最小可复现检查清单

在新主机上满足以下条件后，即可认为批量容器环境可用：

1. `docker compose version` 正常。
2. `docker images` 中存在 `tunnel_comparison:20260415-ipcfix-cpu`。
3. `python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py --help` 正常。
4. `docker compose -f docker-compose.tunnel.yml run --rm tunnel_batch bash -lc 'source /opt/ros/noetic/setup.bash && source /root/slope_ws/devel/setup.bash && source /root/catkin_ws/devel/setup.bash && rospack find navigation_runner'` 能输出 `/root/catkin_ws/src/navigation_runner`。
5. RL smoke batch 成功生成 `ros1/results/smoke_rl_seed5716/host_logs/*.status.json`，且 returncode 为 0。
6. 如果要跑 IPC，默认 slope build 成功，`slope_build.status.json` returncode 为 0。
