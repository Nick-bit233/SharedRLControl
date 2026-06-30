运行 ROS1 隧道对比实验
======================

默认推荐从宿主机启动 **headless batch-by-batch** 工作流。长时间批量实验不要再用
`docker exec tunnel_debug ...` 或默认挂载宿主 X11 显示器；手动可视化调试时再显式启动
`tunnel_debug`。

新主机从零配置 Docker、Python、镜像和 `navigation_runner` 编译环境时，先参考
`navigation_runner/docs/BATCH_CONTAINER_HOST_SETUP.md`。

## 方式一：宿主机长时间批量实验（推荐）

以下命令会为每个 batch 启动一个一次性 `tunnel_batch` 容器，默认不挂载宿主显示器，
由容器内 Xvfb + software rendering 支撑 Gazebo headless 运行。日志写入结果目录下的
`host_logs/`。

```bash
cd /home/haoming/wht/IsaacLab_drones_5.1/SharedRLControl

python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
     --run \
     --num-batches 10 \
     --output-dir /root/results/batch_replay_h5_mapconstrained_seed5716 \
     --methods rl \
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

use other checkpoints (with same actor net structure)
- 最终实验配置（master seed，地图生成参数不再改变）

主要模型
```python
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
     --run \
     --checkpoint /root/catkin_ws/src/navigation_runner/cfg/ckpts/checkpoint_tunnel_M3_21500.pt \
     --num-batches 100 \
     --output-dir /root/results/batch_1000_for_report_04m3ours_seed5716 \
     --methods rl \
     --master-seed 5716 \
     --runs-per-batch 10 \
     --input-source offline \
     --replay-dataset-path /root/catkin_ws/src/navigation_runner/cfg/ckpts/trajectories_tunnel.h5 \
     --replay-start-offset 0 \
     --map-sampling-mode constrained \
     --min-obstacle-spacing 1.0 \
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
- ipc

````python
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
     --run \
     --num-batches 100 \
     --output-dir /root/results/batch_1k_REPORT_ipc_seed5716 \
     --methods ipc \
     --master-seed 5716 \
     --runs-per-batch 10 \
     --input-source offline \
     --replay-dataset-path /root/catkin_ws/src/navigation_runner/cfg/ckpts/trajectories_tunnel.h5 \
     --replay-start-offset 0 \
     --map-sampling-mode constrained \
     --min-obstacle-spacing 1.0 \
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

- 零安全基线

````python
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
     --run \
     --num-batches 100 \
     --output-dir /root/results/batch_1k_REPORT_naive_baseline_seed5716 \
     --methods naive_raw,naive_safe \
     --master-seed 5716 \
     --runs-per-batch 10 \
     --input-source offline \
     --replay-dataset-path /root/catkin_ws/src/navigation_runner/cfg/ckpts/trajectories_tunnel.h5 \
     --replay-start-offset 0 \
     --map-sampling-mode constrained \
     --min-obstacle-spacing 1.0 \
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


[TODO] use other checkpoints with no residual (for ablation)

如果输出目录已经存在，runner 会自动让后续 batch 使用 `--resume-from`，复用
`batch_config.json`、地图和 seed plan，并跳过完整 run。

只恢复某个 batch：

```bash
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
     --run \
     --resume-from /root/results/batch_replay_h5_mapconstrained_seed5716 \
     --num-batches 10 \
     --start-batch 7 \
     --end-batch 9
     --methods rl,ipc \
     --master-seed 5716 \
     --runs-per-batch 10 \
     --run-retries 1
```

先检查 Docker 命令但不运行：

```bash
python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
     --num-batches 10 \
     --output-dir /root/results/dryrun_example \
     --methods rl,ipc \
     --master-seed 5716 \
     --runs-per-batch 10
```

## 方式二：容器内单次/小规模 headless 测试

```bash
docker compose -f docker-compose.tunnel.yml run --rm tunnel_batch bash

python3 /root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py \
     --num-batches 1 \
     --runs-per-batch 1 \
     --methods rl,ipc \
     --output-dir /root/results/smoke_headless

# 从中断的批量结果原地续跑；会复用已有 batch_config/地图/seed，跳过完整轨迹并重跑缺失或损坏的 run
python3 /root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py \
     --resume-from /root/catkin_ws/results/batch_20260427_142203 \
     --master-seed 325

# 100-run safety_min_dist pilot: minimal safety-margin ablation, do not overwrite baseline results
python3 /root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py \
     --num-batches 10 \
     --runs-per-batch 10 \
     --safety-min-dist 0.30 \
     --launch-timeout 100 \
     --output-dir /root/catkin_ws/results/batch_safety030_pilot \
     --master-seed 325 \
     --gazebo-z-mode policy
```

`batch_tunnel_experiments.py` 支持只运行完整 seed plan 中的某个 batch：

```bash
python3 /root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py \
     --resume-from /root/results/replay_h5_mapconstrained_seed5716 \
     --batch-index 1 \
     --run-retries 1
```

## 方式三：手动可视化调试（显式 X11）

只有需要 Gazebo GUI/RViz 时才启动 debug 容器：

```bash
cd /home/haoming/wht/IsaacLab_drones_5.1/SharedRLControl
xhost +local:docker
docker compose -f docker-compose.tunnel.yml --profile debug up -d tunnel_debug
docker exec -it tunnel_debug bash
```

容器内：

```bash
# RL 策略（带 GUI）
roslaunch navigation_runner tunnel_comparison.launch method:=rl gui:=true rviz:=true

# IPC 算法（带 GUI）
roslaunch navigation_runner tunnel_comparison.launch method:=ipc gui:=true rviz:=true
```

## 结果分析

```bash
python3 /root/catkin_ws/src/navigation_runner/scripts/analyze_results.py \
     --data-dir /root/results/replay_h5_mapconstrained_seed5716 \
     --output-dir /root/results/replay_h5_mapconstrained_seed5716/analysis
```

宿主机路径与容器路径映射：

- 宿主机 `ros1/results/`
- 容器内 `/root/results`
- 容器内 `/root/catkin_ws/results`

## 注意事项

- `tunnel_batch` 默认 `TUNNEL_RENDER_MODE=headless`，不会使用宿主 `DISPLAY`。
- `tunnel_debug` 默认 `TUNNEL_RENDER_MODE=x11`，只有手动调试时使用。
- `docker-compose.tunnel.yml` 默认挂载
  `/home/haoming/wht/IsaacLab_drones_5.1/slope_inspection` 到 `/root/slope_ws/src`；
  如路径不同，可设置 `SLOPE_INSPECTION_HOST_PATH=/path/to/slope_inspection`。
- host orchestrator 默认只在开始时重编译一次 IPC 相关包并限制并发：
  `catkin_make -C /root/slope_ws -j1 -l1 -DCATKIN_WHITELIST_PACKAGES=...`，
  构建产物保存在 Docker named volumes 中供后续 batch 容器复用，避免每个 batch
  重复编译触发 gcc ICE 或资源峰值。
- 离线输入回放数据默认路径为
  `/root/catkin_ws/src/navigation_runner/cfg/ckpts/trajectories_tunnel.h5`。
- 约束地图采样会记录 spacing、local density、connectivity 等指标到每个 batch 的
  `obstacles.json` 和分析输出。
- 安全停止阈值：默认 `safety_min_dist=0.2m`；小规模 safety pilot 可通过 batch 参数 `--safety-min-dist 0.30` 覆盖
- 实验数据现在通过 bind mount 同步到宿主机 `ros1/results/`
- 容器内 `/root/results` 与 `/root/catkin_ws/results` 都会映射到同一个宿主机目录

## 真机 PX4 SRLC 部署（实验前无桨验证）

1. 新增 launch：

   roslaunch navigation_runner tunnel_real_px4.launch \
        checkpoint:=/root/catkin_ws/src/navigation_runner/cfg/ckpts/checkpoint_tunnel_M3_21500.pt \
        pcd_file:=/root/catkin_ws/src/navigation_runner/cfg/real_maps/merged/real_map_merged_ascii.pcd \
        start_mavros:=false

2. 默认假设：

   - MAVROS 已在机载电脑上连接 PX4，状态来自 `/mavros/local_position/odom`
   - 预扫描 PCD map 与 MAVROS local ENU 在实验前已标定对齐
   - SRLC 推理只订阅 `/srlc/lidar/range_image`，该话题由 `map_lidar_node.py` 从 PCD+odom 实时生成
   - human_action 来自 `/mavros/rc/in`，由 `rc_input_node.py` 映射到 `/srlc/human_action`

3. 安全默认：

   - `auto_arm=false`、`auto_offboard=false`，节点启动不会默认解锁或切 OFFBOARD
   - RC e-stop 通过 `/experiment_control/stop` 锁定停止
   - 停止后请求 PX4 `AUTO.LOITER` 悬停，不默认空中 disarm/kill motors
   - RC、LiDAR、odom 任一超时会停止 RL 输出

4. 上机前必须校准：

   - `cfg/tunnel/rc_input_real_px4.yaml` 中 RC 通道、方向、deadband、reset/estop/assist 开关
   - `cfg/tunnel/map_lidar_real_px4.yaml` 中 `map_origin_xyz` 与 `map_yaw_deg`
   - 真机限速、限高、geofence 与 `safety_min_dist`


## 转换pcd地图
python3 ros1/navigation_runner/scripts/tunnel_deployment/merge_real_pcd_maps.py \
 --inputs ros1/real_maps/room601/0517.pcd \
 --output ros1/real_maps/room601/0517_crop_xyz_0p05_ascii.pcd \
 --crop-min -3.5 -3.0 0.0 \
 --crop-max 3.5 3.0 3.0 \
 --voxel-size 0.0