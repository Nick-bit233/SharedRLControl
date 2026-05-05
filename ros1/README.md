运行ros1对比实验

进入容器终端：

 docker exec -it tunnel_debug bash

生成地图

python3 generate_tunnel_map.py \
-o ../../cfg/tunnel/tunnel_map_default.pcd \
-w ../../../uav_simulator/worlds/generated_env/tunnel_pcd_match_static.world \
--seed <多次实验随机生成> -n 15 --cuboid-ratio 0.5

方式一：手动单独测试

 # RL 策略（带 GUI）
 roslaunch navigation_runner tunnel_comparison.launch method:=rl gui:=true
 
 # IPC 算法（带 GUI）
 roslaunch navigation_runner tunnel_comparison.launch method:=ipc gui:=true

方式二：自动化多轮对比

python3 /root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py \
     --num-batches 1 \
     --runs-per-batch 1 

# 从中断的批量结果原地续跑；会复用已有 batch_config/地图/seed，跳过完整轨迹并重跑缺失或损坏的 run
python3 /root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py \
     --resume-from /root/catkin_ws/results/batch_20260427_142203 \
     --master-seed 325

# 100-run safety_min_dist pilot: minimal safety-margin ablation, do not overwrite baseline results
python3 /root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py \
     --num-batches 10 \
     --runs-per-batch 10 \
     --methods rl,ipc \
     --master-seed 326 \
     --safety-min-dist 0.30 \
     --launch-timeout 100 \
     --output-dir /root/catkin_ws/results/batch_safety030_pilot

# 100-run recovery-shield pilot: same safety distance, active escape instead of hold-only stop
python3 /root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py \
     --num-batches 10 \
     --runs-per-batch 10 \
     --methods rl,ipc \
     --master-seed 327 \
     --safety-min-dist 0.30 \
     --safety-mode recover \
     --safety-recover-speed 0.35 \
     --safety-recover-forward-speed 0.15 \
     --launch-timeout 100 \
     --output-dir /root/catkin_ws/results/batch_safety030_recover_pilot

# Offline HDF5 replay + constrained feasible-map pilot
python3 /root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py \
     --num-batches 10 \
     --runs-per-batch 10 \
     --methods rl,ipc \
     --master-seed 5716 \
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
     --safety-min-dist 0.20 \
     --launch-timeout 100 \
     --output-dir /root/catkin_ws/results/replay_h5_mapconstrained_seed5716

# 上述 offline replay batch 从断点续跑示例
python3 /root/catkin_ws/src/navigation_runner/scripts/batch_tunnel_experiments.py \
     --resume-from /root/catkin_ws/results/replay_h5_mapconstrained_seed5716 \
     --master-seed 5716

方式三：分析结果

 python3 /root/catkin_ws/src/navigation_runner/scripts/analyze_results.py \
     --data-dir /root/results

注意事项

 - 安全停止阈值：默认 `safety_min_dist=0.2m`；小规模 safety pilot 可通过 batch 参数 `--safety-min-dist 0.30` 覆盖
 - 安全介入模式：默认 `--safety-mode hold` 保持旧行为；`--safety-mode recover` 会在触发安全距离时发布低速脱困命令，用于验证 stop-only trap 是否可减少
 - 离线输入回放：`docker compose -f docker-compose.tunnel.yml build tunnel_debug && docker compose -f docker-compose.tunnel.yml up -d --force-recreate` 后，容器镜像内才会包含 `h5py`；否则 offline replay 节点会在启动时直接报错退出
 - 离线输入回放：容器内默认可通过 `/root/catkin_ws/src/navigation_runner/cfg/ckpts/trajectories_tunnel.h5` 访问 HDF5；不要依赖指向宿主机绝对路径的软链接，因为该目标路径默认不在容器挂载范围内
 - 离线输入回放：`--input-source offline --replay-dataset-path <h5>` 会让 RL/IPC 复用同一条 HDF5 pilot velocity 序列；`--replay-start-offset 0` 可避免随机窗口落在旧数据集的低前进速度段
 - 约束地图采样：`--map-sampling-mode constrained` 会记录 spacing、local density、connectivity 等 feasibility 指标到每个 batch 的 `obstacles.json` 和分析 `summary.json`
 - 实验数据现在通过 bind mount 同步到宿主机 `ros1/results/`
 - 容器内 `/root/results` 与 `/root/catkin_ws/results` 都会映射到同一个宿主机目录
