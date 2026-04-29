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
     --safety-min-dist 0.30 \
     --launch-timeout 100 \
     --output-dir /root/catkin_ws/results/batch_safety030_pilot \
     --master-seed 325 \
     --gazebo-z-mode policy \


方式三：分析结果

 python3 /root/catkin_ws/src/navigation_runner/scripts/analyze_results.py \
     --data-dir /root/results

注意事项

 - 安全停止阈值：默认 `safety_min_dist=0.2m`；小规模 safety pilot 可通过 batch 参数 `--safety-min-dist 0.30` 覆盖
 - 实验数据现在通过 bind mount 同步到宿主机 `ros1/results/`
 - 容器内 `/root/results` 与 `/root/catkin_ws/results` 都会映射到同一个宿主机目录

真机 PX4 SRLC 部署（实验前无桨验证）

1. 新增 launch：

   roslaunch navigation_runner tunnel_real_px4.launch \
        checkpoint:=/path/to/checkpoint_tunnel_M3_21500.pt \
        pcd_file:=/path/to/pre_scanned_map.pcd \
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
