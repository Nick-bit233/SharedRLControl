运行ros1对比实验

进入容器终端：

 docker exec -it tunnel_debug bash

生成地图

python3 generate_tunnel_map.py \
-o ../../cfg/tunnel/tunnel_map_default.pcd \
-w ../../../uav_simulator/worlds/generated_env/tunnel_pcd_match_static.world \
--seed 288 -n 30 --cuboid-ratio 0.

方式一：手动单独测试

 # RL 策略（带 GUI）
 roslaunch navigation_runner tunnel_comparison.launch method:=rl gui:=true
 
 # IPC 算法（带 GUI）
 roslaunch navigation_runner tunnel_comparison.launch method:=ipc gui:=true

方式二：自动化多轮对比

 # 先启动 Gazebo（保持运行）
 roslaunch uav_simulator start_headless.launch gui:=true &
 
 # 然后运行自动对比脚本（各跑 5 轮，每轮 60s 超时）
 python3 /root/catkin_ws/src/navigation_runner/scripts/run_comparison.py \
     --methods rl,ipc --n-trials 5 --timeout 60 \
     --output-dir /root/results

方式三：分析结果

 python3 /root/catkin_ws/src/navigation_runner/scripts/analyze_results.py \
     --data-dir /root/results

注意事项

 - 安全停止阈值：当前 safety_min_dist=0.3m，如果触发过频，可在 launch 时降低或在 cfg/tunnel/tunnel_nav_param.yaml 中修改
 - 实验数据保存在 /root/results（Docker volume 持久化）