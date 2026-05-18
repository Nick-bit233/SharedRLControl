隧道任务消融实验训练与 ROS1 批量评估指南

1. 消融实验设计原则

论文中的消融实验目标是证明完整方法中各组件的必要性，而不是单纯调参。所有消融都应尽量保持 
地图、飞手输入、训练预算、评估协议、外部安全保护 一致，只改变一个因素。

┌─────────────────┬─────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
│ 实验行          │ 目的                                    │ 改动点                                                                │
├─────────────────┼─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ ours            │ 完整方法基线                            │ 默认复用当前论文最佳 checkpoint，不重新三阶段训练                     │
├─────────────────┼─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ no_residual     │ 验证 residual action parameterization   │ 改 PPO action-head 为 direct                                          │
│                 │ 是否重要                                │ policy；仍接收飞手指令且保留同权重意图正则，但动作不再以飞手指令为中心 │
├─────────────────┼─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ no_curriculum   │ 验证课程学习是否必要                    │ 直接在 hard/stage3 隧道上训练，预算等于三阶段总预算                   │
├─────────────────┼─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ follow_only     │ 验证 safety reward 是否必要             │ 保留跟随奖励，关闭局部 LiDAR safety reward                            │
├─────────────────┼─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ safety_reg      │ 验证 following reward 是否必要          │ 保留 safety reward + residual regularization，关闭 following reward   │
└─────────────────┴─────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘

公平性要求：

 1. 所有新训练模型使用同一个 data/trajectories_tunnel.h5。
 2. 每个新训练消融最好至少跑 3 个 seed。
 3. ours 可以直接使用当前最佳模型，但最终表格必须用同一套 ROS1/Gazebo batch protocol 评估。
 4. 不要比较 frozen ours 和新训练模型的训练曲线；只比较同协议最终测试结果。
 5. NoResidual 在 ROS1 测试时必须使用 --policy-mode direct。
 6. Isaac 侧统一评估使用 hard/stage3 隧道分布：80 obstacles、width [0.4, 0.9]、height [8.0, 18.0]。
 7. 旧版 NoResidual 若是在“direct policy 无意图正则”的代码上训练，不能作为最终公平消融结果，需要重训。

------------------------------------------------------------------------------------------------------------------------------------

2. 训练前准备

进入 Isaac 训练目录：

 cd /cpfs/user/wanghaotian/SharedRLControl/isaac-training

确认离线飞手数据存在：

 ls data/trajectories_tunnel.h5

如果不存在，先生成：

 python src/datasets/trajectory_generator.py --config-name trajectory_gen_tunnel

------------------------------------------------------------------------------------------------------------------------------------

3. 注册 ours 最佳模型

ours 不需要重训，直接注册当前最佳 checkpoint：

 python experiments/04a_tunnel_ablation/run_matrix.py \
   --variants ours \
   --tag paper_ablation_v1 \
   --ours-checkpoint /path/to/current_best_checkpoint.pt \
   --ours-config tunnel_m3_finetune

这会生成 manifest，供后续统一 eval 使用。

------------------------------------------------------------------------------------------------------------------------------------

4. 训练各消融模型

4.1 一次性启动完整消融矩阵

> only use seed 42 for quick turnaround; can expand to more seeds if time allows

 python experiments/04a_tunnel_ablation/run_matrix.py \
   --variants no_residual no_curriculum follow_only safety_reg \
   --seeds 42 \
   --tag paper_ablation_v1

4.2 单独训练 NoResidual

 python experiments/04a_tunnel_ablation/run_matrix.py \
   --variants no_residual \
   --seeds 42 \
   --tag paper_ablation_v1

4.3 单独训练 NoCurriculum

 python experiments/04a_tunnel_ablation/run_matrix.py \
   --variants no_curriculum \
   --seeds 42 \
   --tag paper_ablation_v1

4.4 单独训练 FollowOnly

 python experiments/04a_tunnel_ablation/run_matrix.py \
   --variants follow_only \
   --seeds 42 \
   --tag paper_ablation_v1

4.5 单独训练 SafetyRegOnly

 python experiments/04a_tunnel_ablation/run_matrix.py \
   --variants safety_reg \
   --seeds 42 \
   --tag paper_ablation_v1

训练完成后，每个 run 的 manifest 会写到：

 isaac-training/outputs/tunnel_ablation/manifests/

------------------------------------------------------------------------------------------------------------------------------------

5. Isaac 侧快速评估与汇总

用统一 eval 脚本检查 checkpoint 是否能正常加载和 rollout：

 python experiments/04a_tunnel_ablation/run_eval_matrix.py \
   --manifests outputs/tunnel_ablation/manifests \
   --output-dir outputs/tunnel_ablation/eval \
   --eval-seeds 101 102 103 104 105 \
   --num-envs 1024

汇总 Isaac eval：

 python experiments/04a_tunnel_ablation/summarize_results.py \
   --eval-dir outputs/tunnel_ablation/eval \
   --csv outputs/tunnel_ablation/ablation_summary.csv

------------------------------------------------------------------------------------------------------------------------------------

6. 将 checkpoint 放入 ROS1 批测路径

ROS1 Docker 内推荐使用：

 /root/catkin_ws/src/navigation_runner/cfg/ckpts/

宿主机对应放置目录通常是：

 /cpfs/user/wanghaotian/SharedRLControl/ros1/navigation_runner/cfg/ckpts/

例如：

 cp /path/to/no_residual/checkpoint_best.pt \
   /cpfs/user/wanghaotian/SharedRLControl/ros1/navigation_runner/cfg/ckpts/no_residual_seed42.pt

------------------------------------------------------------------------------------------------------------------------------------

7. ROS1 批量测试命令

从仓库根目录运行：

 cd /cpfs/user/wanghaotian/SharedRLControl

7.1 通用 batch 参数

建议所有消融使用完全一致的参数：

 COMMON_ARGS="\
   --run \
   --num-batches 64 \
   --runs-per-batch 10 \
   --methods rl \
   --master-seed 325 \
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
   --run-retries 1"

7.2 测试 Ours

 python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
   $COMMON_ARGS \
   --checkpoint /root/catkin_ws/src/navigation_runner/cfg/ckpts/ours_best.pt \
   --policy-mode residual \
   --output-dir /root/results/ablation_ours

7.3 测试 NoResidual

注意这里必须是 --policy-mode direct：

 python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
   $COMMON_ARGS \
   --checkpoint /root/catkin_ws/src/navigation_runner/cfg/ckpts/no_residual_seed42.pt \
   --policy-mode direct \
   --output-dir /root/results/ablation_no_residual_seed42

7.4 测试 NoCurriculum

 python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
   $COMMON_ARGS \
   --checkpoint /root/catkin_ws/src/navigation_runner/cfg/ckpts/no_curriculum_seed42.pt \
   --policy-mode residual \
   --output-dir /root/results/ablation_no_curriculum_seed42

7.5 测试 FollowOnly

 python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
   $COMMON_ARGS \
   --checkpoint /root/catkin_ws/src/navigation_runner/cfg/ckpts/follow_only_seed42.pt \
   --policy-mode residual \
   --output-dir /root/results/ablation_follow_only_seed42

7.6 测试 SafetyRegOnly

 python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
   $COMMON_ARGS \
   --checkpoint /root/catkin_ws/src/navigation_runner/cfg/ckpts/safety_reg_seed42.pt \
   --policy-mode residual \
   --output-dir /root/results/ablation_safety_reg_seed42

------------------------------------------------------------------------------------------------------------------------------------

8. 结果分析

每个 batch run 完成后，runner 默认会调用 analyze_results.py。如果需要手动分析：

 python3 ros1/navigation_runner/scripts/run_tunnel_batch_containers.py \
   --resume-from /root/results/ablation_no_residual_seed42 \
   --num-batches 64 \
   --no-rebuild-slope

或在容器内直接运行：

 python3 /root/catkin_ws/src/navigation_runner/scripts/analyze_results.py \
   --data-dir /root/results/ablation_no_residual_seed42 \
   --output-dir /root/results/ablation_no_residual_seed42/analysis

主要输出：

 ros1/results/<run_name>/analysis/summary.json
 ros1/results/<run_name>/analysis/metrics.csv
 ros1/results/<run_name>/analysis/comparison_plots.png

重点读取指标：

┌─────────────────────────────────────────────────┬────────────────────────────┐
│ 指标                                            │ 含义                       │
├─────────────────────────────────────────────────┼────────────────────────────┤
│ success_rate                                    │ 到达终点线比例             │
├─────────────────────────────────────────────────┼────────────────────────────┤
│ collision_rate                                  │ 碰撞比例                   │
├─────────────────────────────────────────────────┼────────────────────────────┤
│ non_collision_failure_rate                      │ 非碰撞失败，主要是 timeout │
├─────────────────────────────────────────────────┼────────────────────────────┤
│ avg_speed_mean                                  │ 平均速度                   │
├─────────────────────────────────────────────────┼────────────────────────────┤
│ tcr_at_1_mean / tcr_at_2_mean / tcr_at_5_mean   │ 轨迹覆盖率                 │
├─────────────────────────────────────────────────┼────────────────────────────┤
│ min_obstacle_dist_mean                          │ 最小障碍距离               │
├─────────────────────────────────────────────────┼────────────────────────────┤
│ pct_close_*                                     │ 危险距离内停留比例         │
├─────────────────────────────────────────────────┼────────────────────────────┤
│ likely_safety_hold_trap                         │ 是否疑似被安全停滞困住     │
└─────────────────────────────────────────────────┴────────────────────────────┘

------------------------------------------------------------------------------------------------------------------------------------

9. 建议的论文表格组织

最终表格建议每个消融行汇总 ROS1/Gazebo 结果：

┌───────────────┬─────────────┬───────────┬─────────────┬───────────┬─────────────┬─────────┬─────────┬─────────┐
│ Variant       │ Policy mode │ Success ↑ │ Collision ↓ │ Timeout ↓ │ Avg speed ↑ │ TCR@1 ↑ │ TCR@2 ↑ │ TCR@5 ↑ │
├───────────────┼─────────────┼───────────┼─────────────┼───────────┼─────────────┼─────────┼─────────┼─────────┤
│ Ours          │ residual    │           │             │           │             │         │         │         │
├───────────────┼─────────────┼───────────┼─────────────┼───────────┼─────────────┼─────────┼─────────┼─────────┤
│ NoResidual    │ direct      │           │             │           │             │         │         │         │
├───────────────┼─────────────┼───────────┼─────────────┼───────────┼─────────────┼─────────┼─────────┼─────────┤
│ NoCurriculum  │ residual    │           │             │           │             │         │         │         │
├───────────────┼─────────────┼───────────┼─────────────┼───────────┼─────────────┼─────────┼─────────┼─────────┤
│ FollowOnly    │ residual    │           │             │           │             │         │         │         │
├───────────────┼─────────────┼───────────┼─────────────┼───────────┼─────────────┼─────────┼─────────┼─────────┤
│ SafetyRegOnly │ residual    │           │             │           │             │         │         │         │
└───────────────┴─────────────┴───────────┴─────────────┴───────────┴─────────────┴─────────┴─────────┴─────────┘

核心解释逻辑：

 - NoResidual 若 success/CRR 较高但 TCR/CTE 变差，说明 direct policy 更像自主避障器，residual centering 对共享控制中的意图保持重要。
 - NoCurriculum 若成功率低或碰撞高，说明 staged curriculum 有助于稳定学习。
 - FollowOnly 若碰撞高，说明局部 safety reward 必要。
 - SafetyRegOnly 若速度慢、timeout 高或 TCR 差，说明 following reward 对飞手意图满足必要。
 - Ours 应体现成功率、效率、安全性之间的最佳折中。


## 重跑恢复指令
- NoCurriculum 可从你现有的最新 checkpoint 继续：

python experiments/04a_tunnel_ablation/run_matrix.py \
  --variants no_curriculum \
  --seeds 42 \
  --tag paper_ablation_v1 \
  --resume-checkpoint outputs/tunnel_ablation/no_curriculum/paper_ablation_v1_seed42/2026-05-14_03-05-40/wandb/run-20260514_030557-x8wta239/files/checkpoint_6250.pt

python experiments/04a_tunnel_ablation/run_curriculum.py \
   --variant no_residual --seed 42 --tag paper_ablation_v1_fixed \
   --start-stage 2 \
   --checkpoint "$(cat outputs/tunnel_ablation/no_residual/stage1/paper_ablation_v1_seed42/2026-05-13_17-57-01/final_checkpoint_path.txt)"
