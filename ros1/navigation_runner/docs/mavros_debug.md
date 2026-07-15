已定位根因：PX4 当前 MAVLink 实例没有提供 LOCAL_POSITION_NED（消息 ID 32），所以 MAVROS 无法生成 /mavros/local_position/pose。这不是
  Docker/MAVROS 依赖问题，也不是 vision_pose 输入中断。

  关键证据：

  - /mavros/vision_pose/pose：约 47–50 Hz，输入正常。
  - FCU 已连接，MAVLink 丢包为 0，时间同步正常。
  - EKF2_EV_CTRL=15、EKF2_HGT_REF=3，PX4 估计器的位置和速度有效标志均为 True。
  - 10 秒原始 MAVLink 数据中没有：
      - ATTITUDE，ID 30
      - LOCAL_POSITION_NED，ID 32

  - 协议级查询结果：
      - ID 32：interval_us=0
      - ID 30：interval_us=0
      - 对照 ID 1：200000 µs
      - 对照 ID 230：1000000 µs

  按照 MAVLink 定义，interval_us=0 表示该消息在当前 stream 中不可用。MAVLink MESSAGE_INTERVAL 定义
  (https://mavlink.io/en/messages/common.html#MESSAGE_INTERVAL)

  数据实际卡在这里：

  PX4 vehicle_local_position
    └─ LOCAL_POSITION_NED (ID 32)  ← 当前未输出
         └─ MAVROS local_position plugin
              └─ /mavros/local_position/pose

  MAVROS 的 local_position 插件只有收到 ID 32 才发布 ROS 位姿。MAVROS 1.20.1 源码
  (https://github.com/mavlink/mavros/blob/1.20.1/mavros/src/plugins/local_position.cpp)

  飞控参数中 MAV_0_MODE=8，即 External Vision 模式；PX4 v1.14.2 官方实现本应配置 LOCAL_POSITION_NED=30 Hz、ATTITUDE=20 Hz。PX4 参数说明
  (https://docs.px4.io/v1.14/en/advanced_config/parameter_reference#MAV_0_MODE)、PX4 v1.14.2 stream 源码
  (https://github.com/PX4/PX4-Autopilot/blob/v1.14.2/src/modules/mavlink/mavlink_main.cpp)

  因此进一步结论是：飞控当前运行中的 stream 配置与保存的 MAV_0_MODE=8 默认配置不一致。最可能的来源是：

  - 修改 MAV_0_MODE 后没有重启飞控；该参数明确要求重启。
  - QGC、其他 MAVLink 客户端或自定义 PX4 启动逻辑运行时移除了 ID 30/32。

  容器里的 soccer.launch 和工程代码没有修改 stream rate，MAVROS 相关包及动态库也完整。

  可以用下面两条命令临时恢复并验证，但我尚未执行，避免未经确认修改真实飞控：

  rosservice call /mavros/set_message_interval \
    "{message_id: 32, message_rate: 30.0}"

  rosservice call /mavros/set_message_interval \
    "{message_id: 30, message_rate: 20.0}"

  rostopic hz /mavros/local_position/pose

  永久处理应在未解锁状态下确认 MAV_0_MODE=8 后重启飞控，并在不启动其他 GCS 的情况下复测；若重启后仍为 0，则应检查当前定制 PX4 固件的
  MAVLink 启动配置。