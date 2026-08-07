cmake_minimum_required(VERSION 3.0.2)
project(srlc_real)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
  visualization_msgs
)

catkin_python_setup()

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_add_nosetests(test/test_mavlink_stream_guard_core.py)
  catkin_add_nosetests(test/test_launch_contracts.py)
  catkin_add_nosetests(test/test_one_shot_flight.py)
  catkin_add_nosetests(test/test_pcd_raycast_safety.py)
  catkin_add_nosetests(test/test_recorded_rc_replay.py)
  catkin_add_nosetests(test/test_recorder_contract.py)
  catkin_add_nosetests(test/test_flight_replay.py)
  catkin_add_nosetests(test/test_flight_replay_contract.py)
  add_rostest(test/flight_replay_node.test)
  add_rostest(test/fake_runtime_stream_guard.test)
  add_rostest(test/mavlink_stream_guard.test)
  add_rostest(test/one_shot_flight.test)
endif()

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    mavros_msgs
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
    std_srvs
    tf2_ros
    visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/flight_replay_node.py
  scripts/mavlink_stream_guard_node.py
  scripts/real_navigation_node.py
  scripts/map_lidar_node.py
  scripts/pcd_map_publisher.py
  scripts/inspect_recorded_rc.py
  scripts/recorded_rc_replay_node.py
  scripts/render_flight_replay.py
  scripts/rc_input_node.py
  scripts/srlc_real_fake_runtime_node.py
  scripts/srlc_alignment_viz_node.py
  scripts/srlc_real_recorder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY cfg launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
