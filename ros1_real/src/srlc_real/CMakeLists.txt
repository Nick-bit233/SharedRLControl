cmake_minimum_required(VERSION 3.0.2)
project(srlc_real)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  message_generation
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  ObstacleClearance.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_add_nosetests(test/test_mavlink_stream_guard_core.py)
  catkin_add_nosetests(test/test_clearance_ros_integration_contract.py)
  catkin_add_nosetests(test/test_launch_contracts.py)
  catkin_add_nosetests(test/test_one_shot_flight.py)
  catkin_add_nosetests(test/test_pcd_raycast_safety.py)
  catkin_add_nosetests(test/test_recorder_contract.py)
  add_rostest(test/fake_runtime_stream_guard.test)
  add_rostest(test/mavlink_stream_guard.test)
  add_rostest(test/one_shot_flight.test)
endif()

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    mavros_msgs
    message_runtime
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/mavlink_stream_guard_node.py
  scripts/real_navigation_node.py
  scripts/map_lidar_node.py
  scripts/pcd_map_publisher.py
  scripts/rc_input_node.py
  scripts/srlc_real_fake_runtime_node.py
  scripts/srlc_alignment_viz_node.py
  scripts/srlc_real_recorder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY cfg launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
