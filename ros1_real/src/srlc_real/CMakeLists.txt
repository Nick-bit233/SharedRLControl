cmake_minimum_required(VERSION 3.0.2)
project(srlc_real)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    mavros_msgs
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

catkin_install_python(PROGRAMS
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
