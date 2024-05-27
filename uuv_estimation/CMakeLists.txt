cmake_minimum_required(VERSION 3.0.2)
project(uuv_estimation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
     src/uuv_estimation/estimator.py
     src/uuv_estimation/kalman_base.py
     src/uuv_estimation/uuv_kalman.py
     DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)