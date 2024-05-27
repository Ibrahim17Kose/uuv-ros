cmake_minimum_required(VERSION 3.0.2)
project(uuv_sensor_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
    Camera.msg
    DVL.msg
    IMU.msg
    Pressure.msg
    Sonar.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories()

