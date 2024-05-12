cmake_minimum_required(VERSION 3.0.2)
project(uuv_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
)

add_message_files(
  FILES
    ControlSignal.msg
    ReferenceSignal.msg
    States.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories()
