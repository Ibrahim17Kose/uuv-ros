cmake_minimum_required(VERSION 3.0.2)
project(uuv_gazebo)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_ros
  uuv_msgs
)

find_package(gazebo REQUIRED)

catkin_package(
  DEPENDS 
    roscpp
    gazebo_ros
)

include_directories(
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(uuv_gazebo_plugin 
  src/uuv_gazebo_plugin.cpp
)

target_link_libraries(uuv_gazebo_plugin 
  ${catkin_LIBRARIES} 
  ${GAZEBO_LIBRARIES}
)