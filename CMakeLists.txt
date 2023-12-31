cmake_minimum_required(VERSION 3.0.2)
project(multiple_turtlebots_sim)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs geometry_msgs pcl_ros message_generation)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

add_message_files(
   FILES
   SyncMsg.msg
 )

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${CERES_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
 ${Sophus_INCLUDE_DIRS}
 ${Pangolin_INCLUDE_DIRS}
 ${Python3_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

catkin_install_python(PROGRAMS scripts/get_local_se2_pose.py scripts/move_turtlebot.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)



add_executable(relative_measrement src/relative_measrement.cpp)
target_link_libraries(relative_measrement ${catkin_LIBRARIES})

add_executable(my_kf_node src/my_kf_node.cpp)
target_link_libraries(my_kf_node ${catkin_LIBRARIES} Eigen3::Eigen)

add_executable(timesync src/timesync.cpp)
target_link_libraries(timesync ${catkin_LIBRARIES} )

add_executable(my_nls_node src/my_nls_node.cpp)
target_link_libraries(my_nls_node ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${Python3_LIBRARIES})

add_executable(plot_node src/plot_node.cpp)
target_link_libraries(plot_node ${catkin_LIBRARIES} ${Python3_LIBRARIES})

add_executable(my_sliding_nls_node src/my_sliding_nls_node.cpp)
target_link_libraries(my_sliding_nls_node ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${Python3_LIBRARIES})

add_executable(my_sliding_filtering_nls_node src/my_sliding_filtering_nls_node.cpp)
target_link_libraries(my_sliding_filtering_nls_node ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${Python3_LIBRARIES})
