cmake_minimum_required(VERSION 3.0)
project(lidar_pointpillars LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nuport_perception_msgs
  geometry_msgs
  nodelet
  pluginlib
  tf
)

find_package(CUDA REQUIRED)

# Set TensorRT path (override via TENSORRT_ROOT environment variable)
if(DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
else()
  set(TENSORRT_ROOT "/usr/local/TensorRT")
endif()

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs nuport_perception_msgs geometry_msgs nodelet pluginlib tf
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_ROOT}/include
)

link_directories(${TENSORRT_ROOT}/lib)

set(SOURCE_FILES
  src/pointpillars_node.cpp
  src/voxelization_trt.cpp
  src/utils.cpp
  src/nms_trt.cpp
  src/voxelization_trt.cu
)

# Nodelet shared library
add_library(pointpillars_nodelet ${SOURCE_FILES})
add_dependencies(pointpillars_nodelet ${catkin_EXPORTED_TARGETS})
target_link_libraries(pointpillars_nodelet
  ${catkin_LIBRARIES}
  nvinfer
  nvinfer_plugin
  ${CUDA_LIBRARIES}
)

# Standalone executable (wraps the nodelet)
add_executable(pointpillars_node ${SOURCE_FILES})
add_dependencies(pointpillars_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(pointpillars_node
  ${catkin_LIBRARIES}
  nvinfer
  nvinfer_plugin
  ${CUDA_LIBRARIES}
)

install(TARGETS pointpillars_nodelet pointpillars_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES pointpillars_nodelet.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
