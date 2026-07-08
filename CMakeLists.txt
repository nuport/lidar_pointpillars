cmake_minimum_required(VERSION 3.16)
project(lidar_pointpillars CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nuport_perception_msgs
  geometry_msgs
  nodelet
  pluginlib
)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs nuport_perception_msgs geometry_msgs nodelet pluginlib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

if(VIBRANTE OR NOT ${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "aarch64-linux-gnu")

  #-----------------------------------------------------------------------------
  # CUDA setup — must be configured before enable_language(CUDA)
  #-----------------------------------------------------------------------------
  if(NOT CUDA_DIR)
    set(CUDA_DIR "/usr/local/cuda" CACHE PATH "CUDA Toolkit location.")
  endif()
  if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDA_DIR}/bin/nvcc")
  endif()
  if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()

  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)

  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  # cuDLA is a transitive link dependency of libnvinfer on DRIVE OS (VIBRANTE).
  # Find it first so FindTensorRT.cmake can attach it to TensorRT::nvinfer.
  find_package(cuDLA MODULE)
  find_package(TensorRT MODULE REQUIRED)

  include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${TensorRT_INCLUDE_DIRS}
  )

  set(SOURCE_FILES
    src/pointpillars_node.cpp
    src/voxelization_trt.cpp
    src/utils.cpp
    src/nms_trt.cpp
    src/pp_postprocess_cuda.cu
    src/voxelization_trt.cu
  )

  # Nodelet shared library
  add_library(pointpillars_nodelet ${SOURCE_FILES})
  add_dependencies(pointpillars_nodelet ${catkin_EXPORTED_TARGETS})
  target_link_libraries(pointpillars_nodelet
    ${catkin_LIBRARIES}
    TensorRT::TensorRT
    cudart
  )

  # Standalone executable (wraps the nodelet)
  add_executable(pointpillars_node ${SOURCE_FILES})
  add_dependencies(pointpillars_node ${catkin_EXPORTED_TARGETS})
  target_link_libraries(pointpillars_node
    ${catkin_LIBRARIES}
    TensorRT::TensorRT
    cudart
  )

  install(TARGETS pointpillars_nodelet pointpillars_node
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  if(VIBRANTE)
    # After cross-compilation, stage the binaries into build/ so the production
    # target (which has no CUDA/TRT headers) can deploy them via configure_file.
    add_custom_command(
      TARGET pointpillars_nodelet POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pointpillars_nodelet>
        ${CMAKE_CURRENT_SOURCE_DIR}/build/
      COMMENT "Staging libpointpillars_nodelet.so -> build/"
    )
    add_custom_command(
      TARGET pointpillars_node POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pointpillars_node>
        ${CMAKE_CURRENT_SOURCE_DIR}/build/
      COMMENT "Staging pointpillars_node -> build/"
    )
  endif()

else()

  # Production aarch64 target: no CUDA/TRT development headers available.
  # Deploy pre-built binaries staged in build/ during cross-compilation.
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/build/libpointpillars_nodelet.so
    ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_LIB_DESTINATION}
    COPYONLY
  )
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/build/pointpillars_node
    ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}
    COPYONLY
  )

endif()

install(FILES pointpillars_nodelet.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
