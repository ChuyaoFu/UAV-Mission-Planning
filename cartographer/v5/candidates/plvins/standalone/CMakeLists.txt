cmake_minimum_required(VERSION 3.10)
project(plvins_lsd_sensor_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(PLVINS_SOURCE "" CACHE PATH "Pinned PL-VINS source tree")
if(NOT EXISTS "${PLVINS_SOURCE}/feature_tracker/src/line_descriptor/src/LSDDetector_custom.cpp")
  message(FATAL_ERROR "Supply -DPLVINS_SOURCE=/path/to/patched/pinned/PL-VINS")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d)
set(LINE_ROOT ${PLVINS_SOURCE}/feature_tracker/src/line_descriptor)
add_executable(plvins_lsd_probe lsd_probe.cpp
  ${LINE_ROOT}/src/binary_descriptor_custom.cpp
  ${LINE_ROOT}/src/binary_descriptor_matcher.cpp
  ${LINE_ROOT}/src/draw_custom.cpp
  ${LINE_ROOT}/src/LSDDetector_custom.cpp)
target_include_directories(plvins_lsd_probe PRIVATE ${OpenCV_INCLUDE_DIRS} ${LINE_ROOT}/include)
target_link_libraries(plvins_lsd_probe ${OpenCV_LIBS})
# No prebuilt .so, -march=native, x86 intrinsics, ROS, truth, depth or pose input.

if(EXISTS "${LINE_ROOT}/src/plvins_lsd.cpp")
  target_sources(plvins_lsd_probe PRIVATE ${LINE_ROOT}/src/plvins_lsd.cpp)
endif()
