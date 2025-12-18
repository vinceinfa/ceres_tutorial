cmake_minimum_required(VERSION 3.10.2)
project(ceres_tutorial)

set(CMAKE_CXX_FLAGS "-std=c++17")

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# Position 1D Tests
add_executable(position_1d src/position_1d.cpp)

target_link_libraries(position_1d
  gtest
  gtest_main
  ${EIGEN3_LIBRARIES}
  ${GTEST_LIBRARIES}
  ${CERES_LIBRARIES}
)

# Slam 1D Tests
add_executable(slam_1d src/slam_1d.cpp)

target_link_libraries(slam_1d
  gtest
  gtest_main
  ${EIGEN3_LIBRARIES}
  ${GTEST_LIBRARIES}
  ${CERES_LIBRARIES}
)

# Position 3D Tests
add_executable(position_3d src/position_3d.cpp)

target_link_libraries(position_3d
  gtest
  gtest_main
  ${EIGEN3_LIBRARIES}
  ${GTEST_LIBRARIES}
  ${CERES_LIBRARIES}
)

# Registra il test con CTest
enable_testing()
add_test(NAME Position1DTest COMMAND position_1d)
add_test(NAME Slam1dTest     COMMAND slam_1d)
add_test(NAME Position3DTest COMMAND position_3d)

