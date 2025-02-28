cmake_minimum_required(VERSION 3.21)
project(dual_lidar_cali)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "release")

find_package(Eigen3 REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)

find_package(PkgConfig)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp>=0.5)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

include_directories(
        include
        ${PCL_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
        ${YAMLCPP_INCLUDE_DIRS}
)
link_directories(
        include
        ${PCL_LIBRARY_DIRS}
        ${GTSAM_LIBRARY_DIRS}
)
add_executable(dual_lidar_cali main.cpp BA.h)
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${GTSAM_LIBRARIES} ${YAMLCPP_LIBRARIES} gtsam)