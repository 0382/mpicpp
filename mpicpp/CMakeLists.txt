cmake_minimum_required(VERSION 3.16)

add_library(mpicpp INTERFACE)
target_include_directories(mpicpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

find_package(MPI REQUIRED)

target_compile_features(mpicpp INTERFACE cxx_std_17)
target_include_directories(mpicpp INTERFACE ${MPI_INCLUDE_PATH})
target_link_libraries(mpicpp INTERFACE ${MPI_C_LIBRARIES})