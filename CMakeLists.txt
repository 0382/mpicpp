cmake_minimum_required(VERSION 3.16)

project(mpicpp_test)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)

add_executable(test test/test.cpp)

add_subdirectory(mpicpp)

target_link_libraries(test PRIVATE mpicpp)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(test PRIVATE Threads::Threads)