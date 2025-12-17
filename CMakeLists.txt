cmake_minimum_required(VERSION 3.18)
project(optix_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Disable AVX-512 for WSL compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-avx512f -mno-avx512dq -mno-avx512cd -mno-avx512bw")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-mno-avx512f -Xcompiler=-mno-avx512dq -Xcompiler=-mno-avx512cd -Xcompiler=-mno-avx512bw")

# OptiX 路径
set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64")
set(OptiX_INCLUDE "${OptiX_INSTALL_DIR}/include")

# CUDA
find_package(CUDAToolkit REQUIRED)

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Add root src directory for includes like "core/optix_context.h"
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 理论计算库
add_library(theory STATIC
    src/theory/theory.cpp
)

target_include_directories(theory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 编译 CUDA 为 PTX
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_tracer.ptx
    COMMAND ${CMAKE_CUDA_COMPILER}
        -ptx
        -arch=sm_75
        -I${OptiX_INCLUDE}
        -I${CUDAToolkit_INCLUDE_DIRS}
        -I${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/forward_tracer.cu
        -o ${CMAKE_CURRENT_BINARY_DIR}/forward_tracer.ptx
    DEPENDS src/simulation/forward_tracer.cu include/device_params.h
    COMMENT "Compiling forward_tracer.cu to PTX"
)

add_custom_target(ptx_target ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward_tracer.ptx
)

# 积分球模拟器
add_executable(ideal_sphere_sim
    src/app/main.cpp
    src/core/optix_context.cpp
    src/scene/scene.cpp
    src/simulation/path_tracer.cpp
)

add_dependencies(ideal_sphere_sim ptx_target)

target_include_directories(ideal_sphere_sim PRIVATE
    ${OptiX_INCLUDE}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ideal_sphere_sim PRIVATE
    theory
    ${CMAKE_DL_LIBS}
    CUDA::cudart
)

# Google Test 单元测试
add_executable(unit_tests
    tests/test_theory.cpp
)
target_include_directories(
    unit_tests PUBLIC
    src/theory/
)

target_link_libraries(unit_tests PRIVATE
    theory
    GTest::gtest_main
)


# 注册测试
include(GoogleTest)
gtest_discover_tests(unit_tests)

message(STATUS "")
message(STATUS "Build: cd build && cmake .. && make")
message(STATUS "Test:  cd build && ctest --verbose")
message(STATUS "Run:   ./test_env")
message(STATUS "")
