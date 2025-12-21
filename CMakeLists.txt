cmake_minimum_required(VERSION 3.18)
project(optix_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# OptiX 路径
if(WIN32)
    set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
else()
    set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64")
endif()
set(OptiX_INCLUDE "${OptiX_INSTALL_DIR}/include")

# CUDA
find_package(CUDAToolkit REQUIRED)

# External dependencies (googletest, spdlog, etc.)
include(cmake/dependencies.cmake)

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
        -I${CMAKE_CURRENT_SOURCE_DIR}/src/simulation
        ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/kernels.cu
        -o ${CMAKE_CURRENT_BINARY_DIR}/forward_tracer.ptx
    DEPENDS
        src/simulation/kernels.cu
        src/simulation/kernels/kernel_utils.cuh
        src/simulation/kernels/intersection.cu
        src/simulation/kernels/radiance.cu
        src/simulation/kernels/shadow.cu
        src/simulation/kernels/raygen.cu
        include/device_params.h
    COMMENT "Compiling kernels to PTX"
)

add_custom_target(ptx_target ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward_tracer.ptx
)

# 将 PTX 嵌入为 C++ 头文件
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/embedded_ptx.h
    COMMAND ${CMAKE_COMMAND}
        -DPTX_FILE=${CMAKE_CURRENT_BINARY_DIR}/forward_tracer.ptx
        -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/embedded_ptx.h
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward_tracer.ptx
    COMMENT "Embedding PTX into C++ header"
)

add_custom_target(embedded_ptx_target ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/embedded_ptx.h
)

# 积分球模拟器
add_executable(ideal_sphere_sim
    src/app/main.cpp
    src/core/optix_context.cpp
    src/scene/scene.cpp
    src/simulation/path_tracer.cpp
    src/simulation/optix_pipeline_builder.cpp
    src/simulation/optix_sbt_builder.cpp
)

add_dependencies(ideal_sphere_sim ptx_target embedded_ptx_target)

target_include_directories(ideal_sphere_sim PRIVATE
    ${OptiX_INCLUDE}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}  # For embedded_ptx.h
)

target_link_libraries(ideal_sphere_sim PRIVATE
    theory
    spdlog::spdlog
    ${CMAKE_DL_LIBS}
    CUDA::cudart
)

# 复制 PTX 文件到可执行文件目录
add_custom_command(TARGET ideal_sphere_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/forward_tracer.ptx
        $<TARGET_FILE_DIR:ideal_sphere_sim>/forward_tracer.ptx
    COMMENT "Copying PTX file to executable directory"
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
gtest_add_tests(TARGET unit_tests)

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)
if (SKBUILD)
    message(STATUS "Building Python bindings...")

    # Find pybind11
    find_package(pybind11 CONFIG REQUIRED)

    # Python module
    pybind11_add_module(_core
        src/python/bindings.cpp
        src/core/optix_context.cpp
        src/scene/scene.cpp
        src/simulation/path_tracer.cpp
        src/simulation/optix_pipeline_builder.cpp
        src/simulation/optix_sbt_builder.cpp
    )

    add_dependencies(_core ptx_target embedded_ptx_target)

    target_include_directories(_core PRIVATE
        ${OptiX_INCLUDE}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}  # For embedded_ptx.h
    )

    target_link_libraries(_core PRIVATE
        theory
        spdlog::spdlog
        ${CMAKE_DL_LIBS}
        CUDA::cudart
    )

    # Install Python module (PTX is now embedded, no need to copy)
    install(TARGETS _core LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/optix_sphere)
endif()

message(STATUS "")
message(STATUS "Build: cd build && cmake .. && make")
message(STATUS "Test:  cd build && ctest --verbose")
message(STATUS "Run:   ./test_env")
message(STATUS "")
