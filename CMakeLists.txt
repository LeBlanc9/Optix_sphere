cmake_minimum_required(VERSION 3.18)
project(optix_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING)
endif()

# OptiX 路径
if(WIN32)
    set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
else()
    set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64")
endif()
set(OptiX_INCLUDE "${OptiX_INSTALL_DIR}/include")

# CUDA
find_package(CUDAToolkit REQUIRED)

include(cmake/dependencies.cmake)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/optix_sphere
)

# 编译 CUDA 为 PTX
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_tracer.ptx
    COMMAND ${CMAKE_CUDA_COMPILER}
        -ptx
        -arch=sm_75
        -I${OptiX_INCLUDE}
        -I${CUDAToolkit_INCLUDE_DIRS}
        -I${CMAKE_CURRENT_SOURCE_DIR}/optix_sphere
        -I${CMAKE_CURRENT_SOURCE_DIR}/optix_sphere/simulation
        ${CMAKE_CURRENT_SOURCE_DIR}/optix_sphere/simulation/kernels.cu
        -o ${CMAKE_CURRENT_BINARY_DIR}/forward_tracer.ptx
    DEPENDS
        optix_sphere/simulation/kernels.cu
        optix_sphere/simulation/kernels/kernel_utils.cuh
        optix_sphere/simulation/kernels/intersection.cu
        optix_sphere/simulation/kernels/radiance.cu
        optix_sphere/simulation/kernels/shadow.cu
        optix_sphere/simulation/kernels/raygen.cu
        optix_sphere/simulation/device_params.h
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

# ============================================
# Core Library: libOptixSphere
# ============================================
add_library(OptixSphere STATIC
    optix_sphere/core/optix_context.cpp
    optix_sphere/scene/scene.cpp
    optix_sphere/simulation/path_tracer.cpp
    optix_sphere/simulation/optix_pipeline_builder.cpp
    optix_sphere/simulation/optix_sbt_builder.cpp
    optix_sphere/geometry/mesh_loader.cpp
    # optix_sphere/geometry/fbx_loader.cpp
    optix_sphere/theory/theory.cpp
    optix_sphere/material.cpp
)

add_dependencies(OptixSphere ptx_target embedded_ptx_target)

target_include_directories(OptixSphere PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/optix_sphere
    ${OptiX_INCLUDE}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}  # For embedded_ptx.h
)

target_link_libraries(OptixSphere PUBLIC
    spdlog::spdlog
    tinyobjloader
    ${CMAKE_DL_LIBS}
    CUDA::cudart
)


if (SKBUILD)
    # ============================================
    # Python Bindings (pybind11)
    # ============================================
    message(STATUS "Building Python bindings...")

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_core
        optix_sphere/python/bindings.cpp
    )

    target_link_libraries(_core PRIVATE
        OptixSphere
    )

    install(TARGETS _core LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/optix_sphere)

else()
    # ============================================
    #  Test 
    # ============================================
    message(STATUS "Building Tests...")

    enable_testing()

    add_executable(main_test
        tests/main.cpp
    )
    target_link_libraries(main_test PRIVATE OptixSphere)
    add_test(NAME IntegrationTest COMMAND main_test)


endif()

message(STATUS "")
message(STATUS "====================================")
message(STATUS "Optix Sphere")
message(STATUS "====================================")
if (SKBUILD)
    message(STATUS "Mode: Python Package Build")
    message(STATUS "Targets:")
    message(STATUS "  - libOptixSphere.a (core library)")
    message(STATUS "  - _core.pyd (Python module)")
else()
    message(STATUS "Mode: C++ Development Build")
    message(STATUS "Targets:")
    message(STATUS "  - libOptixSphere.a (core library)")
    message(STATUS "  - integration_test.exe (main.cpp)")
    message(STATUS "")
    message(STATUS "Commands:")
    message(STATUS "  Build:   cmake -B build && cmake --build build --config Debug")
    message(STATUS "  Test:    cd build && ctest -C Debug --verbose")
    message(STATUS "  Run:     build/Debug/integration_test.exe")
endif()
message(STATUS "====================================")
message(STATUS "")
