cmake_minimum_required(VERSION 3.18)
project(optix_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Option to control building Python bindings (also set automatically when using scikit-build)
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)

if(MSVC)
    add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING NOMINMAX)
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
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

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
        optix_sphere/simulation/kernels/source_generators.cu
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
# Library Definitions (Modular Approach)
# ============================================

# --------------------------------------------
# 1. 'photon' Library (Core utilities)
#    - Photon data structures (SoA)
#    - GPU-side photon source generators
# --------------------------------------------
add_library(photon STATIC
    optix_sphere/photon/batch.cu
    optix_sphere/photon/launchers.cu
)
set_target_properties(photon PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 75
)
target_include_directories(photon PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/optix_sphere
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(photon PUBLIC
    nlohmann_json::nlohmann_json
    CUDA::curand
)
message(STATUS "Library 'photon' configured.")

# --------------------------------------------
# 2. 'media' Library (Layered Media Simulation)
#    - Depends on the 'photon' library
# --------------------------------------------
add_library(media STATIC
    optix_sphere/layered_media/media_simulator.cu
)
set_target_properties(media PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 75
)
target_include_directories(media PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/optix_sphere
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(media PRIVATE
    photon
)
message(STATUS "Library 'media' configured, depends on 'photon'.")

# --------------------------------------------
# 3. 'sphere' Library (OptiX Integrating Sphere)
#    - The main path tracing engine
#    - Depends on the 'photon' library
# --------------------------------------------
add_library(sphere STATIC
    optix_sphere/simulation/optix_context.cpp
    optix_sphere/scene/scene.cpp
    optix_sphere/simulation/path_tracer.cpp
    optix_sphere/simulation/optix_pipeline_builder.cpp
    optix_sphere/simulation/optix_sbt_builder.cpp
    optix_sphere/geometry/mesh_loader.cpp
    optix_sphere/theory/theory.cpp
    optix_sphere/material.cpp
    optix_sphere/simulator.cpp
)
set_target_properties(sphere PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 75
)
add_dependencies(sphere ptx_target embedded_ptx_target)
target_include_directories(sphere PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/optix_sphere
    ${OptiX_INCLUDE}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}  # For embedded_ptx.h
)
target_link_libraries(sphere PUBLIC
    photon
    spdlog::spdlog
    tinyobjloader
    ${CMAKE_DL_LIBS}
    CUDA::cudart
)
message(STATUS "Library 'sphere' configured, depends on 'photon'.")

# ============================================
# Final Target Linking
# ============================================

if (SKBUILD OR BUILD_PYTHON_BINDINGS)
    # Python Bindings
    message(STATUS "Configuring Python bindings...")
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core
        optix_sphere/python/bindings.cpp
    )
    # The Python module needs to link to all relevant libraries
    target_link_libraries(_core PRIVATE
        sphere
        media
    )
    if (SKBUILD)
        install(TARGETS _core LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/optix_sphere)

        # --- Bundle dependent DLLs ---
        # Find and install the CUDA runtime DLL next to the _core.pyd module.
        # This makes the wheel self-contained and avoids DLL load errors on Windows.
        if(WIN32)
            message(STATUS "Searching for CUDA runtime DLL to bundle in wheel...")
            # CUDAToolkit_BIN_DIR is provided by find_package(CUDAToolkit)
            file(GLOB CUDA_RUNTIME_DLL "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
            if(CUDA_RUNTIME_DLL)
                message(STATUS "  -> Found and bundling: ${CUDA_RUNTIME_DLL}")
                install(FILES ${CUDA_RUNTIME_DLL} DESTINATION ${SKBUILD_PLATLIB_DIR}/optix_sphere)
            else()
                message(WARNING "Could not find CUDA runtime DLL to bundle. The installed wheel may require CUDA runtime to be in the system PATH.")
            endif()
        endif()
    endif()
else()
    # Test Executables
    message(STATUS "Configuring Test executables...")
    enable_testing()

    # Main integration test (uses both sphere and media modules)
    add_executable(main_test tests/main.cpp)
    target_link_libraries(main_test PRIVATE
        sphere
        media
    )
    add_test(NAME IntegrationTest COMMAND main_test)

    # Layered medium test (only needs media module)
    add_executable(test_layered_medium tests/test_layered_medium.cu)
    target_link_libraries(test_layered_medium PRIVATE
        media
    )
    add_test(NAME LayeredMediumTest COMMAND test_layered_medium)

endif()

message(STATUS "")
message(STATUS "====================================")
message(STATUS "Optix Sphere - Modular Build")
message(STATUS "====================================")
if (SKBUILD OR BUILD_PYTHON_BINDINGS)
    message(STATUS "Mode: Python Package Build")
    message(STATUS "Libraries: photon, media, sphere")
    message(STATUS "Final Target: _core.pyd (Python module)")
else()
    message(STATUS "Mode: C++ Development Build")
    message(STATUS "Libraries: photon, media, sphere")
    message(STATUS "Final Targets: main_test.exe, test_layered_medium.exe")
    message(STATUS "")
    message(STATUS "Commands:")
    message(STATUS "  Build:   cmake -B build && cmake --build build --config Debug")
    message(STATUS "  Test:    cd build && ctest -C Debug --verbose")
    message(STATUS "  Run main:    build/Debug/main_test.exe")
    message(STATUS "  Run media:   build/Debug/test_layered_medium.exe")
endif()
message(STATUS "====================================")
message(STATUS "")

