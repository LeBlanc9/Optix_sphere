# Python bindings for optix_sphere
# This subdirectory encapsulates pybind11 module setup.

message(STATUS "Configuring Python bindings...")
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
    module.cu
    logging_bindings.cu
    types_bindings.cu
    sim_bindings.cu
    photon_bindings.cu
    material_bindings.cu
    media_bindings.cu
    voxel_bindings.cu
)

target_link_libraries(_core PRIVATE
    sphere
    media
    voxel_media
    spdlog::spdlog
)

target_compile_definitions(_core PRIVATE FMT_UNICODE=0)

if(SKBUILD)
    message(STATUS "Configuring pybind11-stubgen to generate type stubs...")

    # Generate stubs right after building _core.so
    add_custom_command(
        TARGET _core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Generating Python stub files next to _core.so..."
        COMMAND "${Python3_EXECUTABLE}" -m pybind11_stubgen
            _core
            -o "${CMAKE_CURRENT_BINARY_DIR}"
            --ignore-all-errors
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating type stubs for _core module"
        VERBATIM
    )

    # Install _core.so to site-packages
    install(TARGETS _core LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/optix_sphere)

    # Install generated stub files if they exist
    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_core/
        DESTINATION ${SKBUILD_PLATLIB_DIR}/optix_sphere
        OPTIONAL
        FILES_MATCHING
        PATTERN "*.pyi"
    )

    # --- Bundle dependent DLLs on Windows ---
    if(WIN32)
        message(STATUS "Searching for CUDA runtime DLL to bundle in wheel...")
        # CUDAToolkit_BIN_DIR is provided by find_package(CUDAToolkit) in the root CMakeLists
        file(GLOB CUDA_RUNTIME_DLL "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
        if(CUDA_RUNTIME_DLL)
            message(STATUS "  -> Found and bundling: ${CUDA_RUNTIME_DLL}")
            install(FILES ${CUDA_RUNTIME_DLL} DESTINATION ${SKBUILD_PLATLIB_DIR}/optix_sphere)
        else()
            message(WARNING "Could not find CUDA runtime DLL to bundle. The installed wheel may require CUDA runtime to be in the system PATH.")
        endif()
    endif()
endif()