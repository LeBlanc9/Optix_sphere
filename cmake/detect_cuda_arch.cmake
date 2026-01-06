# Automatically detect CUDA GPU architecture
# This script runs a small CUDA program to detect the compute capability of the local GPU

function(detect_cuda_architectures output_var)
    # First, try using nvidia-smi (fastest method)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE COMPUTE_CAP
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(COMPUTE_CAP)
            # Remove the dot: 8.9 -> 89
            string(REPLACE "." "" ARCH_NUMBER "${COMPUTE_CAP}")
            # Get the first GPU if multiple GPUs exist
            string(REGEX MATCH "^[0-9]+" ARCH_NUMBER "${ARCH_NUMBER}")
            message(STATUS "Detected GPU compute capability: ${COMPUTE_CAP} (arch: ${ARCH_NUMBER})")
            set(${output_var} ${ARCH_NUMBER} PARENT_SCOPE)
            return()
        endif()
    endif()

    # Fallback: Create and run a small CUDA program to detect architecture
    set(DETECT_SOURCE "${CMAKE_BINARY_DIR}/detect_cuda_arch.cu")
    set(DETECT_BINARY "${CMAKE_BINARY_DIR}/detect_cuda_arch")

    file(WRITE ${DETECT_SOURCE}
"#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        return 1;
    }
    // Print compute capability as: major.minor
    printf(\"%d%d\\\\n\", prop.major, prop.minor);
    return 0;
}
")

    # Try to compile and run the detection program
    try_run(
        RUN_RESULT
        COMPILE_RESULT
        ${CMAKE_BINARY_DIR}
        ${DETECT_SOURCE}
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
        RUN_OUTPUT_VARIABLE RUN_OUTPUT
    )

    if(COMPILE_RESULT AND RUN_RESULT EQUAL 0)
        string(STRIP "${RUN_OUTPUT}" ARCH_NUMBER)
        message(STATUS "Detected GPU architecture via CUDA program: sm_${ARCH_NUMBER}")
        set(${output_var} ${ARCH_NUMBER} PARENT_SCOPE)
    else()
        # Ultimate fallback: use a common architecture
        message(WARNING "Could not auto-detect GPU architecture. Falling back to sm_75 (Turing). "
                        "You may need to manually set CMAKE_CUDA_ARCHITECTURES.")
        set(${output_var} "75" PARENT_SCOPE)
    endif()
endfunction()
