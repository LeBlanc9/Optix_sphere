#include "optix_context.h"
#include "utils/device/core.cuh"
#include <optix_function_table_definition.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

OptixContext::OptixContext(int device_id) : device_id_(device_id) {
    try {
        init_cuda();
        init_optix();
    } catch (const std::exception& e) {
        spdlog::error("Error during OptiX context initialization: {}", e.what());
        exit(1);
    }
}

OptixContext::~OptixContext() {
    try {
        if (context_) {
            OPTIX_CHECK(optixDeviceContextDestroy(context_));
            spdlog::info("✅ OptiX context destroyed");
        }
    } catch (const std::exception& e) {
        spdlog::error("Error during OptiX context destruction: {}", e.what());
    }
}

void OptixContext::init_cuda() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA-capable devices found.");
    }

    if (device_id_ < 0 || device_id_ >= device_count) {
        throw std::runtime_error(fmt::format(
            "Invalid device ID {}. Available devices: 0-{}", device_id_, device_count - 1));
    }

    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaFree(0)); // Initializes the CUDA context

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id_));
    spdlog::info("✅ CUDA Device {}: {}", device_id_, props.name);
}

void OptixContext::init_optix() {
    OPTIX_CHECK(optixInit());
    spdlog::info("✅ OptiX initialized");

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &OptixContext::context_log_cb;
    options.logCallbackLevel = 4; // All messages

    // Note: The first argument is the CUcontext. 0 means use the current one.
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, &options, &context_));
    spdlog::info("✅ OptiX context created");
}

void OptixContext::context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
    // OptiX log levels: 1=Fatal, 2=Error, 3=Warning, 4=Info
    std::string formatted_msg = fmt::format("[{}]: {}", tag, message);

    switch (level) {
        case 1: // Fatal
        case 2: // Error
            spdlog::error("{}", formatted_msg);
            break;
        case 3: // Warning
            spdlog::warn("{}", formatted_msg);
            break;
        case 4: // Info
        default:
            spdlog::debug("{}", formatted_msg);
            break;
    }
}
