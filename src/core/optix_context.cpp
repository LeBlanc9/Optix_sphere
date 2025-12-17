#include "optix_context.h"
#include "device_buffer.h" // For CUDA_CHECK
#include <iostream>

OptixContext::OptixContext() {
    try {
        init_cuda();
        init_optix();
    } catch (const std::exception& e) {
        std::cerr << "Error during OptiX context initialization: " << e.what() << std::endl;
        // In a real application, you might want to re-throw or handle this more gracefully
        exit(1);
    }
}

OptixContext::~OptixContext() {
    try {
        if (context_) {
            OPTIX_CHECK(optixDeviceContextDestroy(context_));
            std::cout << "✅ OptiX context destroyed" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during OptiX context destruction: " << e.what() << std::endl;
    }
}

void OptixContext::init_cuda() {
    CUDA_CHECK(cudaFree(0)); // Initializes the CUDA context
    
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA-capable devices found.");
    }

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "✅ CUDA Device: " << props.name << std::endl;
    
    // The CUcontext is implicitly managed by the CUDA runtime API
    // We can get it if needed, but for now we'll pass 0 to OptiX
}

void OptixContext::init_optix() {
    OPTIX_CHECK(optixInit());
    std::cout << "✅ OptiX initialized" << std::endl;

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &OptixContext::context_log_cb;
    options.logCallbackLevel = 4; // All messages

    // Note: The first argument is the CUcontext. 0 means use the current one.
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, &options, &context_));
    std::cout << "✅ OptiX context created" << std::endl;
}

void OptixContext::context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
    std::cerr << "[" << std::to_string(level) << "][" << tag << "]: " << message << std::endl;
}
