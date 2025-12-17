#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

// OptiX 错误检查宏
#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            throw std::runtime_error("OptiX error at " + std::string(__FILE__) \
                                     + ":" + std::to_string(__LINE__)          \
                                     + " - " + optixGetErrorString(res));      \
        }                                                                      \
    } while (0)

/**
 * @brief A RAII wrapper for the OptiX device context.
 *
 * This class handles the initialization and destruction of the OptiX context,
 * providing a central place for OptiX-related setup.
 */
class OptixContext {
public:
    OptixContext();
    ~OptixContext();

    // Disable copy and move
    OptixContext(const OptixContext&) = delete;
    OptixContext& operator=(const OptixContext&) = delete;
    OptixContext(OptixContext&&) = delete;
    OptixContext& operator=(OptixContext&&) = delete;

    OptixDeviceContext get() const { return context_; }

private:
    void init_cuda();
    void init_optix();

    static void context_log_cb(unsigned int level, const char* tag, const char* message, void*);

    OptixDeviceContext context_ = nullptr;
    CUcontext cuda_context_ = 0; // Not owned, just a handle
};

