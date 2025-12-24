#pragma once
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

// This is a CUDA-side header.
// It defines the data structures for holding photon batches on the device.

namespace phonder {

// Forward-declare the C++ host-side batch for conversion functions
struct HostPhotonBatch;

/**
 * @struct PhotonBatchView
 * @brief A non-owning, POD view of photon batch data for use in CUDA kernels.
 */
struct PhotonBatchView {
    float3* positions;
    float3* directions;
    double* weights;
};

/**
 * @struct DevicePhotonBatch
 * @brief A structure to hold a batch of photons on the GPU using SoA layout.
 */
struct DevicePhotonBatch {
    thrust::device_vector<float3> positions;
    thrust::device_vector<float3> directions;
    thrust::device_vector<double> weights;

    void resize(size_t new_size);
    void clear();
    size_t size() const;
    bool empty() const;

    // Conversion to host
    HostPhotonBatch to_host() const;

    // Get a view for passing to kernels (non-const version for writing)
    PhotonBatchView get_view();

    // Get a view for passing to kernels (const version for reading)
    PhotonBatchView get_view() const;
};

} // namespace phonder
