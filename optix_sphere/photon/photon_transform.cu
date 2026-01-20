#include "photon_transform.cuh"
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

namespace phonder {

// CUDA kernel: translate photon positions
__global__ void translate_positions_kernel(
    float3* positions,
    size_t num_photons,
    float3 offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    // Simple translation: new_pos = old_pos + offset
    positions[idx].x += offset.x;
    positions[idx].y += offset.y;
    positions[idx].z += offset.z;
}

PhotonBatch translate_photons(
    const PhotonBatch& input_batch,
    float3 offset
) {
    size_t num_photons = input_batch.size();
    if (num_photons == 0) {
        return input_batch; // Empty batch, return a copy
    }

    // Create a copy of the input batch. The copy constructor handles
    // ensuring the data is consistent.
    PhotonBatch result = input_batch;

    // Translate the copy in-place
    translate_photons_inplace(result, offset);

    return result;
}

void translate_photons_inplace(
    PhotonBatch& batch,
    float3 offset
) {
    size_t num_photons = batch.size();
    if (num_photons == 0) {
        return; // Nothing to do for empty batch
    }

    // Get a writeable pointer to the device-side data.
    float3* pos_ptr = batch.positions();

    // Launch kernel to translate positions
    const int block_size = 256;
    const int grid_size = (num_photons + block_size - 1) / block_size;

    translate_positions_kernel<<<grid_size, block_size>>>(
        pos_ptr,
        num_photons,
        offset
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel error in translate_photons_inplace: ") +
            cudaGetErrorString(err)
        );
    }

    // Wait for kernel completion
    cudaDeviceSynchronize();
}

} // namespace phonder
