#pragma once
#include <cuda_runtime.h>

namespace phonder::voxel {

/**
 * @brief Device-side grid structure used in CUDA kernels
 *
 * This is a lightweight structure containing pointers to device memory.
 * Used for passing grid data to CUDA kernels.
 *
 * Note: Ambient properties are read from c_materials[0] (material 0).
 */
struct DeviceGrid {
    unsigned char* material_ids;  // Device pointer to material ID array
    int3 dims;                    // Grid dimensions (nx, ny, nz)
    float3 voxel_size;            // Voxel size in mm
    int num_materials;            // Number of material types
};

} // namespace phonder::voxel
