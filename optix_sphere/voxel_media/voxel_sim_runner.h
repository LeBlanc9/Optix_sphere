#pragma once
#include <cuda_runtime.h>
#include "voxel_grid.cuh"

namespace phonder::voxel {

/**
 * @brief Upload material properties to constant memory
 * erials Packed float4 array (n, mua, mus, g)
 * @param num_materials Number of materials
 */
void upload_materials_to_constant_memory(const float4* materials, int num_materials);

/**
 * @brief Run voxel Monte Carlo simulation kernel
 *
 * This function encapsulates the kernel launch and constant memory access.
 * All parameters are device pointers.
 */
void run_voxel_simulation(
    const Grid* grid,
    const float3* input_positions,
    const float3* input_directions,
    const double* input_weights,
    int input_size,
    float3* specular_positions,
    float3* specular_directions,
    double* specular_weights,
    float3* reflected_positions,
    float3* reflected_directions,
    double* reflected_weights,
    float3* transmitted_positions,
    float3* transmitted_directions,
    double* transmitted_weights,
    int* specular_counter,
    int* reflected_counter,
    int* transmitted_counter,
    int output_capacity,
    float exit_z_min,
    float exit_z_max,
    unsigned long long seed
);

} // namespace phonder::voxel
