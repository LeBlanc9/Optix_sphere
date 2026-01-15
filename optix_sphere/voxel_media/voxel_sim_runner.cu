#include "voxel_sim_runner.h"
#include "voxel_grid.cuh"
#include "kernels/voxel_mc_kernel.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace phonder::voxel {

// Constant memory definition (only defined once here)
__constant__ float4 c_materials[MAX_MATERIALS];

void upload_materials_to_constant_memory(const float4* materials, int num_materials) {
    if (num_materials > MAX_MATERIALS) {
        throw std::runtime_error("Number of materials exceeds MAX_MATERIALS (256)");
    }

    cudaError_t err = cudaMemcpyToSymbol(c_materials, materials,
                                          num_materials * sizeof(float4));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to upload materials to constant memory: ")
                                 + cudaGetErrorString(err));
    }
}

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
) {
    int block_size = 64;
    int grid_size = (input_size + block_size - 1) / block_size;

    mc_kernel<<<grid_size, block_size>>>(
        grid,
        input_positions,
        input_directions,
        input_weights,
        input_size,
        specular_positions,
        specular_directions,
        specular_weights,
        reflected_positions,
        reflected_directions,
        reflected_weights,
        transmitted_positions,
        transmitted_directions,
        transmitted_weights,
        specular_counter,
        reflected_counter,
        transmitted_counter,
        output_capacity,
        exit_z_min,
        exit_z_max,
        seed
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace phonder::voxel
