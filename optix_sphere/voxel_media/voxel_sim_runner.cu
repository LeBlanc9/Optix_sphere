#include "voxel_sim_runner.h"
#include "voxel_grid.cuh"
#include "kernels/voxel_mc_kernel.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace phonder::voxel {

// Constant memory definition (only defined once here)
__constant__ float4 c_materials[MAX_MATERIALS];

SimulationResult run_voxel_simulation(
    const Grid* grid_struct,
    const PhotonBatch& input_batch,
    const BoundaryCollectionConfig& boundary_config,
    bool enable_specular,
    bool merge_specular,
    unsigned long long seed
) {
    int input_size = input_batch.size();

    // Allocate output batches (on device)
    // If merge_specular=true, boundary batches need 2x capacity (for specular + boundary photons)
    int boundary_capacity = merge_specular ? (input_size * 2) : input_size;
    int specular_capacity = merge_specular ? 0 : input_size;

    SimulationResult result;
    result.specular_batch = PhotonBatch(specular_capacity);
    result.negative_boundary_batch = PhotonBatch(boundary_capacity);
    result.positive_boundary_batch = PhotonBatch(boundary_capacity);

    // Allocate and initialize device counters
    int* d_specular_counter;
    int* d_negative_counter;
    int* d_positive_counter;
    cudaMalloc(&d_specular_counter, sizeof(int));
    cudaMalloc(&d_negative_counter, sizeof(int));
    cudaMalloc(&d_positive_counter, sizeof(int));
    cudaMemset(d_specular_counter, 0, sizeof(int));
    cudaMemset(d_negative_counter, 0, sizeof(int));
    cudaMemset(d_positive_counter, 0, sizeof(int));

    // Prepare boundary config on host
    DeviceBoundaryConfig host_bc;
    for (int i = 0; i < 6; i++) {
        host_bc.collect_faces[i] = boundary_config.collect_faces[i];
    }
    host_bc.collection_center = boundary_config.collection_center;
    host_bc.collection_radius_negative = boundary_config.collection_radius_negative;
    host_bc.collection_radius_positive = boundary_config.collection_radius_positive;

    // Auto-set collection center to grid center if requested
    if (boundary_config.use_grid_center) {
        Grid host_grid;
        cudaMemcpy(&host_grid, grid_struct, sizeof(Grid), cudaMemcpyDeviceToHost);
        host_bc.collection_center = make_float3(
            host_grid.dims.x * host_grid.voxel_size.x * 0.5f,
            host_grid.dims.y * host_grid.voxel_size.y * 0.5f,
            0.0f  // Z center is typically 0 or grid surface
        );
    }

    // Build kernel parameters
    MCKernelParams params;

    // Grid - pass device pointer directly
    params.grid = grid_struct;

    // Input
    params.input.positions = input_batch.positions();
    params.input.directions = input_batch.directions();
    params.input.weights = input_batch.weights();
    params.input.size = input_size;

    // Specular batch
    params.specular_batch.positions = result.specular_batch.positions();
    params.specular_batch.directions = result.specular_batch.directions();
    params.specular_batch.weights = result.specular_batch.weights();
    params.specular_batch.counter = d_specular_counter;
    params.specular_batch.capacity = specular_capacity;

    // Negative boundary batch
    params.negative_boundary_batch.positions = result.negative_boundary_batch.positions();
    params.negative_boundary_batch.directions = result.negative_boundary_batch.directions();
    params.negative_boundary_batch.weights = result.negative_boundary_batch.weights();
    params.negative_boundary_batch.counter = d_negative_counter;
    params.negative_boundary_batch.capacity = boundary_capacity;

    // Positive boundary batch
    params.positive_boundary_batch.positions = result.positive_boundary_batch.positions();
    params.positive_boundary_batch.directions = result.positive_boundary_batch.directions();
    params.positive_boundary_batch.weights = result.positive_boundary_batch.weights();
    params.positive_boundary_batch.counter = d_positive_counter;
    params.positive_boundary_batch.capacity = boundary_capacity;

    // Boundary config (already prepared on host, copy directly)
    params.boundary_config = host_bc;

    // Simulation parameters
    params.enable_specular = enable_specular;
    params.merge_specular = merge_specular;
    params.seed = seed;

    // Launch kernel
    int block_size = 256;
    int grid_size = (input_size + block_size - 1) / block_size;
    voxel_kernel<<<grid_size, block_size>>>(params);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_specular_counter);
        cudaFree(d_negative_counter);
        cudaFree(d_positive_counter);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    // Synchronize
    cudaDeviceSynchronize();

    // Read back counters
    int h_specular_count, h_negative_count, h_positive_count;
    cudaMemcpy(&h_specular_count, d_specular_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_negative_count, d_negative_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_positive_count, d_positive_counter, sizeof(int), cudaMemcpyDeviceToHost);

    // Resize output batches to actual counts
    result.specular_batch.resize(h_specular_count);
    result.negative_boundary_batch.resize(h_negative_count);
    result.positive_boundary_batch.resize(h_positive_count);

    // Cleanup
    cudaFree(d_specular_counter);
    cudaFree(d_negative_counter);
    cudaFree(d_positive_counter);

    return result;
}

} // namespace phonder::voxel
