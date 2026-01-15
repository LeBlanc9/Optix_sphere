#include "voxel_simulator.h"
#include "photon/launchers.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ctime>

#include "utils/device/math.cuh"

namespace phonder::voxel {

// Constructor
Simulator::Simulator(
    GridBuilder& grid_builder,
    const PhotonSource& source,
    int gpu_id
) {
    config_.device_grid = grid_builder.get_device_grid();
    config_.source = source;
    config_.gpu_id = gpu_id;

    // Set exit boundaries
    const auto& host_grid = grid_builder.get_host_grid();
    config_.exit_z_min = 0.0f;
    config_.exit_z_max = host_grid.nz * host_grid.dz;

    // Set GPU device
    cudaSetDevice(gpu_id);
}

// Run simulation with specified number of photons
SimulationResult Simulator::run(int num_photons) {
    // Generate initial photon batch from source
    PhotonBatch input_batch(num_photons);
    generate_photons_on_device(
        config_.source,
        input_batch,
        num_photons,
        static_cast<unsigned long long>(std::time(nullptr))
    );

    // Run simulation with the batch
    return run(input_batch);
}

// Run simulation with input batch
SimulationResult Simulator::run(const PhotonBatch& input_batch) {
    int input_size = input_batch.size();
    int output_capacity = input_size;  // Output buffer size equals input size

    // Allocate output buffers for all three batches
    PhotonBatch specular_batch(output_capacity);
    PhotonBatch reflected_batch(output_capacity);
    PhotonBatch transmitted_batch(output_capacity);

    // Counters for output
    int* d_specular_counter;
    int* d_reflected_counter;
    int* d_transmitted_counter;
    cudaMalloc(&d_specular_counter, sizeof(int));
    cudaMalloc(&d_reflected_counter, sizeof(int));
    cudaMalloc(&d_transmitted_counter, sizeof(int));
    cudaMemset(d_specular_counter, 0, sizeof(int));
    cudaMemset(d_reflected_counter, 0, sizeof(int));
    cudaMemset(d_transmitted_counter, 0, sizeof(int));

    // Launch kernel
    int block_size = 256;
    int grid_size = (input_size + block_size - 1) / block_size;

    mc_kernel<<<grid_size, block_size>>>(
        config_.device_grid,
        input_batch.c_positions_ptr(),
        input_batch.c_directions_ptr(),
        input_batch.c_weights_ptr(),
        input_size,
        specular_batch.positions_ptr(),
        specular_batch.directions_ptr(),
        specular_batch.weights_ptr(),
        reflected_batch.positions_ptr(),
        reflected_batch.directions_ptr(),
        reflected_batch.weights_ptr(),
        transmitted_batch.positions_ptr(),
        transmitted_batch.directions_ptr(),
        transmitted_batch.weights_ptr(),
        d_specular_counter,
        d_reflected_counter,
        d_transmitted_counter,
        output_capacity,
        config_.exit_z_min,
        config_.exit_z_max,
        static_cast<unsigned long long>(std::time(nullptr))
    );

    cudaDeviceSynchronize();

    // Get actual counts
    int specular_count, reflected_count, transmitted_count;
    cudaMemcpy(&specular_count, d_specular_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&reflected_count, d_reflected_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&transmitted_count, d_transmitted_counter, sizeof(int), cudaMemcpyDeviceToHost);


    // Resize batches to actual size
    specular_batch.resize(std::min(specular_count, output_capacity));
    reflected_batch.resize(std::min(reflected_count, output_capacity));
    transmitted_batch.resize(std::min(transmitted_count, output_capacity));

    // Cleanup
    cudaFree(d_specular_counter);
    cudaFree(d_reflected_counter);
    cudaFree(d_transmitted_counter);

    return {specular_batch, reflected_batch, transmitted_batch};
}

}; // namespace phonder::voxel
