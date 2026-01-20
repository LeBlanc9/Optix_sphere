#include "layered_media/media_simulator.cuh"
#include "layered_media/media_kernel.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdexcept>
#include <algorithm>

namespace phonder {



static MediaSimulationResult _run_simulation(const MediaSimConfig& config, const PhotonBatch& input_batch) {
    size_t num_photons = input_batch.size();
    if (num_photons == 0) {
        return MediaSimulationResult{};
    }

    MediaSimulationResult result;
    // Allocate 2x capacity directly on the device
    result.reflected_batch.resize(num_photons * 2);
    result.transmitted_batch.resize(num_photons * 2);

    double* d_specular_reflection_weight;
    cudaMalloc((void**)&d_specular_reflection_weight, sizeof(double));
    cudaMemset(d_specular_reflection_weight, 0, sizeof(double));

    MediaKernelParams params;
    params.output_buffer_capacity = num_photons * 2;
    
    // Get const pointers for the input batch
    params.input_positions = input_batch.positions();
    params.input_directions = input_batch.directions();
    params.input_weights = input_batch.weights();
    params.input_batch_size = num_photons;

    // Pass writable raw pointers for output buffers
    params.reflected_positions = result.reflected_batch.positions();
    params.reflected_directions = result.reflected_batch.directions();
    params.reflected_weights = result.reflected_batch.weights();

    params.transmitted_positions = result.transmitted_batch.positions();
    params.transmitted_directions = result.transmitted_batch.directions();
    params.transmitted_weights = result.transmitted_batch.weights();
    
    params.specular_reflection_weight = d_specular_reflection_weight;

    params.seed = static_cast<unsigned long long>(time(nullptr)) + config.gpu_id * 1000000;
    params.reflected_radius = config.reflected_radius;
    params.transmitted_radius = config.transmitted_radius;

    cudaMalloc((void**)&params.reflected_counter, sizeof(int));
    cudaMalloc((void**)&params.transmitted_counter, sizeof(int));
    cudaMemset(params.reflected_counter, 0, sizeof(int));
    cudaMemset(params.transmitted_counter, 0, sizeof(int));

    cudaMalloc((void**)&params.medium, sizeof(LayeredMedium));
    cudaMemcpy((void*)params.medium, &config.medium, sizeof(LayeredMedium), cudaMemcpyHostToDevice);

    MediaKernelParams* d_params;
    cudaMalloc((void**)&d_params, sizeof(MediaKernelParams));
    cudaMemcpy(d_params, &params, sizeof(MediaKernelParams), cudaMemcpyHostToDevice);

    int block_size = 256;
    int max_grid_size = 1024;
    int grid_size = std::min((static_cast<int>(num_photons) + block_size - 1) / block_size, max_grid_size);
    media_simulation_kernel<<<grid_size, block_size>>>(d_params);

    cudaDeviceSynchronize();

    int reflected_count, transmitted_count;
    cudaMemcpy(&reflected_count, params.reflected_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&transmitted_count, params.transmitted_counter, sizeof(int), cudaMemcpyDeviceToHost);

    double h_specular_reflection_weight;
    cudaMemcpy(&h_specular_reflection_weight, d_specular_reflection_weight, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree((void*)params.medium);
    cudaFree(params.reflected_counter);
    cudaFree(params.transmitted_counter);
    cudaFree(d_params);
    cudaFree(d_specular_reflection_weight);

    // Resize the device vectors to the actual number of photons.
    result.reflected_batch.resize(reflected_count);
    result.transmitted_batch.resize(transmitted_count);

    result.specular_reflection_weight = h_specular_reflection_weight;
    return result;
}


__host__ MediaSimulationResult MediaSimulator::run(int num_photons) {
    cudaSetDevice(config_.gpu_id);

    PhotonBatch input_batch;
    // Create a seed for this run. Note: time(nullptr) has 1-second resolution.
    // For rapid subsequent calls, a better random seed source might be needed.
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr)) + config_.gpu_id;

    if (config_.source) {
        config_.source->generate(input_batch, num_photons, seed);
    } else {
        throw std::runtime_error("No photon source configured");
    }

    return _run_simulation(config_, input_batch);
}

__host__ MediaSimulationResult MediaSimulator::run(const PhotonBatch& input_batch) {
    cudaSetDevice(config_.gpu_id);
    return _run_simulation(config_, input_batch);
}

} // namespace phonder
