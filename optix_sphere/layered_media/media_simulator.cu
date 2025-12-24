#include "layered_media/media_simulator.cuh"
#include "layered_media/media_kernel.cuh"
#include "photon/launchers.h" // New C-style launcher API
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdexcept>
#include <algorithm> // For std::min

namespace phonder {

MediaSimulationResult MediaSimulationDeviceResult::to_host() const {
    MediaSimulationResult cpu_result;
    cpu_result.reflected_batch = this->reflected_batch.to_host();
    cpu_result.transmitted_batch = this->transmitted_batch.to_host();
    cpu_result.specular_reflection_weight = this->specular_reflection_weight;
    return cpu_result;
}

static MediaSimulationDeviceResult _run_simulation(const MediaSimConfig& config, const DevicePhotonBatch& input_batch) {
    int num_photons = input_batch.size();
    if (num_photons == 0) {
        return MediaSimulationDeviceResult{};
    }

    DevicePhotonBatch reflected_batch_out;
    DevicePhotonBatch transmitted_batch_out;
    reflected_batch_out.resize(num_photons);
    transmitted_batch_out.resize(num_photons);

    double* d_specular_reflection_weight;
    cudaMalloc((void**)&d_specular_reflection_weight, sizeof(double));
    cudaMemset(d_specular_reflection_weight, 0, sizeof(double));

    MediaKernelParams params;
    params.input_batch_size = num_photons;
    params.output_buffer_capacity = num_photons;
    
    // The get_view() method now returns a non-const view, which is what the kernel needs
    // But for an *input* batch, it should be const. Let's make it const.
    // The issue is that the input_batch is passed as const, so get_view() must be const.
    // And get_view() must return a view with const pointers.
    // However, the kernel `media_simulation_kernel` takes a `const MediaKernelParams*`, 
    // and its `input_batch` member is a `PhotonBatchView` with non-const pointers.
    // This is a design flaw from the original code. Let's fix PhotonBatchView to use const pointers.
    // And make get_view() return a const view.
    // And make the kernel expect a const view.
    // For now, I will assume the view has non-const pointers and the kernel is okay with it.
    // This will likely cause a compile error, which I will fix next.
    // 现在get_view()有const重载版本，可以安全调用
    params.input_batch = input_batch.get_view();
    
    // Pass writable raw pointers for output buffers
    params.reflected_positions = thrust::raw_pointer_cast(reflected_batch_out.positions.data());
    params.reflected_directions = thrust::raw_pointer_cast(reflected_batch_out.directions.data());
    params.reflected_weights = thrust::raw_pointer_cast(reflected_batch_out.weights.data());
    params.transmitted_positions = thrust::raw_pointer_cast(transmitted_batch_out.positions.data());
    params.transmitted_directions = thrust::raw_pointer_cast(transmitted_batch_out.directions.data());
    params.transmitted_weights = thrust::raw_pointer_cast(transmitted_batch_out.weights.data());
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
    int grid_size = std::min((num_photons + block_size - 1) / block_size, max_grid_size);
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

    reflected_batch_out.resize(reflected_count);
    transmitted_batch_out.resize(transmitted_count);
    
    MediaSimulationDeviceResult result;
    result.reflected_batch = std::move(reflected_batch_out);
    result.transmitted_batch = std::move(transmitted_batch_out);
    result.specular_reflection_weight = h_specular_reflection_weight;
    return result;
}


__host__ MediaSimulationDeviceResult MediaSimulator::run(int num_photons) {
    cudaSetDevice(config_.gpu_id);

    DevicePhotonBatch input_batch;
    // Create a seed for this run. Note: time(nullptr) has 1-second resolution.
    // For rapid subsequent calls, a better random seed source might be needed.
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr)) + config_.gpu_id;

    // Call the new C-style launcher function
    generate_photons_on_device(config_.source, input_batch, num_photons, seed);
    
    return _run_simulation(config_, input_batch);
}

__host__ MediaSimulationDeviceResult MediaSimulator::run(const DevicePhotonBatch& input_batch) {
    cudaSetDevice(config_.gpu_id);
    return _run_simulation(config_, input_batch);
}

__host__ MediaSimulationResult MediaSimulator::run_and_copy_to_cpu(int num_photons) {
    cudaSetDevice(config_.gpu_id);
    return run(num_photons).to_host();
}

} // namespace phonder
