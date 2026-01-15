#include "voxel_simulator.h"
#include "voxel_sim_runner.h"
#include "photon/launchers.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ctime>
#include <stdexcept>
#include <cstring>

namespace phonder::voxel {

// Constructor from SimConfig
Simulator::Simulator(const SimConfig& config) {
    if (!config.is_valid()) {
        throw std::invalid_argument("Invalid SimConfig");
    }

    initialize_from_config(config);
}

Simulator::~Simulator() {
    free_device_memory();
}

void Simulator::initialize_from_config(const SimConfig& config) {
    config_ = config;

    // Set GPU device
    cudaSetDevice(config_.get_gpu_id());

    // Make owned copies of the data if needed
    int total_voxels = config_.get_nx() * config_.get_ny() * config_.get_nz();
    owned_grid_.resize(total_voxels);
    std::memcpy(owned_grid_.data(), config_.get_grid(), total_voxels);

    int material_floats = config_.get_num_materials() * 4;
    owned_materials_.resize(material_floats);
    std::memcpy(owned_materials_.data(), config_.get_materials(), material_floats * sizeof(float));

    owns_data_ = true;

    // Upload to device
    upload_grid_to_device();
}

void Simulator::upload_grid_to_device() {
    int nx = config_.get_nx();
    int ny = config_.get_ny();
    int nz = config_.get_nz();
    int total_voxels = nx * ny * nz;
    int num_materials = config_.get_num_materials();

    // Check materials count
    if (num_materials > MAX_MATERIALS) {
        throw std::runtime_error("Number of materials exceeds MAX_MATERIALS (256)");
    }

    // Allocate device memory (only grid and grid_struct, no materials array)
    cudaMalloc(&device_data_.grid, total_voxels * sizeof(unsigned char));
    cudaMalloc(&device_data_.grid_struct, sizeof(Grid));

    // Copy grid data
    cudaMemcpy(device_data_.grid, owned_grid_.data(),
               total_voxels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Pack materials into float4 format
    std::vector<float4> packed_materials(num_materials);
    for (int i = 0; i < num_materials; i++) {
        packed_materials[i] = make_float4(
            owned_materials_[i * 4 + 0],  // n
            owned_materials_[i * 4 + 1],  // mua
            owned_materials_[i * 4 + 2],  // mus
            owned_materials_[i * 4 + 3]   // g
        );
    }

    // Upload to constant memory using runner function
    upload_materials_to_constant_memory(packed_materials.data(), num_materials);

    // Create Grid structure (no material_table pointer needed)
    Grid host_grid;
    host_grid.dims = make_int3(nx, ny, nz);
    host_grid.voxel_size = make_float3(config_.get_dx(), config_.get_dy(), config_.get_dz());
    host_grid.material_ids = device_data_.grid;
    host_grid.num_materials = num_materials;
    host_grid.ambient_n = config_.get_ambient_n();

    cudaMemcpy(device_data_.grid_struct, &host_grid, sizeof(Grid), cudaMemcpyHostToDevice);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}

void Simulator::free_device_memory() {
    if (device_data_.grid) cudaFree(device_data_.grid);
    if (device_data_.grid_struct) cudaFree(device_data_.grid_struct);

    device_data_.grid = nullptr;
    device_data_.grid_struct = nullptr;
    // Note: constant memory doesn't need explicit free
}

SimulationResult Simulator::run(int num_photons) {
    // Generate initial photon batch from source
    PhotonBatch input_batch(num_photons);
    generate_photons_on_device(
        config_.get_source(),
        input_batch,
        num_photons,
        config_.get_seed() ? config_.get_seed() : static_cast<unsigned long long>(std::time(nullptr))
    );

    return run(input_batch);
}

SimulationResult Simulator::run(const PhotonBatch& input_batch) {
    int input_size = input_batch.size();
    int output_capacity = input_size;

    // Allocate output buffers
    PhotonBatch specular_batch(output_capacity);
    PhotonBatch reflected_batch(output_capacity);
    PhotonBatch transmitted_batch(output_capacity);

    // Counters
    int* d_specular_counter;
    int* d_reflected_counter;
    int* d_transmitted_counter;
    cudaMalloc(&d_specular_counter, sizeof(int));
    cudaMalloc(&d_reflected_counter, sizeof(int));
    cudaMalloc(&d_transmitted_counter, sizeof(int));
    cudaMemset(d_specular_counter, 0, sizeof(int));
    cudaMemset(d_reflected_counter, 0, sizeof(int));
    cudaMemset(d_transmitted_counter, 0, sizeof(int));

    // Get exit boundaries
    float exit_z_min = config_.get_exit_z_min();
    float exit_z_max = config_.get_exit_z_max();

    // Auto-compute if not set
    if (exit_z_min < 0) exit_z_min = 0.0f;
    if (exit_z_max < 0) exit_z_max = config_.get_nz() * config_.get_dz();

    // Run simulation using runner function
    run_voxel_simulation(
        device_data_.grid_struct,
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
        exit_z_min,
        exit_z_max,
        config_.get_seed() ? config_.get_seed() : static_cast<unsigned long long>(std::time(nullptr))
    );

    cudaDeviceSynchronize();

    // Get actual counts
    int specular_count, reflected_count, transmitted_count;
    cudaMemcpy(&specular_count, d_specular_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&reflected_count, d_reflected_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&transmitted_count, d_transmitted_counter, sizeof(int), cudaMemcpyDeviceToHost);

    // Resize batches
    specular_batch.resize(std::min(specular_count, output_capacity));
    reflected_batch.resize(std::min(reflected_count, output_capacity));
    transmitted_batch.resize(std::min(transmitted_count, output_capacity));

    // Cleanup
    cudaFree(d_specular_counter);
    cudaFree(d_reflected_counter);
    cudaFree(d_transmitted_counter);

    return {specular_batch, reflected_batch, transmitted_batch};
}

void Simulator::update_materials(const float* materials, int num_materials) {
    // Check materials count
    if (num_materials > MAX_MATERIALS) {
        throw std::runtime_error("Number of materials exceeds MAX_MATERIALS (256)");
    }

    // Update owned copy
    owned_materials_.resize(num_materials * 4);
    std::memcpy(owned_materials_.data(), materials, num_materials * 4 * sizeof(float));

    // Pack and upload to constant memory
    std::vector<float4> packed(num_materials);
    for (int i = 0; i < num_materials; i++) {
        packed[i] = make_float4(
            materials[i * 4 + 0],
            materials[i * 4 + 1],
            materials[i * 4 + 2],
            materials[i * 4 + 3]
        );
    }

    // Upload to constant memory using runner function
    upload_materials_to_constant_memory(packed.data(), num_materials);

    // Update grid struct (only num_materials field)
    Grid host_grid;
    cudaMemcpy(&host_grid, device_data_.grid_struct, sizeof(Grid), cudaMemcpyDeviceToHost);
    host_grid.num_materials = num_materials;
    cudaMemcpy(device_data_.grid_struct, &host_grid, sizeof(Grid), cudaMemcpyHostToDevice);

    // Update config
    config_.set_materials(owned_materials_.data(), num_materials);
}

void Simulator::update_source(const PhotonSource& source) {
    // Just update in config - will be used in next run()
    SimConfig new_config = config_;
    new_config.set_source(source);
    config_ = new_config;
}

} // namespace phonder::voxel
