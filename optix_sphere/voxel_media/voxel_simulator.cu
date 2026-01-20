#include "voxel_simulator.h"
#include "voxel_sim_runner.h"
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
    config_ = config;  // Copy entire config (includes std::vector data)

    // Set GPU device
    cudaSetDevice(config_.gpu_id);

    // Upload to device
    upload_grid_to_device();
}

void Simulator::upload_grid_to_device() {
    const int3 dims = config_.get_dims();
    const float3& voxel_size = config_.voxel_size;
    int total_voxels = dims.x * dims.y * dims.z;

    // Flatten 3D nested vector to 1D array for GPU upload
    // Row-major order: grid[x][y][z] -> flat[x * ny * nz + y * nz + z]
    std::vector<uint8_t> flat_grid(total_voxels);
    for (int x = 0; x < dims.x; x++) {
        for (int y = 0; y < dims.y; y++) {
            for (int z = 0; z < dims.z; z++) {
                int idx = x * (dims.y * dims.z) + y * dims.z + z;
                flat_grid[idx] = config_.grid[x][y][z];
            }
        }
    }

    // Allocate device memory for grid
    cudaMalloc(&device_data_.grid, total_voxels);
    cudaMemcpy(device_data_.grid, flat_grid.data(), total_voxels, cudaMemcpyHostToDevice);

    // Flatten materials for upload to constant memory
    int num_materials = config_.get_num_materials();
    std::vector<float> flat_materials(num_materials * 4);
    for (int i = 0; i < num_materials; i++) {
        for (int j = 0; j < 4; j++) {
            flat_materials[i * 4 + j] = config_.materials[i][j];
        }
    }

    // Upload materials to constant memory
    cudaMemcpyToSymbol(c_materials, flat_materials.data(),
                       num_materials * 4 * sizeof(float));

    // Create Grid structure
    Grid host_grid;
    host_grid.dims = dims;
    host_grid.voxel_size = voxel_size;
    host_grid.material_ids = device_data_.grid;
    host_grid.num_materials = num_materials;
    // Note: Ambient properties are read from c_materials[0] in device code

    // Allocate and copy Grid structure to device
    cudaMalloc(&device_data_.grid_struct, sizeof(Grid));
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
}

SimulationResult Simulator::run(int num_photons) {
    // Generate initial photon batch from source
    PhotonBatch input_batch;
    if (config_.source) {
        config_.source->generate(
            input_batch,
            num_photons,
            config_.seed ? config_.seed : static_cast<unsigned long long>(std::time(nullptr))
        );
    } else {
        throw std::runtime_error("No photon source configured");
    }

    return run(input_batch);
}

SimulationResult Simulator::run(const PhotonBatch& input_batch) {
    // Simply call the runner - it handles all device operations
    return run_voxel_simulation(
        device_data_.grid_struct,
        input_batch,
        config_.boundary_collection,
        config_.enable_specular,
        config_.merge_specular,
        config_.seed ? config_.seed : static_cast<unsigned long long>(std::time(nullptr))
    );
}

void Simulator::update_materials(const float* materials, int num_materials) {
    // Update host-side config (convert flat array to 2D nested vector)
    config_.materials.resize(num_materials);
    for (int i = 0; i < num_materials; i++) {
        config_.materials[i].resize(4);
        for (int j = 0; j < 4; j++) {
            config_.materials[i][j] = materials[i * 4 + j];
        }
    }

    // Update constant memory on device
    cudaMemcpyToSymbol(c_materials, materials, num_materials * 4 * sizeof(float));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error updating materials: ") + cudaGetErrorString(err));
    }
}

void Simulator::update_source(std::shared_ptr<PhotonSource> source) {
    config_.source = source;
}

} // namespace phonder::voxel
