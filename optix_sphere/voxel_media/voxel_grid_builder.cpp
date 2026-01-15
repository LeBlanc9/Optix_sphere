#include "voxel_grid_builder.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace phonder::voxel {

GridBuilder::GridBuilder(int nx, int ny, int nz, float dx, float dy, float dz, float ambient_n)
    : nx_(nx), ny_(ny), nz_(nz), dx_(dx), dy_(dy), dz_(dz), ambient_n_(ambient_n)
{
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }
    if (dx <= 0.0f || dy <= 0.0f || dz <= 0.0f) {
        throw std::invalid_argument("Voxel sizes must be positive");
    }

    total_voxels_ = nx * ny * nz;

    // Allocate host memory - initialize all voxels to material ID 0
    host_material_ids_.resize(total_voxels_, 0);

    // Add default material 0 (ambient/air-like: no absorption, minimal scattering)
    host_material_table_.push_back(make_float4(1.0f, 0.0f, 1e-6f, 0.0f));

    // Initialize host grid structure
    host_grid_.nx = nx;
    host_grid_.ny = ny;
    host_grid_.nz = nz;
    host_grid_.dx = dx;
    host_grid_.dy = dy;
    host_grid_.dz = dz;
    host_grid_.ambient_n = ambient_n;
    host_grid_.material_ids = host_material_ids_.data();
    host_grid_.material_table = host_material_table_.data();
    host_grid_.num_materials = 1;
}

GridBuilder::~GridBuilder() {
    free_device_memory();
}

int GridBuilder::get_index(int x, int y, int z) const {
    if (x < 0 || x >= nx_ || y < 0 || y >= ny_ || z < 0 || z >= nz_) {
        throw std::out_of_range("Voxel index out of bounds");
    }
    return x * (ny_ * nz_) + y * nz_ + z;
}

int GridBuilder::add_material(float n, float mua, float mus, float g) {
    if (host_material_table_.size() >= 256) {
        throw std::runtime_error("Maximum number of materials (256) exceeded");
    }

    int material_id = static_cast<int>(host_material_table_.size());
    host_material_table_.push_back(make_float4(n, mua, mus, g));

    // Update host grid
    host_grid_.material_table = host_material_table_.data();
    host_grid_.num_materials = static_cast<int>(host_material_table_.size());

    device_dirty_ = true;
    return material_id;
}

void GridBuilder::set_voxel(int x, int y, int z, int material_id) {
    if (material_id < 0 || material_id >= static_cast<int>(host_material_table_.size())) {
        throw std::out_of_range("Material ID out of range");
    }
    int idx = get_index(x, y, z);
    host_material_ids_[idx] = static_cast<unsigned char>(material_id);
    device_dirty_ = true;
}

void GridBuilder::fill_uniform(int material_id) {
    if (material_id < 0 || material_id >= static_cast<int>(host_material_table_.size())) {
        throw std::out_of_range("Material ID out of range");
    }
    std::fill(host_material_ids_.begin(), host_material_ids_.end(),
              static_cast<unsigned char>(material_id));
    device_dirty_ = true;
}

void GridBuilder::fill_region(int x0, int x1, int y0, int y1, int z0, int z1, int material_id) {
    if (material_id < 0 || material_id >= static_cast<int>(host_material_table_.size())) {
        throw std::out_of_range("Material ID out of range");
    }

    // Clamp to valid range
    x0 = std::max(0, std::min(x0, nx_));
    x1 = std::max(0, std::min(x1, nx_));
    y0 = std::max(0, std::min(y0, ny_));
    y1 = std::max(0, std::min(y1, ny_));
    z0 = std::max(0, std::min(z0, nz_));
    z1 = std::max(0, std::min(z1, nz_));

    unsigned char mid = static_cast<unsigned char>(material_id);
    for (int x = x0; x < x1; ++x) {
        for (int y = y0; y < y1; ++y) {
            for (int z = z0; z < z1; ++z) {
                int idx = get_index(x, y, z);
                host_material_ids_[idx] = mid;
            }
        }
    }
    device_dirty_ = true;
}

void GridBuilder::upload_to_device() {
    size_t material_ids_bytes = total_voxels_ * sizeof(unsigned char);
    size_t material_table_bytes = host_material_table_.size() * sizeof(float4);

    // Allocate device arrays if not already allocated
    if (!device_material_ids_) {
        cudaMalloc(&device_material_ids_, material_ids_bytes);
        cudaMalloc(&device_material_table_, material_table_bytes);
        cudaMalloc(&device_grid_, sizeof(Grid));
    } else {
        // Reallocate material table if size changed
        cudaFree(device_material_table_);
        cudaMalloc(&device_material_table_, material_table_bytes);
    }

    // Copy data to device
    cudaMemcpy(device_material_ids_, host_material_ids_.data(),
               material_ids_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_material_table_, host_material_table_.data(),
               material_table_bytes, cudaMemcpyHostToDevice);

    // Create device Grid structure
    Grid device_grid_host = host_grid_;
    device_grid_host.material_ids = device_material_ids_;
    device_grid_host.material_table = device_material_table_;
    device_grid_host.num_materials = static_cast<int>(host_material_table_.size());

    cudaMemcpy(device_grid_, &device_grid_host, sizeof(Grid), cudaMemcpyHostToDevice);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in upload_to_device: ") + cudaGetErrorString(err));
    }

    device_dirty_ = false;
}

void GridBuilder::free_device_memory() {
    if (device_material_ids_) cudaFree(device_material_ids_);
    if (device_material_table_) cudaFree(device_material_table_);
    if (device_grid_) cudaFree(device_grid_);

    device_material_ids_ = nullptr;
    device_material_table_ = nullptr;
    device_grid_ = nullptr;
}

Grid* GridBuilder::get_device_grid() {
    if (device_dirty_) {
        upload_to_device();
    }
    return device_grid_;
}

}; // namespace phonder::voxel
