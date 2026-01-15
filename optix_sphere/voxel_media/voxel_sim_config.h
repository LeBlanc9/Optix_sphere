#pragma once
#include <cuda_runtime.h>
#include "photon/sources.h"

namespace phonder::voxel {

/**
 * @brief Configuration for voxel media simulation
 *
 * This class holds all parameters needed for simulation.
 * Data is stored as const pointers (non-owning).
 */
class SimConfig {
public:
    SimConfig() = default;

    /**
     * @brief Set grid data (material IDs)
     * @param grid Material ID array, shape: (nx, ny, nz), row-major (z fastest)
     * @param nx, ny, nz Grid dimensions
     * @param voxel_size Voxel size in mm (default: 1x1x1)
     */
    void set_grid(
        const unsigned char* grid,
        int nx, int ny, int nz,
        const float3& voxel_size = make_float3(1.0f, 1.0f, 1.0f)
    ) {
        grid_ = grid;
        nx_ = nx;
        ny_ = ny;
        nz_ = nz;
        dx_ = voxel_size.x;
        dy_ = voxel_size.y;
        dz_ = voxel_size.z;
    }

    /**
     * @brief Set material properties table
     * @param materials Flattened array, shape: (num_materials, 4)
     *                  Each row: [n, mua, mus, g]
     * @param num_materials Number of material types
     */
    void set_materials(const float* materials, int num_materials) {
        materials_ = materials;
        num_materials_ = num_materials;
    }

    /**
     * @brief Set light source
     */
    void set_source(const PhotonSource& source) {
        source_ = source;
    }

    /**
     * @brief Set ambient (outside) refractive index
     */
    void set_ambient_n(float n) { ambient_n_ = n; }

    /**
     * @brief Set GPU device ID
     */
    void set_gpu_id(int id) { gpu_id_ = id; }

    /**
     * @brief Set random seed (0 = auto-generate)
     */
    void set_seed(unsigned long long s) { seed_ = s; }

    /**
     * @brief Set exit detection boundaries
     */
    void set_exit_boundaries(float z_min, float z_max) {
        exit_z_min_ = z_min;
        exit_z_max_ = z_max;
    }

    // Getters
    const unsigned char* get_grid() const { return grid_; }
    int get_nx() const { return nx_; }
    int get_ny() const { return ny_; }
    int get_nz() const { return nz_; }
    float get_dx() const { return dx_; }
    float get_dy() const { return dy_; }
    float get_dz() const { return dz_; }
    const float* get_materials() const { return materials_; }
    int get_num_materials() const { return num_materials_; }
    const PhotonSource& get_source() const { return source_; }
    float get_ambient_n() const { return ambient_n_; }
    int get_gpu_id() const { return gpu_id_; }
    unsigned long long get_seed() const { return seed_; }
    float get_exit_z_min() const { return exit_z_min_; }
    float get_exit_z_max() const { return exit_z_max_; }

    /**
     * @brief Validate configuration
     */
    bool is_valid() const {
        return grid_ != nullptr &&
               materials_ != nullptr &&
               nx_ > 0 && ny_ > 0 && nz_ > 0 &&
               dx_ > 0 && dy_ > 0 && dz_ > 0 &&
               num_materials_ > 0;
    }

private:
    // Grid (non-owning pointers)
    const unsigned char* grid_ = nullptr;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    float dx_ = 1.0f, dy_ = 1.0f, dz_ = 1.0f;

    // Materials (non-owning pointer)
    const float* materials_ = nullptr;
    int num_materials_ = 0;

    // Source
    PhotonSource source_;

    // Options
    float ambient_n_ = 1.0f;
    int gpu_id_ = 0;
    unsigned long long seed_ = 0;

    // Exit detection
    float exit_z_min_ = -1.0f;
    float exit_z_max_ = -1.0f;
};

} // namespace phonder::voxel
