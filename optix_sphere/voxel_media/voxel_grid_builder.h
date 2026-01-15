#pragma once
#include "voxel_grid.cuh"
#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace phonder::voxel {

/**
 * @brief Host-side builder for Grid (MCX-style with material table)
 *
 * This class manages memory allocation and provides convenient methods
 * to construct voxel grids using material IDs and a lookup table.
 *
 * Usage:
 *   GridBuilder builder(nx, ny, nz, dx, dy, dz);
 *   int skin_id = builder.add_material(1.42, 0.01, 20.0, 0.7);
 *   int muscle_id = builder.add_material(1.42, 0.3, 80.0, 0.7);
 *   builder.fill_region(0, 100, 0, 100, 0, 1, skin_id);
 *   builder.fill_region(0, 100, 0, 100, 1, 3, muscle_id);
 */
class GridBuilder {
public:
    GridBuilder(int nx, int ny, int nz, float dx, float dy, float dz, float ambient_n = 1.0f);
    ~GridBuilder();

    // Disable copy, enable move
    GridBuilder(const GridBuilder&) = delete;
    GridBuilder& operator=(const GridBuilder&) = delete;
    GridBuilder(GridBuilder&&) = default;
    GridBuilder& operator=(GridBuilder&&) = default;

    /**
     * @brief Add a material type and return its ID
     * @return Material ID (0-255)
     */
    int add_material(float n, float mua, float mus, float g);

    /**
     * @brief Set material ID for a single voxel
     */
    void set_voxel(int x, int y, int z, int material_id);

    /**
     * @brief Fill all voxels with a material ID
     */
    void fill_uniform(int material_id);

    /**
     * @brief Fill a rectangular region with a material ID
     */
    void fill_region(int x0, int x1, int y0, int y1, int z0, int z1, int material_id);

    /**
     * @brief Get the device pointer to the Grid structure
     * This uploads the data to GPU if needed
     */
    Grid* get_device_grid();

    /**
     * @brief Get the host-side Grid structure (for inspection)
     */
    const Grid& get_host_grid() const { return host_grid_; }

    // Getters
    int get_nx() const { return nx_; }
    int get_ny() const { return ny_; }
    int get_nz() const { return nz_; }
    float get_dx() const { return dx_; }
    float get_dy() const { return dy_; }
    float get_dz() const { return dz_; }

private:
    int nx_, ny_, nz_;
    float dx_, dy_, dz_;
    float ambient_n_;
    int total_voxels_;

    // Host-side storage
    std::vector<unsigned char> host_material_ids_;  // Material ID for each voxel
    std::vector<float4> host_material_table_;       // Material properties lookup table

    // Host-side Grid structure
    Grid host_grid_;

    // Device-side storage
    unsigned char* device_material_ids_ = nullptr;
    float4* device_material_table_ = nullptr;
    Grid* device_grid_ = nullptr;

    bool device_dirty_ = true; // Whether device data needs update

    void upload_to_device();
    void free_device_memory();
    int get_index(int x, int y, int z) const;
};

}; // namespace phonder::voxel
