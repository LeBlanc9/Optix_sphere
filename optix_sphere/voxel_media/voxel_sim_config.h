#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <memory>
#include "photon/sources.h"
#include "boundary_config.h"

namespace phonder::voxel {

/**
 * @brief Host-side configuration for voxel media simulation
 *
 * All data is owned and stored in std::vector for automatic memory management.
 * All members are public for direct access from Python.
 */
struct SimConfig {
    // grid[x][y][z] = material_id
    float3 voxel_size = make_float3(1.0f, 1.0f, 1.0f);
    std::vector<std::vector<std::vector<uint8_t>>> grid;
    std::vector<std::vector<float>> materials;

    std::shared_ptr<PhotonSource> source;

    int gpu_id = 0;                   // GPU device ID
    unsigned long long seed = 0;      // Random seed (0 = auto)
    bool enable_specular = true;      // Enable specular reflection at entry
    bool merge_specular = false;      // Merge specular into negative_boundary_batch (saves memory)

    // Boundary photon collection configuration
    BoundaryCollectionConfig boundary_collection;

    /**
     * @brief Get ambient medium refractive index
     *
     * By convention, materials[0] represents the ambient medium (outside grid).
     * The refractive index is stored in materials[0][3].
     */
    float get_ambient_n() const {
        if (materials.empty() || materials[0].size() < 4) {
            return 1.0f;  // Default: air
        }
        return materials[0][3];  // n is the 4th component
    }

    /**
     * @brief Get grid dimensions
     */
    int3 get_dims() const {
        if (grid.empty()) return make_int3(0, 0, 0);
        int nx = grid.size();
        int ny = grid[0].empty() ? 0 : grid[0].size();
        int nz = (ny == 0 || grid[0][0].empty()) ? 0 : grid[0][0].size();
        return make_int3(nx, ny, nz);
    }

    /**
     * @brief Get number of materials
     */
    int get_num_materials() const {
        return materials.size();
    }

    /**
     * @brief Get grid center X coordinate
     */
    float get_center_x() const {
        int nx = grid.size();
        return nx * voxel_size.x * 0.5f;
    }

    /**
     * @brief Get grid center Y coordinate
     */
    float get_center_y() const {
        if (grid.empty()) return 0.0f;
        int ny = grid[0].size();
        return ny * voxel_size.y * 0.5f;
    }

    /**
     * @brief Validate configuration
     */
    bool is_valid() const {
        if (grid.empty() || materials.empty()) return false;
        if (voxel_size.x <= 0 || voxel_size.y <= 0 || voxel_size.z <= 0) return false;

        // Check grid is rectangular (all rows have same size)
        int ny = grid[0].size();
        int nz = grid[0][0].size();
        for (const auto& plane : grid) {
            if (plane.size() != static_cast<size_t>(ny)) return false;
            for (const auto& row : plane) {
                if (row.size() != static_cast<size_t>(nz)) return false;
            }
        }

        // Check materials have 4 properties each
        for (const auto& mat : materials) {
            if (mat.size() != 4) return false;
        }

        return true;
    }
};

} // namespace phonder::voxel
