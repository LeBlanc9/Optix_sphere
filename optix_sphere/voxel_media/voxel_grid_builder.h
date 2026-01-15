#pragma once
#include <vector>
#include <stdexcept>

namespace phonder::voxel {

/**
 * @brief Lightweight utility to build material ID grids
 *
 * This class ONLY constructs the grid of material IDs (labels).
 * It does NOT manage material properties - those are handled separately
 * in SimConfig.
 *
 * Usage:
 *   GridBuilder builder(nx, ny, nz);
 *   builder.set_voxel(x, y, z, material_id);
 *   builder.fill_region(x0, x1, y0, y1, z0, z1, material_id);
 *
 *   // Then pass to SimConfig
 *   SimConfig config;
 *   config.set_grid(builder.get_grid(), nx, ny, nz, dx, dy, dz);
 */
class GridBuilder {
public:
    /**
     * @brief Constructor
     * @param nx, ny, nz Grid dimensions
     */
    GridBuilder(int nx, int ny, int nz);

    /**
     * @brief Set material ID for a single voxel
     * @param x, y, z Voxel coordinates
     * @param material_id Material ID (0-255)
     */
    void set_voxel(int x, int y, int z, int material_id);

    /**
     * @brief Fill all voxels with a material ID
     * @param material_id Material ID to fill
     */
    void fill_uniform(int material_id);

    /**
     * @brief Fill a rectangular region with a material ID
     * @param x0, x1 X range [x0, x1) (exclusive end)
     * @param y0, y1 Y range [y0, y1)
     * @param z0, z1 Z range [z0, z1)
     * @param material_id Material ID to fill
     */
    void fill_region(int x0, int x1, int y0, int y1, int z0, int z1, int material_id);

    /**
     * @brief Fill a sphere with a material ID
     * @param cx, cy, cz Sphere center (in voxel indices)
     * @param radius Sphere radius (in voxels)
     * @param material_id Material ID to fill
     */
    void fill_sphere(int cx, int cy, int cz, float radius, int material_id);

    /**
     * @brief Fill a cylinder along Z axis
     * @param cx, cy Center (in voxel indices)
     * @param radius Cylinder radius (in voxels)
     * @param z0, z1 Z range
     * @param material_id Material ID to fill
     */
    void fill_cylinder_z(int cx, int cy, float radius, int z0, int z1, int material_id);

    // Getters
    const unsigned char* get_grid() const { return grid_.data(); }
    int get_nx() const { return nx_; }
    int get_ny() const { return ny_; }
    int get_nz() const { return nz_; }

private:
    int nx_, ny_, nz_;
    std::vector<unsigned char> grid_;  // Material IDs only

    int get_index(int x, int y, int z) const;
};

} // namespace phonder::voxel
