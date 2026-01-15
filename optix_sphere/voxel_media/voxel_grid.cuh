#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace phonder::voxel {

/**
 * @brief Optical properties for a single voxel
 */
struct OpticalProperties {
    float n;    // refractive index
    float mua;  // absorption coefficient (mm^-1)
    float mus;  // scattering coefficient (mm^-1)
    float g;    // anisotropy factor [-1, 1]

    __host__ __device__ OpticalProperties()
        : n(1.0f), mua(0.0f), mus(0.0f), g(0.0f) {}

    __host__ __device__ OpticalProperties(float n, float mua, float mus, float g)
        : n(n), mua(mua), mus(mus), g(g) {}

    __host__ __device__ float get_inv_mus() const {
        return (mus > 0.0f) ? (1.0f / mus) : 0.0f;
    }
};

/**
 * @brief 3D Voxel Grid with material-based optical properties (MCX-style)
 *
 * This structure stores optical properties using a two-level approach:
 * 1. Each voxel stores a material ID (uint8_t, 0-255)
 * 2. A material table maps IDs to optical properties (float4)
 *
 * Benefits:
 * - Memory efficient: 1 byte per voxel instead of 16 bytes
 * - Fast property lookup: single float4 read from material table
 * - Easy material management: change material properties globally
 */
struct Grid {
    // Grid dimensions
    int nx, ny, nz;           // number of voxels in each dimension
    float dx, dy, dz;         // voxel size in mm

    // Material ID for each voxel (0-255)
    unsigned char* material_ids;  // 1D array: material_ids[x*ny*nz + y*nz + z]

    // Material properties lookup table (packed as float4)
    // Each float4 contains: {n, mua, mus, g}
    float4* material_table;
    int num_materials;        // number of materials in the table

    // Ambient (outside) refractive index
    float ambient_n;

    __host__ __device__ Grid()
        : nx(0), ny(0), nz(0),
          dx(0.0f), dy(0.0f), dz(0.0f),
          material_ids(nullptr), material_table(nullptr),
          num_materials(0), ambient_n(1.0f) {}

    /**
     * @brief Convert world position to voxel indices
     * @param pos World position in mm
     * @return Voxel indices (can be out of bounds!)
     */
    __host__ __device__ int3 world_to_voxel(const float3& pos) const {
        return make_int3(
            static_cast<int>(floorf(pos.x / dx)),
            static_cast<int>(floorf(pos.y / dy)),
            static_cast<int>(floorf(pos.z / dz))
        );
    }

    /**
     * @brief Convert voxel indices to 1D array index
     * @param voxel_idx Voxel indices
     * @return 1D array index
     */
    __host__ __device__ int voxel_to_index(const int3& voxel_idx) const {
        return voxel_idx.x * (ny * nz) + voxel_idx.y * nz + voxel_idx.z;
    }

    /**
     * @brief Check if voxel indices are within bounds
     */
    __host__ __device__ bool is_inside(const int3& voxel_idx) const {
        return voxel_idx.x >= 0 && voxel_idx.x < nx &&
               voxel_idx.y >= 0 && voxel_idx.y < ny &&
               voxel_idx.z >= 0 && voxel_idx.z < nz;
    }

    /**
     * @brief Get optical properties at a voxel index (no bounds checking!)
     *
     * Fast path: single material table lookup using float4
     */
    __device__ OpticalProperties get_properties_at_index(int idx) const {
        unsigned char material_id = material_ids[idx];
        float4 props = material_table[material_id];
        return OpticalProperties(props.x, props.y, props.z, props.w);
    }

    /**
     * @brief Get optical properties at voxel indices (with bounds checking)
     * Returns ambient properties if out of bounds
     */
    __device__ OpticalProperties get_properties(const int3& voxel_idx) const {
        if (!is_inside(voxel_idx)) {
            // Outside the grid: return ambient properties (no absorption/scattering)
            return OpticalProperties(ambient_n, 0.0f, 1e-6f, 0.0f);
        }
        int idx = voxel_to_index(voxel_idx);
        return get_properties_at_index(idx);
    }

    /**
     * @brief Get the center position of a voxel
     */
    __host__ __device__ float3 voxel_center(const int3& voxel_idx) const {
        return make_float3(
            (voxel_idx.x + 0.5f) * dx,
            (voxel_idx.y + 0.5f) * dy,
            (voxel_idx.z + 0.5f) * dz
        );
    }
};

}; // namespace phonder::voxel
