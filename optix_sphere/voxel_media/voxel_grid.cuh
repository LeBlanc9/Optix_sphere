#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace phonder::voxel {

// Constant memory for material properties (MCX-style)
// Maximum 256 materials (matching uint8 material ID range)
constexpr int MAX_MATERIALS = 256;

// External declaration (defined in voxel_sim_runner.cu)
extern __constant__ float4 c_materials[MAX_MATERIALS];

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
 * 2. Material properties stored in constant memory (c_materials)
 *
 * Benefits:
 * - Memory efficient: 1 byte per voxel instead of 16 bytes
 * - Fast property lookup: constant memory cached, broadcast to warp
 * - Easy material management: change material properties globally
 */
struct Grid {
    // Grid dimensions and voxel size
    int3 dims;                // number of voxels (nx, ny, nz)
    float3 voxel_size;        // voxel size in mm (dx, dy, dz)

    // Material ID for each voxel (0-255)
    unsigned char* material_ids;  // 1D array: material_ids[x*ny*nz + y*nz + z]

    // Number of materials (properties stored in c_materials constant memory)
    int num_materials;

    __host__ __device__ Grid()
        : dims(make_int3(0, 0, 0)),
          voxel_size(make_float3(0.0f, 0.0f, 0.0f)),
          material_ids(nullptr),
          num_materials(0) {}

    /**
     * @brief Convert world position to voxel indices
     * @param pos World position in mm
     * @return Voxel indices (can be out of bounds!)
     */
    __host__ __device__ int3 world_to_voxel(const float3& pos) const {
        return make_int3(
            static_cast<int>(floorf(pos.x / voxel_size.x)),
            static_cast<int>(floorf(pos.y / voxel_size.y)),
            static_cast<int>(floorf(pos.z / voxel_size.z))
        );
    }

    /**
     * @brief Convert voxel indices to 1D array index
     * @param voxel_idx Voxel indices
     * @return 1D array index
     */
    __host__ __device__ int voxel_to_index(const int3& voxel_idx) const {
        return voxel_idx.x * (dims.y * dims.z) + voxel_idx.y * dims.z + voxel_idx.z;
    }

    /**
     * @brief Check if voxel indices are within bounds
     */
    __host__ __device__ bool is_inside(const int3& voxel_idx) const {
        return voxel_idx.x >= 0 && voxel_idx.x < dims.x &&
               voxel_idx.y >= 0 && voxel_idx.y < dims.y &&
               voxel_idx.z >= 0 && voxel_idx.z < dims.z;
    }

    /**
     * @brief Get optical properties at a voxel index (no bounds checking!)
     *
     * Fast path: reads from constant memory c_materials
     * Constant memory is cached and broadcasts to all threads in a warp
     */
    __device__ OpticalProperties get_properties_at_index(int idx) const {
        unsigned char material_id = material_ids[idx];
        float4 props = c_materials[material_id];
        return OpticalProperties(props.x, props.y, props.z, props.w);
    }

    /**
     * @brief Get optical properties at voxel indices (with bounds checking)
     * Returns ambient properties if out of bounds
     */
    __device__ OpticalProperties get_properties(const int3& voxel_idx) const {
        if (!is_inside(voxel_idx)) {
            // Outside the grid: return ambient properties (material 0)
            float4 ambient_props = c_materials[0];
            return OpticalProperties(ambient_props.x, ambient_props.y, ambient_props.z, ambient_props.w);
        }
        int idx = voxel_to_index(voxel_idx);
        return get_properties_at_index(idx);
    }

    /**
     * @brief Get the center position of a voxel
     */
    __host__ __device__ float3 voxel_center(const int3& voxel_idx) const {
        return make_float3(
            (voxel_idx.x + 0.5f) * voxel_size.x,
            (voxel_idx.y + 0.5f) * voxel_size.y,
            (voxel_idx.z + 0.5f) * voxel_size.z
        );
    }
};

}; // namespace phonder::voxel
