#pragma once
#include <cuda_runtime.h>

namespace phonder::voxel {

/**
 * @brief Device-side grid structure used in CUDA kernels
 *
 * This is a lightweight structure containing pointers to device memory.
 */
struct DeviceGrid {
    unsigned char* material_ids;  // Device pointer to material ID array
    int3 dims;                    // Grid dimensions (nx, ny, nz)
    float3 voxel_size;            // Voxel size in mm
    int num_materials;            // Number of material types

    /**
     * @brief Get ambient refractive index
     *
     * By convention, materials[0] is the ambient medium.
     * Access via constant memory: c_materials[0].n
     */
};

/**
 * @brief Boundary collection configuration (device side)
 */
struct DeviceBoundaryConfig {
    // Which faces collect photons (6 bools: -X, -Y, -Z, +X, +Y, +Z)
    bool collect_faces[6];

    // Collection center point
    float3 collection_center;

    // Collection radii (negative = no limit)
    float collection_radius_negative;  // For -X/-Y/-Z faces
    float collection_radius_positive;  // For +X/+Y/+Z faces

    /**
     * @brief Check if a face is collecting photons
     */
    __device__ __forceinline__ bool is_collecting(int face_index) const {
        return collect_faces[face_index];
    }

    /**
     * @brief Check if photon is within collection radius
     *
     * @param pos Photon position
     * @param is_positive_face True if checking +X/+Y/+Z face
     * @return True if within radius (or no radius limit)
     */
    __device__ __forceinline__ bool within_radius(const float3& pos, bool is_positive_face) const {
        float radius = is_positive_face ? collection_radius_positive : collection_radius_negative;

        // No radius limit
        if (radius < 0.0f) return true;

        // Calculate distance from collection center
        float dx = pos.x - collection_center.x;
        float dy = pos.y - collection_center.y;
        float radial_dist = sqrtf(dx * dx + dy * dy);

        return radial_dist <= radius;
    }
};

/**
 * @brief Input photon batch (device side)
 */
struct DevicePhotonInput {
    const float3* positions;
    const float3* directions;
    const double* weights;
    int size;
};

/**
 * @brief Output photon batch buffers (device side)
 */
struct DevicePhotonOutput {
    float3* positions;
    float3* directions;
    double* weights;
    int* counter;           // Atomic counter for this batch
    int capacity;           // Max capacity

    /**
     * @brief Try to add a photon to this output batch
     *
     * @return True if photon was added, false if batch is full
     */
    __device__ __forceinline__ bool add_photon(
        const float3& pos,
        const float3& dir,
        double weight
    ) const {
        int idx = atomicAdd(counter, 1);
        if (idx < capacity) {
            positions[idx] = pos;
            directions[idx] = dir;
            weights[idx] = weight;
            return true;
        }
        return false;
    }
};

/**
 * @brief Complete kernel parameters (packed into one structure)
 */
struct MCKernelParams {
    // Grid (device pointer)
    const Grid* grid;

    // Input photons
    DevicePhotonInput input;

    // Output batches
    DevicePhotonOutput specular_batch;
    DevicePhotonOutput negative_boundary_batch;
    DevicePhotonOutput positive_boundary_batch;

    // Boundary collection config
    DeviceBoundaryConfig boundary_config;

    // Simulation parameters
    bool enable_specular;
    bool merge_specular;  // Merge specular into boundary batches
    unsigned long long seed;
};

} // namespace phonder::voxel
