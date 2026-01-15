#pragma once
#include <cuda_runtime.h>
#include "voxel_media/voxel_grid.cuh"

namespace phonder::voxel {

// Constants
#define EPS 1e-6f
#define ONE_PI 3.1415926535897932f
#define TWO_PI 6.28318530717959f

/**
 * @brief Calculate ray-voxel intersection distance
 *
 * Finds which voxel face the ray hits first and the distance to it.
 *
 * @param pos Current position in voxel coordinates
 * @param dir Direction vector (normalized)
 * @param inv_dir Inverse direction (1/dir) for fast calculation
 * @param voxel_state [in] voxel indices (x,y,z), [out] hit face index at [3]
 * @return Distance to nearest boundary
 */
__device__ inline float hitgrid(float3* pos, float3* dir, float* inv_dir, short voxel_state[4]) {
    float dist;
    float time_to_face[3];

    // Time-of-flight to hit each face (x, y, z)
    time_to_face[0] = fabsf((voxel_state[0] + (dir->x > 0.f) - pos->x) * inv_dir[0]);
    time_to_face[1] = fabsf((voxel_state[1] + (dir->y > 0.f) - pos->y) * inv_dir[1]);
    time_to_face[2] = fabsf((voxel_state[2] + (dir->z > 0.f) - pos->z) * inv_dir[2]);

    // Find the nearest face (minimum time-of-flight)
    dist = fminf(fminf(time_to_face[0], time_to_face[1]), time_to_face[2]);
    voxel_state[3] = (dist == time_to_face[0] ? 0 : (dist == time_to_face[1] ? 1 : 2));

    return dist;
}

/**
 * @brief Compute inverse direction for fast ray-AABB tests
 *
 * Uses hardware-accelerated division. Always inlined for performance.
 */
__device__ __forceinline__ float3 calc_inverse_dir(const float3& dir) {
    return make_float3(
        __fdividef(1.f, dir.x),
        __fdividef(1.f, dir.y),
        __fdividef(1.f, dir.z)
    );
}





}; // namespace phonder::voxel 
