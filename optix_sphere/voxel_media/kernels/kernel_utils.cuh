#pragma once
#include <cuda_runtime.h>
#include "voxel_media/voxel_grid.cuh"
#include "voxel_media/kernels/device_types.cuh"
#include "utils/device/math.cuh"

namespace phonder::voxel {

/**
 * @brief Normal axis for axis-aligned boundary faces
 */
enum NormalAxis {
    NORMAL_AXIS_X = 0,  // Face normal aligned with X axis
    NORMAL_AXIS_Y = 1,  // Face normal aligned with Y axis
    NORMAL_AXIS_Z = 2   // Face normal aligned with Z axis
};

/**
 * @brief Grid boundary structure (world coordinates)
 */
struct GridBounds {
    float3 min;
    float3 max;

    __device__ __forceinline__ GridBounds(const Grid* grid) {
        min = make_float3(0.f, 0.f, 0.f);
        max = make_float3(
            grid->voxel_size.x * grid->dims.x,
            grid->voxel_size.y * grid->dims.y,
            grid->voxel_size.z * grid->dims.z
        );
    }

    __device__ __forceinline__ bool is_outside(const float3& pos) const {
        return (pos.x < min.x || pos.x >= max.x ||
                pos.y < min.y || pos.y >= max.y ||
                pos.z < min.z || pos.z >= max.z);
    }

    __device__ __forceinline__ float3 center() const {
        return make_float3(
            (min.x + max.x) * 0.5f,
            (min.y + max.y) * 0.5f,
            (min.z + max.z) * 0.5f
        );
    }
};

/**
 * @brief Check intersection with axis-aligned boundary face
 *
 * @param pos_coord Position component (x, y, or z)
 * @param dir_coord Direction component (x, y, or z)
 * @param boundary Boundary coordinate value
 * @param axis Axis index (NORMAL_AXIS_X/Y/Z)
 * @param t_min Current minimum t value (modified if closer intersection found)
 * @param normal_axis Current normal axis (modified if closer intersection found)
 */
__device__ __forceinline__ void check_boundary_intersection(
    float pos_coord,
    float dir_coord,
    float boundary,
    int axis,
    float& t_min,
    int& normal_axis
) {
    if ((pos_coord < boundary && dir_coord > 0.f) ||
        (pos_coord >= boundary && dir_coord < 0.f)) {
        float t = (boundary - pos_coord) / dir_coord;
        if (t < t_min) {
            t_min = t;
            normal_axis = axis;
        }
    }
}

/**
 * @brief Find entry point into grid for photon starting outside
 *
 * @param pos Photon position
 * @param dir Photon direction
 * @param bounds Grid boundaries
 * @param normal_axis Output: normal axis of entry face
 * @return Entry position, or original position if no intersection
 */
__device__ inline float3 find_grid_entry(
    const float3& pos,
    const float3& dir,
    const GridBounds& bounds,
    int& normal_axis
) {
    float t_min = 1e30f;
    normal_axis = -1;

    // Check all 6 faces (2 per axis)
    check_boundary_intersection(pos.x, dir.x, bounds.min.x, NORMAL_AXIS_X, t_min, normal_axis);
    check_boundary_intersection(pos.x, dir.x, bounds.max.x, NORMAL_AXIS_X, t_min, normal_axis);
    check_boundary_intersection(pos.y, dir.y, bounds.min.y, NORMAL_AXIS_Y, t_min, normal_axis);
    check_boundary_intersection(pos.y, dir.y, bounds.max.y, NORMAL_AXIS_Y, t_min, normal_axis);
    check_boundary_intersection(pos.z, dir.z, bounds.min.z, NORMAL_AXIS_Z, t_min, normal_axis);
    check_boundary_intersection(pos.z, dir.z, bounds.max.z, NORMAL_AXIS_Z, t_min, normal_axis);

    if (normal_axis < 0) {
        return pos;  // No intersection (photon moving away)
    }

    return pos + dir * (t_min + 1e-5f);
}

/**
 * @brief Detect normal axis for photon at grid boundary
 */
__device__ inline int detect_boundary_normal_axis(
    const float3& pos,
    const float3& dir,
    const GridBounds& bounds
) {
    if (pos.x <= bounds.min.x + EPS && dir.x > 0.f) {
        return NORMAL_AXIS_X;
    } else if (pos.x >= bounds.max.x - EPS && dir.x < 0.f) {
        return NORMAL_AXIS_X;
    } else if (pos.y <= bounds.min.y + EPS && dir.y > 0.f) {
        return NORMAL_AXIS_Y;
    } else if (pos.y >= bounds.max.y - EPS && dir.y < 0.f) {
        return NORMAL_AXIS_Y;
    } else if (pos.z <= bounds.min.z + EPS && dir.z > 0.f) {
        return NORMAL_AXIS_Z;
    } else if (pos.z >= bounds.max.z - EPS && dir.z < 0.f) {
        return NORMAL_AXIS_Z;
    } else {
        return NORMAL_AXIS_Z;  // Default to Z
    }
}

/**
 * @brief Convert world position to voxel coordinates
 */
__device__ __forceinline__ float3 world_to_voxel(
    const float3& world_pos,
    const Grid* grid
) {
    return make_float3(
        world_pos.x / grid->voxel_size.x,
        world_pos.y / grid->voxel_size.y,
        world_pos.z / grid->voxel_size.z
    );
}

/**
 * @brief Convert voxel position to world coordinates
 */
__device__ __forceinline__ float3 voxel_to_world(
    const float3& voxel_pos,
    const Grid* grid
) {
    return make_float3(
        voxel_pos.x * grid->voxel_size.x,
        voxel_pos.y * grid->voxel_size.y,
        voxel_pos.z * grid->voxel_size.z
    );
}

/**
 * @brief Calculate specular reflection direction
 */
__device__ __forceinline__ float3 calculate_specular_direction(
    const float3& incident_dir,
    int normal_axis
) {
    float3 flip = make_float3(
        normal_axis == NORMAL_AXIS_X ? -1.0f : 1.0f,
        normal_axis == NORMAL_AXIS_Y ? -1.0f : 1.0f,
        normal_axis == NORMAL_AXIS_Z ? -1.0f : 1.0f
    );
    return make_float3(
        incident_dir.x * flip.x,
        incident_dir.y * flip.y,
        incident_dir.z * flip.z
    );
}

/**
 * @brief Determine if position is on positive face
 */
__device__ __forceinline__ bool is_positive_face(
    const float3& pos,
    const GridBounds& bounds,
    int normal_axis
) {
    if (normal_axis == NORMAL_AXIS_X) {
        return (pos.x >= bounds.max.x - EPS);
    } else if (normal_axis == NORMAL_AXIS_Y) {
        return (pos.y >= bounds.max.y - EPS);
    } else if (normal_axis == NORMAL_AXIS_Z) {
        return (pos.z >= bounds.max.z - EPS);
    }
    return false;
}

/**
 * @brief Check if position is within collection radius
 */
__device__ __forceinline__ bool within_collection_radius(
    const float3& pos,
    const GridBounds& bounds,
    float radius
) {
    if (radius < 0.0f) return true;  // No limit

    float3 center = bounds.center();
    float dx = pos.x - center.x;
    float dy = pos.y - center.y;
    float radial_dist = sqrtf(dx * dx + dy * dy);

    return radial_dist <= radius;
}

/**
 * @brief Store photon in output batch with atomic counter
 */
__device__ __forceinline__ bool store_photon(
    const float3& pos,
    const float3& dir,
    double weight,
    float3* positions,
    float3* directions,
    double* weights,
    int* counter,
    int capacity
) {
    int out_idx = atomicAdd(counter, 1);
    if (out_idx < capacity) {
        positions[out_idx] = pos;
        directions[out_idx] = dir;
        weights[out_idx] = weight;
        return true;
    }
    return false;
}

/**
 * @brief Store specular photon (handles merge_specular logic)
 */
__device__ inline void store_specular_photon(
    const float3& pos,
    const float3& dir,
    double weight,
    const GridBounds& bounds,
    int normal_axis,
    bool merge_specular,
    DevicePhotonOutput& specular_batch,
    DevicePhotonOutput& negative_batch,
    DevicePhotonOutput& positive_batch
) {
    if (merge_specular) {
        // Merge into boundary batches
        if (is_positive_face(pos, bounds, normal_axis)) {
            store_photon(pos, dir, weight,
                        positive_batch.positions,
                        positive_batch.directions,
                        positive_batch.weights,
                        positive_batch.counter,
                        positive_batch.capacity);
        } else {
            store_photon(pos, dir, weight,
                        negative_batch.positions,
                        negative_batch.directions,
                        negative_batch.weights,
                        negative_batch.counter,
                        negative_batch.capacity);
        }
    } else {
        // Store in separate specular batch
        store_photon(pos, dir, weight,
                    specular_batch.positions,
                    specular_batch.directions,
                    specular_batch.weights,
                    specular_batch.counter,
                    specular_batch.capacity);
    }
}

/**
 * @brief Determine exit face from voxel indices
 *
 * @param voxel_idx Voxel indices (may be out of bounds)
 * @param dims Grid dimensions
 * @param exit_face Output: face index (0=-X, 1=-Y, 2=-Z, 3=+X, 4=+Y, 5=+Z)
 * @param is_positive Output: true if positive face, false if negative
 * @return true if photon exited through a boundary
 */
__device__ __forceinline__ bool get_exit_face(
    const int3& voxel_idx,
    const int3& dims,
    int& exit_face,
    bool& is_positive
) {
    if (voxel_idx.x < 0) {
        exit_face = 0;  // -X
        is_positive = false;
        return true;
    } else if (voxel_idx.x >= dims.x) {
        exit_face = 3;  // +X
        is_positive = true;
        return true;
    } else if (voxel_idx.y < 0) {
        exit_face = 1;  // -Y
        is_positive = false;
        return true;
    } else if (voxel_idx.y >= dims.y) {
        exit_face = 4;  // +Y
        is_positive = true;
        return true;
    } else if (voxel_idx.z < 0) {
        exit_face = 2;  // -Z
        is_positive = false;
        return true;
    } else if (voxel_idx.z >= dims.z) {
        exit_face = 5;  // +Z
        is_positive = true;
        return true;
    }

    return false;  // Not at boundary
}

} // namespace phonder::voxel
