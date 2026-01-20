#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "voxel_media/voxel_grid.cuh"
#include "voxel_media/kernels/voxel_ray_tracing.cuh"
#include "voxel_media/kernels/voxel_physics.cuh"
#include "voxel_media/kernels/device_types.cuh"
#include "voxel_media/kernels/kernel_utils.cuh"
#include "utils/device/math.cuh"

namespace phonder::voxel {

/**
 * @brief Handle photon entry into grid from outside
 *
 * Complete entry logic:
 * 1. Move photon to grid boundary if outside (skipvoid)
 * 2. Calculate Fresnel reflection at air-tissue interface
 * 3. Store specular reflection component
 * 4. Apply refraction to transmitted component
 *
 * @param grid Voxel grid
 * @param pos Position (modified to grid entry point, then to voxel coordinates)
 * @param dir Direction (modified by refraction)
 * @param weight Weight (reduced by reflection)
 * @param specular_positions,specular_directions,specular_weights Specular output arrays
 * @param specular_counter Atomic counter
 * @param output_capacity Buffer capacity
 * @param reflected_radius Max radial distance for reflected photons (negative = no limit)
 * @return true if photon enters successfully, false otherwise
 */
__device__ inline bool enter_grid_from_outside(
    const Grid* grid,
    float3& pos,           // Will be modified to voxel coordinates
    float3& dir,
    double& weight,
    float3* specular_positions,
    float3* specular_directions,
    double* specular_weights,
    int* specular_counter,
    int specular_capacity,
    float3* negative_positions,
    float3* negative_directions,
    double* negative_weights,
    int* negative_counter,
    int negative_capacity,
    float3* positive_positions,
    float3* positive_directions,
    double* positive_weights,
    int* positive_counter,
    int positive_capacity,
    float reflected_radius,
    bool merge_specular
) {
    // Grid boundaries
    GridBounds bounds(grid);

    // Step 1: Find entry point
    int normal_axis;
    float3 entry_pos;

    if (bounds.is_outside(pos)) {
        // Move photon to grid entrance
        entry_pos = find_grid_entry(pos, dir, bounds, normal_axis);
        if (normal_axis < 0) {
            return false;  // Photon moving away from grid
        }
        pos = entry_pos;
    } else {
        // Already inside - detect entry face
        entry_pos = pos;
        normal_axis = detect_boundary_normal_axis(pos, dir, bounds);
    }

    // Convert to voxel coordinates
    float3 pos_voxel = world_to_voxel(pos, grid);

    // Small epsilon push to avoid being exactly on boundary
    pos_voxel = pos_voxel + dir * EPS;

    // Get voxel index
    int3 voxel_idx = make_int3(
        (int)floorf(pos_voxel.x),
        (int)floorf(pos_voxel.y),
        (int)floorf(pos_voxel.z)
    );

    // Check if voxel is valid
    if (!grid->is_inside(voxel_idx)) {
        return false;
    }

    // Step 2: Get optical properties at entry voxel
    OpticalProperties entry_props = grid->get_properties(voxel_idx);
    float n1 = c_materials[0].x;  // outside (air) - ambient refractive index
    float n2 = entry_props.n;     // inside (tissue)

    // Step 3: Calculate Fresnel reflection coefficient
    float r = reflectcoeff(&dir, n1, n2, normal_axis);

    // Step 4: Store specular reflection photon
    if (r > 1e-6f && within_collection_radius(entry_pos, bounds, reflected_radius)) {
        // Calculate specular reflection direction
        float3 spec_dir = calculate_specular_direction(dir, normal_axis);

        // Package output batches
        DevicePhotonOutput spec_batch = {
            specular_positions, specular_directions, specular_weights,
            specular_counter, specular_capacity
        };
        DevicePhotonOutput neg_batch = {
            negative_positions, negative_directions, negative_weights,
            negative_counter, negative_capacity
        };
        DevicePhotonOutput pos_batch = {
            positive_positions, positive_directions, positive_weights,
            positive_counter, positive_capacity
        };

        // Store (handles merge_specular logic)
        store_specular_photon(
            entry_pos, spec_dir, weight * r,
            bounds, normal_axis, merge_specular,
            spec_batch, neg_batch, pos_batch
        );
    }

    // Step 5: Reduce weight by transmission coefficient (1-r)
    weight *= (1.0 - r);

    // Step 6: Apply refraction to transmitted photon
    if (fabsf(n1 - n2) > 1e-6f) {
        transmit(&dir, n1, n2, normal_axis);
    }

    // Step 7: Update position to voxel coordinates (output)
    pos = pos_voxel;

    return (weight > 1e-6);
}

/**
 * @brief Monte Carlo photon transport kernel for voxel media
 *
 * All parameters are packed into MCKernelParams for cleaner API.
 */
__global__ void voxel_kernel(const MCKernelParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    curandState state;
    curand_init(params.seed + idx * 97, idx, 0, &state);

    const int MAX_ITER = 10000;
    const float WEIGHT_THRESHOLD = 1e-6f;
    const float ROULETTE_WEIGHT = 0.001f;
    const float SURVIVAL_PROB = 0.1f;

    // Unpack commonly used parameters
    const Grid* grid = params.grid;
    const DeviceBoundaryConfig& bc = params.boundary_config;

    for (int i = idx; i < params.input.size; i += stride) {
        // Load photon from input batch
        float3 pos = params.input.positions[i];
        float3 dir = params.input.directions[i];
        double weight = params.input.weights[i];

        // Skip photons with zero weight
        if (weight <= WEIGHT_THRESHOLD) continue;

        // Enter grid from outside: handles skipvoid, Fresnel, specular reflection, and refraction
        // After this call, pos is in voxel coordinates, dir is refracted, weight is reduced
        bool entered = true;
        if (params.enable_specular) {
            entered = enter_grid_from_outside(
                grid, pos, dir, weight,
                params.specular_batch.positions,
                params.specular_batch.directions,
                params.specular_batch.weights,
                params.specular_batch.counter,
                params.specular_batch.capacity,
                params.negative_boundary_batch.positions,
                params.negative_boundary_batch.directions,
                params.negative_boundary_batch.weights,
                params.negative_boundary_batch.counter,
                params.negative_boundary_batch.capacity,
                params.positive_boundary_batch.positions,
                params.positive_boundary_batch.directions,
                params.positive_boundary_batch.weights,
                params.positive_boundary_batch.counter,
                params.positive_boundary_batch.capacity,
                bc.collection_radius_negative,
                params.merge_specular
            );
        } else {
            // No specular - just move to grid entrance and convert to voxel coordinates
            float3 grid_min = make_float3(0.f, 0.f, 0.f);
            float3 grid_max = make_float3(grid->voxel_size.x * grid->dims.x,
                                           grid->voxel_size.y * grid->dims.y,
                                           grid->voxel_size.z * grid->dims.z);

            bool outside = (pos.x < grid_min.x || pos.x >= grid_max.x ||
                            pos.y < grid_min.y || pos.y >= grid_max.y ||
                            pos.z < grid_min.z || pos.z >= grid_max.z);

            if (outside) {
                float t_min = 1e30f;
                bool can_enter = false;

                if (pos.x < grid_min.x && dir.x > 0.f) {
                    float t = (grid_min.x - pos.x) / dir.x;
                    if (t < t_min) { t_min = t; can_enter = true; }
                }
                if (pos.x >= grid_max.x && dir.x < 0.f) {
                    float t = (grid_max.x - pos.x) / dir.x;
                    if (t < t_min) { t_min = t; can_enter = true; }
                }
                if (pos.y < grid_min.y && dir.y > 0.f) {
                    float t = (grid_min.y - pos.y) / dir.y;
                    if (t < t_min) { t_min = t; can_enter = true; }
                }
                if (pos.y >= grid_max.y && dir.y < 0.f) {
                    float t = (grid_max.y - pos.y) / dir.y;
                    if (t < t_min) { t_min = t; can_enter = true; }
                }
                if (pos.z < grid_min.z && dir.z > 0.f) {
                    float t = (grid_min.z - pos.z) / dir.z;
                    if (t < t_min) { t_min = t; can_enter = true; }
                }
                if (pos.z >= grid_max.z && dir.z < 0.f) {
                    float t = (grid_max.z - pos.z) / dir.z;
                    if (t < t_min) { t_min = t; can_enter = true; }
                }

                if (!can_enter) {
                    continue;
                }

                pos = pos + dir * (t_min + 1e-5f);
            }

            // Convert to voxel coordinates
            pos = make_float3(
                pos.x / grid->voxel_size.x,
                pos.y / grid->voxel_size.y,
                pos.z / grid->voxel_size.z
            );
            pos = pos + dir * EPS;
        }

        if (!entered) {
            continue;  // Photon didn't enter (weight too low or outside grid)
        }

        // pos is now in voxel coordinates
        float3 pos_voxel = pos;

        // Inverse direction for fast ray-voxel intersection (1/dir per component)
        float3 inv_dir = calc_inverse_dir(dir);

        // Voxel state: [0-2] = voxel indices (x,y,z), [3] = last hit face (0=x, 1=y, 2=z, -1=none)
        short voxel_state[4];
        voxel_state[0] = (short)floorf(pos_voxel.x);
        voxel_state[1] = (short)floorf(pos_voxel.y);
        voxel_state[2] = (short)floorf(pos_voxel.z);
        voxel_state[3] = -1;  // No face hit yet

        // Remaining scattering path length (triggers sampling when <= 0)
        float scatter_len = 0.f;

        // Flag to skip direction change on first scattering event (preserve launch direction)
        bool first_scatter = true;

        // Precompute minimum voxel size (used for physical distance calculations)
        const float voxel_size_mm = fminf(grid->voxel_size.x,
                                          fminf(grid->voxel_size.y, grid->voxel_size.z));

        // Main transport loop
        int iter = 0;
        bool exited = false;
        int3 voxel_idx;  // Current voxel index
        OpticalProperties optical_props;  // Will be set in first iteration

        while (!exited && weight > WEIGHT_THRESHOLD && iter < MAX_ITER) {
            iter++;

            // Sample scattering event
            if (scatter_len <= 0.f) {
                scatter_len = sample_scatter_length(&state);

                // Apply scattering (skip on first iteration to preserve launch direction)
                if (!first_scatter) {
                    scatter(&dir, optical_props.g, &state);
                    inv_dir = calc_inverse_dir(dir);
                }

                first_scatter = false;
            }

            // Get current voxel properties
            voxel_idx = make_int3(voxel_state[0], voxel_state[1], voxel_state[2]);
            optical_props = grid->get_properties(voxel_idx);

            // Find distance to next voxel boundary
            float dist_to_boundary = hitgrid(&pos_voxel, &dir, (float*)&inv_dir, voxel_state);

            // Convert to physical units for scattering calculation
            float scatter_len_to_boundary = dist_to_boundary * voxel_size_mm * optical_props.mus;

            // Determine if photon scatters or crosses boundary
            scatter_len_to_boundary = fminf(scatter_len_to_boundary, scatter_len);

            // Convert back to voxel units for movement
            float travel_dist = (optical_props.mus > EPS) ?
                (scatter_len_to_boundary / (voxel_size_mm * optical_props.mus)) : dist_to_boundary;

            // Move photon
            pos_voxel = pos_voxel + dir * travel_dist;

            // Update voxel index (only if boundary crossed, not if scattered)
            bool scattered = (fabsf(scatter_len_to_boundary - scatter_len) < EPS);

            if (voxel_state[3] == 0) {  // Last hit X face
                voxel_state[0] += scattered ? 0 : (dir.x > 0.f ? 1 : -1);
            }
            if (voxel_state[3] == 1) {  // Last hit Y face
                voxel_state[1] += scattered ? 0 : (dir.y > 0.f ? 1 : -1);
            }
            if (voxel_state[3] == 2) {  // Last hit Z face
                voxel_state[2] += scattered ? 0 : (dir.z > 0.f ? 1 : -1);
            }

            // Apply absorption (Beer-Lambert law)
            float travel_dist_mm = travel_dist * voxel_size_mm;
            weight *= expf(-optical_props.mua * travel_dist_mm);

            // Reduce remaining scattering path length
            scatter_len -= scatter_len_to_boundary;

            // Handle interface physics (reflection/refraction at boundaries)
            if (!scattered) {
                int3 next_voxel_idx = make_int3(voxel_state[0], voxel_state[1], voxel_state[2]);
                bool at_boundary = !grid->is_inside(next_voxel_idx);

                // Determine next medium's refractive index
                float n_next = at_boundary ?
                    c_materials[0].x :  // Ambient refractive index from material 0
                    grid->get_properties(next_voxel_idx).n;

                // Handle refractive index change
                if (fabsf(optical_props.n - n_next) > EPS) {
                    float n_current = optical_props.n;
                    float R = reflectcoeff(&dir, n_current, n_next, voxel_state[3]);

                    // Russian roulette: reflect or transmit
                    if (curand_uniform(&state) < R) {
                        reflect_at_interface(&dir, voxel_state, &inv_dir);
                    } else {
                        transmit(&dir, n_current, n_next, voxel_state[3]);
                        inv_dir = calc_inverse_dir(dir);

                        if (at_boundary) {
                            exited = true;
                        }
                    }
                } else if (at_boundary) {
                    exited = true;
                }
            }

            // Russian roulette for low-weight photons
            if (weight < ROULETTE_WEIGHT) {
                if (curand_uniform(&state) > SURVIVAL_PROB) {
                    weight = 0.0;
                    exited = true;
                }
                weight /= SURVIVAL_PROB;
            }
        }

        // Store exited photons based on which boundary face they crossed
        if (weight > WEIGHT_THRESHOLD && exited) {
            // Convert to world coordinates
            pos = voxel_to_world(pos_voxel, grid);

            // Determine which face the photon exited from
            int3 voxel_idx = make_int3(voxel_state[0], voxel_state[1], voxel_state[2]);
            int exit_face;
            bool is_positive_face;

            // Check if this face is configured to collect photons
            if (get_exit_face(voxel_idx, grid->dims, exit_face, is_positive_face) &&
                bc.is_collecting(exit_face)) {
                // Check radius constraint
                if (bc.within_radius(pos, is_positive_face)) {
                    // Add to appropriate batch (negative or positive)
                    if (is_positive_face) {
                        params.positive_boundary_batch.add_photon(pos, dir, weight);
                    } else {
                        params.negative_boundary_batch.add_photon(pos, dir, weight);
                    }
                }
            }
        }
    }
}


}; // namespace phonder::voxel