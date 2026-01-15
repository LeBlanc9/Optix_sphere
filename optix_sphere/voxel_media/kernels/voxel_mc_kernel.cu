#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "voxel_media/voxel_grid.cuh"
#include "voxel_media/kernels/voxel_ray_tracing.cuh"
#include "voxel_media/kernels/voxel_physics.cuh"
#include "utils/device/math.cuh"

namespace phonder::voxel {

/**
 * @brief Handle photon entry into grid
 *
 * Applies Fresnel reflection at entry interface, stores specular component,
 * and refracts transmitted component.
 *
 * @param grid Voxel grid
 * @param voxel_idx Entry voxel index
 * @param dir Direction (modified by refraction)
 * @param weight Weight (reduced by reflection)
 * @param pos Entry position
 * @param specular_positions,specular_directions,specular_weights Specular output arrays
 * @param specular_counter Atomic counter
 * @param output_capacity Buffer capacity
 * @return true if photon enters successfully
 */
__device__ inline bool launch_into_grid(
    const Grid* grid,
    const int3& voxel_idx,
    float3& dir,
    double& weight,
    const float3& pos,
    float3* specular_positions,
    float3* specular_directions,
    double* specular_weights,
    int* specular_counter,
    int output_capacity
) {
    if (!grid->is_inside(voxel_idx)) {
        return false;
    }

    OpticalProperties entry_props = grid->get_properties(voxel_idx);
    float n1 = grid->ambient_n;  // outside (air)
    float n2 = entry_props.n;     // inside (tissue)

    // all photons enter from -z face (flipdir=2 for z-direction)
    const int entry_flipdir = 2;

    // calculate fresnel reflection coefficient
    float r = reflectcoeff(&dir, n1, n2, entry_flipdir);

    // store specular reflection photon
    if (r > 1e-6f) {
        int out_idx = atomicAdd(specular_counter, 1);
        if (out_idx < output_capacity) {
            float3 spec_dir = dir;
            spec_dir.z = -spec_dir.z;  // reflect about z-axis
            specular_positions[out_idx] = pos;
            specular_directions[out_idx] = spec_dir;
            specular_weights[out_idx] = weight * r;
        }
    }

    // reduce weight by transmission coefficient (1-r)
    weight *= (1.0 - r);

    // apply refraction to transmitted photon
    if (fabsf(n1 - n2) > 1e-6f) {
        transmit(&dir, n1, n2, entry_flipdir);
    }

    return (weight > 1e-6);
}


/**
 * @brief Monte Carlo photon transport kernel for voxel media
 */
__global__ void mc_kernel(
    const Grid* grid,
    const float3* input_positions,
    const float3* input_directions,
    const double* input_weights,
    int input_size,
    float3* specular_positions,
    float3* specular_directions,
    double* specular_weights,
    float3* reflected_positions,
    float3* reflected_directions,
    double* reflected_weights,
    float3* transmitted_positions,
    float3* transmitted_directions,
    double* transmitted_weights,
    int* specular_counter,
    int* reflected_counter,
    int* transmitted_counter,
    int output_capacity,
    float exit_z_min,
    float exit_z_max,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    curandState state;
    curand_init(seed + idx * 97, idx, 0, &state);

    const int MAX_ITER = 10000;
    const float WEIGHT_THRESHOLD = 1e-6f;
    const float ROULETTE_WEIGHT = 0.001f;
    const float SURVIVAL_PROB = 0.1f;

    for (int i = idx; i < input_size; i += stride) {
        // Load photon from input batch
        float3 pos = input_positions[i];
        float3 dir = input_directions[i];
        double weight = input_weights[i];


        // Skip photons with zero weight
        if (weight <= WEIGHT_THRESHOLD) continue;

        // Move photon to grid entrance if starting outside (MCX skipvoid style)
        float3 grid_min = make_float3(0.f, 0.f, exit_z_min);
        float3 grid_max = make_float3(grid->dx * grid->nx, grid->dy * grid->ny, exit_z_max);

        bool outside = (pos.x < grid_min.x || pos.x >= grid_max.x ||
                        pos.y < grid_min.y || pos.y >= grid_max.y ||
                        pos.z < grid_min.z || pos.z >= grid_max.z);

        if (outside) {
            float t_min = 1e30f;
            bool can_enter = false;

            // Check all 6 faces
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
                continue;  // Photon moving away from grid
            }

            // Move to entry point with small epsilon (using operator+)
            pos = pos + dir * (t_min + 1e-5f);
        }

        // Convert position to voxel coordinates (MCX works in voxel units)
        float3 pos_voxel = make_float3(
            pos.x / grid->dx,
            pos.y / grid->dy,
            pos.z / grid->dz
        );

        // Small epsilon push to avoid being exactly on a voxel boundary
        pos_voxel = pos_voxel + dir * EPS;

        // Inverse direction for fast ray-voxel intersection (1/dir per component)
        float3 inv_dir = calc_inverse_dir(dir);

        // Voxel state: [0-2] = voxel indices (x,y,z), [3] = last hit face (0=x, 1=y, 2=z, -1=none)
        short voxel_state[4];
        voxel_state[0] = (short)floorf(pos_voxel.x);
        voxel_state[1] = (short)floorf(pos_voxel.y);
        voxel_state[2] = (short)floorf(pos_voxel.z);
        voxel_state[3] = -1;  // No face hit yet

        int3 voxel_idx = make_int3(voxel_state[0], voxel_state[1], voxel_state[2]);

        // Launch photon into voxel grid (handles specular reflection and refraction)
        bool entered = launch_into_grid(
            grid, voxel_idx, dir, weight, pos,
            specular_positions, specular_directions, specular_weights,
            specular_counter, output_capacity
        );


        if (!entered) {
            continue;  // Photon didn't enter (weight too low or outside grid)
        }

        // Update inverse direction after refraction (direction may have changed)
        inv_dir = calc_inverse_dir(dir);

        // Remaining scattering path length (triggers sampling when <= 0)
        float scatter_len = 0.f;

        // Number of scattering events (skip direction change on first iteration)
        int scatter_count = 0;

        // Get initial voxel properties (will be updated in loop)
        voxel_idx = make_int3(voxel_state[0], voxel_state[1], voxel_state[2]);
        OpticalProperties optical_props = grid->get_properties(voxel_idx);

        // Main transport loop
        int iter = 0;
        bool exited = false;

        while (!exited && weight > WEIGHT_THRESHOLD && iter < MAX_ITER) {
            iter++;

            // Sample scattering event
            if (scatter_len <= 0.f) {
                scatter_len = sample_scatter_length(&state);

                // Apply scattering (skip on first iteration to preserve launch direction)
                if (scatter_count > 0) {
                    scatter(&dir, optical_props.g, &state);
                    inv_dir = calc_inverse_dir(dir);
                }

                scatter_count++;
            }

            // Get current voxel properties
            voxel_idx = make_int3(voxel_state[0], voxel_state[1], voxel_state[2]);
            optical_props = grid->get_properties(voxel_idx);

            // Find distance to next voxel boundary
            float dist_to_boundary = hitgrid(&pos_voxel, &dir, (float*)&inv_dir, voxel_state);

            // Convert to physical units for scattering calculation
            float voxel_size_mm = fminf(grid->dx, fminf(grid->dy, grid->dz));
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
            float travel_dist_mm = travel_dist * fminf(grid->dx, fminf(grid->dy, grid->dz));
            weight *= expf(-optical_props.mua * travel_dist_mm);

            // Reduce remaining scattering path length
            scatter_len -= scatter_len_to_boundary;

            // Handle interface physics (reflection/refraction at boundaries)
            if (!scattered) {
                int3 next_voxel_idx = make_int3(voxel_state[0], voxel_state[1], voxel_state[2]);
                bool at_boundary = !grid->is_inside(next_voxel_idx);

                // Determine next medium's refractive index
                float n_next = at_boundary ?
                    grid->ambient_n :
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

        // Store exited photons
        if (weight > WEIGHT_THRESHOLD && exited) {
            // Convert to world coordinates
            pos = make_float3(
                pos_voxel.x * grid->dx,
                pos_voxel.y * grid->dy,
                pos_voxel.z * grid->dz
            );

            // Classify by exit face
            if (voxel_state[2] < 0) {
                // -Z face (reflected)
                int out_idx = atomicAdd(reflected_counter, 1);
                if (out_idx < output_capacity) {
                    reflected_positions[out_idx] = pos;
                    reflected_directions[out_idx] = dir;
                    reflected_weights[out_idx] = weight;
                }
            } else if (voxel_state[2] >= grid->nz) {
                // +Z face (transmitted)
                int out_idx = atomicAdd(transmitted_counter, 1);
                if (out_idx < output_capacity) {
                    transmitted_positions[out_idx] = pos;
                    transmitted_directions[out_idx] = dir;
                    transmitted_weights[out_idx] = weight;
                }
            }
        }
    }
}


}; // namespace phonder::voxel