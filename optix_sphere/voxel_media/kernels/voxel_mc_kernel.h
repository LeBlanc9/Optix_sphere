#pragma once
#include <cuda_runtime.h>
#include "voxel_media/voxel_grid.cuh"

namespace phonder::voxel {

/**
 * @brief Main Monte Carlo simulation kernel for voxel media
 *
 * This kernel simulates photon transport through a 3D voxel grid using
 * Monte Carlo ray tracing with MCX-style physics.
 *
 * @param grid Device pointer to voxel grid structure
 * @param input_positions Input photon positions
 * @param input_directions Input photon directions (normalized)
 * @param input_weights Input photon weights
 * @param input_size Number of input photons
 * @param specular_positions Output: specular reflection positions
 * @param specular_directions Output: specular reflection directions
 * @param specular_weights Output: specular reflection weights
 * @param reflected_positions Output: diffuse reflection positions (exit from -Z face)
 * @param reflected_directions Output: diffuse reflection directions
 * @param reflected_weights Output: diffuse reflection weights
 * @param transmitted_positions Output: transmitted positions (exit from +Z face)
 * @param transmitted_directions Output: transmitted directions
 * @param transmitted_weights Output: transmitted weights
 * @param specular_counter Atomic counter for specular photons
 * @param reflected_counter Atomic counter for reflected photons
 * @param transmitted_counter Atomic counter for transmitted photons
 * @param output_capacity Maximum output buffer capacity
 * @param exit_z_min Minimum Z coordinate for exit detection
 * @param exit_z_max Maximum Z coordinate for exit detection
 * @param seed Random seed for curand
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
);

} // namespace phonder::voxel
