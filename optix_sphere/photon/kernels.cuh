#pragma once
#include <cuda_runtime.h>

// This header declares the __global__ CUDA kernels for photon generation.
// It should only be included by .cu files that need to launch these kernels.

namespace phonder {

__global__ void generate_isotropic_point_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 source_pos, double weight, unsigned long long seed);

__global__ void generate_collimated_beam_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 source_pos, float3 source_dir, double weight);

__global__ void generate_spot_source_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 center_position, float3 direction, float radius, double weight, unsigned long long seed);

__global__ void generate_gaussian_source_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 center_position, float3 direction, float beam_waist, double weight, unsigned long long seed);

__global__ void generate_focused_spot_source_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 spot_center, float spot_radius, float convergence_half_angle, float3 main_axis, float source_distance, double weight, unsigned long long seed);

} // namespace phonder
