#pragma once
#include <cuda_runtime.h>

namespace phonder {

__global__ void generate_isotropic_point_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 source_pos, double weight, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    curandState state;
    curand_init(seed + idx * 97ULL, idx, 0, &state);

    float u1 = curand_uniform(&state);
    float u2 = curand_uniform(&state);
    float theta = two_pi * u1;
    float phi = acosf(2.0f * u2 - 1.0f);

    float3 dir = make_float3(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));

    positions[idx] = source_pos;
    directions[idx] = dir;
    weights[idx] = weight;
}

__global__ void generate_collimated_beam_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 source_pos, float3 source_dir, double weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    positions[idx] = source_pos;
    directions[idx] = source_dir;
    weights[idx] = weight;
}

__global__ void generate_spot_source_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 center_position, float3 disk_normal, float3 direction,
    float radius, double weight, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    curandState state;
    curand_init(seed + idx * 97ULL, idx, 0, &state);

    // Build orthonormal basis for the disk
    float3 w = normalized(disk_normal);
    float3 u, v;
    if (fabsf(w.x) > 0.9f) {
        u = normalized(cross(make_float3(0.0f, 1.0f, 0.0f), w));
    } else {
        u = normalized(cross(make_float3(1.0f, 0.0f, 0.0f), w));
    }
    v = cross(w, u);

    // Sample uniformly on disk
    float r = radius * sqrtf(curand_uniform(&state));
    float theta = two_pi * curand_uniform(&state);
    float3 pos_offset = r * (cosf(theta) * u + sinf(theta) * v);

    positions[idx] = center_position + pos_offset;
    directions[idx] = direction;
    weights[idx] = weight;
}


__global__ void generate_gaussian_source_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 center_position, float3 direction,
    float beam_waist, double weight, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    curandState state;
    curand_init(seed + idx * 97ULL, idx, 0, &state);

    // Build orthonormal basis
    float3 w = normalized(direction);
    float3 u, v;
    if (fabsf(w.x) > 0.9f) {
        u = normalized(cross(make_float3(0.0f, 1.0f, 0.0f), w));
    } else {
        u = normalized(cross(make_float3(1.0f, 0.0f, 0.0f), w));
    }
    v = cross(w, u);

    // Sample from 2D Gaussian
    float r1 = curand_normal(&state) * beam_waist;
    float r2 = curand_normal(&state) * beam_waist;
    float3 pos_offset = r1 * u + r2 * v;

    positions[idx] = center_position + pos_offset;
    directions[idx] = direction;
    weights[idx] = weight;
}

__global__ void generate_focused_spot_source_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 spot_center, float spot_radius,
    float convergence_half_angle, float3 main_axis, float source_distance,
    double weight, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    curandState state;
    curand_init(seed + idx * 97ULL, idx, 0, &state);

    // Build orthonormal basis
    float3 w = normalized(main_axis);
    float3 u, v;
    if (fabsf(w.x) > 0.9f) {
        u = normalized(cross(make_float3(0.0f, 1.0f, 0.0f), w));
    } else {
        u = normalized(cross(make_float3(1.0f, 0.0f, 0.0f), w));
    }
    v = cross(w, u);

    // Source disk
    float3 source_center = spot_center - w * source_distance;
    float source_radius = source_distance * tanf(convergence_half_angle);

    // Sample on source disk
    float r_source = source_radius * sqrtf(curand_uniform(&state));
    float theta_source = two_pi * curand_uniform(&state);
    float3 source_offset = r_source * (cosf(theta_source) * u + sinf(theta_source) * v);
    float3 photon_pos = source_center + source_offset;

    // Sample on target spot
    float r_target = spot_radius * sqrtf(curand_uniform(&state));
    float theta_target = two_pi * curand_uniform(&state);
    float3 target_offset = r_target * (cosf(theta_target) * u + sinf(theta_target) * v);
    float3 target_pos = spot_center + target_offset;

    positions[idx] = photon_pos;
    directions[idx] = normalized(target_pos - photon_pos);
    weights[idx] = weight;
}


}; // namespace phonder
