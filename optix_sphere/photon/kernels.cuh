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
    float3 disk_normal, float convergence_half_angle, float3 main_axis,
    double weight, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    curandState state;
    curand_init(seed + idx * 97ULL, idx, 0, &state);

    // Build orthonormal basis for disk position (perpendicular to disk_normal)
    float3 w_disk = normalized(disk_normal);
    float3 u_disk, v_disk;
    if (fabsf(w_disk.x) > 0.9f) {
        u_disk = normalized(cross(make_float3(0.0f, 1.0f, 0.0f), w_disk));
    } else {
        u_disk = normalized(cross(make_float3(1.0f, 0.0f, 0.0f), w_disk));
    }
    v_disk = cross(w_disk, u_disk);

    // Sample position uniformly on disk centered at spot_center
    float r_pos = spot_radius * sqrtf(curand_uniform(&state));
    float theta_pos = two_pi * curand_uniform(&state);
    float3 pos_offset = r_pos * (cosf(theta_pos) * u_disk + sinf(theta_pos) * v_disk);

    // Build orthonormal basis for direction (perpendicular to main_axis)
    float3 w_dir = normalized(main_axis);
    float3 u_dir, v_dir;
    if (fabsf(w_dir.x) > 0.9f) {
        u_dir = normalized(cross(make_float3(0.0f, 1.0f, 0.0f), w_dir));
    } else {
        u_dir = normalized(cross(make_float3(1.0f, 0.0f, 0.0f), w_dir));
    }
    v_dir = cross(w_dir, u_dir);

    // Sample direction uniformly within cone around main_axis
    // cos_theta uniformly distributed in [cos(convergence_half_angle), 1]
    float cos_theta_max = cosf(convergence_half_angle);
    float cos_theta = cos_theta_max + (1.0f - cos_theta_max) * curand_uniform(&state);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    float phi = two_pi * curand_uniform(&state);

    // Direction in local coordinate system (cone around main_axis)
    float3 direction = sin_theta * cosf(phi) * u_dir +
                      sin_theta * sinf(phi) * v_dir +
                      cos_theta * w_dir;

    positions[idx] = spot_center + pos_offset;
    directions[idx] = normalized(direction);
    weights[idx] = weight;
}

// Lambertian disk source - cosine-weighted hemisphere emission
// Position: uniform on disk, Direction: cosine-weighted hemisphere
__global__ void generate_lambertian_disk_source_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 center_position, float3 disk_normal,
    float radius, double weight, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    curandState state;
    curand_init(seed + idx * 97ULL, idx, 0, &state);

    // Build orthonormal basis with disk_normal as the hemisphere orientation
    float3 w = normalized(disk_normal);
    float3 u, v;
    if (fabsf(w.x) > 0.9f) {
        u = normalized(cross(make_float3(0.0f, 1.0f, 0.0f), w));
    } else {
        u = normalized(cross(make_float3(1.0f, 0.0f, 0.0f), w));
    }
    v = cross(w, u);

    // Sample position uniformly on disk
    float r = radius * sqrtf(curand_uniform(&state));
    float theta = two_pi * curand_uniform(&state);
    float3 pos_offset = r * (cosf(theta) * u + sinf(theta) * v);

    // Sample direction from cosine-weighted hemisphere
    // Using Malley's method: uniform disk -> project to hemisphere
    float r_dir = sqrtf(curand_uniform(&state));
    float theta_dir = two_pi * curand_uniform(&state);
    float x = r_dir * cosf(theta_dir);
    float y = r_dir * sinf(theta_dir);
    float z = sqrtf(1.0f - x*x - y*y);  // Ensures normalized and cosine-weighted

    float3 direction = x * u + y * v + z * w;

    positions[idx] = center_position + pos_offset;
    directions[idx] = direction;
    weights[idx] = weight;
}


}; // namespace phonder
