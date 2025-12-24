#include "photon/launchers.h"
#include "photon/kernels.cuh"
#include "photon/batch.cuh"
#include "utils/device/math.cuh"
#include "constants.h"
#include <curand_kernel.h>

// This file implements the C-style launcher function and contains the
// definitions of the CUDA kernels. It is the core of the CUDA implementation.

namespace phonder {

// ============================================
// CUDA Kernel Definitions
// ============================================

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
    float theta = TWO_PI * u1;
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

// NOTE: The other kernels (spot, gaussian, focused) were missing from the
// original .cu files. They need to be implemented here. For now, they will
// be placeholders that do nothing. This will allow the project to compile.
// A proper implementation would sample points on a disk, etc.

__global__ void generate_spot_source_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 center_position, float3 direction, float radius, double weight, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    curandState state;
    curand_init(seed + idx * 97ULL, idx, 0, &state);

    // Build orthonormal basis for the disk
    float3 w = normalized(direction);
    float3 u, v;
    if (fabsf(w.x) > 0.9f) {
        u = normalized(cross(make_float3(0.0f, 1.0f, 0.0f), w));
    } else {
        u = normalized(cross(make_float3(1.0f, 0.0f, 0.0f), w));
    }
    v = cross(w, u);

    // Sample a random point on the disk
    float r = radius * sqrtf(curand_uniform(&state));
    float theta = TWO_PI * curand_uniform(&state);

    float3 pos_offset = r * (cosf(theta) * u + sinf(theta) * v);

    positions[idx] = center_position + pos_offset;
    directions[idx] = direction;
    weights[idx] = weight;
}

__global__ void generate_gaussian_source_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 center_position, float3 direction, float beam_waist, double weight, unsigned long long seed)
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

    // Sample from a 2D Gaussian distribution using the Box-Muller transform
    // or cuRAND's built-in normal distribution generator.
    // Using curand_normal for simplicity and performance.
    float r1 = curand_normal(&state) * beam_waist;
    float r2 = curand_normal(&state) * beam_waist;

    float3 pos_offset = r1 * u + r2 * v;
    
    positions[idx] = center_position + pos_offset;
    directions[idx] = direction;
    weights[idx] = weight;
}

__global__ void generate_focused_spot_source_kernel(
    float3* positions, float3* directions, double* weights,
    int num_photons, float3 spot_center, float spot_radius, float convergence_half_angle, float3 main_axis, float source_distance, double weight, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    curandState state;
    curand_init(seed + idx * 97ULL, idx, 0, &state);

    // Build orthonormal basis from the main axis
    float3 w = normalized(main_axis);
    float3 u, v;
    if (fabsf(w.x) > 0.9f) {
        u = normalized(cross(make_float3(0.0f, 1.0f, 0.0f), w));
    } else {
        u = normalized(cross(make_float3(1.0f, 0.0f, 0.0f), w));
    }
    v = cross(w, u);

    // 1. Calculate the source disk properties
    float3 source_center = spot_center - w * source_distance;
    float source_radius = source_distance * tanf(convergence_half_angle);

    // 2. Sample a random point on the source disk
    float r_source = source_radius * sqrtf(curand_uniform(&state));
    float theta_source = TWO_PI * curand_uniform(&state);
    float3 source_offset = r_source * (cosf(theta_source) * u + sinf(theta_source) * v);
    float3 photon_pos = source_center + source_offset;

    // 3. Sample a random point on the target spot disk
    float r_target = spot_radius * sqrtf(curand_uniform(&state));
    float theta_target = TWO_PI * curand_uniform(&state);
    float3 target_offset = r_target * (cosf(theta_target) * u + sinf(theta_target) * v);
    float3 target_pos = spot_center + target_offset;

    // 4. Set photon properties
    positions[idx] = photon_pos;
    directions[idx] = normalized(target_pos - photon_pos);
    weights[idx] = weight;
}


// ============================================
// C-style Launcher Implementation
// ============================================

template<typename KernelFunc, typename... Args>
void launch_kernel(int num_photons, KernelFunc kernel, Args... args) {
    const int block_size = 256;
    const int grid_size = (num_photons + block_size - 1) / block_size;
    kernel<<<grid_size, block_size>>>(args...);
    // Note: No cudaDeviceSynchronize here. It's better to let the caller decide
    // when to synchronize, for example after a batch of operations.
}

void generate_photons_on_device(
    const PhotonSource& source,
    DevicePhotonBatch& batch_out,
    int num_photons,
    unsigned long long seed)
{
    batch_out.resize(num_photons);
    PhotonBatchView view = batch_out.get_view();

    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, IsotropicPointSource>) {
            launch_kernel(num_photons, generate_isotropic_point_kernel, view.positions, view.directions, view.weights, num_photons, arg.position, arg.weight, seed);
        } else if constexpr (std::is_same_v<T, CollimatedBeamSource>) {
            launch_kernel(num_photons, generate_collimated_beam_kernel, view.positions, view.directions, view.weights, num_photons, arg.position, arg.direction, arg.weight);
        } else if constexpr (std::is_same_v<T, SpotSource>) {
            launch_kernel(num_photons, generate_spot_source_kernel, view.positions, view.directions, view.weights, num_photons, arg.center_position, arg.direction, arg.radius, arg.weight, seed);
        } else if constexpr (std::is_same_v<T, GaussianBeamSource>) {
            launch_kernel(num_photons, generate_gaussian_source_kernel, view.positions, view.directions, view.weights, num_photons, arg.center_position, arg.direction, arg.beam_waist, arg.weight, seed);
        } else if constexpr (std::is_same_v<T, FocusedSpotSource>) {
            launch_kernel(num_photons, generate_focused_spot_source_kernel, view.positions, view.directions, view.weights, num_photons, arg.spot_center, arg.spot_radius, arg.convergence_half_angle_rad, arg.main_axis, arg.source_distance, arg.weight, seed);
        }
    }, source);

    // Synchronize after the kernel launch to ensure data is ready.
    cudaDeviceSynchronize();
}


} // namespace phonder
