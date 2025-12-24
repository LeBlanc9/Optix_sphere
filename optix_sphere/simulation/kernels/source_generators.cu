#include "../device_params.h"
#include "kernel_utils.cuh"

// ============================================
// GPU-side photon source generators
// ============================================
// These kernels generate InputPhoton arrays directly on the GPU,
// avoiding CPU->GPU transfer overhead. The generated photons are
// then consumed by the data-driven __raygen__ program.

/**
 * @brief Isotropic point source generator (4π steradian emission)
 *
 * Generates photons emitted uniformly in all directions from a point.
 * Uses spherical coordinate sampling with uniform distribution.
 *
 * @param output_photons Output array (device memory)
 * @param num_photons Number of photons to generate
 * @param position Source position (mm)
 * @param power_per_photon Power carried by each photon (W)
 * @param seed_offset Random seed offset for reproducibility
 */
__global__ void generate_isotropic_point_source(
    InputPhoton* output_photons,
    unsigned int num_photons,
    float3 position,
    float power_per_photon,
    unsigned int seed_offset)
{
    unsigned int photon_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (photon_index >= num_photons) {
        return;
    }

    // Initialize random seed (unique per photon)
    unsigned int seed = seed_offset + photon_index;

    // Sample uniform direction using spherical coordinates
    // θ ∈ [0, 2π], φ ∈ [0, π]
    float u1 = random_float(&seed);
    float u2 = random_float(&seed);
    float theta = 2.0f * M_PIf * u1;           // Azimuthal angle
    float phi = acosf(2.0f * u2 - 1.0f);       // Polar angle (uniform on sphere)

    float3 direction = make_float3(
        sinf(phi) * cosf(theta),
        sinf(phi) * sinf(theta),
        cosf(phi)
    );

    // Write to output array
    output_photons[photon_index].position = position;
    output_photons[photon_index].direction = direction;
    output_photons[photon_index].weight = power_per_photon;
    output_photons[photon_index].seed = seed;
}

/**
 * @brief Collimated laser beam source with Gaussian divergence
 *
 * Generates photons forming a laser beam with optional divergence.
 * Uses small-angle approximation for divergence (cone sampling).
 *
 * @param output_photons Output array (device memory)
 * @param num_photons Number of photons to generate
 * @param position Beam origin (mm)
 * @param direction Beam central axis (normalized)
 * @param divergence_angle Half-angle divergence in radians (0 = perfect collimation)
 * @param power_per_photon Power carried by each photon (W)
 * @param seed_offset Random seed offset
 */
__global__ void generate_laser_beam_source(
    InputPhoton* output_photons,
    unsigned int num_photons,
    float3 position,
    float3 direction,
    float divergence_angle,
    float power_per_photon,
    unsigned int seed_offset)
{
    unsigned int photon_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (photon_index >= num_photons) {
        return;
    }

    unsigned int seed = seed_offset + photon_index;

    // Sample direction within cone (using small-angle approximation)
    float3 perturbed_direction;
    if (divergence_angle > 1e-6f) {
        // Sample random point in unit disk (for small-angle cone)
        float u1 = random_float(&seed);
        float u2 = random_float(&seed);
        float r = sqrtf(u1) * tanf(divergence_angle);  // Radius in tangent space
        float theta = 2.0f * M_PIf * u2;               // Azimuthal angle

        // Build orthonormal basis (Frisvad method - efficient for GPU)
        float3 up, right;
        if (fabsf(direction.z) < 0.999f) {
            float3 tmp = make_float3(0, 0, 1);
            right = normalize(cross(tmp, direction));
        } else {
            float3 tmp = make_float3(1, 0, 0);
            right = normalize(cross(tmp, direction));
        }
        up = normalize(cross(direction, right));

        // Perturb direction
        float dx = r * cosf(theta);
        float dy = r * sinf(theta);
        perturbed_direction = normalize(
            direction + dx * right + dy * up
        );
    } else {
        // Perfect collimation
        perturbed_direction = direction;
    }

    output_photons[photon_index].position = position;
    output_photons[photon_index].direction = perturbed_direction;
    output_photons[photon_index].weight = power_per_photon;
    output_photons[photon_index].seed = seed;
}

/**
 * @brief Lambertian surface emitter (cosine-weighted hemisphere)
 *
 * Generates photons emitted from a surface with Lambertian emission pattern.
 * Useful for modeling diffuse light sources or secondary emission.
 *
 * @param output_photons Output array (device memory)
 * @param num_photons Number of photons to generate
 * @param position Emission surface center (mm)
 * @param normal Surface normal (outward, normalized)
 * @param power_per_photon Power carried by each photon (W)
 * @param seed_offset Random seed offset
 */
__global__ void generate_lambertian_emitter(
    InputPhoton* output_photons,
    unsigned int num_photons,
    float3 position,
    float3 normal,
    float power_per_photon,
    unsigned int seed_offset)
{
    unsigned int photon_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (photon_index >= num_photons) {
        return;
    }

    unsigned int seed = seed_offset + photon_index;

    // Sample cosine-weighted hemisphere using Malley's method
    float3 direction = sample_lambertian(normal, &seed);

    output_photons[photon_index].position = position;
    output_photons[photon_index].direction = direction;
    output_photons[photon_index].weight = power_per_photon;
    output_photons[photon_index].seed = seed;
}

/**
 * @brief Ring/annular beam source (e.g., fiber output with central obscuration)
 *
 * Generates photons emitted from a ring-shaped aperture with configurable
 * inner/outer radius and angular divergence.
 *
 * @param output_photons Output array (device memory)
 * @param num_photons Number of photons to generate
 * @param center Ring center position (mm)
 * @param axis Ring axis direction (normalized)
 * @param inner_radius Inner radius of ring (mm)
 * @param outer_radius Outer radius of ring (mm)
 * @param divergence_angle Half-angle divergence (radians)
 * @param power_per_photon Power carried by each photon (W)
 * @param seed_offset Random seed offset
 */
__global__ void generate_ring_beam_source(
    InputPhoton* output_photons,
    unsigned int num_photons,
    float3 center,
    float3 axis,
    float inner_radius,
    float outer_radius,
    float divergence_angle,
    float power_per_photon,
    unsigned int seed_offset)
{
    unsigned int photon_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (photon_index >= num_photons) {
        return;
    }

    unsigned int seed = seed_offset + photon_index;

    // Build orthonormal basis
    float3 up, right;
    if (fabsf(axis.z) < 0.999f) {
        float3 tmp = make_float3(0, 0, 1);
        right = normalize(cross(tmp, axis));
    } else {
        float3 tmp = make_float3(1, 0, 0);
        right = normalize(cross(tmp, axis));
    }
    up = normalize(cross(axis, right));

    // Sample position on annular disk
    float u1 = random_float(&seed);
    float u2 = random_float(&seed);

    // Uniform sampling in annular region
    float r_squared = inner_radius * inner_radius +
                      u1 * (outer_radius * outer_radius - inner_radius * inner_radius);
    float r = sqrtf(r_squared);
    float theta = 2.0f * M_PIf * u2;

    float3 position = center + r * (cosf(theta) * right + sinf(theta) * up);

    // Sample direction with divergence
    float u3 = random_float(&seed);
    float u4 = random_float(&seed);
    float div_r = sqrtf(u3) * tanf(divergence_angle);
    float div_theta = 2.0f * M_PIf * u4;

    float3 direction = normalize(
        axis + div_r * (cosf(div_theta) * right + sinf(div_theta) * up)
    );

    output_photons[photon_index].position = position;
    output_photons[photon_index].direction = direction;
    output_photons[photon_index].weight = power_per_photon;
    output_photons[photon_index].seed = seed;
}
