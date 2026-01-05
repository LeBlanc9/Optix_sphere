#pragma once

#include <optix.h>
#include "simulation/device_params.h"
#include "constants.h"

// ============================================
// Russian Roulette Configuration
// ============================================
// Weight-based Russian Roulette strategy (PBRT-style):
// 1. Always apply BRDF attenuation (physical correctness)
// 2. When weight drops below threshold, probabilistically terminate low-contribution paths
// 3. Survival probability proportional to weight (self-adaptive to material reflectance)

#define MIN_WEIGHT_FOR_RR 0.1f    // Enable RR when weight < 0.01 (1% of original contribution)
#define MIN_SURVIVAL_PROB 0.05f    // Minimum survival probability (prevent premature termination)

// Launch params (passed from CPU)
extern "C" {
    __constant__ DeviceParams params;
}

// Vector math helpers
__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 operator-(const float3& a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator*(const float3& a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 operator*(float s, const float3& a) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float length(const float3& v) { return sqrtf(dot(v, v)); }
__device__ __forceinline__ float3 normalize(const float3& v) { float len = length(v); return len > 0.0f ? v * (1.0f / len) : v; }
__device__ __forceinline__ float3 cross(const float3& a, const float3& b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

// CUDA 随机数生成器 (简单的 LCG)
__device__ float random_float(unsigned int* seed) {
    *seed = (*seed * 1664525u + 1013904223u);
    return (float)(*seed) / (float)0xFFFFFFFFu;
}

// 生成随机整数 [0, max)
__device__ unsigned int random_uint(unsigned int* seed, unsigned int max) {
    *seed = (*seed * 1664525u + 1013904223u);
    return *seed % max;
}

// 生成漫反射方向 (Lambertian BRDF)
__device__ float3 sample_lambertian(const float3& normal, unsigned int* seed) {
    float u1 = random_float(seed);
    float u2 = random_float(seed);
    float r = sqrtf(u1);
    float theta = 2.0f * M_PIf * u2;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(1.0f - u1);
    float3 up = fabsf(normal.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return normalize(tangent * x + bitangent * y + normal * z);
}

// ============================================
// Russian Roulette Helper Function
// ============================================
/**
 * Apply Russian Roulette path termination
 *
 * @param payload Ray payload containing weight and seed
 * @return true if ray should continue, false if terminated
 */
__device__ __forceinline__ bool apply_russian_roulette(RayPayload* payload) {
    if (payload->weight < MIN_WEIGHT_FOR_RR) {
        float survival_prob = fmaxf(payload->weight / MIN_WEIGHT_FOR_RR, MIN_SURVIVAL_PROB);
        if (random_float(&payload->seed) > survival_prob) {
            payload->active = 0;
            return false;  // Terminated
        }
        payload->weight /= survival_prob;
    }
    return true;  // Continue
}

// ============================================
// Next Event Estimation (NEE) Helper Function
// ============================================
/**
 * Accumulate NEE contribution by shooting a shadow ray to the detector
 *
 * @param hit_point Surface hit point
 * @param shading_normal Surface normal (facing ray origin)
 * @param brdf_reflectance BRDF reflectance term (ρ/π for Lambertian, adjusted for mixed materials)
 * @param weight Ray weight/throughput
 * @param shadow_sbt_offset SBT offset for shadow rays (material-dependent)
 */
__device__ __forceinline__ void accumulate_nee_contribution(
    const float3& hit_point,
    const float3& shading_normal,
    double brdf_reflectance,
    double weight,
    unsigned int shadow_sbt_offset
) {
    // Compute direction and distance to detector
    float3 to_detector = params.detector.position - hit_point;
    float distance = length(to_detector);
    float3 dir_to_detector = to_detector * (1.0f / distance);

    // Check if detector is in surface normal's positive hemisphere
    float cos_theta_surface = dot(dir_to_detector, shading_normal);
    if (cos_theta_surface > 0.0f) {
        // Check detector normal direction (detector only receives front-facing light)
        float cos_theta_detector = dot(-dir_to_detector, params.detector.normal);

        if (cos_theta_detector > 0.0f) {
            // Compute geometric factor (solid angle projection)
            float detector_area = M_PIf * params.detector.radius * params.detector.radius;
            float geometric_factor = (detector_area * cos_theta_detector) / (distance * distance);

            // BRDF * cos(θ) term
            double brdf_cosine = brdf_reflectance * cos_theta_surface;

            // NEE contribution = weight * BRDF * cos(θ) * geometric_factor
            double nee_contribution = weight * brdf_cosine * geometric_factor;

            // Shoot shadow ray to check visibility
            ShadowPayload shadow_payload;
            shadow_payload.occluded = false;

            unsigned long long shadow_ptr = reinterpret_cast<unsigned long long>(&shadow_payload);
            unsigned int s0 = static_cast<unsigned int>(shadow_ptr);
            unsigned int s1 = static_cast<unsigned int>(shadow_ptr >> 32);

            optixTrace(
                params.traversable,
                hit_point + shading_normal * 1e-4f,  // Offset to avoid self-intersection
                dir_to_detector,
                1e-4f,                               // tmin
                distance - 1e-4f,                    // tmax (to detector)
                0.0f,                                // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,  // Stop at first hit (occlusion test)
                shadow_sbt_offset,                   // SBT offset (shadow rays)
                1,                                   // SBT stride
                1,                                   // missSBTIndex (shadow miss)
                s0, s1                               // payload
            );

            // Accumulate contribution if not occluded
            if (!shadow_payload.occluded) {
                atomicAdd(params.flux_buffer, nee_contribution);
            }
        }
    }
}
