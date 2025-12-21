#pragma once

#include <optix.h>
#include <vector_types.h>

// This header defines data structures shared between the host (C++)
// and the device (CUDA). It should be C-compatible.

// SBT record for sphere primitive
struct SphereSbtData {
    float3 center;
    float radius;
    float reflectance;
};

// SBT record for disk primitive (detector)
struct DiskSbtData {
    float3 center;
    float3 normal;
    float radius;
};

// Data passed along with a ray
struct RayPayload {
    float3 origin;
    float3 direction;
    double weight;      // 无量纲权重 (初始=1.0, 代表光子存活概率) - 使用double提高精度
    int bounce_count;
    bool active;
    unsigned int seed;
};

// Shadow ray payload for NEE (Next Event Estimation)
struct ShadowPayload {
    bool occluded;      // 是否被遮挡
};

// The main parameter block passed to the optixLaunch kernel
struct DeviceParams {
    // Scene geometry
    OptixTraversableHandle traversable;

    // Statistic gathering
    double* flux_buffer;
    unsigned long long* detected_rays_buffer;
    unsigned long long* total_bounces_buffer;
    unsigned int* seed_buffer;

    // Configuration
    unsigned int num_rays;
    unsigned int max_bounces;
    double power_per_ray;   // 使用double精度
    bool use_nee;           // 是否启用Next Event Estimation (NEE) 方差优化

    // Scene objects
    // For a simple scene, we can pass small objects by value.
    // For complex scenes, these would be pointers to device memory.
    struct {
        float3 position;
    } light_source;

    struct {
        float3 position;
        float3 normal;
        float radius;
        // Use radius for intersection check, not area
    } detector;
};
