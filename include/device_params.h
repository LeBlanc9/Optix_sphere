#pragma once

#include <optix.h>
#include <vector_types.h>

// This header defines data structures shared between the host (C++)
// and the device (CUDA). It should be C-compatible.

// SBT record for our sphere primitive
struct SphereSbtData {
    float3 center;
    float radius;
    float reflectance;
};

// Data passed along with a ray
struct RayPayload {
    float3 origin;
    float3 direction;
    float power;
    int bounce_count;
    bool active;
    unsigned int seed;
};

// The main parameter block passed to the optixLaunch kernel
struct DeviceParams {
    // Scene geometry
    OptixTraversableHandle traversable;

    // We pass pointers to device buffers for dynamic data
    unsigned int* seed_buffer;
    float* flux_buffer;

    // Simulation settings
    int num_rays;
    int max_bounces;
    float power_per_ray;

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
