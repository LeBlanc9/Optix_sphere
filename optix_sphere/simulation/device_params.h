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

// Multi-material SBT data structures for triangle meshes

// SBT data for normal sphere wall (high reflectance)
struct SphereWallSbtData {
    float reflectance;  // 反射率 (e.g., 0.98)
    float3 center;      // 球心（用于计算法线）
};

// SBT data for detector surface
struct DetectorSbtData {
    float3 position;
    float3 normal;
    float radius;
    float sensitivity;  // 灵敏度系数 (通常为 1.0)
};

// SBT data for baffle (low reflectance)
struct BaffleSbtData {
    float reflectance;  // 低反射率 (e.g., 0.1-0.3)
    float3 center;      // 球心
};

// SBT data for port hole (complete absorption)
struct PortHoleSbtData {
    float3 center;      // 球心（可能不需要，但保持一致性）
};

// ============================================
// SoA (Structure of Arrays) 光子数据
// 用于高性能 GPU 数据驱动模拟
// ============================================
// 注意：我们统一使用 SoA 格式，不再使用 InputPhoton (AoS)
// SoA 格式对 GPU 合并访问更友好

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

    // Data-driven mode (SoA format for GPU efficiency)
    // 当这些指针非空时，raygen 从外部光源数据读取
    const float3* photon_positions;         // 光子起始位置数组 (device pointer)
    const float3* photon_directions;        // 光子起始方向数组 (device pointer)
    const double* photon_weights;           // 光子权重数组 (device pointer)
    unsigned long long photon_seed_base;  // 基础随机种子（每个光子 seed = base + idx * 97）
    unsigned int num_input_photons;   // 输入光子数量

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

    // Detector triangles for NEE (mesh scene)
    float3* detector_triangles;          // Array of vertices [v0, v1, v2, v0, v1, v2, ...]
    unsigned int num_detector_triangles; // Number of triangles
    float detector_total_area;           // Total area for PDF calculation
};
