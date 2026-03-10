#pragma once

#include <optix.h>
#include <vector_types.h>

// This header defines data structures shared between the host (C++)
// and the device (CUDA). It should be C-compatible.


// SBT record for disk primitive (used for custom detector geometry)
struct DiskSbtData {
    float3 center;
    float3 normal;
    float radius;
};

// ============================================
// Triangle Mesh Material SBT Structures
// ============================================
// These materials use triangle geometric normals

struct LambertianSbtData {
    float reflectance;  // 反射率 (0-1)
};

// SBT data for detector surface
struct DetectorSbtData {
    float3 position;
    float3 normal;
    float radius;
    float sensitivity;  // 灵敏度系数 (通常为 1.0)
};

struct AbsorberSbtData {
    // Empty - absorber doesn't need any parameters
    int dummy;
};

struct MixedMaterialSbtData {
    float diffuse_ratio;   // Lambertian 散射比例 (0-1)
    float specular_ratio;  // 镜面反射比例 (0-1)
    float reflectance;     // 总反射率 (0-1)
};

// ============================================
// Spherical Material SBT Structures
// ============================================
// These materials use spherical normal calculation (from center)
// instead of triangle geometric normals. Faster but only accurate
// for perfectly spherical surfaces.

struct SphericalLambertianSbtData {
    float reflectance;  // 反射率 (e.g., 0.98)
    float3 center;      // 球心（用于计算球面法线）
};

struct SphericalMixedSbtData {
    float diffuse_ratio;   // Lambertian 散射比例 (0-1)
    float specular_ratio;  // 镜面反射比例 (0-1)
    float reflectance;     // 总反射率 (0-1)
    float3 center;         // 球心（用于计算球面法线）
};

// ============================================
// SoA (Structure of Arrays) 光子数据
// ============================================

// Data passed along with a ray
struct RayPayload {
    float3 origin;
    float3 direction;
    double weight;
    int bounce_count;
    bool active;
    unsigned int seed;
    bool last_bounce_was_specular;  // 上一次反弹是否为镜面反射（用于NEE正确处理）
};

// Shadow ray payload for NEE (Next Event Estimation)
struct ShadowPayload {
    bool occluded;      // 是否被遮挡
};

// The main parameter block passed to the optixLaunch kernel
struct DeviceParams {
    // Scene geometry
    OptixTraversableHandle traversable;

    // Triangle mesh data (for computing geometric normals)
    const float3* vertex_buffer;   // Vertex positions (device pointer)
    const uint3* index_buffer;     // Triangle indices (device pointer)

    // Statistic gathering
    double* flux_buffer;
    unsigned long long* detected_rays_buffer;
    unsigned long long* total_bounces_buffer;
    unsigned int* seed_buffer;

    // Configuration
    unsigned int num_rays;
    unsigned int max_bounces;
    double power_per_ray;
    bool use_nee;           // 是否启用Next Event Estimation (NEE) 方差优化
    unsigned int num_materials;  // Number of materials in the scene (for dynamic SBT offset calculation)

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
        float na;              // Numerical Aperture (数值孔径, 默认 1.0 表示无限制)
        float cos_max_angle;   // cos(θ_max) = sqrt(1 - (NA/n)^2), 预计算用于快速比较
        // Use radius for intersection check, not area
    } detector;

    // Detector triangles for NEE (mesh scene)
    float3* detector_triangles;          // Array of vertices [v0, v1, v2, v0, v1, v2, ...]
    unsigned int num_detector_triangles; // Number of triangles
    float detector_total_area;           // Total area for PDF calculation
};
