#include <optix.h>
#include "device_params.h"
#include "constants.h"

// Launch params (passed from CPU)
extern "C" {
    __constant__ DeviceParams params;
}

// Vector math helpers (could be in a shared header, but kept here for simplicity)
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

// 圆盘相交函数（用于探测器）
extern "C" __global__ void __intersection__disk() {
    const DiskSbtData* disk = (DiskSbtData*)optixGetSbtDataPointer();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    // 圆盘定义：中心在 disk->center，法线为 disk->normal
    float3 oc = ray_orig - disk->center;
    float denom = dot(ray_dir, disk->normal);

    // 检查射线是否平行于圆盘平面
    if (fabsf(denom) > 1e-6f) {
        float t = -dot(oc, disk->normal) / denom;

        if (t >= ray_tmin && t <= ray_tmax) {
            // 计算交点
            float3 hit_point = ray_orig + t * ray_dir;
            float3 to_center = hit_point - disk->center;
            float dist_sq = dot(to_center, to_center);

            // 检查是否在圆盘半径内
            if (dist_sq <= disk->radius * disk->radius) {
                optixReportIntersection(t, 0);
            }
        }
    }
}

// 解析球面相交函数
extern "C" __global__ void __intersection__sphere() {
    const SphereSbtData* sphere = (SphereSbtData*)optixGetSbtDataPointer();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    float3 oc = ray_orig - sphere->center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0f * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere->radius * sphere->radius;

    float discriminant = b * b - 4.0f * a * c;
    if (discriminant >= 0.0f) {
        float sqrt_d = sqrtf(discriminant);
        float t1 = (-b - sqrt_d) / (2.0f * a);
        float t2 = (-b + sqrt_d) / (2.0f * a);
        float t = t1;
        if (t < ray_tmin) t = t2;
        if (t >= ray_tmin && t <= ray_tmax) {
            // 计算击中点
            float3 hit_point = ray_orig + t * ray_dir;

            // 排除探测器区域：检查击中点是否在探测器圆盘内
            float3 to_detector = hit_point - params.detector.position;
            float dist_to_detector_sq = dot(to_detector, to_detector);

            // 如果击中点在探测器半径内，不报告球面相交（让探测器处理）
            if (dist_to_detector_sq > params.detector.radius * params.detector.radius) {
                optixReportIntersection(t, 0);
            }
        }
    }
}

// 探测器命中程序（简单：记录能量并终止）
extern "C" __global__ void __closesthit__detector() {
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    // 探测器吸收所有入射光子，直接累积权重
    atomicAdd(params.flux_buffer, payload->weight);
    atomicAdd(params.detected_rays_buffer, 1ull);

    // 终止路径
    payload->active = 0;
}

// 最近命中程序（球体反射）
extern "C" __global__ void __closesthit__sphere() {
    const SphereSbtData* sphere = (SphereSbtData*)optixGetSbtDataPointer();
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) | 
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    // 1. 计算命中点和法线
    float t_hit = optixGetRayTmax();
    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_point = ray_orig + t_hit * ray_dir;
    float3 geometric_normal = normalize(hit_point - sphere->center);

    // 法线应该总是与射线方向相反（用于内部反射）
    float3 shading_normal = dot(ray_dir, geometric_normal) < 0 ? geometric_normal : -geometric_normal;

    // 2. Next Event Estimation (NEE): 使用 shadow ray 显式采样探测器
    // TODO: 稍后实现基于 shadow ray 的正确 NEE
    // if (params.use_nee) {
    //     // 朝向探测器发射 shadow ray
    //     // 如果未被遮挡，累积 NEE 贡献
    // }

    // 3. 反射次数检查 & 俄罗斯轮盘赌
    atomicAdd(params.total_bounces_buffer, 1ull); // Increment total bounces
    payload->bounce_count++;
    if (payload->bounce_count >= params.max_bounces) {
        payload->active = 0;
        return;
    }
    float survival_prob = sphere->reflectance;
    if (random_float(&payload->seed) >= survival_prob) {
        payload->active = 0;
        return;
    }

    // 5. 更新光线状态以进行下一次反弹（间接光照路径）
    payload->weight *= sphere->reflectance / survival_prob; // 更新权重（无偏估计）
    payload->origin = hit_point + shading_normal * 1e-3f; // 避免自相交 (1微米 in mm)
    payload->direction = sample_lambertian(shading_normal, &payload->seed);
    payload->active = 1;
}

// Miss 程序（光线逃逸）
extern "C" __global__ void __miss__sphere() {
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) | 
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);
    payload->active = 0;
}

// Ray generation 程序
extern "C" __global__ void __raygen__forward_trace() {
    const uint3 idx = optixGetLaunchIndex();
    unsigned int seed = params.seed_buffer[idx.x];

    // 从光源均匀采样方向 (各向同性点光源)
    float u1 = random_float(&seed);
    float u2 = random_float(&seed);
    float theta = 2.0f * M_PIf * u1;
    float phi = acosf(2.0f * u2 - 1.0f);
    float3 direction = make_float3(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));

    // 初始化 payload (无量纲weight)
    RayPayload payload;
    payload.origin = params.light_source.position;
    payload.direction = direction;
    payload.weight = 1.0f;  // 初始权重=1 (无量纲)
    payload.bounce_count = 0;
    payload.active = 1;
    payload.seed = seed;

    // 追踪光线 (循环多次反射)
    for (int i = 0; i < params.max_bounces && payload.active; ++i) {
        unsigned long long payload_ptr = reinterpret_cast<unsigned long long>(&payload);
        unsigned int p0 = static_cast<unsigned int>(payload_ptr);
        unsigned int p1 = static_cast<unsigned int>(payload_ptr >> 32);
        optixTrace(
            params.traversable,
            payload.origin, payload.direction,
            1e-4f, 1e16f, 0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0, // SBT offsets
            p0, p1
        );
    }

    params.seed_buffer[idx.x] = payload.seed;
}