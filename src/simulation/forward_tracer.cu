#include <optix.h>
#include "device_params.h"

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
            optixReportIntersection(t, 0);
        }
    }
}

// 最近命中程序
extern "C" __global__ void __closesthit__sphere() {
    const SphereSbtData* sphere = (SphereSbtData*)optixGetSbtDataPointer();
    RayPayload* payload = (RayPayload*)optixGetPayload_0();

    // 1. 计算命中点和法线
    float t_hit = optixGetRayTmax();
    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_point = ray_orig + t_hit * ray_dir;
    float3 normal = normalize(hit_point - sphere->center);

    // 2. 检查是否命中探测器 (Ray-Disk Intersection)
    // Is the hit point on the same plane as the detector?
    float3 to_detector = hit_point - params.detector.position;
    float dist_to_plane = dot(to_detector, params.detector.normal);

    // Check if the ray is behind the detector plane and going towards it
    if (dot(ray_dir, params.detector.normal) < 0 && fabsf(dist_to_plane) < 1e-5f) {
        // Is the hit point within the detector's radius?
        if (dot(to_detector, to_detector) < params.detector.radius * params.detector.radius) {
            float cos_theta = fmaxf(0.0f, dot(-ray_dir, params.detector.normal));
            float detected_flux = payload->power * cos_theta;
            atomicAdd(params.flux_buffer, detected_flux);
            payload->active = false;
            return;
        }
    }

    // 3. 反射次数检查 & 俄罗斯轮盘赌
    payload->bounce_count++;
    if (payload->bounce_count >= params.max_bounces) {
        payload->active = false;
        return;
    }
    float survival_prob = sphere->reflectance;
    if (random_float(&payload->seed) >= survival_prob) {
        payload->active = false;
        return;
    }

    // 4. 更新光线状态以进行下一次反弹
    payload->power /= survival_prob; // 能量守恒
    payload->origin = hit_point + normal * 1e-4f; // 避免自相交
    payload->direction = sample_lambertian(normal, &payload->seed);
    payload->active = true;
}

// Miss 程序（光线逃逸）
extern "C" __global__ void __miss__sphere() {
    RayPayload* payload = (RayPayload*)optixGetPayload_0();
    payload->active = false;
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

    // 初始化 payload
    RayPayload payload;
    payload.origin = params.light_source.position;
    payload.direction = direction;
    payload.power = params.power_per_ray;
    payload.bounce_count = 0;
    payload.active = true;
    payload.seed = seed;

    // 追踪光线 (循环多次反射)
    for (int i = 0; i < params.max_bounces && payload.active; ++i) {
        unsigned int p0 = (unsigned int)&payload;
        optixTrace(
            params.traversable,
            payload.origin, payload.direction,
            1e-4f, 1e16f, 0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0, // SBT offsets
            p0
        );
    }

    params.seed_buffer[idx.x] = payload.seed;
}